import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from twintrack.config import Config
from twintrack.models.backbone import Backbone
from twintrack.models.tlcfs import TLCFS
from twintrack.models.dltc import DLTC
from twintrack.losses.tel import tracking_enhancement_loss
from twintrack.datasets.dct import DCTDataset
from twintrack.datasets.animaltrack import AnimalTrackDataset
from twintrack.datasets.bucktales import BuckTalesDataset
from twintrack.datasets.harvardcow import HarvardCowDataset
from twintrack.utils.logger import Logger
import random
import numpy as np

def get_dataset(name, root, split, cfg):
    if name == 'DCT':
        return DCTDataset(root, split, cfg.input_size, frame_interval=cfg.frame_interval)
    elif name == 'AnimalTrack':
        return AnimalTrackDataset(root, split, cfg.input_size, frame_interval=cfg.frame_interval)
    elif name == 'BuckTales':
        return BuckTalesDataset(root, split, cfg.input_size, frame_interval=cfg.frame_interval)
    elif name == 'HarvardCow':
        return HarvardCowDataset(root, split, cfg.input_size, frame_interval=cfg.frame_interval)
    else:
        raise ValueError(f"Unknown dataset: {name}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    cfg = Config()
    set_seed(cfg.seed)
    logger = Logger(os.path.join(cfg.log_dir, 'train.log'))
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    # Dataset
    train_set = get_dataset(cfg.dataset, cfg.data_root, 'train', cfg)
    val_set = get_dataset(cfg.dataset, cfg.data_root, 'val', cfg)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    # Model
    backbone = Backbone(cfg.backbone, out_dim=cfg.out_dim).to(device)
    tlcfs = TLCFS(in_channels=cfg.out_dim, out_channels=cfg.out_dim).to(device) if cfg.use_tlcfs else None
    dltc = DLTC(embed_dim=cfg.out_dim, memory_len=cfg.memory_len).to(device) if cfg.use_dltc else None
    # Optimizer
    params = list(backbone.parameters())
    if tlcfs: params += list(tlcfs.parameters())
    if dltc: params += list(dltc.parameters())
    optimizer = optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if cfg.amp else None
    # Training loop
    best_loss = float('inf')
    for epoch in range(cfg.epochs):
        backbone.train()
        if tlcfs: tlcfs.train()
        if dltc: dltc.train()
        total_loss = 0
        for batch in train_loader:
            imgs = batch['image'].to(device)
            # Forward
            feats = backbone(imgs)
            if tlcfs:
                feats, _ = tlcfs(feats)
            # Dummy: flatten features for DLTC (simulate per-object embedding)
            B, C, H, W = feats.shape
            obj_embeds = feats.mean(dim=[2,3])  # (B, C)
            if dltc:
                # For demo, use previous as zeros
                O_prev = torch.zeros_like(obj_embeds)
                O_t, _, _, _ = dltc(obj_embeds, O_prev)
            else:
                O_t = obj_embeds
            # Dummy: fake aug_feats/trajs for TEL
            aug_feats = O_t.detach() + 0.01 * torch.randn_like(O_t)
            trajs = torch.randn(B, 2, 2, device=device)  # (N, T, 2)
            # Loss
            loss, loss_dict = tracking_enhancement_loss(O_t.unsqueeze(1), aug_feats.unsqueeze(1), trajs.unsqueeze(1), lambda_feat=cfg.tel_lambda)
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{cfg.epochs} | Train Loss: {avg_loss:.4f}")
        # Save checkpoint
        if avg_loss < best_loss and cfg.save_best:
            best_loss = avg_loss
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            torch.save({
                'backbone': backbone.state_dict(),
                'tlcfs': tlcfs.state_dict() if tlcfs else None,
                'dltc': dltc.state_dict() if dltc else None,
                'epoch': epoch,
                'loss': avg_loss
            }, os.path.join(cfg.checkpoint_dir, 'best.pth'))
            logger.info(f"Best model saved at epoch {epoch+1}")

if __name__ == '__main__':
    main() 