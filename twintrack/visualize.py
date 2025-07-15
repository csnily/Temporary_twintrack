import os
import argparse
import torch
from torch.utils.data import DataLoader
from twintrack.config import Config
from twintrack.models.backbone import Backbone
from twintrack.models.tlcfs import TLCFS
from twintrack.models.dltc import DLTC
from twintrack.datasets.dct import DCTDataset
from twintrack.datasets.animaltrack import AnimalTrackDataset
from twintrack.datasets.bucktales import BuckTalesDataset
from twintrack.datasets.harvardcow import HarvardCowDataset
from twintrack.utils.visualization import gradcam_visualization, plot_tracking_results, plot_failure_cases

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--mode', type=str, default='gradcam', choices=['gradcam', 'tracking', 'failure'])
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    cfg = Config()
    if args.dataset:
        cfg.dataset = args.dataset
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    # Dataset
    val_set = get_dataset(cfg.dataset, cfg.data_root, 'val', cfg)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    # Model
    backbone = Backbone(cfg.backbone, out_dim=cfg.out_dim).to(device)
    tlcfs = TLCFS(in_channels=cfg.out_dim, out_channels=cfg.out_dim).to(device) if cfg.use_tlcfs else None
    dltc = DLTC(embed_dim=cfg.out_dim, memory_len=cfg.memory_len).to(device) if cfg.use_dltc else None
    # Load checkpoint
    ckpt_path = args.checkpoint or os.path.join(cfg.checkpoint_dir, 'best.pth')
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(ckpt['backbone'])
    if tlcfs and ckpt['tlcfs']:
        tlcfs.load_state_dict(ckpt['tlcfs'])
    if dltc and ckpt['dltc']:
        dltc.load_state_dict(ckpt['dltc'])
    backbone.eval()
    if tlcfs: tlcfs.eval()
    if dltc: dltc.eval()
    # Visualization
    for batch in val_loader:
        img = batch['image'].to(device)
        if args.mode == 'gradcam':
            # Grad-CAM on backbone last layer
            cam = gradcam_visualization(backbone, img, backbone.layer4)
            import matplotlib.pyplot as plt
            plt.imshow(cam, cmap='jet')
            plt.title('Grad-CAM')
            plt.show()
            break
        elif args.mode == 'tracking':
            # Dummy: visualize tracking results (replace with real tracking)
            frames = [img[0].cpu().permute(1,2,0).numpy() * 255.0]
            tracks = [{0: [10,10,100,100]}]  # Dummy box
            plot_tracking_results(frames, tracks)
            break
        elif args.mode == 'failure':
            # Dummy: visualize failure cases
            frames = [img[0].cpu().permute(1,2,0).numpy() * 255.0]
            fail_events = [{'type': 'ID switch', 'frame': 0, 'info': None}]
            plot_failure_cases(frames, fail_events)
            break

if __name__ == '__main__':
    main() 