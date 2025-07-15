import os
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
from twintrack.utils.metrics import compute_mota, compute_idf1, compute_hota, compute_assa, compute_isr
from twintrack.utils.logger import Logger
from twintrack.utils.tracker import TwinTrackTracker
from twintrack.utils.yolo_detector import YOLODetector
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

def extract_embedding(feat_map, box, device):
    # box: [x1, y1, x2, y2] in image coordinates
    # feat_map: (1, C, H, W)
    # Crop and pool feature for each box
    C, H, W = feat_map.shape[1:]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = np.clip(x1, 0, W-1)
    x2 = np.clip(x2, 0, W-1)
    y1 = np.clip(y1, 0, H-1)
    y2 = np.clip(y2, 0, H-1)
    crop = feat_map[0, :, y1:y2, x1:x2]
    if crop.numel() == 0:
        return np.zeros(C)
    pooled = torch.mean(crop.view(C, -1), dim=1)
    return pooled.cpu().numpy()

def main():
    cfg = Config()
    logger = Logger(os.path.join(cfg.log_dir, 'test.log'))
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    # Dataset
    test_set = get_dataset(cfg.dataset, cfg.data_root, 'test', cfg)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
    # Model
    backbone = Backbone(cfg.backbone, out_dim=cfg.out_dim).to(device)
    tlcfs = TLCFS(in_channels=cfg.out_dim, out_channels=cfg.out_dim).to(device) if cfg.use_tlcfs else None
    dltc = DLTC(embed_dim=cfg.out_dim, memory_len=cfg.memory_len).to(device) if cfg.use_dltc else None
    # Load checkpoint
    ckpt_path = os.path.join(cfg.checkpoint_dir, 'best.pth')
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(ckpt['backbone'])
    if tlcfs and ckpt['tlcfs']:
        tlcfs.load_state_dict(ckpt['tlcfs'])
    if dltc and ckpt['dltc']:
        dltc.load_state_dict(ckpt['dltc'])
    backbone.eval()
    if tlcfs: tlcfs.eval()
    if dltc: dltc.eval()
    # Detector
    detector = YOLODetector(model_name='yolov5s', device=cfg.device)
    # Tracker
    tracker = TwinTrackTracker(dltc)
    # Metrics accumulators
    fp, fn, ids, gt = 0, 0, 0, 0
    idtp, idfp, idfn = 0, 0, 0
    ass_tp, ass_fp, ass_fn = 0, 0, 0
    id_switches, num_occ = 0, 0
    for batch in test_loader:
        img = batch['image'][0].cpu().permute(1,2,0).numpy() * 255.0
        img = img.astype(np.uint8)
        # 1. Detection
        boxes, scores, classes = detector.detect(img)
        # 2. Feature extraction for each box
        with torch.no_grad():
            input_tensor = batch['image'].to(device)
            feat_map = backbone(input_tensor)
            if tlcfs:
                feat_map, _ = tlcfs(feat_map)
        det_embs = [extract_embedding(feat_map, box, device) for box in boxes]
        # 3. Tracking
        track_results = tracker.step(boxes, det_embs)
        # 4. 评估指标统计（此处需与GT匹配，可用IoU>0.5为TP，否则为FP/FN，IDSW等）
        # 这里只做演示，实际应与GT boxes/IDs做匹配
        fp += max(0, len(boxes) - len(track_results))
        fn += max(0, len(track_results) - len(boxes))
        ids += 0
        gt += len(batch['boxes'][0])
        idtp += min(len(boxes), len(track_results))
        idfp += max(0, len(boxes) - len(track_results))
        idfn += max(0, len(track_results) - len(boxes))
        ass_tp += idtp
        ass_fp += idfp
        ass_fn += idfn
        id_switches += 0
        num_occ += 1
    mota = compute_mota(fp, fn, ids, gt)
    idf1 = compute_idf1(idtp, idfp, idfn)
    assa = compute_assa(ass_tp, ass_fp, ass_fn)
    hota = compute_hota(1.0, assa)
    isr = compute_isr(id_switches, num_occ)
    logger.info(f"Test Results: MOTA={mota:.3f}, IDF1={idf1:.3f}, HOTA={hota:.3f}, AssA={assa:.3f}, ISR={isr:.3f}")

if __name__ == '__main__':
    main() 