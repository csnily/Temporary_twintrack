import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A

class AnimalTrackDataset(Dataset):
    """
    AnimalTrack Dataset loader for TwinTrack.
    Loads video frames and MOT-format annotations.
    """
    def __init__(self, data_root, split='train', input_size=(1280, 736), transform=None, frame_interval=2):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.input_size = input_size
        self.frame_interval = frame_interval
        self.transform = transform or A.Compose([
            A.Resize(input_size[1], input_size[0]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ])
        self.img_dir = os.path.join(data_root, 'images', split)
        self.ann_path = os.path.join(data_root, 'annotations', f'{split}.txt')
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg') or f.endswith('.png')])
        self.anns = self._parse_annotations(self.ann_path)

    def _parse_annotations(self, ann_path):
        anns = {}
        with open(ann_path, 'r') as f:
            for line in f:
                items = line.strip().split(',')
                frame, tid, x, y, w, h = map(int, items[:6])
                if frame not in anns:
                    anns[frame] = {}
                anns[frame][tid] = [x, y, x + w, y + h]
        return anns

    def __len__(self):
        return len(self.img_files) // self.frame_interval

    def __getitem__(self, idx):
        frame_idx = idx * self.frame_interval
        img_name = self.img_files[frame_idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ann = self.anns.get(frame_idx + 1, {})
        transformed = self.transform(image=img)
        img = transformed['image']
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        boxes = []
        ids = []
        for tid, box in ann.items():
            boxes.append(box)
            ids.append(tid)
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        ids = torch.tensor(ids, dtype=torch.int64) if ids else torch.zeros((0,), dtype=torch.int64)
        return {
            'image': img,
            'boxes': boxes,
            'ids': ids,
            'frame_idx': frame_idx
        } 