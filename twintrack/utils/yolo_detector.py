import torch
import numpy as np

class YOLODetector:
    """
    YOLOv5/YOLOv8 detector wrapper for TwinTrack.
    Automatically downloads weights, supports inference.
    """
    def __init__(self, model_name='yolov5s', device='cuda'):
        try:
            from yolov5 import YOLOv5
            self.model = YOLOv5(model_name, device=device)
            self.is_v5 = True
        except ImportError:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_name)
                self.is_v5 = False
            except ImportError:
                raise ImportError('Please install yolov5 or ultralytics package!')
        self.device = device

    def detect(self, image):
        """
        Args:
            image: np.ndarray (H, W, 3), RGB, 0-255
        Returns:
            boxes: list of [x1, y1, x2, y2]
            scores: list of float
            classes: list of int
        """
        if self.is_v5:
            results = self.model.predict(image)
            boxes = results.xyxy[0][:, :4].cpu().numpy() if len(results.xyxy) > 0 else np.zeros((0, 4))
            scores = results.xyxy[0][:, 4].cpu().numpy() if len(results.xyxy) > 0 else np.zeros((0,))
            classes = results.xyxy[0][:, 5].cpu().numpy().astype(int) if len(results.xyxy) > 0 else np.zeros((0,), dtype=int)
        else:
            results = self.model(image)
            boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else np.zeros((0, 4))
            scores = results[0].boxes.conf.cpu().numpy() if len(results[0].boxes) > 0 else np.zeros((0,))
            classes = results[0].boxes.cls.cpu().numpy().astype(int) if len(results[0].boxes) > 0 else np.zeros((0,), dtype=int)
        return boxes, scores, classes 