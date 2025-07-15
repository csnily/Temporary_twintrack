import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

class TwinTrackTracker:
    """
    Simple multi-object tracker for TwinTrack.
    Supports IoU+embedding association, Hungarian assignment, ID management, DLTC memory.
    """
    def __init__(self, dltc, iou_thresh=0.3, emb_thresh=0.5, max_age=30):
        self.dltc = dltc
        self.iou_thresh = iou_thresh
        self.emb_thresh = emb_thresh
        self.max_age = max_age
        self.tracks = []  # List of dict: {id, box, emb, age, memory}
        self.next_id = 1

    def iou(self, boxA, boxB):
        # box: [x1, y1, x2, y2]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = areaA + areaB - inter + 1e-6
        return inter / union

    def associate(self, det_boxes, det_embs):
        N, M = len(self.tracks), len(det_boxes)
        if N == 0 or M == 0:
            return [], list(range(M)), list(range(N))
        cost = np.zeros((N, M))
        for i, trk in enumerate(self.tracks):
            for j, (box, emb) in enumerate(zip(det_boxes, det_embs)):
                iou_score = self.iou(trk['box'], box)
                emb_score = np.dot(trk['emb'], emb) / (np.linalg.norm(trk['emb']) * np.linalg.norm(emb) + 1e-6)
                cost[i, j] = 1 - (0.5 * iou_score + 0.5 * emb_score)
        row, col = linear_sum_assignment(cost)
        matches, unmatched_det, unmatched_trk = [], [], []
        for i in range(N):
            if i not in row:
                unmatched_trk.append(i)
        for j in range(M):
            if j not in col:
                unmatched_det.append(j)
        for r, c in zip(row, col):
            if cost[r, c] < (1 - self.iou_thresh):
                matches.append((r, c))
            else:
                unmatched_trk.append(r)
                unmatched_det.append(c)
        return matches, unmatched_det, unmatched_trk

    def step(self, det_boxes, det_embs):
        # det_boxes: list of [x1, y1, x2, y2], det_embs: list of np.array
        matches, unmatched_det, unmatched_trk = self.associate(det_boxes, det_embs)
        # Update matched tracks
        for trk_idx, det_idx in matches:
            self.tracks[trk_idx]['box'] = det_boxes[det_idx]
            self.tracks[trk_idx]['emb'] = det_embs[det_idx]
            self.tracks[trk_idx]['age'] = 0
            # DLTC memory update (可选)
        # Create new tracks
        for det_idx in unmatched_det:
            self.tracks.append({
                'id': self.next_id,
                'box': det_boxes[det_idx],
                'emb': det_embs[det_idx],
                'age': 0,
                'memory': None
            })
            self.next_id += 1
        # Age and remove lost tracks
        new_tracks = []
        for trk in self.tracks:
            trk['age'] += 1
            if trk['age'] <= self.max_age:
                new_tracks.append(trk)
        self.tracks = new_tracks
        # Return current tracks
        return [{'id': trk['id'], 'box': trk['box']} for trk in self.tracks] 