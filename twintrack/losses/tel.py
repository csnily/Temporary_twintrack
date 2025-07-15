import torch
import torch.nn.functional as F

def tracking_enhancement_loss(pred_feats, aug_feats, trajs, lambda_feat=0.6, eps=1e-6):
    """
    Tracking Enhancement Loss (TEL)
    Args:
        pred_feats: (N, T, D) predicted features for each object and frame
        aug_feats: (N, T, D) augmented/occlusion-aware features
        trajs: (N, T, 2) trajectory positions (x, y)
        lambda_feat: weight for feature consistency loss
        eps: small value to avoid division by zero
    Returns:
        total_loss: scalar
        dict: {'traj_loss': ..., 'feat_loss': ...}
    """
    # Trajectory smoothness loss
    diff = trajs[:, 1:, :] - trajs[:, :-1, :]
    norm = torch.norm(trajs[:, :-1, :], dim=-1) + eps
    traj_loss = torch.mean(torch.tanh(torch.norm(diff, dim=-1) / norm) ** 2)
    # Feature consistency loss (log-L2)
    feat_diff = torch.norm(pred_feats - aug_feats, dim=-1) ** 2
    feat_loss = torch.mean(torch.log(1 + feat_diff))
    total_loss = traj_loss + lambda_feat * feat_loss
    return total_loss, {'traj_loss': traj_loss, 'feat_loss': feat_loss} 