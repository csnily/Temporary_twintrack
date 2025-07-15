import torch
import torch.nn as nn
import torch.nn.functional as F

class TLCFS(nn.Module):
    """
    Twin-Level Contextual Feature Synthesizer (TLCFS)
    - Fine-grained branch: 3x3 conv for low-level details
    - Semantic branch: 5x5 conv for high-level context
    - Dynamic fusion: MLP + softmax to adaptively weight branches per frame
    - Optional: Multi-head self-attention within each branch
    """
    def __init__(self, in_channels=3, out_channels=256, attn_heads=4, attn_dim=64):
        super().__init__()
        # Fine-grained branch
        self.fine_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # Semantic branch
        self.semantic_conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        # Multi-head self-attention for each branch
        self.fine_attn = nn.MultiheadAttention(out_channels, attn_heads, batch_first=True)
        self.semantic_attn = nn.MultiheadAttention(out_channels, attn_heads, batch_first=True)
        # Dynamic fusion weights (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(2 * out_channels, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )
        self.out_channels = out_channels

    def forward(self, x):
        # Fine-grained features
        f1 = self.fine_conv(x)  # (B, C, H, W)
        # Semantic features
        f2 = self.semantic_conv(x)
        # Flatten spatial for attention: (B, C, H, W) -> (B, HW, C)
        B, C, H, W = f1.shape
        f1_flat = f1.flatten(2).transpose(1, 2)  # (B, HW, C)
        f2_flat = f2.flatten(2).transpose(1, 2)
        # Self-attention within each branch
        f1_attn, _ = self.fine_attn(f1_flat, f1_flat, f1_flat)
        f2_attn, _ = self.semantic_attn(f2_flat, f2_flat, f2_flat)
        # Restore spatial
        f1_attn = f1_attn.transpose(1, 2).reshape(B, C, H, W)
        f2_attn = f2_attn.transpose(1, 2).reshape(B, C, H, W)
        # Global Average Pooling + L2 norm
        g1 = F.normalize(f1_attn.mean(dim=[2, 3]), p=2, dim=1)  # (B, C)
        g2 = F.normalize(f2_attn.mean(dim=[2, 3]), p=2, dim=1)
        # Dynamic fusion weights
        h = torch.cat([g1, g2], dim=1)  # (B, 2C)
        a = self.mlp(h)  # (B, 2)
        alpha = F.softmax(a, dim=1)  # (B, 2)
        # Weighted sum
        alpha1 = alpha[:, 0].view(B, 1, 1, 1)
        alpha2 = alpha[:, 1].view(B, 1, 1, 1)
        fused = alpha1 * f1_attn + alpha2 * f2_attn
        return fused, (f1_attn, f2_attn, alpha) 