import torch
import torch.nn as nn
import torch.nn.functional as F

class DLTC(nn.Module):
    """
    Dynamic Long-Term Temporal Consistency (DLTC) Module
    - Dynamic frame weighting (MLP + Tanh)
    - LSTM-gated update with two-frame concat
    - Memory bank with attention retrieval
    - Gated memory update
    """
    def __init__(self, embed_dim=256, memory_len=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_len = memory_len
        # Dynamic frame weighting
        self.weight_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.Tanh()
        )
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size=embed_dim * 2, hidden_size=embed_dim, batch_first=True)
        # Memory bank (initialized empty)
        self.register_buffer('memory', torch.zeros(memory_len, embed_dim))
        # Attention projection
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # Gated memory update
        self.z_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.r_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.h_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, O_t, O_prev, memory=None):
        # O_t: (B, D), O_prev: (B, D)
        B, D = O_t.shape
        if memory is None:
            memory = self.memory.unsqueeze(0).expand(B, -1, -1)  # (B, L, D)
        # Dynamic frame weighting (dummy features for demo)
        rho_norm = O_t.norm(dim=1, keepdim=True)
        rho_delta = (O_t - O_prev).norm(dim=1, keepdim=True)
        rho_assoc = (O_t * O_prev).sum(dim=1, keepdim=True) / (O_t.norm(dim=1) * O_prev.norm(dim=1) + 1e-6)
        weight_input = torch.cat([rho_norm, rho_delta, rho_assoc], dim=1)
        W_t = (self.weight_mlp(weight_input) + 1) / 2  # (B, D), in [0,1]
        O_t_weighted = O_t * W_t
        # LSTM-gated update
        lstm_in = torch.cat([O_prev, O_t_weighted], dim=1).unsqueeze(1)  # (B, 1, 2D)
        lstm_out, _ = self.lstm(lstm_in)
        O_t_prime = lstm_out.squeeze(1)  # (B, D)
        # Attention retrieval from memory
        Q = self.q_proj(O_t_prime).unsqueeze(1)  # (B, 1, D)
        K = self.k_proj(memory)  # (B, L, D)
        V = self.v_proj(memory)  # (B, L, D)
        attn = torch.softmax((Q @ K.transpose(1, 2)) / (D ** 0.5), dim=-1)  # (B, 1, L)
        O_t_updated = (attn @ V).squeeze(1)  # (B, D)
        # Gated memory update
        mem_cat = torch.cat([memory, O_t_updated.unsqueeze(1).expand(-1, self.memory_len, -1)], dim=2)  # (B, L, 2D)
        z_t = torch.sigmoid(self.z_proj(mem_cat))  # (B, L, D)
        r_t = torch.sigmoid(self.r_proj(mem_cat))  # (B, L, D)
        m_tilde = torch.tanh(self.h_proj(torch.cat([r_t * memory, O_t_updated.unsqueeze(1).expand(-1, self.memory_len, -1)], dim=2)))
        memory_new = z_t * memory + (1 - z_t) * m_tilde  # (B, L, D)
        # Optionally, roll memory and insert new embedding at the end
        memory_new = torch.cat([memory_new[:, 1:], O_t_updated.unsqueeze(1)], dim=1)
        # Tracking embedding refinement (can be extended)
        E_track = O_t_updated
        return O_t_updated, memory_new, E_track, W_t 