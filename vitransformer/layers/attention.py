import torch.nn as nn

from .drop_path import DropPath


class AttentionBlock(nn.Module):
    def __init__(
        self, emb_dim: int, num_heads: int, hidden_dim: int = 2048, dropout: float = 0.0, drop_path: float = 0.0
    ):
        """
        Inputs:
            emb_dim - Dimensionality of input and attention feature vectors
            num_heads - Number of heads to use in the Multi-Head Attention block
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than emb_dim)
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_norm_2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.drop_path(self.attn(inp_x, inp_x, inp_x)[0])
        x = x + self.drop_path(self.mlp(self.layer_norm_2(x)))
        return x
