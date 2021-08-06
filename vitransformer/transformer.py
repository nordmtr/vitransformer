import torch
import torch.nn as nn

from vitransformer.layers import PatchEmbedding, AttentionBlock


class VisionTransformer(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, emb_dim: int = 512, mlp_dim: int = 2048,
                 depth: int = 6, num_heads: int = 8, num_channels: int = 3, num_classes: int = 10,
                 dropout: float = 0.1, emb_dropout: float = 0.1, drop_path: float = 0.):
        super().__init__()

        self.embedding = PatchEmbedding(img_size, patch_size, emb_dim, num_channels, dropout=emb_dropout)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.transformer = nn.Sequential(*[
            AttentionBlock(emb_dim, num_heads, mlp_dim, dropout=dropout, drop_path=dpr[i]) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim, eps=1e-6)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.norm(x)
        x = self.classifier(x[:, 0])
        return x