from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbedding(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        emb_dim: int = 768,
        num_channels: int = 3,
        norm_layer: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.img_size = img_size = (img_size, img_size)
        self.patch_size = patch_size = (patch_size, patch_size)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(num_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(emb_dim) if norm_layer else nn.Identity()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1 + self.num_patches, emb_dim))  # + cls_token
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        _, _, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.norm(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.dropout(x + self.pos_embedding)
        return x
