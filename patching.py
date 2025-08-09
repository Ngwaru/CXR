from torch import nn
from einops.layers.torch import Rearrange
from torch import tensor

class PatchEmbedding(nn.Module):
    def __init__(self, num_channels=3, patch_size = 8, embed_size =128):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size*patch_size*num_channels, embed_size)
        )

    def forward(self, x ):
        x = self.projection(x)
        return x
