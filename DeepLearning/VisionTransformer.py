from typing import Union, List

import torch
from einops import rearrange
from torch import nn, Tensor

from DeepLearning.Transformer import Encoder as TransformerEncoder


class ViT(nn.Module):
    """
    Vision Transformer: An image is worth 16x16 words, transformer for image recognition at scale

    References:
        1. https://arxiv.org/abs/2010.11929
    """

    def __init__(self, net_type: str, img_size: Union[int, List[int]], patch_size: int, d_model: int, nheads: int,
                 num_layers: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.net_type = net_type
        self.encoder = TransformerEncoder(d_model, nheads, num_layers=num_layers, dropout=dropout)
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.LazyLinear(d_model)
        self.cls_encoding = nn.Parameter(torch.randn(1, 1, d_model))
        # number of patches: (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, d_model))
        self.fc = nn.LazyLinear(num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # shape of x: [N, C, H, W], N -- batch size, C -- channels, H -- height, W -- width
        patches = rearrange(x, "n c (h px) (w py) -> n (h w) (px py c)", px=self.patch_size, py=self.patch_size)

        # project the patches: [N, num_patches, d_model]
        patches_proj = self.proj(patches)
        # print(patches_proj.shape, self.cls_encoding.shape)

        # please find the vit paper, there are 4 equations
        # now, equation 1: cat + pos
        # cls_token is at the first position
        out = torch.cat([self.cls_encoding.repeat(x.shape[0], 1, 1), patches_proj], dim=1) + self.pos_embedding

        # equation 2 and 3: msa + mlp = transformer encoder
        # note that in the original transformer paper, layer norm is at the end of msa and mlp (ffn)
        # in vit, layer norm is at first
        out = self.encoder(out)

        # equation 4
        # for classification task, take out the cls_token and give it to fc layer
        if self.net_type.lower() in ["classification", "cls", "clf"]:
            return self.fc(out[:, 0])
        else:
            return out[:, 1:]


if __name__ == "__main__":
    x = torch.randn(2, 3, 256, 256)
    f = ViT("clss", 256, 16, 512, 8, 6, 1000)
    y = f(x)
    print(y.shape)
