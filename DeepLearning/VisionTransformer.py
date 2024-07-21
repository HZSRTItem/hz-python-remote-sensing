from functools import partial
from typing import Union, List

import torch
from einops import rearrange
from torch import nn, Tensor
from torchsummary import summary
from torchvision.models import VisionTransformer

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


class VisionTransformerChannel(VisionTransformer):
    """
    https://blog.csdn.net/damadashen/article/details/

    _vision_transformer(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=weights,
        progress=progress,
        **kwargs,
    )

    """

    def __init__(
            self,
            in_channels: int,
            image_size: int,
            patch_size: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            num_classes: int = 1000,
            representation_size=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            conv_stem_configs=None
    ):
        super().__init__(image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, dropout,
                         attention_dropout, num_classes, representation_size, norm_layer, conv_stem_configs)
        self.conv_proj = nn.Conv2d(
            in_channels=in_channels, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor):
        return super(VisionTransformerChannel, self).forward(x)


if __name__ == "__main__":
    # x = torch.randn(2, 3, 256, 256)
    # f = ViT("clss", 256, 16, 512, 8, 6, 1000)
    # y = f(x)
    # print(y.shape)
    x = torch.rand(10, 18, 18, 18).to("cuda")
    model = VisionTransformerChannel(
        in_channels=18,
        image_size=18,
        patch_size=6,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        num_classes=4
    ).to("cuda")
    out_x = model(x)
    params = list(model.parameters())
    print(out_x.shape)
    # torch.save(model.state_dict(), r"F:\Week\20240721\Data\model.mod")

