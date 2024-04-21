# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHHMmif.py
@Time    : 2024/3/7 14:14
@Author  : Zheng Han
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHHMmif

code G:\src\MMIF-CDDFuse\MMIF-CDDFuse-main
-----------------------------------------------------------------------------"""

import numbers
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from osgeo import gdal
from torch.utils.data import Dataset, DataLoader

from SRTCodes.GDALRasterClassification import GDALRasterPrediction
from SRTCodes.GDALRasterIO import getArraySize, saveGDALRaster, tiffAddColorTable
from SRTCodes.PytorchModelTraining import PytorchCategoryTraining
from SRTCodes.Utils import changext
from Shadow.Hierarchical import SHHConfig
from Shadow.Hierarchical.SHHRunMain import SHHMainInit
from Shadow.Hierarchical.ShadowHSample import ShadowHierarchicalSampleCollection, loadSHHSamples


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
                    torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False, ):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.proj(out)
        return out


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 ffn_expansion_factor=2,
                 bias=False):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,
                 qkv_bias=False, ):
        super(BaseFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias, )
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim, ffn_expansion_factor=int(ffn_expansion_factor), )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
                                    stride=1, padding=0, bias=True)

    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)

    def forward(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


def to_3d(x):
    """ Layer Norm """
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    """ Gated-Dconv Feed-Forward Network (GDFN) """

    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    """ Multi-DConv Head Transposed Self-Attention (MDTA) """

    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Overlapped image patch embedding with 3x3 Conv """

    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Restormer_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=None,
                 heads=None,
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(Restormer_Encoder, self).__init__()

        if heads is None:
            heads = [8, 8, 8]
        if num_blocks is None:
            num_blocks = [4, 4]
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        self.detailFeature = DetailFeatureExtraction()

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        base_feature = self.baseFeature(out_enc_level1)
        detail_feature = self.detailFeature(out_enc_level1)
        return base_feature, detail_feature, out_enc_level1


class Restormer_Decoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=None,
                 heads=None,
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Decoder, self).__init__()
        if heads is None:
            heads = [8, 8, 8]
        if num_blocks is None:
            num_blocks = [4, 4]

        self.reduce_channel = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)

        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim) // 2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim) // 2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias), )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp_img, base_feature, detail_feature):
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1), out_enc_level0


# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Net_model(nn.Module):

    def __init__(self):
        super(Net_model, self).__init__()

        def func_conv2d(in_c):
            return nn.Sequential(
                nn.Conv2d(in_c, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
            )

        self.conv_as = func_conv2d(2)
        self.conv_de = func_conv2d(2)
        self.conv_rgb = func_conv2d(3)
        self.conv_nir = func_conv2d(1)

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=4)
        self.fc = nn.Linear(64, 8)

    def forward(self, x):
        x_as, x_de = x[:, 0:2, :, :], x[:, 3:5, :, :]
        x_rgb, x_nir = x[:, 6:9, :, :], x[:, 10:11, :, :]

        x_as = self.conv_as(x_as)
        x_de = self.conv_de(x_de)
        x_rgb = self.conv_rgb(x_rgb)
        x_nir = self.conv_nir(x_nir)

        x = torch.concatenate([x_as, x_de, x_rgb, x_nir, ], dim=1)
        x = self.relu1(x)
        x = self.conv1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


def t_func():
    x = torch.rand(32, 12, 13, 13)
    mod = Net_model()
    out_d = mod(x)
    torch.save(mod, r"F:\ProjectSet\Shadow\Hierarchical\Mods\MMIF\mod2.pth")
    return


class MMIF_ASDE(nn.Module):

    def __init__(self):
        super(MMIF_ASDE, self).__init__()

        inp_channels = 1
        out_channels = 1
        dim = 64
        num_blocks = [4, 4]
        heads = [8, 8, 8]
        ffn_expansion_factor = 2
        bias = False
        LayerNorm_type = 'WithBias'

        self.conv1_as = nn.Conv2d(2, 64, kernel_size=1, stride=1)
        self.conv1_de = nn.Conv2d(2, 64, kernel_size=1, stride=1)

        self.transformers_front = nn.Sequential(
            *[TransformerBlock(
                dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type
            ) for i in range(num_blocks[0])]
        )

        self.full_feat = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        self.full_feat_conv = nn.Conv2d(128, 64, kernel_size=1, stride=1)
        self.detail_feat = DetailFeatureExtraction()
        self.detail_feat_conv = nn.Conv2d(128, 64, kernel_size=1, stride=1)

        self.transformers_end = nn.Sequential(
            *[TransformerBlock(
                dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type
            ) for i in range(num_blocks[1])]
        )

    def forward(self, x_as, x_de):
        x_as = self.conv1_as(x_as)
        x_as = self.transformers_front(x_as)
        x_as_full = self.full_feat(x_as)
        x_as_detail = self.full_feat(x_as)

        x_de = self.conv1_de(x_de)
        x_de = self.transformers_front(x_de)
        x_de_full = self.full_feat(x_de)
        x_de_detail = self.full_feat(x_de)

        x_full = torch.cat([x_as_full, x_de_full], dim=1)
        x_full = self.full_feat_conv(x_full)
        x_detail = torch.cat([x_as_detail, x_de_detail], dim=1)
        x_detail = self.detail_feat_conv(x_detail)

        x_full = self.transformers_end(x_full)
        x_detail = self.transformers_end(x_detail)

        return x_full, x_detail


class MMIF_Opt(nn.Module):

    def __init__(self):
        super(MMIF_Opt, self).__init__()

        inp_channels = 1
        out_channels = 1
        dim = 64
        num_blocks = [4, 4]
        heads = [8, 8, 8]
        ffn_expansion_factor = 2
        bias = False
        LayerNorm_type = 'WithBias'

        self.conv1_opt_rgb = nn.Conv2d(3, 64, kernel_size=1, stride=1)
        self.conv1_opt_nir = nn.Conv2d(1, 64, kernel_size=1, stride=1)

        self.transformers_front = nn.Sequential(
            *[TransformerBlock(
                dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type
            ) for i in range(num_blocks[0])]
        )

        self.full_feat = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        self.full_feat_conv = nn.Conv2d(128, 64, kernel_size=1, stride=1)
        self.detail_feat = DetailFeatureExtraction()
        self.detail_feat_conv = nn.Conv2d(128, 64, kernel_size=1, stride=1)

        self.transformers_end = nn.Sequential(
            *[TransformerBlock(
                dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type
            ) for i in range(num_blocks[1])]
        )

    def forward(self, x_rgb, x_nir):
        x_rgb = self.conv1_opt_rgb(x_rgb)
        x_rgb = self.transformers_front(x_rgb)
        x_rgb_full = self.full_feat(x_rgb)
        x_rgb_detail = self.full_feat(x_rgb)

        x_nir = self.conv1_opt_nir(x_nir)
        x_nir = self.transformers_front(x_nir)
        x_nir_full = self.full_feat(x_nir)
        x_nir_detail = self.full_feat(x_nir)

        x_full = torch.cat([x_rgb_full, x_nir_full], dim=1)
        x_full = self.full_feat_conv(x_full)
        x_detail = torch.cat([x_rgb_detail, x_nir_detail], dim=1)
        x_detail = self.detail_feat_conv(x_detail)

        x_full = self.transformers_end(x_full)
        x_detail = self.transformers_end(x_detail)

        return x_full, x_detail


class MMIF_Cat(nn.Module):

    def __init__(self):
        super(MMIF_Cat, self).__init__()

        self.sar_net = MMIF_ASDE()
        self.opt_net = MMIF_Opt()

        self.full_feat = BaseFeatureExtraction(dim=64, num_heads=8)
        self.detail_feat = DetailFeatureExtraction()

        def func_cscr(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid(),
                nn.Conv2d(out_c, out_c, kernel_size=2, stride=2),
                nn.ReLU(),
            )

        # self.cscr1 = func_cscr(128, 256)
        # self.cscr2 = func_cscr(256, 512)
        # self.conv_end = nn.Conv2d(512, 512, kernel_size=3, stride=1)
        #
        # self.fc1 = nn.Linear(512, 8, bias=False)

    def forward(self, x):
        x_as, x_de = x[:, 0:2, :, :], x[:, 3:5, :, :]
        x_sar_full, x_sar_detail = self.sar_net(x_as, x_de)

        x_rgb, x_nir = x[:, 6:9, :, :], x[:, 10:11, :, :]
        x_opt_full, x_opt_detail = self.opt_net(x_rgb, x_nir)

        x_full = x_sar_full + x_opt_full
        x_full = self.full_feat(x_full)
        x_detail = x_sar_detail + x_opt_detail
        x_detail = self.detail_feat(x_detail)

        x = torch.cat([x_full, x_detail], dim=1)

        # x = self.cscr1(x)
        # x = self.cscr2(x)
        # x = self.conv_end(x)
        #
        # x = torch.flatten(x, start_dim=1)
        # x = self.fc1(x)
        return x


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc_data = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(
        img1 ** 2, dim=-1)) * torch.sqrt(torch.sum(img2 ** 2, dim=-1)))
    cc_data = torch.clamp(cc_data, -1., 1.)
    return cc_data.mean()


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_vis, image_ir, generate_img):
        image_y = image_vis[:, :1, :, :]
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, generate_img)
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
        loss_total = loss_in + 10 * loss_grad
        return loss_total, loss_in, loss_grad


def splitData(data: np.ndarray, stride=128, win_size=128):
    data_list = []
    for i in range(0, data.shape[1], stride):
        for j in range(0, data.shape[2], stride):
            d = data[:, i:i + win_size, j:j + win_size]
            if d.shape[1:] == (win_size, win_size):
                data_list.append(d)
    return data_list


def splitData2D(data: np.ndarray, stride=128, win_size=128):
    data_list = []
    n_rows, n_columns = 0, 0
    for i in range(0, data.shape[1], stride):
        row_list = []
        n_columns = 0
        for j in range(0, data.shape[2], stride):
            d = data[:, i:i + win_size, j:j + win_size]
            if d.shape[1:] == (win_size, win_size):
                row_list.append(d)
        if row_list:
            data_list.append(row_list)
            n_rows += 1
            n_columns = len(row_list)
    return data_list, n_rows, n_columns


class MMIF_ImageFuse:

    def __init__(self, ckpt_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
        self.Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
        self.BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
        self.DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

        mod_dict = torch.load(ckpt_path)
        self.Encoder.load_state_dict(mod_dict['DIDF_Encoder'])
        self.Decoder.load_state_dict(mod_dict['DIDF_Decoder'])
        self.BaseFuseLayer.load_state_dict(mod_dict['BaseFuseLayer'])
        self.DetailFuseLayer.load_state_dict(mod_dict['DetailFuseLayer'])

        self.Encoder.eval()
        self.Decoder.eval()
        self.BaseFuseLayer.eval()
        self.DetailFuseLayer.eval()

    def fit(self, data_IR, data_VIS):
        data_IR, data_VIS = torch.FloatTensor(data_IR), torch.FloatTensor(data_VIS)
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

        feature_V_B, feature_V_D, feature_V = self.Encoder(data_VIS)
        feature_I_B, feature_I_D, feature_I = self.Encoder(data_IR)
        feature_F_B = self.BaseFuseLayer(feature_V_B + feature_I_B)
        feature_F_D = self.DetailFuseLayer(feature_V_D + feature_I_D)
        data_Fuse, _ = self.Decoder(data_VIS, feature_F_B, feature_F_D)
        data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))
        fi = np.squeeze((data_Fuse * 255).cpu().numpy())

        return fi


def saveGEORaster(d, geo_fn=None, copy_geo_fn=None, fmt="ENVI", dtype=gdal.GDT_Float32, geo_transform=None,
                  probing=None, interleave='band', options=None, descriptions=None):
    """ Save geo image
    \
    :param geo_fn: save geo file name
    :param copy_geo_fn: get geo_transform probing in this geo file
    :param descriptions: descriptions
    :param options: save options list
    :param interleave: The data is organized as `band`:(b,y,x) or `pixel`:(x,y,b)
    :param probing: projection information
    :param geo_transform: projection transformation information
    :param d: data
    :param fmt: save type
    :param dtype: save data type default:gdal.GDT_Float32
    :return: None
    """

    ds = gdal.Open(copy_geo_fn)
    if options is None:
        options = []
    if geo_transform is None:
        geo_transform = ds.GetGeoTransform()
    if probing is None:
        probing = ds.GetProjection()
    band_count, n_column, n_row = getArraySize(d.shape, interleave)
    saveGDALRaster(d, n_row, n_column, band_count, dtype, fmt, geo_transform, interleave, options, probing,
                   geo_fn, descriptions)


class MMIF_GDALImdc:

    def __init__(self, geo_fn=None, win_size=128) -> None:
        if geo_fn is None:
            geo_fn = r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\qd_sh2_1.tif"
        self.geo_fn = geo_fn
        ds: gdal.Dataset = gdal.Open(geo_fn, gdal.GA_ReadOnly)
        self.data = ds.ReadAsArray()
        if len(self.data.shape) == 2:
            self.data = np.array([self.data])
        self.imdc = np.zeros(self.data.shape[1:])
        data_list, n_rows, n_columns = splitData2D(self.data, stride=win_size, win_size=win_size)
        self.data = data_list
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.win_size = win_size

    def get(self, row, column):
        return self.data[row][column]

    def set(self, data, row, column):
        win_size = self.win_size
        self.imdc[row * win_size:(row + 1) * win_size, column * win_size:(column + 1) * win_size] = data

    def save(self, to_geo_fn):
        saveGEORaster(self.imdc, to_geo_fn, self.geo_fn, fmt="GTiff", options=["COMPRESS=PACKBITS"])


class MMIF_TrainDataset(Dataset):

    def __init__(self, geo_fn=None, stride=128, win_size=128) -> None:
        super().__init__()

        if geo_fn is None:
            geo_fn = r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\qd_sh2_1.tif"
        ds: gdal.Dataset = gdal.Open(geo_fn, gdal.GA_ReadOnly)
        self.data = ds.ReadAsArray()
        self.data = splitData(self.data, stride=stride, win_size=win_size)

    def __getitem__(self, index):
        d = self.data[index]
        vv_as = np.array([d[0]]) / 40.0
        vv_de = np.array([d[3]]) / 40.0
        return vv_as, vv_de

    def __len__(self):
        return len(self.data)


def loadTrainDataLoader(batch_size=8):
    ds = MMIF_TrainDataset()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return dl


def mmif_dealData(x: np.ndarray, y=None):
    if np.sum(x) == 0:
        return
    x = x + 0
    x[0:2, :, :] = (x[0:2, :, :] + 40.0) / 80.0
    x[3:5, :, :] = (x[3:5, :, :] + 40.0) / 80.0
    x[6:, :, :] = x[6:, :, :] / 3000.0

    if y is not None:
        y = y - 1
        return x, y
    return x


class MMIF_SHHDataset(Dataset):

    def __init__(self, shh_sc: ShadowHierarchicalSampleCollection):
        super(MMIF_SHHDataset, self).__init__()

        self.shh_sc: ShadowHierarchicalSampleCollection = shh_sc
        self.shh_sc.ndc.__init__(3, (13, 13), (21, 21))
        # self.shh_sc.initVegHighLowCategoryCollMap()

    def __getitem__(self, index):
        x = self.shh_sc.data(index, is_center=True)
        y = self.shh_sc.category(index)
        x, y = mmif_dealData(x, y)
        x = x.astype("float32")
        return x, y

    def __len__(self):
        return len(self.shh_sc)


class SHHModel_MMIF_ptt(PytorchCategoryTraining):

    def __init__(self, model_dir=None, model_name="model", n_category=2, category_names=None, epochs=10, device=None,
                 n_test=100):
        super().__init__(model_dir=model_dir, model_name=model_name, n_category=n_category,
                         category_names=category_names, epochs=epochs, device=device, n_test=n_test)

    def logisticToCategory(self, logts):
        logts = torch.argmax(logts, dim=1) + 1
        return logts

    def train(self, batch_save=False, epoch_save=True, *args, **kwargs):

        self._initTrain()
        self._printModel()
        self._log.saveHeader()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        lr = 1e-4
        weight_decay = 0

        clip_grad_norm_value = 0.01
        optim_step = 20
        optim_gamma = 0.5

        # self.model = Net_model().to(device)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=optim_step, gamma=optim_gamma)
        # self.criterion = nn.CrossEntropyLoss()

        print("len train loader", len(self.train_loader))
        print("len test loader", len(self.test_loader))

        for epoch in range(self.epochs):

            for batchix, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                x, y = x.float(), y.long()

                self.model.train()
                # self.model.zero_grad()

                logts = self.model(x)
                self.loss = self.criterion(logts, y)

                self.optimizer.zero_grad()
                self.loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                self.optimizer.step()

                self.batchTAcc(batch_save, batchix, epoch)

            self.epochTAcc(epoch, epoch_save)

            if self.scheduler is not None:
                self.scheduler.step()

    def testAccuracy(self, deal_y_func=None):
        return super(SHHModel_MMIF_ptt, self).testAccuracy(lambda y: y + 1)


class SHHModel_MMIF_GRP(GDALRasterPrediction):

    def __init__(self, geo_fn):
        super(SHHModel_MMIF_GRP, self).__init__(geo_fn)
        self.device = "cuda:0"
        self.is_category = False
        self.number_pred = 15000

    def predict(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        # y = np.ones(x.shape[0])
        n = x.shape[0]
        x = torch.from_numpy(x)
        x = x.float()
        x = x.to(self.device)
        y = torch.zeros((n), dtype=torch.float)
        y = y.to(self.device)

        self.model.eval()
        with torch.no_grad():
            for i in range(0, n, self.number_pred):
                y_temp = self.model(x[i:i + self.number_pred, :])
                y_temp = torch.argmax(y_temp, dim=1)
                y[i:i + self.number_pred] = y_temp

        y = y + 1
        y = y.cpu().numpy()

        return y

    def preDeal(self, row, column_start, column_end):
        d_row = self.d[:, row, column_start:column_end]
        ndvi = (d_row[3, :] - d_row[2, :]) / (d_row[3, :] + d_row[2, :])
        # np.ones(d_row.shape[1], dtype="bool")
        return ndvi < 0.5


class SHHModel_MMIF_Main(SHHMainInit):

    def __init__(self):
        super().__init__()

        self.n_category = 8
        self.category_names = ['NOT_KNOW', 'IS', 'VEG', 'SOIL', 'WAT', 'IS_SH', 'VEG_SH', 'SOIL_SH', 'WAT_SH']
        self.epochs = 200
        self.n_test = 10
        self.win_size = 13
        self.batch_size = 64
        self.mod = MMIF_Cat()

        self.geo_raster = self.qd_geo_raster

        self.code_text_filename = __file__

    def loadSamplesDS(self):
        shh_sc_train, shh_sc_test = loadSHHSamples("sample1[21,21]")
        self.n_category = len(shh_sc_test.category_coll) - 1
        self.category_names = shh_sc_test.category_coll.keys()[1:]
        print("n category:", self.n_category)
        print("category names:", self.category_names)
        self.train_ds = MMIF_SHHDataset(shh_sc_train)
        self.test_ds = MMIF_SHHDataset(shh_sc_test)
        return None

    def train(self, *args, **kwargs):
        super(SHHModel_MMIF_Main, self).train(SHHModel_MMIF_ptt)

    def samplesCategory(self, mod_fn=None, logit_cate_func=None, spl_csv_fn=None, *args, **kwargs):
        super(SHHModel_MMIF_Main, self).samplesCategory()

    def imdcOne(self, mod_fn=None, to_imdc_name=None, grp: GDALRasterPrediction = None,
                data_deal=None, code_colors=None,
                *args, **kwargs):

        # mod_fn = r"F:\ProjectSet\Shadow\Hierarchical\Mods\20240315H220508\model_epoch_86.pth"
        # to_imdc_name = "cd_imdc"
        # grp = SHHModel_MMIF_GRP(self.cd_geo_raster)

        data_deal = mmif_dealData
        code_colors = SHHConfig.SHH_COLOR8

        if code_colors is None:
            code_colors = SHHConfig.SHH_COLOR8
        if mod_fn is None:
            mod_fn = sys.argv[1]
        if to_imdc_name is None:
            to_imdc_name = sys.argv[2]
        imdc_fn = changext(mod_fn, "_{0}.tif".format(to_imdc_name) )

        self.mod_dirname = os.path.dirname(mod_fn)
        self.mod.load_state_dict(torch.load(mod_fn))
        self.mod.to(self.device)
        self.mod.eval()

        print("mod_dirname:", self.mod_dirname)
        print("imdc_fn    :", imdc_fn)
        print("mod_fn     :", mod_fn)

        grp.is_category = True
        np_type = "int8"

        grp.run(imdc_fn=imdc_fn, np_type=np_type, mod=self.mod,
                spl_size=[self.win_size, self.win_size],
                row_start=self.win_size + 6, row_end=-(self.win_size + 6),
                column_start=self.win_size + 6, column_end=-(self.win_size + 6),
                n_one_t=15000, data_deal=data_deal)
        tiffAddColorTable(imdc_fn, code_colors=code_colors)


def imdc1():

    n_mod_fn = 50
    if sys.argv[1] == "1":
        n_mod_fn = 50
    elif sys.argv[1] == "2":
        n_mod_fn = 100
    elif sys.argv[1] == "3":
        n_mod_fn = 150
    elif sys.argv[1] == "4":
        n_mod_fn = 198

    print("n_mod_fn:", n_mod_fn)
    mod_fn = r"F:\ProjectSet\Shadow\Hierarchical\Mods\20240315H221134\model_epoch_{0}.pth".format(n_mod_fn)
    print("mod_fn:", mod_fn)

    shh_main = SHHModel_MMIF_Main()

    if sys.argv[2] == "qd":
        to_imdc_name = "qd_imdc"
        grp = SHHModel_MMIF_GRP(shh_main.qd_geo_raster)
    elif sys.argv[2] == "bj":
        to_imdc_name = "bj_imdc"
        grp = SHHModel_MMIF_GRP(shh_main.bj_geo_raster)
    elif sys.argv[2] == "cd":
        to_imdc_name = "cd_imdc"
        grp = SHHModel_MMIF_GRP(shh_main.cd_geo_raster)
    else:
        to_imdc_name = "None"
        grp = SHHModel_MMIF_GRP(shh_main.cd_geo_raster)

    print("to_imdc_name:", to_imdc_name)
    print("grp:", grp)

    shh_main.imdcOne(mod_fn, to_imdc_name=to_imdc_name, grp=grp)


def fuse():
    mod_fn = r"G:\src\MMIF-CDDFuse\MMIF-CDDFuse-main\models\CDDFuse_03-06-22-33.pth"
    mmif_imdc = MMIF_GDALImdc()
    MMIF_imf = MMIF_ImageFuse(mod_fn)
    for i in range(mmif_imdc.n_rows):
        print(i)
        for j in range(mmif_imdc.n_columns):
            data = mmif_imdc.get(i, j)
            out_d = MMIF_imf.fit(data[0], data[3])
            mmif_imdc.set(out_d, i, j)
    mmif_imdc.save("{0}_1.tif".format(mod_fn))
    return


def main():
    # SHHModel_MMIF_Main().train()

    # height = 128
    # width = 128
    # window_size = 8
    # modelE = Restormer_Encoder().cuda()
    # modelD = Restormer_Decoder().cuda()

    # F:\ProjectSet\Shadow\Hierarchical\Mods\MMIF
    # x = torch.rand(8, 64, 13, 13)
    #
    # trans1 = TransformerBlock(dim=64, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
    # out_x = trans1(x)
    # torch.save(trans1, r"G:\src\MMIF-CDDFuse\MMIF-CDDFuse-main\models\mod1.pth")

    # out_x = mod(x)
    # print(out_x.shape)
    # torch.save(mod, r"F:\ProjectSet\Shadow\Hierarchical\Mods\MMIF\mod1.pth")

    mod = MMIF_Cat()
    x = torch.rand(8, 64, 13, 13)
    out_d = mod(x)
    print(out_d.shape)
    print(mod)
    torch.save(mod, r"F:\ProjectSet\Shadow\Hierarchical\Mods\MMIF\mod3.pth")

    # mod = Net_model()
    # x = torch.rand(8, 64, 13, 13)
    # out_d = mod(x)
    # print(out_d.shape)
    # torch.save(mod, r"F:\ProjectSet\Shadow\Hierarchical\Mods\MMIF\mod2.pth")

    # run_line = """python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHMmif import imdc1; imdc1()" """
    # for i in range(1, 5):
    #     for city in ["qd", "bj", "cd"]:
    #         print(run_line, i, city)


if __name__ == '__main__':
    """
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHMmif import SHHModel_MMIF_Main; SHHModel_MMIF_Main().train()"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHMmif import SHHModel_MMIF_Main; SHHModel_MMIF_Main().imdcOne()"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHMmif import imdc1; imdc1()" 
    """
    main()
    pass
