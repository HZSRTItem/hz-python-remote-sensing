# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2DLModels.py
@Time    : 2024/6/25 10:02
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2DLModels
-----------------------------------------------------------------------------"""
import os

import torch
from torch import nn

from SRTCodes.NumpyUtils import TensorSelectNames, NumpyDataCenter
from SRTCodes.PytorchUtils import convBnAct
from Shadow.Hierarchical import SHH2Config
from Shadow.Hierarchical.SHHMmif import TransformerBlock

NAMES = SHH2Config.NAMES


class SHH2DLModelInit(nn.Module):

    def __init__(self, *args, **kwargs):
        if "tsn" in kwargs:
            self.tsn = kwargs["tsn"]
            kwargs.pop("tsn")
        else:
            self.tsn = TensorSelectNames(*NAMES, dim=1)
        super().__init__(*args, **kwargs)
        self.xforward = lambda x: x

    def initTSN(self, *names, dim=1):
        self.tsn = TensorSelectNames(*names, dim=dim)


class Model(SHH2DLModelInit):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.in_channels = self.tsn.length()
        self.cba1 = nn.Sequential(
            convBnAct(self.in_channels, 128, 1, 1, 0),
            convBnAct(128, 128, 3, 1, 1),
        )
        self.max_pooling1 = nn.MaxPool2d(2, 2)
        self.cba2 = nn.Sequential(
            convBnAct(128, 256, 1, 1, 0),
            convBnAct(256, 256, 3, 1, 1),
        )
        self.conv_end = nn.Conv2d(256, 512, 3, 1, 0)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x):
        x = self.xforward(x)
        x = self.cba1(x)
        x = self.max_pooling1(x)
        x = self.cba2(x)
        x = self.conv_end(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class _STD_SpectralCNN(nn.Module):

    def __init__(self, in_channels, embed_dim, num_heads, num_blocks, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.to_ch = nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1, padding=0, bias=False)

        self.out = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=1, padding=0, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=2, stride=1, padding=0, bias=False),
            nn.Flatten(1),
        )

    def forward(self, x):
        x = self.to_ch(x)
        x = self.blocks(x)
        return x


class _STD_SpectralTransformer(nn.Module):

    def __init__(self, in_channels, embed_dim, num_heads, num_blocks, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.to_ch = nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1, padding=0, bias=False)

        self.blocks = nn.Sequential(*(TransformerBlock(
            dim=embed_dim, num_heads=num_heads,
            ffn_expansion_factor=2, bias=False, LayerNorm_type="WithBias"
        ) for _ in range(num_blocks)))

        self.out = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=1, padding=0, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=2, stride=1, padding=0, bias=False),
            nn.Flatten(1),
        )

    def forward(self, x):
        x = self.to_ch(x)
        x = self.blocks(x)
        return x


class _STD_Spectral(nn.Module):

    def __init__(self, blocks_type, in_channels, embed_dim, num_heads, num_blocks, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.to_ch = nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1, padding=0, bias=False)

        if blocks_type == "CNN":
            blocks = []
            for n in range(num_blocks):
                blocks.extend([
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(embed_dim),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(embed_dim * 2),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(embed_dim),
                ])
                if n != num_blocks - 1:
                    blocks.append(nn.ReLU())
            self.blocks = nn.Sequential(*blocks)

        elif blocks_type == "Transformer":
            self.blocks = nn.Sequential(*(TransformerBlock(
                dim=embed_dim, num_heads=num_heads,
                ffn_expansion_factor=2, bias=False, LayerNorm_type="WithBias"
            ) for _ in range(num_blocks)))

        else:
            self.blocks = lambda x: x

    def forward(self, x):
        x = self.to_ch(x)
        x = self.blocks(x)
        return x


class _STD_TextureCNN_ExtOne(nn.Module):

    def __init__(self, in_channels, patch_size, num_down, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def func_conv(kernel_size, ):
            return [
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=int(kernel_size / 2)),
                nn.BatchNorm2d(in_channels),
                nn.Sigmoid(),
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=int(kernel_size / 2)),
                nn.BatchNorm2d(in_channels),
                nn.Sigmoid(),
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=int(kernel_size / 2)),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
            ]

        def func_down():
            if patch_size == 5:
                return [
                    nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=1),
                    nn.BatchNorm2d(in_channels),
                ]
            elif patch_size == 11:
                return [
                    nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0),
                    nn.BatchNorm2d(in_channels),
                    nn.Sigmoid(),
                    nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=1),
                    nn.BatchNorm2d(in_channels),
                ]
            elif patch_size == 21:
                return [
                    nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0),
                    nn.BatchNorm2d(in_channels),
                    nn.Sigmoid(),
                    nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0),
                    nn.BatchNorm2d(in_channels),
                    nn.Sigmoid(),
                    nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=1),
                    nn.BatchNorm2d(in_channels),
                ]
            else:
                return []

        exts_k3, exts_k7, exts_k13 = [], [], []
        for i in range(num_down):
            exts_k3.extend(func_conv(3))
            exts_k7.extend(func_conv(7))
            exts_k13.extend(func_conv(13))

        self.exts_k3 = nn.Sequential(*exts_k3, *func_down())
        self.exts_k7 = nn.Sequential(*exts_k7, *func_down())
        self.exts_k13 = nn.Sequential(*exts_k13, *func_down())

    def forward(self, x):
        x3 = self.exts_k3(x)
        x7 = self.exts_k7(x)
        x13 = self.exts_k13(x)
        x = torch.cat([x3, x7, x13], dim=1)
        return x


class _STD_TextureCNN_Ext(nn.Module):

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        """
        1. size   x -> [b, c, 5, 5] [b, c, 11, 11] [b, c, 21, 21]
        2. kernel x -> conv3d [b, 2, 2, h, w] -> [b, 2, 1, h, w]
           21 10 5 3
              11 5 3
                 5 3
        """

        self.ndc5 = NumpyDataCenter(4, (5, 5), (21, 21))
        self.exts_size5 = _STD_TextureCNN_ExtOne(in_channels=in_channels, patch_size=5, num_down=2)

        self.ndc11 = NumpyDataCenter(4, (11, 11), (21, 21))
        self.exts_size11 = _STD_TextureCNN_ExtOne(in_channels=in_channels, patch_size=11, num_down=2)

        self.exts_size21 = _STD_TextureCNN_ExtOne(in_channels=in_channels, patch_size=21, num_down=2)

    def forward(self, x):
        x5 = self.ndc5.fit(x)
        x5 = self.exts_size5(x5)

        x11 = self.ndc11.fit(x)
        x11 = self.exts_size11(x11)

        x21 = self.exts_size21(x)

        x = torch.cat([x5, x11, x21], dim=1)
        return x


class _STD_TextureCNN(nn.Module):

    def __init__(self, is_3d=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_3d = is_3d
        self.conv = nn.Conv3d(2, 2, kernel_size=(2, 3, 3), stride=1, padding=(0, 1, 1))
        self.ext = _STD_TextureCNN_Ext(2)

    def forward(self, x):
        if self.is_3d:
            x = self.conv(x)
            x = torch.squeeze(x, dim=2)
        x = self.ext(x)
        return x


class SHH2MOD_SpectralTextureDouble(SHH2DLModelInit):

    def __init__(
            self,
            in_channels,
            num_category,
            blocks_type="CNN",
            embed_dim=64,
            num_blocks=4,
            num_heads=8,
            is_texture=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.is_texture = is_texture

        self.spectral = _STD_Spectral(
            blocks_type=blocks_type,
            in_channels=self.in_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
        )
        out_dim = 64

        if self.is_texture:
            self.texture_opt = _STD_TextureCNN(False)
            self.texture_bs = _STD_TextureCNN()
            self.texture_c2 = _STD_TextureCNN()
            self.texture_ha = _STD_TextureCNN()
            out_dim = 136

        self.out = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=2, stride=1, padding=0, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(out_dim, out_dim * 2, kernel_size=2, stride=1, padding=0, bias=False),
            nn.Flatten(1),
        )

        self.fc = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim * 4),
            nn.Linear(out_dim * 4, out_dim),
            nn.Linear(out_dim, num_category),
        )

    def forward(self, x):
        x_coll = self.xforward(x)

        x1 = self.spectral(x_coll[0])

        if self.is_texture:

            x2 = self.texture_opt(x_coll[1])
            x3 = self.texture_bs(x_coll[2])
            x4 = self.texture_c2(x_coll[3])
            x5 = self.texture_ha(x_coll[4])

            x = torch.cat([
                x1,
                x2, x3, x4, x5
            ], dim=1)
        else:
            x = x1

        x = self.out(x)
        x = self.fc(x)
        return x


class VHLModel(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(VHLModel, self).__init__()
        self.conv_end = nn.Conv2d(in_channel, out_channel, 3, 1, 1)

    def forward(self, x):
        x = self.conv_end(x)
        logit = torch.rand(32, 4)
        return x, logit


class ISModel(nn.Module):

    def __init__(self, in_channel):
        super(ISModel, self).__init__()

    def forward(self, x):
        logit = torch.rand(32, 2)
        return logit


class WSModel(nn.Module):

    def __init__(self):
        super(WSModel, self).__init__()

    def forward(self, x):
        logit = torch.rand(32, 2)
        return logit


class FCModel(nn.Module):

    def __init__(self):
        super(FCModel, self).__init__()
        self.vhl_mod = VHLModel()
        self.is_mod = ISModel()
        self.ws_mod = WSModel()


def x_Model():
    mod = Model()
    x = torch.rand(32, 86, 7, 7)
    out_x = mod(x)
    torch.save(mod, r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\Temp\tmp.pth")
    return


def main():
    def func1():
        embed_dim = 6
        model = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=1, padding=0, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=2, stride=1, padding=0, bias=False),
        )
        x = torch.rand(8, 6, 3, 3)
        print(model(x).shape)

    def func2():
        tsn = TensorSelectNames(*SHH2Config.NAMES, dim=1)
        tsn.addTSN("OPT", ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", ])
        tsn.addTSN("BS", ["AS_VV", "DE_VV", "AS_VH", "DE_VH"])
        tsn.addTSN("C2", ["AS_C11", "DE_C11", "AS_C22", "DE_C22"])
        tsn.addTSN("HA", ["AS_H", "DE_H", "AS_Alpha", "DE_Alpha"])

        ndc3 = NumpyDataCenter(4, (3, 3), (21, 21))

        def to_3d(x):
            x1 = torch.zeros(x.shape[0], 2, 2, x.shape[2], x.shape[3])
            x1[:, 0, :, :, :] = x[:, [0, 1], :, :]
            x1[:, 1, :, :, :] = x[:, [2, 3], :, :]
            return x1

        def xforward(x: torch.Tensor):
            x_opt = tsn["OPT"].fit(x)
            x_bs = tsn["BS"].fit(x)
            x_c2 = tsn["C2"].fit(x)
            x_ha = tsn["HA"].fit(x)
            x0 = ndc3.fit(torch.cat([x_opt, x_bs, x_c2, x_ha], dim=1))
            return x0, x_opt[:, [2, 3], :, :], to_3d(x_bs), to_3d(x_c2), to_3d(x_ha)

        model = SHH2MOD_SpectralTextureDouble(tsn.length(), 4)
        model.xforward = xforward

        data = torch.rand(10, 86, 21, 21)
        out_x = model(data)
        print(model)
        print(out_x.shape)
        to_fn = r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\Temp\SHH2MOD_SpectralTextureDouble.pth"
        torch.save(model.state_dict(), to_fn)
        stats = os.stat(to_fn)
        print(stats.st_size / 1024 / 1024, "mb")

    def func3():
        model = _STD_Spectral(
            blocks_type="Transformer",
            in_channels=18,
            embed_dim=64,
            num_heads=8,
            num_blocks=4,
        )
        x = torch.rand(10, 18, 3, 3)
        out_x = model(x)
        print(model)
        print(out_x.shape)
        to_fn = r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\Temp\model.pth"
        torch.save(model, to_fn)
        stats = os.stat(to_fn)
        print(stats.st_size / 1024 / 1024, "mb")

    def func4():
        x = torch.rand(10, 3, 11, 11)
        model = nn.Conv2d(3, 3, 2, 2)
        # x = model(x)
        out_x = model(x)
        print(out_x.shape)

    def func5():
        model = _STD_TextureCNN_Ext(2)
        x = torch.rand(10, 2, 21, 21)
        out_x = model(x)
        print(model)
        print(out_x.shape)
        to_fn = r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\Temp\_STD_TextureCNN_Ext.pth"
        torch.save(model, to_fn)
        stats = os.stat(to_fn)
        print(stats.st_size / 1024 / 1024, "mb")

    func2()
    pass


if __name__ == "__main__":
    main()
