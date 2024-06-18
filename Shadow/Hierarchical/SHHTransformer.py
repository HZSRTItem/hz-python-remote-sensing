# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHHTransformer.py
@Time    : 2024/3/24 18:41
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHHTransformer
-----------------------------------------------------------------------------"""

import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional

import numpy as np
import torch
from osgeo import gdal
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.vision_transformer import ConvStemConfig, Encoder, VisionTransformer

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.Utils import timeDirName, DirFileName, writeCSVLine, copyFile, FN, Jdt, changext
from Shadow.Hierarchical import SHHConfig
from Shadow.Hierarchical.ShadowHSample import SHH2Samples, copySHH2Samples, loadSHH2SamplesDataset


class VIT_WS(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
            self,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            num_classes: int = 1000,
            representation_size: Optional[int] = None,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            conv_stem_configs: Optional[List[ConvStemConfig]] = None,
    ):
        super().__init__()

        self.image_size = 3
        self.patch_size = 1
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        seq_length = (self.image_size // self.patch_size) ** 2

        self.conv1d_1 = nn.Conv1d(in_channels=9, out_channels=self.hidden_dim, kernel_size=2)

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1
        """
        (10,
         12,
         12,
         81,
         1028,
         0.2,
         0.2
        (10,
         12,
         12,
         120,
         1028,
         0.2,
         0.2,
        """
        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        # if isinstance(self.conv_proj, nn.Conv2d):
        #     # Init the patchify stem
        #     fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        #     nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        #     if self.conv_proj.bias is not None:
        #         nn.init.zeros_(self.conv_proj.bias)
        # elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
        #     # Init the last 1x1 conv of the conv stem
        #     nn.init.normal_(
        #         self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
        #     )
        #     if self.conv_proj.conv_last.bias is not None:
        #         nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        # torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        # torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        # x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, c, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)
        x = self.conv1d_1(x)
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)

        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x


class VisionTransformer_Channel(VisionTransformer):

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
        return super(VisionTransformer_Channel, self).forward(x)


class VIT_WS_T1(nn.Module):

    def __init__(self):
        super(VIT_WS_T1, self).__init__()

        self.vit = VIT_WS(
            num_layers=12,
            num_heads=12,
            hidden_dim=120,
            mlp_dim=514,
            dropout=0.2,
            attention_dropout=0.2,
            num_classes=8,
            representation_size=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            conv_stem_configs=None,
        )

    def forward(self, x):
        x = self.vit(x)
        return x


def shh2VITTrain():
    coor_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\11\sh2_spl11_2.csv"

    # to_fn = numberfilename(coor_fn)
    # copyFile(coor_fn, to_fn)
    # for geo_fn in SHHConfig.SHH2_QD1_FNS:
    #     grf = GDALSamplingFast(geo_fn)
    #     grf.csvfile(to_fn, to_fn)
    #     print(to_fn)

    def data_deal(x, y=None):
        out_x = np.zeros((10, x.shape[1], x.shape[2]))
        out_x[0:2] = x[0:2] / 30 + 1
        out_x[2:4] = x[3:5] / 30 + 1
        out_x[4:] = x[6:] / 3000
        return out_x, y

    init_dirname = r"F:\ProjectSet\Shadow\Hierarchical\WSModels"
    batch_size = 64
    epochs = 200
    device = "cuda"
    model = VIT_WS_T1().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    n_test = 10
    print("model:", model)
    print("init_dirname:", init_dirname)

    def train():
        shh_spl = SHH2Samples()
        shh_spl.addCSV(r"F:\ProjectSet\Shadow\Hierarchical\Samples\11\sh2_spl11_21.csv")
        # shh_spl.initXY()
        # shh_spl.sampling(SHHConfig.GRS_SHH2_IMAGE1_FNS(), (21, 21))
        # shh_spl.toNpy(r"F:\ProjectSet\Shadow\Hierarchical\Samples\11\sh2_spl11_21_data.npy")
        shh_spl.loadNpy(r"F:\ProjectSet\Shadow\Hierarchical\Samples\11\sh2_spl11_21_data.npy")
        shh_spl.ndc.__init__(3, (3, 3), (21, 21))
        shh_spl.initCategory("CATEGORY", map_dict={12: 1}, others=0)
        shh_spl = copySHH2Samples(shh_spl).addSamples(shh_spl.filterEQ("SH_IMDC", 2))
        train_ds, test_ds = loadSHH2SamplesDataset(shh_spl, data_deal=data_deal)

        print("length train_ds:", len(train_ds))
        print("length test_ds:", len(test_ds))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        print("length train_loader:", len(train_loader))
        print("length test_loader:", len(test_loader))

        is_fit = True
        test_cm = ConfusionMatrix(2, ["NOIS", "IS"])
        curr_dirname = timeDirName(init_dirname, True)
        cuur_dfn = DirFileName(curr_dirname)
        log_fn = cuur_dfn.fn("log.txt")
        writeCSVLine(log_fn, ["EPOCH", "BATCH", "LOSS", "OA", "FN"])
        copyFile(__file__, FN(__file__).changedirname(curr_dirname))
        print("curr_dirname:", curr_dirname)
        print("log_fn:", log_fn)

        def log():
            print("EPOCH: {0:>3d} BATCH: {1:>5d} LOSS: {2:>10.6f} OA: {3:>10.6f}".format(
                epoch, batchix, float(loss.cpu().item()), test_cm.OA()))
            fn = cuur_dfn.fn("model_{0}_{1}.pth".format(epoch, batchix))
            writeCSVLine(log_fn, [epoch, batchix, float(loss.cpu().item()), test_cm.OA(), fn])
            return fn

        def saveModel(to_fn):
            torch.save(model.state_dict(), to_fn)

        def tAcc():
            test_cm.clear()
            model.eval()
            with torch.no_grad():
                for i, (x, y) in enumerate(test_loader):
                    x = x.to(device).float()
                    _logts = model(x)
                    y1 = torch.argmax(_logts, dim=1)
                    y = y.numpy() + 1
                    test_cm.addData(y, y1)
            model.train()
            return test_cm.OA()

        def epochTAcc(is_save=False):
            if test_loader is not None:
                if not is_save:
                    if batchix % n_test == 0:
                        tAcc()
                        modname = log()
                        if is_save:
                            print("MODEL:", modname)
                            saveModel(modname)
                else:
                    tAcc()
                    modname = log()
                    print("MODEL:", modname)
                    saveModel(modname)

        if not is_fit:
            return

        for epoch in range(epochs):

            for batchix, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                x, y = x.float(), y.long()

                model.train()
                logts = model(x)
                loss = criterion(logts, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epochTAcc(False)

            epochTAcc(True)

    def predict():
        gr = GDALRaster(SHHConfig.SHH2_QD1_FNS[0])
        image_data = gr.readAsArray()
        image_data, y = data_deal(image_data)
        image_data = image_data.astype("float32")
        image_data = image_data[:, 2000:3000, 2000:3000]

        mod_fn = r"F:\ProjectSet\Shadow\Hierarchical\WSModels\20240325H085738\model_4_29.pth"
        model.load_state_dict(torch.load(mod_fn))
        model.to(device)

        imdc = np.zeros(image_data.shape[1:])
        row_start, row_end, = 10, imdc.shape[0] - 10
        column_start, column_end, = 10, imdc.shape[1] - 10
        win_row = 3
        win_column = 3
        win_row_2, win_column_2 = int(win_row / 2), int(win_column / 2)

        jdt = Jdt(row_end - row_start, "predict").start(is_jdt=True)
        for i in range(row_start, row_end):
            data_line = []

            for j in range(column_start, column_end):
                r, c = i, j
                data_line.append(
                    [image_data[:, r - win_row_2:r + win_row_2 + 1, c - win_column_2:c + win_column_2 + 1]])

            data_line = np.concatenate(data_line)
            data_line = torch.from_numpy(data_line).to(device)

            logit = model(data_line)
            y = torch.argmax(logit, dim=1)
            y = y.cpu().numpy()
            imdc[i, column_start: column_end] = y + 1

            jdt.add(is_jdt=True)
        jdt.end(is_jdt=True)

        filename = changext(mod_fn, "_imdc.tif")
        driver = gdal.GetDriverByName("GTiff")  # 申请空间
        dst_ds = driver.Create(filename, imdc.shape[1], imdc.shape[0], 1, gdal.GDT_Byte, [])  # 列数 行数 波段数
        band: gdal.Band = dst_ds.GetRasterBand(1)
        band.WriteArray(imdc)

    predict()


def main():
    x = torch.rand((32, 10, 3, 3))

    vit = VIT_WS(
        num_layers=12,
        num_heads=12,
        hidden_dim=120,
        mlp_dim=514,
        dropout=0.2,
        attention_dropout=0.2,
        num_classes=2,
        representation_size=None,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs=None,
    )
    logits = vit(x)
    torch.save(vit.state_dict(), r"F:\ProjectSet\Shadow\Hierarchical\Mods\Temp\tmp1.pth")

    # mod = VisionTransformer_Channel(
    #     in_channels=10,
    #     image_size=3,
    #     patch_size=1,
    #     num_layers=12,
    #     num_heads=12,
    #     hidden_dim=120,
    #     mlp_dim=1028,
    #     dropout=0.2,
    #     attention_dropout=0.2,
    #     num_classes=2,
    #     representation_size=None,
    #     norm_layer=partial(nn.LayerNorm, eps=1e-6),
    #     conv_stem_configs=None,
    # )
    # logits = mod(x)

    pass


if __name__ == "__main__":
    shh2VITTrain()
