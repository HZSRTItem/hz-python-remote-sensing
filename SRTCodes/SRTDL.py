# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SRTDL.py
@Time    : 2024/6/6 10:59
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SRTDL
-----------------------------------------------------------------------------"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from SRTCodes.GDALRasterIO import GDALRasterChannel
from SRTCodes.GDALUtils import GDALSamplingFast, GDALSampling
from SRTCodes.SRTModelImage import SRTModImPytorch
from SRTCodes.Utils import writeTexts, changext


class DLModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(3, 16, 3)

    def forward(self, x):
        x = self.conv(x)
        return x


class DLDataset(Dataset):

    def __init__(self):
        self.data_list = []
        self.y_list = []

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        x = self.data_list[item]
        y = self.y_list[item]
        x, y = data_deal(x, y)
        return x, y


def data_deal(x, y=None):
    if y is None:
        return x
    return x, y


def loadDS():
    csv_fn = ""

    df = pd.read_csv(csv_fn)
    to_fn = changext(csv_fn, "_data.npy")
    if os.path.isfile(to_fn):
        data = np.load(to_fn)
    else:
        grf = GDALSampling(
            "", # geo raster
        )
        data = grf.sampling2(df["X"].to_list(), df["Y"].to_list(), 39, 39, is_trans=True)
        np.save(to_fn, data)
    train_ds = DLDataset()
    test_ds = DLDataset()

    for i in range(len(df)):
        if df["TEST"][i] == 1:
            train_ds.data_list.append(data[i])
            train_ds.y_list.append(df["CATEGORY"][i] - 1)
        else:
            test_ds.data_list.append(data[i])
            test_ds.y_list.append(df["CATEGORY"][i] - 1)
    return train_ds, test_ds



class DeepLearning:
    """ DeepLearning

    change:
        self.name: save name
        self.smip.model_dirname: model save dir name
        self.smip.class_names: class names
        self.smip.win_size: win size of data predict
        self.color_table: color table of image classification

    """

    def __init__(self):
        self.name = "DL"
        self.smip = SRTModImPytorch()

        def func_predict(model, x: torch.Tensor):
            logit = model(x)
            y = torch.argmax(logit, dim=1) + 1
            return y

        self.func_predict = func_predict
        self.color_table = {1: (0, 0, 0), 2: (255, 255, 255)}
        writeTexts(os.path.join(self.smip.model_dirname, "sys.argv.txt"), sys.argv, mode="a")

    def main(self):
        self.smip.model_dirname = None
        self.smip.model_name = "Model"
        self.smip.epochs = 100
        self.smip.device = "cuda:0"
        self.smip.n_test = 5
        self.smip.batch_size = 32
        self.smip.class_names = ["NO"]
        self.smip.n_class = len(self.smip.class_names)
        self.smip.win_size = (39, 39)
        self.smip.model = DLModel().to(self.smip.device)
        self.smip.func_predict = self.func_predict
        self.smip.func_y_deal = lambda y: y
        self.smip.initColorTable(self.color_table)
        return

    def train(self):
        self.smip.timeDirName()
        self.smip.train_ds, self.smip.test_ds = loadDS()
        self.smip.initTrainLog()
        self.smip.initPytorchTraining()
        self.smip.pt.func_logit_category = self.func_predict
        self.smip.pt.func_y_deal = lambda y: y + 1
        self.smip.initModel()
        self.smip.initDataLoader()
        self.smip.initCriterion(nn.CrossEntropyLoss())
        self.smip.initOptimizer(torch.optim.Adam, lr=0.001, eps=10e-5)
        self.smip.pt.addScheduler(torch.optim.lr_scheduler.StepLR(self.smip.pt.optimizer, 20, gamma=0.5, last_epoch=-1))
        self.smip.copyFile(__file__)
        self.smip.print()
        self.smip.train()

    def imdc(self):
        self.smip.loadPTH(None)
        # gr = GDALRaster(r"H:\ChengJiXiangMu\LTB_2024_14_tif.tif")
        # data = gr.readAsArray()
        # self.smip.imdcData(data, data_deal)

        grc: GDALRasterChannel = GDALRasterChannel()
        grc.addGDALDatas(
            r"H:\ChengJiXiangMu\LTB_2024_14_tif.tif",
            # r"H:\ChengJiXiangMu\clip_510110天府新区0304.tif",
        )
        self.smip.imdc(grc=grc, is_jdt=True, data_deal=data_deal)

        # self.smip.imdcTiles(
        #     # to_fn=r"F:\Week\20240331\Data\20240329H185618\Net2_epoch2_imdc5.tif",
        #     tiles_dirname=r"H:\ChengJiXiangMu\clip_510110_tiles",
        #     data_deal=data_deal,
        # )

        # self.smip.imdcTiles(
        #     # to_fn=r"F:\Week\20240331\Data\20240329H185618\Net2_epoch2_imdc5.tif",
        #     tiles_dirname=r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\cd_sh2_1_retile",
        #     data_deal=data_deal,
        # )


def main():
    pass


if __name__ == "__main__":
    main()
