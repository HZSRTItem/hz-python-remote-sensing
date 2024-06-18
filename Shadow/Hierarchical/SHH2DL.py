# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2DL.py
@Time    : 2024/6/6 10:26
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2DL
-----------------------------------------------------------------------------"""
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from SRTCodes.GDALUtils import GDALSampling
from SRTCodes.NumpyUtils import NumpyDataCenter, TensorSelectNames
from SRTCodes.PytorchUtils import convBnAct
from SRTCodes.SRTModelImage import SRTModImPytorch
from SRTCodes.Utils import changext, SRTLog


NAMES = [
    "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "NDVI", "NDWI",
    "OPT_asm", "OPT_con", "OPT_cor", "OPT_dis", "OPT_ent", "OPT_hom", "OPT_mean", "OPT_var",
    "AS_VV", "AS_VH", "AS_angle", "AS_VHDVV",
    "AS_C11", "AS_C12_imag", "AS_C12_real", "AS_C22", "AS_Lambda1", "AS_Lambda2", "AS_SPAN",
    "AS_Epsilon", "AS_Mu", "AS_RVI", "AS_m", "AS_Beta",
    "AS_H", "AS_A", "AS_Alpha",
    "AS_VH_asm", "AS_VH_con", "AS_VH_cor", "AS_VH_dis", "AS_VH_ent", "AS_VH_hom", "AS_VH_mean", "AS_VH_var",
    "AS_VV_asm", "AS_VV_con", "AS_VV_cor", "AS_VV_dis", "AS_VV_ent", "AS_VV_hom", "AS_VV_mean", "AS_VV_var",
    "DE_VV", "DE_VH", "DE_angle", "DE_VHDVV",
    "DE_C11", "DE_C12_imag", "DE_C12_real", "DE_C22", "DE_SPAN", "DE_Lambda1", "DE_Lambda2",
    "DE_Epsilon", "DE_Mu", "DE_RVI", "DE_m", "DE_Beta",
    "DE_H", "DE_A", "DE_Alpha",
    "DE_VH_asm", "DE_VH_con", "DE_VH_cor", "DE_VH_dis", "DE_VH_ent", "DE_VH_hom", "DE_VH_mean", "DE_VH_var",
    "DE_VV_asm", "DE_VV_con", "DE_VV_cor", "DE_VV_dis", "DE_VV_ent", "DE_VV_hom", "DE_VV_mean", "DE_VV_var",
]


class Model(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tsn = TensorSelectNames(*NAMES, dim=1)
        self.tsn.addTSN("OPT", ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", ])
        self.tsn.addTSN("AS", ["AS_VV", "AS_VH"])
        self.tsn.addTSN("DE", ["DE_VV", "DE_VH"])

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
        x_opt = self.tsn["OPT"].fit(x)
        x_opt = x_opt / 3000
        x_as = self.tsn["AS"].fit(x)
        x_as = (x_as + 30) / 60
        x_de = self.tsn["DE"].fit(x)
        x_de = (x_de + 30) / 60
        x = torch.cat([x_opt, x_as, x_de], dim=1)
        x = self.cba1(x)
        x = self.max_pooling1(x)
        x = self.cba2(x)
        x = self.conv_end(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
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


# x_Model()


class DLDataset(Dataset):

    def __init__(self, win_size, read_size):
        self.data_list = []
        self.y_list = []
        self.ndc = NumpyDataCenter(3, win_size, read_size)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        x = self.data_list[item]
        x = self.ndc.fit(x)
        y = self.y_list[item]
        x, y = data_deal(x, y)
        return x, y


def data_deal(x, y=None):
    if y is None:
        return x
    return x, y


def loadDS(win_size, to_csv_fn=None):
    csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\sh2_spl25_2_spl2_3.csv"
    read_size = 21, 21

    df = pd.read_csv(csv_fn)
    if to_csv_fn is not None:
        df.to_csv(to_csv_fn, index=False)
    to_fn = changext(csv_fn, "_data.npy")
    if os.path.isfile(to_fn):
        data = np.load(to_fn)
    else:
        gsf = GDALSampling()
        gsf.initNPYRaster(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\2\SHH2_QD2_data.npy")
        data = gsf.sampling2DF(df=df, win_row=read_size[0], win_column=read_size[1])
        np.save(to_fn, data.astype("float32"))

    train_ds = DLDataset(win_size, read_size)
    test_ds = DLDataset(win_size, read_size)

    vhl_map_dict = {
        "NOT_KNOW": 0,
        "IS": 1, "VEG": 0, "SOIL": 1, "WAT": 2,
        "IS_SH": 3, "VEG_SH": 3, "SOIL_SH": 3, "WAT_SH": 3
    }
    is_map_dict = {"IS": 1, "SOIL": 0, }
    ws_map_dict = {
        "IS_SH": 0, "VEG_SH": 1, "SOIL_SH": 2, "WAT_SH": 3
    }

    map_dict = ws_map_dict

    def map_category(_category):
        if _category in map_dict:
            return map_dict[_category]
        return None

    for i in range(len(df)):
        if df["TEST"][i] == 1:
            category = map_category(str(df["CNAME"][i]))
            if category is not None:
                train_ds.data_list.append(data[i])
                train_ds.y_list.append(category)
        else:
            category = map_category(str(df["CNAME"][i]))
            if category is not None:
                test_ds.data_list.append(data[i])
                test_ds.y_list.append(category)
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
        self.name = "SHH2DL_OPTASDE"
        self.smip = SRTModImPytorch()

        def func_predict(model, x: torch.Tensor):
            logit = model(x)
            y = torch.argmax(logit, dim=1) + 1
            return y

        self.func_predict = func_predict
        self.color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), }
        self.color_table = {1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 0, 255), 4: (0, 0, 0), }
        self.color_table = {1: (0, 0, 0), 2: (255, 0, 0)}
        self.color_table = {1: (128, 0, 0), 2: (0, 128, 0), 3: (128, 128, 0), 4: (0, 0, 128), }
        # writeTexts(os.path.join(self.smip.model_dirname, "sys.argv.txt"), sys.argv, mode="a")
        self.slog = SRTLog()

    def main(self):
        self.smip.model_dirname = r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods"
        self.smip.model_name = "OPTASDE"
        self.smip.epochs = 100
        self.smip.device = "cuda:0"
        self.smip.n_test = 5
        self.smip.batch_size = 32
        self.smip.class_names = ["NOT_KNOW", "IS", "VEG", "SOIL", "WAT"]
        self.smip.n_class = len(self.smip.class_names)
        self.smip.win_size = (7, 7)
        self.smip.model = Model().to(self.smip.device)
        self.smip.func_predict = self.func_predict
        self.smip.func_y_deal = lambda y: y
        self.smip.initColorTable(self.color_table)
        return

    def train(self):
        self.smip.timeDirName()
        self.slog = SRTLog(os.path.join(self.smip.model_dirname, "log.txt"))
        self.smip.train_ds, self.smip.test_ds = loadDS(self.smip.win_size, self.smip.toCSVFN())
        self.smip.initTrainLog()
        self.smip.initPytorchTraining()
        self.smip.pt.func_logit_category = self.func_predict
        self.smip.pt.func_y_deal = lambda y: y + 1
        self.smip.initModel()
        self.slog.kw("self.smip.model.tsn.names", self.smip.model.tsn.toDict())
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

        # grc: GDALRasterChannel = GDALRasterChannel()
        # grc.addGDALDatas("")
        # self.smip.imdc(grc=grc, is_jdt=True, data_deal=data_deal)

        self.smip.imdcTiles(
            tiles_dirname=r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\qd_sh2_1_retile2",
            data_deal=data_deal,
        )

        # self.smip.imdcTiles(
        #     # to_fn=r"F:\Week\20240331\Data\20240329H185618\Net2_epoch2_imdc5.tif",
        #     tiles_dirname=r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\cd_sh2_1_retile",
        #     data_deal=data_deal,
        # )


def DeepLearning_main(is_train=True):
    dl = DeepLearning()
    dl.main()
    if is_train:
        dl.train()
    else:
        dl.imdc()


def main():
    DeepLearning_main()
    return


if __name__ == "__main__":
    main()
    r"""
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2DL import DeepLearning_main; DeepLearning_main(False)"
    """
