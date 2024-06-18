# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ChengJiXiangMu.py
@Time    : 2024/5/31 10:52
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of ChengJiXiangMu
-----------------------------------------------------------------------------"""
import itertools
import os.path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from osgeo import ogr, gdal
from osgeo_utils.gdal_merge import main as gdal_merge_main
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset

from SRTCodes.GDALDraw import GDALDrawImagesColumns
from SRTCodes.GDALRasterIO import GDALRaster, tiffAddColorTable, GDALRasterChannel
from SRTCodes.GDALUtils import GDALSamplingFast, dictRasterToVRT, GDALSampling, RasterToVRTS
from SRTCodes.ModelTraining import dataModelPredict, TrainLog
from SRTCodes.NumpyUtils import NumpyDataCenter
from SRTCodes.PytorchUtils import convBnAct
from SRTCodes.SRTModelImage import GDALImdc, SRTModImSklearn, SRTModImPytorch
from SRTCodes.SRTSample import GeoJsonPolygonCoor
from SRTCodes.Utils import changext, changefiledirname, getfilenamewithoutext, writeLines, writeTextLine, \
    numberfilename, SRTDataFrame, FRW, SRTLog, timeDirName, DirFileName, printList, writeTexts

TEMP_DFN = DirFileName(r"H:\ChengJiXiangMu\Temp")


def isExit():
    line = input(">")
    if "exit" in line:
        sys.exit(0)


def net1(channels, init_channel=1, kernel_size=3, padding=1, stride=1, act=nn.ReLU()):
    cba_list = []
    out_channel = channels[-1] * init_channel
    for i in range(len(channels) - 1):
        cba_list.append(convBnAct(
            in_channels=channels[i] * init_channel,
            out_channels=channels[i + 1] * init_channel,
            kernel_size=kernel_size, padding=padding,
            stride=stride, act=act,
        ))
    return nn.Sequential(*tuple(cba_list)), out_channel


def net2(channels, kernel_size=3, padding=1, stride=1, act=nn.ReLU()):
    cba_list = []

    for i in range(len(channels) - 1):
        cba_list.append(convBnAct(
            in_channels=channels[i],
            out_channels=channels[i + 1],
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            act=act,
        ))
    return nn.Sequential(*tuple(cba_list)), channels[-1]


def getSingle(n_pool, in_channel=1):
    convs = []

    if n_pool >= 1:
        cbas, in_channel = net1([1, 2, 8, 16], in_channel)
        conv_pool = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=2, stride=2)
        convs.append(cbas)
        convs.append(conv_pool)
    n_pool -= 1
    if n_pool >= 1:
        cbas, in_channel = net1([1, 2, 4, 8], in_channel)
        conv_pool = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=2, stride=2)
        convs.append(cbas)
        convs.append(conv_pool)
    n_pool -= 1
    if n_pool >= 1:
        cbas, in_channel = net1([1, 2, 2, 2], in_channel)
        conv_pool = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=2, stride=2)
        convs.append(cbas)
        convs.append(conv_pool)
    n_pool -= 1
    return nn.Sequential(*convs)


def getSingle2(n_pool, in_channel=1):
    convs = []

    if n_pool >= 1:
        cbas, in_channel = net2([in_channel, 8, 16, 16, 32])
        conv_pool = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=2, stride=2)
        convs.append(cbas)
        convs.append(conv_pool)
    n_pool -= 1
    if n_pool >= 1:
        cbas, in_channel = net2([32, 64, 128])
        conv_pool = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=2, stride=2)
        convs.append(cbas)
        convs.append(conv_pool)
    n_pool -= 1
    if n_pool >= 1:
        cbas, in_channel = net2([128, 264])
        conv_pool = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=2, stride=2)
        convs.append(cbas)
        convs.append(conv_pool)
    n_pool -= 1
    return nn.Sequential(*convs)


class ChengJiModel(nn.Module):

    def __init__(self, in_channel=0, find_channels=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if (in_channel == 0) and (find_channels == -1):
            in_channel = int(sys.argv[1])
            find_channels = int(sys.argv[2])

        self.in_channel = in_channel
        self.find_channels = find_channels
        self.mod1 = getSingle2(1, in_channel)
        self.mod2 = getSingle2(2, in_channel)
        self.mod3 = getSingle2(3, in_channel)
        self.avg_pool2d = nn.AvgPool2d(3)
        self.fc1 = nn.Linear(424, 2)

        self.ndc1 = NumpyDataCenter(3, (7, 7), (39, 39))
        self.ndc2 = NumpyDataCenter(3, (15, 15), (39, 39))
        self.ndc3 = NumpyDataCenter(3, (31, 31), (39, 39))

    def forward(self, x):
        if self.in_channel == 1:
            x = x[:, self.find_channels:self.find_channels + 1, :, :]
        elif self.in_channel == 4:
            x = x

        x1 = self.ndc1.fit2(x)
        x2 = self.ndc2.fit2(x)
        x3 = self.ndc3.fit2(x)
        out_x1 = self.mod1(x1)
        out_x2 = self.mod2(x2)
        out_x3 = self.mod3(x3)
        out_x = torch.concat([out_x1, out_x2, out_x3], dim=1)
        out_x = self.avg_pool2d(out_x)
        out_x = torch.flatten(out_x, 1)
        out_x = self.fc1(out_x)
        return out_x


def x_forward_ChengJiModel():
    x = torch.rand(32, 4, 39, 39)
    mod = ChengJiModel(3)
    out_x = mod(x)
    torch.save(mod, r"H:\ChengJiXiangMu\Temp\tmp2.mod")
    return


# x_forward_ChengJiModel()


def data_deal(x, y=None):
    x = x / 255
    if y is not None:
        return x, y
    return x


class CJDataset(Dataset):

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


def loadDS():
    csv_fn = r"H:\ChengJiXiangMu\Samples\2\zyh_spl2_1.csv"

    df = pd.read_csv(csv_fn)
    to_fn = changext(csv_fn, "_data.npy")
    if os.path.isfile(to_fn):
        data = np.load(to_fn)
    else:
        grf = GDALSamplingFast(
            r"H:\ChengJiXiangMu\LTB_2024_14_tif.tif",
            # r"H:\ChengJiXiangMu\clip_510110天府新区0304.tif"
        )
        data = grf.sampling2(df["X"].to_list(), df["Y"].to_list(), 39, 39, is_trans=True)
        np.save(to_fn, data)
    train_ds = CJDataset()
    test_ds = CJDataset()

    for i in range(len(df)):
        if df["TEST"][i] == 1:
            train_ds.data_list.append(data[i])
            train_ds.y_list.append(df["CATEGORY"][i] - 1)
        else:
            test_ds.data_list.append(data[i])
            test_ds.y_list.append(df["CATEGORY"][i] - 1)
    return train_ds, test_ds


class CJTrainImdc:

    def __init__(self):
        self.name = "CJTrainImdc"

        self.smip = SRTModImPytorch()

        def func_predict(model, x: torch.Tensor):
            logit = model(x)
            y = torch.argmax(logit, dim=1) + 1
            return y

        self.func_predict = func_predict
        self.city_name = "cd"

    def main(self):
        self.smip.model_dirname = r"H:\ChengJiXiangMu\CJTrainImdc_Mods"
        self.smip.model_name = "ChengJiModel"
        self.smip.epochs = 100
        self.smip.device = "cuda:0"
        self.smip.n_test = 5
        self.smip.batch_size = 32
        self.smip.class_names = ["NO", "IS"]
        self.smip.n_class = len(self.smip.class_names)
        self.smip.win_size = (39, 39)
        self.smip.model = ChengJiModel().to(self.smip.device)
        self.smip.func_predict = self.func_predict
        self.smip.func_y_deal = lambda y: y
        self.smip.initColorTable({1: (0, 0, 0), 2: (255, 255, 255)})
        return

    def train(self):
        self.smip.timeDirName()
        writeTexts(os.path.join(self.smip.model_dirname, "log_tmp.txt"), sys.argv, mode="a")
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


def readShape(shh_fn):
    datasource = ogr.Open(shh_fn, 0)  # 0 表示只读
    layer = datasource.GetLayerByIndex(0)
    layer_defn = layer.GetLayerDefn()
    field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]
    to_dict = {"X": [], "Y": [], **{field_name: [] for field_name in field_names}}
    for feature in layer:
        geometry = feature.GetGeometryRef()
        x = geometry.GetX()
        y = geometry.GetY()
        to_dict["X"].append(x)
        to_dict["Y"].append(y)
        fields = feature.items()
        for field_name, field_value in fields.items():
            to_dict[field_name].append(field_value)
    return to_dict


def samplingPoint(geo_fn, coor_dict):
    gsf = GDALSamplingFast(geo_fn)
    to_dict = gsf.sampling(coor_dict["X"], coor_dict["Y"])
    coor_dict = {**coor_dict, **to_dict}
    return coor_dict, gsf.gr


class CJSampling:

    def __init__(self, dirname=None):
        self.dirname = dirname
        # if not os.path.isdir(self.dirname):
        #     os.mkdir(dirname)
        self.sdf = SRTDataFrame()
        self.sdf.addFields("X", "Y", "CATEGORY")

    def addCSV(self, csv_fn, c_field_name="CATEGORY", x_field_name="X", y_field_name="Y", category=None):
        sdf = SRTDataFrame().read_csv(csv_fn, is_auto_type=True)
        for i in range(len(sdf)):
            line = sdf.rowToDict(i)
            if category is not None:
                cate = category
            else:
                cate = line[c_field_name]
            line = {"X": line[x_field_name], "Y": line[y_field_name], "CATEGORY": cate, **line}
            self.sdf.addLine(line)

    def addShape(self, shp_fn, c_field_name="CATEGORY", category=None):
        to_dict = readShape(shp_fn)
        for i in range(len(to_dict["X"])):
            if category is not None:
                cate = category
            else:
                cate = to_dict[c_field_name][i]
            line = {"CATEGORY": cate, **{k: to_dict[k][i] for k in to_dict}}
            self.sdf.addLine(line)

    def sampling(self, geo_fn):
        def func1():
            gsf = GDALSamplingFast(geo_fn)
            to_dict = gsf.sampling(self.sdf["X"], self.sdf["Y"])
            for k, data in to_dict.items():
                self.sdf[k] = data

        def func2():
            gs = GDALSampling(geo_fn)
            to_dict = gs.sampling(self.sdf["X"], self.sdf["Y"])
            for k, data in to_dict.items():
                self.sdf[k] = data

        func2()

    def toCSV(self, csv_fn):
        return self.sdf.toCSV(csv_fn)


def main():
    def func1():
        x = torch.rand(32, 1, 13, 13)
        model = ChengJiModel()
        x = model(x)
        torch.save(model, r"H:\ChengJiXiangMu\Temp\mod.pth")

    def func2():
        filelist_dict = {
            "LTB1": r"H:\ChengJiXiangMu\LTB_20240115_TFXQ_SARGLCM_VRTS\filelist.txt",
            "LTB2": r"H:\ChengJiXiangMu\LTB_202402_TFXQ_SARGLCM_VRTS\filelist.txt",
            "LTB3": r"H:\ChengJiXiangMu\LTB_20240319_TFXQ_SARGLCM_VRTS\filelist.txt",
            "LTB4": r"H:\ChengJiXiangMu\LTB_202404_TFXQ_SARGLCM_VRTS\filelist.txt",
        }

        n_str = ["n{}".format(i + 1) for i in range(8)]
        des = ["mean", "var", "hom", "con", "dis", "ent", "asm", "cor"]
        des_change = dict(zip(n_str, des))
        n_start_list = [67, 63, 67, 63]

        def func_des_change(_str):
            for _n_str in des_change:
                if _n_str in _str:
                    return _str.replace(_n_str, des_change[_n_str])
            return _str

        file_dict = {}
        i_n_start_list = 0
        for k, fn in filelist_dict.items():
            lines = FRW(fn).readLines(is_remove_empty=True)
            n_start = n_start_list[i_n_start_list]
            i_n_start_list += 1
            file_dict[k] = lines[0]
            for line in lines[1:]:
                name = "{0}_{1}".format(k, func_des_change(line[n_start:].strip(".vrt")))
                file_dict[name] = line
                print(name)
        dictRasterToVRT(r"H:\ChengJiXiangMu\LTB_1_4_GLCM.vrt", file_dict)
        FRW(r"H:\ChengJiXiangMu\Temp\tmp.json").saveJson(file_dict)

    def func3():
        cjs = CJSampling()
        # cjs.addShape(r"H:\ChengJiXiangMu\Samples\Sample\Sample.shp", c_field_name="Id")
        # cjs.addShape(r"H:\ChengJiXiangMu\Samples\4041sample\4041sample\sample404_randspl1.shp", category=1)
        cjs.addCSV(r"H:\ChengJiXiangMu\Samples\4041sample\cj_randspl2_random1_spl1.csv")
        cjs.sampling(r"H:\ChengJiXiangMu\LTB_1_4_GLCM.vrt")
        cjs.toCSV(r"H:\ChengJiXiangMu\Samples\4041sample\cj_randspl2_random1_spl2.csv")

    def func4():
        cj_rf = CJ_RFMain(
            r"H:\ChengJiXiangMu\Samples\zyh_sample3.csv",
            # r"H:\ChengJiXiangMu\LTB_1_4_GLCM.vrt"
        )
        keys = [
            "LTB1",
            "LTB1_01_mean", "LTB1_01_var", "LTB1_01_hom", "LTB1_01_con", "LTB1_01_dis", "LTB1_01_ent", "LTB1_01_asm",
            "LTB1_01_cor",
            "LTB1_1_1_mean", "LTB1_1_1_var", "LTB1_1_1_hom", "LTB1_1_1_con", "LTB1_1_1_dis", "LTB1_1_1_ent",
            "LTB1_1_1_asm", "LTB1_1_1_cor",
            "LTB1_10_mean", "LTB1_10_var", "LTB1_10_hom", "LTB1_10_con", "LTB1_10_dis", "LTB1_10_ent", "LTB1_10_asm",
            "LTB1_10_cor",
            "LTB1_11_mean", "LTB1_11_var", "LTB1_11_hom", "LTB1_11_con", "LTB1_11_dis", "LTB1_11_ent", "LTB1_11_asm",
            "LTB1_11_cor",
            "LTB2",
            "LTB2_01_mean", "LTB2_01_var", "LTB2_01_hom", "LTB2_01_con", "LTB2_01_dis", "LTB2_01_ent", "LTB2_01_asm",
            "LTB2_01_cor",
            "LTB2_1_1_mean", "LTB2_1_1_var", "LTB2_1_1_hom", "LTB2_1_1_con", "LTB2_1_1_dis", "LTB2_1_1_ent",
            "LTB2_1_1_asm", "LTB2_1_1_cor",
            "LTB2_10_mean", "LTB2_10_var", "LTB2_10_hom", "LTB2_10_con", "LTB2_10_dis", "LTB2_10_ent", "LTB2_10_asm",
            "LTB2_10_cor",
            "LTB2_11_mean", "LTB2_11_var", "LTB2_11_hom", "LTB2_11_con", "LTB2_11_dis", "LTB2_11_ent", "LTB2_11_asm",
            "LTB2_11_cor",
            "LTB3",
            "LTB3_01_mean", "LTB3_01_var", "LTB3_01_hom", "LTB3_01_con", "LTB3_01_dis", "LTB3_01_ent", "LTB3_01_asm",
            "LTB3_01_cor",
            "LTB3_1_1_mean", "LTB3_1_1_var", "LTB3_1_1_hom", "LTB3_1_1_con", "LTB3_1_1_dis", "LTB3_1_1_ent",
            "LTB3_1_1_asm", "LTB3_1_1_cor",
            "LTB3_10_mean", "LTB3_10_var", "LTB3_10_hom", "LTB3_10_con", "LTB3_10_dis", "LTB3_10_ent", "LTB3_10_asm",
            "LTB3_10_cor",
            "LTB3_11_mean", "LTB3_11_var", "LTB3_11_hom", "LTB3_11_con", "LTB3_11_dis", "LTB3_11_ent", "LTB3_11_asm",
            "LTB3_11_cor",
            "LTB4",
            "LTB4_01_mean", "LTB4_01_var", "LTB4_01_hom", "LTB4_01_con", "LTB4_01_dis", "LTB4_01_ent", "LTB4_01_asm",
            "LTB4_01_cor",
            "LTB4_1_1_mean", "LTB4_1_1_var", "LTB4_1_1_hom", "LTB4_1_1_con", "LTB4_1_1_dis", "LTB4_1_1_ent",
            "LTB4_1_1_asm", "LTB4_1_1_cor",
            "LTB4_10_mean", "LTB4_10_var", "LTB4_10_hom", "LTB4_10_con", "LTB4_10_dis", "LTB4_10_ent", "LTB4_10_asm",
            "LTB4_10_cor",
            "LTB4_11_mean", "LTB4_11_var", "LTB4_11_hom", "LTB4_11_con", "LTB4_11_dis", "LTB4_11_ent", "LTB4_11_asm",
            "LTB4_11_cor",
        ]
        cj_rf.train("SAR", ["LTB1", "LTB2", "LTB3", "LTB4", ])
        cj_rf.train(
            "SAR-GLCM",
            ["LTB1",
             "LTB1_01_mean", "LTB1_01_var", "LTB1_01_hom", "LTB1_01_con", "LTB1_01_dis", "LTB1_01_ent", "LTB1_01_asm",
             "LTB1_01_cor",
             "LTB1_1_1_mean", "LTB1_1_1_var", "LTB1_1_1_hom", "LTB1_1_1_con", "LTB1_1_1_dis", "LTB1_1_1_ent",
             "LTB1_1_1_asm", "LTB1_1_1_cor",
             "LTB1_10_mean", "LTB1_10_var", "LTB1_10_hom", "LTB1_10_con", "LTB1_10_dis", "LTB1_10_ent", "LTB1_10_asm",
             "LTB1_10_cor",
             "LTB1_11_mean", "LTB1_11_var", "LTB1_11_hom", "LTB1_11_con", "LTB1_11_dis", "LTB1_11_ent", "LTB1_11_asm",
             "LTB1_11_cor",
             "LTB2",
             "LTB2_01_mean", "LTB2_01_var", "LTB2_01_hom", "LTB2_01_con", "LTB2_01_dis", "LTB2_01_ent", "LTB2_01_asm",
             "LTB2_01_cor",
             "LTB2_1_1_mean", "LTB2_1_1_var", "LTB2_1_1_hom", "LTB2_1_1_con", "LTB2_1_1_dis", "LTB2_1_1_ent",
             "LTB2_1_1_asm", "LTB2_1_1_cor",
             "LTB2_10_mean", "LTB2_10_var", "LTB2_10_hom", "LTB2_10_con", "LTB2_10_dis", "LTB2_10_ent", "LTB2_10_asm",
             "LTB2_10_cor",
             "LTB2_11_mean", "LTB2_11_var", "LTB2_11_hom", "LTB2_11_con", "LTB2_11_dis", "LTB2_11_ent", "LTB2_11_asm",
             "LTB2_11_cor",
             "LTB3",
             "LTB3_01_mean", "LTB3_01_var", "LTB3_01_hom", "LTB3_01_con", "LTB3_01_dis", "LTB3_01_ent", "LTB3_01_asm",
             "LTB3_01_cor",
             "LTB3_1_1_mean", "LTB3_1_1_var", "LTB3_1_1_hom", "LTB3_1_1_con", "LTB3_1_1_dis", "LTB3_1_1_ent",
             "LTB3_1_1_asm", "LTB3_1_1_cor",
             "LTB3_10_mean", "LTB3_10_var", "LTB3_10_hom", "LTB3_10_con", "LTB3_10_dis", "LTB3_10_ent", "LTB3_10_asm",
             "LTB3_10_cor",
             "LTB3_11_mean", "LTB3_11_var", "LTB3_11_hom", "LTB3_11_con", "LTB3_11_dis", "LTB3_11_ent", "LTB3_11_asm",
             "LTB3_11_cor",
             "LTB4",
             "LTB4_01_mean", "LTB4_01_var", "LTB4_01_hom", "LTB4_01_con", "LTB4_01_dis", "LTB4_01_ent", "LTB4_01_asm",
             "LTB4_01_cor",
             "LTB4_1_1_mean", "LTB4_1_1_var", "LTB4_1_1_hom", "LTB4_1_1_con", "LTB4_1_1_dis", "LTB4_1_1_ent",
             "LTB4_1_1_asm", "LTB4_1_1_cor",
             "LTB4_10_mean", "LTB4_10_var", "LTB4_10_hom", "LTB4_10_con", "LTB4_10_dis", "LTB4_10_ent", "LTB4_10_asm",
             "LTB4_10_cor",
             "LTB4_11_mean", "LTB4_11_var", "LTB4_11_hom", "LTB4_11_con", "LTB4_11_dis", "LTB4_11_ent", "LTB4_11_asm",
             "LTB4_11_cor",
             ])
        cj_rf.train(
            "SAR-GLCM2",
            ["LTB1",
             "LTB1_01_mean", "LTB1_01_var", "LTB1_01_hom", "LTB1_01_con", "LTB1_01_dis", "LTB1_01_ent", "LTB1_01_asm",
             "LTB1_01_cor",
             "LTB1_1_1_mean", "LTB1_1_1_var", "LTB1_1_1_hom", "LTB1_1_1_con", "LTB1_1_1_dis", "LTB1_1_1_ent",
             "LTB1_1_1_asm", "LTB1_1_1_cor",
             "LTB1_10_mean", "LTB1_10_var", "LTB1_10_hom", "LTB1_10_con", "LTB1_10_dis", "LTB1_10_ent", "LTB1_10_asm",
             "LTB1_10_cor",
             "LTB1_11_mean", "LTB1_11_var", "LTB1_11_hom", "LTB1_11_con", "LTB1_11_dis", "LTB1_11_ent", "LTB1_11_asm",
             "LTB1_11_cor",

             "LTB4",
             "LTB4_01_mean", "LTB4_01_var", "LTB4_01_hom", "LTB4_01_con", "LTB4_01_dis", "LTB4_01_ent", "LTB4_01_asm",
             "LTB4_01_cor",
             "LTB4_1_1_mean", "LTB4_1_1_var", "LTB4_1_1_hom", "LTB4_1_1_con", "LTB4_1_1_dis", "LTB4_1_1_ent",
             "LTB4_1_1_asm", "LTB4_1_1_cor",
             "LTB4_10_mean", "LTB4_10_var", "LTB4_10_hom", "LTB4_10_con", "LTB4_10_dis", "LTB4_10_ent", "LTB4_10_asm",
             "LTB4_10_cor",
             "LTB4_11_mean", "LTB4_11_var", "LTB4_11_hom", "LTB4_11_con", "LTB4_11_dis", "LTB4_11_ent", "LTB4_11_asm",
             "LTB4_11_cor",
             ])
        cj_rf.train(
            "SAR-GLCM3",
            [
                "LTB3",
                "LTB3_01_mean", "LTB3_01_var", "LTB3_01_hom", "LTB3_01_con", "LTB3_01_dis", "LTB3_01_ent",
                "LTB3_01_asm",
                "LTB3_01_cor",
                "LTB3_1_1_mean", "LTB3_1_1_var", "LTB3_1_1_hom", "LTB3_1_1_con", "LTB3_1_1_dis", "LTB3_1_1_ent",
                "LTB3_1_1_asm", "LTB3_1_1_cor",
                "LTB3_10_mean", "LTB3_10_var", "LTB3_10_hom", "LTB3_10_con", "LTB3_10_dis", "LTB3_10_ent",
                "LTB3_10_asm",
                "LTB3_10_cor",
                "LTB3_11_mean", "LTB3_11_var", "LTB3_11_hom", "LTB3_11_con", "LTB3_11_dis", "LTB3_11_ent",
                "LTB3_11_asm",
                "LTB3_11_cor",
                "LTB4",
                "LTB4_01_mean", "LTB4_01_var", "LTB4_01_hom", "LTB4_01_con", "LTB4_01_dis", "LTB4_01_ent",
                "LTB4_01_asm",
                "LTB4_01_cor",
                "LTB4_1_1_mean", "LTB4_1_1_var", "LTB4_1_1_hom", "LTB4_1_1_con", "LTB4_1_1_dis", "LTB4_1_1_ent",
                "LTB4_1_1_asm", "LTB4_1_1_cor",
                "LTB4_10_mean", "LTB4_10_var", "LTB4_10_hom", "LTB4_10_con", "LTB4_10_dis", "LTB4_10_ent",
                "LTB4_10_asm",
                "LTB4_10_cor",
                "LTB4_11_mean", "LTB4_11_var", "LTB4_11_hom", "LTB4_11_con", "LTB4_11_dis", "LTB4_11_ent",
                "LTB4_11_asm",
                "LTB4_11_cor",
            ])
        cj_rf.train(
            "SAR-GLCM4",
            ["LTB1",
             "LTB1_01_mean", "LTB1_01_var", "LTB1_01_hom", "LTB1_01_con", "LTB1_01_dis", "LTB1_01_ent", "LTB1_01_asm",
             "LTB1_01_cor",
             "LTB1_1_1_mean", "LTB1_1_1_var", "LTB1_1_1_hom", "LTB1_1_1_con", "LTB1_1_1_dis", "LTB1_1_1_ent",
             "LTB1_1_1_asm", "LTB1_1_1_cor",
             "LTB1_10_mean", "LTB1_10_var", "LTB1_10_hom", "LTB1_10_con", "LTB1_10_dis", "LTB1_10_ent", "LTB1_10_asm",
             "LTB1_10_cor",
             "LTB1_11_mean", "LTB1_11_var", "LTB1_11_hom", "LTB1_11_con", "LTB1_11_dis", "LTB1_11_ent", "LTB1_11_asm",
             "LTB1_11_cor",
             "LTB2",
             "LTB2_01_mean", "LTB2_01_var", "LTB2_01_hom", "LTB2_01_con", "LTB2_01_dis", "LTB2_01_ent", "LTB2_01_asm",
             "LTB2_01_cor",
             "LTB2_1_1_mean", "LTB2_1_1_var", "LTB2_1_1_hom", "LTB2_1_1_con", "LTB2_1_1_dis", "LTB2_1_1_ent",
             "LTB2_1_1_asm", "LTB2_1_1_cor",
             "LTB2_10_mean", "LTB2_10_var", "LTB2_10_hom", "LTB2_10_con", "LTB2_10_dis", "LTB2_10_ent", "LTB2_10_asm",
             "LTB2_10_cor",
             "LTB2_11_mean", "LTB2_11_var", "LTB2_11_hom", "LTB2_11_con", "LTB2_11_dis", "LTB2_11_ent", "LTB2_11_asm",
             "LTB2_11_cor",

             ])
        cj_rf.train(
            "GLCM",
            [
                "LTB1_01_mean", "LTB1_01_var", "LTB1_01_hom", "LTB1_01_con", "LTB1_01_dis", "LTB1_01_ent",
                "LTB1_01_asm",
                "LTB1_01_cor",
                "LTB1_1_1_mean", "LTB1_1_1_var", "LTB1_1_1_hom", "LTB1_1_1_con", "LTB1_1_1_dis", "LTB1_1_1_ent",
                "LTB1_1_1_asm", "LTB1_1_1_cor",
                "LTB1_10_mean", "LTB1_10_var", "LTB1_10_hom", "LTB1_10_con", "LTB1_10_dis", "LTB1_10_ent",
                "LTB1_10_asm",
                "LTB1_10_cor",
                "LTB1_11_mean", "LTB1_11_var", "LTB1_11_hom", "LTB1_11_con", "LTB1_11_dis", "LTB1_11_ent",
                "LTB1_11_asm",
                "LTB1_11_cor",

                "LTB2_01_mean", "LTB2_01_var", "LTB2_01_hom", "LTB2_01_con", "LTB2_01_dis", "LTB2_01_ent",
                "LTB2_01_asm",
                "LTB2_01_cor",
                "LTB2_1_1_mean", "LTB2_1_1_var", "LTB2_1_1_hom", "LTB2_1_1_con", "LTB2_1_1_dis", "LTB2_1_1_ent",
                "LTB2_1_1_asm", "LTB2_1_1_cor",
                "LTB2_10_mean", "LTB2_10_var", "LTB2_10_hom", "LTB2_10_con", "LTB2_10_dis", "LTB2_10_ent",
                "LTB2_10_asm",
                "LTB2_10_cor",
                "LTB2_11_mean", "LTB2_11_var", "LTB2_11_hom", "LTB2_11_con", "LTB2_11_dis", "LTB2_11_ent",
                "LTB2_11_asm",
                "LTB2_11_cor",

                "LTB3_01_mean", "LTB3_01_var", "LTB3_01_hom", "LTB3_01_con", "LTB3_01_dis", "LTB3_01_ent",
                "LTB3_01_asm",
                "LTB3_01_cor",
                "LTB3_1_1_mean", "LTB3_1_1_var", "LTB3_1_1_hom", "LTB3_1_1_con", "LTB3_1_1_dis", "LTB3_1_1_ent",
                "LTB3_1_1_asm", "LTB3_1_1_cor",
                "LTB3_10_mean", "LTB3_10_var", "LTB3_10_hom", "LTB3_10_con", "LTB3_10_dis", "LTB3_10_ent",
                "LTB3_10_asm",
                "LTB3_10_cor",
                "LTB3_11_mean", "LTB3_11_var", "LTB3_11_hom", "LTB3_11_con", "LTB3_11_dis", "LTB3_11_ent",
                "LTB3_11_asm",
                "LTB3_11_cor",

                "LTB4_01_mean", "LTB4_01_var", "LTB4_01_hom", "LTB4_01_con", "LTB4_01_dis", "LTB4_01_ent",
                "LTB4_01_asm",
                "LTB4_01_cor",
                "LTB4_1_1_mean", "LTB4_1_1_var", "LTB4_1_1_hom", "LTB4_1_1_con", "LTB4_1_1_dis", "LTB4_1_1_ent",
                "LTB4_1_1_asm", "LTB4_1_1_cor",
                "LTB4_10_mean", "LTB4_10_var", "LTB4_10_hom", "LTB4_10_con", "LTB4_10_dis", "LTB4_10_ent",
                "LTB4_10_asm",
                "LTB4_10_cor",
                "LTB4_11_mean", "LTB4_11_var", "LTB4_11_hom", "LTB4_11_con", "LTB4_11_dis", "LTB4_11_ent",
                "LTB4_11_asm",
                "LTB4_11_cor",
            ])

        return

    def func5():
        def save(name, fit_keys):
            to_fn = os.path.join(r"H:\ChengJiXiangMu\Models\1", name)
            with open(to_fn, "w", encoding="utf-8") as f:
                for k in fit_keys:
                    f.write(k)
                    f.write("\n")

    def func6():
        to_dict, feat_iter = splitKeys()
        cj_rf = CJ_RFMain(
            r"H:\ChengJiXiangMu\Samples\zyh_sample3.csv",
            # r"H:\ChengJiXiangMu\LTB_1_4_GLCM.vrt"
        )
        for feat_names in feat_iter:
            feat_keys = ["LTB1", "LTB2", "LTB3", "LTB4", ]
            for feat_name in feat_names:
                feat_keys.extend(to_dict[feat_name])
            cj_rf.train("-".join(feat_names), feat_keys)

    def func7():
        # mean - var - hom - asm - cor
        cj_rf = CJ_RFMain(
            # r"H:\ChengJiXiangMu\Samples\zyh_sample3.csv",
            r"H:\ChengJiXiangMu\Samples\zyh_sample4.csv",
            r"H:\ChengJiXiangMu\LTB_1_4_GLCM.vrt"
        )
        to_dict, feat_iter = splitKeys()

        # cj_rf.train(
        #     "IMDC1", ["LTB1", "LTB2", "LTB3", "LTB4", ] + \
        #             to_dict["mean"] + \
        #             to_dict["var"] + \
        #             to_dict["hom"] + \
        #             to_dict["asm"] + \
        #             to_dict["cor"]
        #             )

        cj_rf.train("IMDC2", to_dict["KEYS"])

        cj_rf.imdcTiles()

    def func8():
        gr1 = GDALRaster(r"H:\ChengJiXiangMu\LTB_1_4_GLCM.vrt")
        gr2 = GDALRaster(r"H:\ChengJiXiangMu\LTB_1_4_GLCM_retiles\LTB_1_4_GLCM_1_1.tif")
        for i in range(gr2.n_channels):
            band: gdal.Band = gr2.raster_ds.GetRasterBand(i + 1)
            band.SetDescription(gr1.names[i])
        printList("gr1", gr1.names)
        printList("gr2", gr2.names)
        return

    def func9():
        to_list = []
        # gj = GeoJsonPolygonCoor(r"H:\ChengJiXiangMu\Samples\4041sample\4041sample\sample404.geojson")
        # to_list.extend(gj.random(500, field_names={"CATEGORY": 1, "TAG":"SCG"}))
        gj = GeoJsonPolygonCoor(r"H:\ChengJiXiangMu\Samples\4041sample\4041sample\sample404_randspl2.geojson")
        to_list.extend(gj.random(1000, field_names={"CATEGORY": 0, "TAG": "sample404_randspl2"}))
        gj = GeoJsonPolygonCoor(r"H:\ChengJiXiangMu\Samples\4041sample\4041sample\sample404_randspl3.geojson")
        to_list.extend(gj.random(1500, field_names={"CATEGORY": 1, "TAG": "sample404_randspl3"}))
        gj = GeoJsonPolygonCoor(r"H:\ChengJiXiangMu\Samples\4041sample\4041sample\sample404_randspl4.geojson")
        to_list.extend(gj.random(300, field_names={"CATEGORY": 0, "TAG": "sample404_randspl4"}))
        gj = GeoJsonPolygonCoor(r"H:\ChengJiXiangMu\Samples\4041sample\4041sample\sample404_randspl5.geojson")
        to_list.extend(gj.random(1000, field_names={"CATEGORY": 0, "TAG": "sample404_randspl5"}))
        df = pd.DataFrame(to_list)
        df.to_csv(r"H:\ChengJiXiangMu\Samples\4041sample\cj_randspl2_random1.csv")
        print(df)

    def func10():
        def func10_1():
            x = torch.rand(32, 1, 7, 7)
            mod = getSingle(1)
            out_x1 = mod(x)

            x = torch.rand(32, 1, 15, 15)
            mod = getSingle(2)
            out_x2 = mod(x)

            x = torch.rand(32, 1, 31, 31)
            mod = getSingle(3)
            out_x3 = mod(x)

            out_x = torch.concat([out_x1, out_x2, out_x3], dim=1)
            conv1 = nn.Conv2d(400, 800, 3)
            out_x = conv1(out_x)

            out_x = torch.flatten(out_x, 1)

            fc1 = nn.Linear(800, 80)
            fc2 = nn.Linear(80, 2)

            out_x = fc1(out_x)
            out_x = fc2(out_x)

        return

    def func11():
        # mod = ChengJiModel()
        # x = torch.rand(32, 1, 39, 39)
        # out_x = mod(x)

        cjti = CJTrainImdc()
        cjti.main()
        # cjti.train()
        cjti.imdc()

    def func12():
        fn_list = [
            r"J:\data\cdtf_sar1_01"
            , r"J:\data\cdtf_sar1_1_1"
            , r"J:\data\cdtf_sar1_10"
            , r"J:\data\cdtf_sar1_11"
            , r"J:\data\cdtf_sar2_01"
            , r"J:\data\cdtf_sar2_1_1"
            , r"J:\data\cdtf_sar2_10"
            , r"J:\data\cdtf_sar2_11"
            , r"J:\data\cdtf_sar3_01"
            , r"J:\data\cdtf_sar3_1_1"
            , r"J:\data\cdtf_sar3_10"
            , r"J:\data\cdtf_sar3_11"
            , r"J:\data\cdtf_sar4_01"
            , r"J:\data\cdtf_sar4_1_1"
            , r"J:\data\cdtf_sar4_10"
            , r"J:\data\cdtf_sar4_11"
        ]
        for fn in fn_list:
            RasterToVRTS(fn).frontStr(os.path.split(fn)[1]).save(r"J:\data\vrts")

    func12()
    return


class CJModImSklearn(SRTModImSklearn):

    def __init__(self):
        super().__init__()

    def train(self, is_print=True, sample_weight=None, *args, **kwargs):
        x_test, x_train, y_test, y_train = self.getTrainingData()
        self.clf = RandomForestClassifier(60)
        self.clf.fit(x_train.values, y_train, sample_weight=sample_weight)
        if is_print:
            print("train accuracy: {0}".format(self.clf.score(x_train.values, y_train)))
            if self.y_test is not None:
                print("test accuracy: {0}".format(self.clf.score(x_test.values, y_test)))


class CJ_RFMain:

    def __init__(self, csv_fn=None, raster_fn=None, raster_tile_fns=None):
        if raster_tile_fns is None:
            raster_tile_fns = [
                r"H:\ChengJiXiangMu\LTB_1_4_GLCM_retiles\LTB_1_4_GLCM_1_1.tif",
                r"H:\ChengJiXiangMu\LTB_1_4_GLCM_retiles\LTB_1_4_GLCM_1_2.tif",
                r"H:\ChengJiXiangMu\LTB_1_4_GLCM_retiles\LTB_1_4_GLCM_1_3.tif",
                r"H:\ChengJiXiangMu\LTB_1_4_GLCM_retiles\LTB_1_4_GLCM_2_1.tif",
                r"H:\ChengJiXiangMu\LTB_1_4_GLCM_retiles\LTB_1_4_GLCM_2_2.tif",
                r"H:\ChengJiXiangMu\LTB_1_4_GLCM_retiles\LTB_1_4_GLCM_2_3.tif",
                r"H:\ChengJiXiangMu\LTB_1_4_GLCM_retiles\LTB_1_4_GLCM_3_1.tif",
                r"H:\ChengJiXiangMu\LTB_1_4_GLCM_retiles\LTB_1_4_GLCM_3_2.tif",
                r"H:\ChengJiXiangMu\LTB_1_4_GLCM_retiles\LTB_1_4_GLCM_3_3.tif",
            ]

        self.fit_keys = None
        self.name = "_CJRF"
        self.smis = SRTModImSklearn()
        self.slog = SRTLog()
        self.model_dirname = r"H:\ChengJiXiangMu\Models"
        self.model_dirname = timeDirName(self.model_dirname, is_mk=True)

        self.slog.__init__(os.path.join(self.model_dirname, "{0}_log.txt".format(self.name)), mode="a", )
        self.slog.kw("NAME", self.name)
        self.slog.kw("MODEL_DIRNAME", self.model_dirname)

        self.cnames = ["NOT_KNOW", "NO", "IS"]

        self.saveCodeFile()

        self.df = None
        self.datas = None
        self.datas_tile = {}
        self.gr = None
        self.initDF(csv_fn=csv_fn)
        self.initRaster(raster_fn=raster_fn)
        self.initRasterTiles(raster_tile_fns=raster_tile_fns)
        self.raster_tile_fns = raster_tile_fns

        self.train_log = TrainLog(
            save_csv_file=os.path.join(self.model_dirname, "train_save{}.csv".format(self.name)),
            log_filename=os.path.join(self.model_dirname, "train_log{}.csv".format(self.name)),
        )
        self.addTrainLogFields()
        self.train_log.saveHeader()
        self.n = 1

    def initRaster(self, raster_fn=None):
        if raster_fn is None:
            return
        self.gr = GDALRaster(raster_fn)
        self.datas = {k: None for k in self.gr.names}

    def initRasterTiles(self, raster_tile_fns=None):
        if raster_tile_fns is None:
            return
        for fn in raster_tile_fns:
            gr = GDALRaster(fn)
            self.datas_tile[fn] = {"gr": GDALRaster(fn), "datas": {k: None for k in gr.names}}

    def initDF(self, csv_fn=None):
        if csv_fn is not None:
            self.df = pd.read_csv(csv_fn)
            self.saveCSVFile()

    def addTrainLogFields(self):
        self.train_log.addField("NUMBER", "int")
        self.train_log.addField("NAME", "string")
        self.train_log.addField("OA_TEST", "float")
        self.train_log.addField("KAPPA_TEST", "float")
        self.train_log.addField("UA_TEST", "float")
        self.train_log.addField("PA_TEST", "float")
        self.train_log.addField("OA_TRAIN", "float")
        self.train_log.addField("KAPPA_TRAIN", "float")
        self.train_log.addField("UA_TRAIN", "float")
        self.train_log.addField("PA_TRAIN", "float")

    def train(self, name=None, fit_keys=None):
        fit_keys, name = self._initFitKeysName(fit_keys, name)

        self.train_log.updateField("NAME", name)
        self.slog.wl("-" * 80)
        self.slog.wl("** Training --", name)
        self.name = name

        self.smis = SRTModImSklearn()
        self.smis.initColorTable({1: (0, 0, 0), 2: (255, 255, 255)})
        self.smis.category_names = self.cnames
        self.slog.kw("CATEGORY_NAMES", self.smis.category_names)
        self.slog.kw("COLOR_TABLE", self.smis.color_table)
        self.smis.initPandas(self.df)
        self.slog.kw("shh_mis.df.keys()", list(self.smis.df.keys()))
        self.slog.kw("Category Field Name:", self.smis.initCategoryField())
        self.fit_keys = fit_keys
        self.slog.kw("fit_keys", fit_keys)
        self.smis.initXKeys(fit_keys)
        self.smis.testFieldSplitTrainTest()
        self.slog.kw("LEN X", len(self.smis.x))
        self.slog.kw("LEN Train", len(self.smis.x_train))
        self.slog.kw("LEN Test", len(self.smis.y_test))
        self.smis.initCLF(RandomForestClassifier(150))
        self.smis.train()
        self.smis.scoreTrainCM()
        self.slog.kw("Train CM", self.smis.train_cm.fmtCM(), sep="\n")
        self.train_log.updateField("OA_TRAIN", self.smis.train_cm.OA())
        self.train_log.updateField("KAPPA_TRAIN", self.smis.train_cm.getKappa())
        self.train_log.updateField("UA_TRAIN", self.smis.train_cm.UA(2))
        self.train_log.updateField("PA_TRAIN", self.smis.train_cm.PA(2))
        self.smis.scoreTestCM()
        self.slog.kw("Test CM", self.smis.test_cm.fmtCM(), sep="\n")
        self.train_log.updateField("OA_TEST", self.smis.test_cm.OA())
        self.train_log.updateField("KAPPA_TEST", self.smis.test_cm.getKappa())
        self.train_log.updateField("UA_TEST", self.smis.test_cm.UA(2))
        self.train_log.updateField("PA_TEST", self.smis.test_cm.PA(2))
        mod_fn = self.slog.kw("Model FileName", os.path.join(self.model_dirname, "{0}.model".format(name)))
        self.smis.saveModel(mod_fn)
        self.train_log.updateField("NUMBER", self.n)
        self.train_log.saveLine()
        self.train_log.newLine()
        self.n += 1
        return self.smis.clf

    def imdc(self, fit_keys=None, name=None):
        self.slog.wl("** Image Classification --", name)
        fit_keys, name = self._initFitKeysName(fit_keys, name)
        to_imdc_fn = self.slog.kw("to_imdc_fn", os.path.join(self.model_dirname, "{}_imdc.tif".format(name)))
        data = np.zeros((len(fit_keys), self.gr.n_rows, self.gr.n_columns))
        for i, k in enumerate(fit_keys):
            if self.datas[k] is None:
                self.datas[k] = self.gr.readGDALBand(k)
            data[i] = self.datas[k]
        self.gr.d = data
        self.smis.imdcGR(to_imdc_fn, self.gr)

    def imdcTiles(self, fit_keys=None, name=None):
        self.slog.wl("** Image Classification --", name)
        fit_keys, name = self._initFitKeysName(fit_keys, name)
        to_imdc_fn = self.slog.kw("to_imdc_fn", os.path.join(self.model_dirname, "{}_imdc.tif".format(name)))
        to_imdc_dirname = changext(to_imdc_fn, "_tiles")
        if not os.path.isdir(to_imdc_dirname):
            os.mkdir(to_imdc_dirname)
        to_fn_tmps = []

        for fn in self.datas_tile:
            self.slog.wl("{}".format(fn))
            gr = self.datas_tile[fn]["gr"]
            data = np.zeros((len(fit_keys), gr.n_rows, gr.n_columns))
            for i, k in enumerate(fit_keys):
                if self.datas_tile[fn]["datas"][k] is None:
                    self.datas_tile[fn]["datas"][k] = gr.readGDALBand(k)
                data[i] = self.datas_tile[fn]["datas"][k]
            gr.d = data
            to_imdc_fn_tmp = os.path.join(to_imdc_dirname, changext(os.path.split(fn)[1], "_imdc.tif"))
            to_fn_tmps.append(to_imdc_fn_tmp)
            self.slog.wl("TO_IMDC_FN_TMP: {}".format(to_imdc_fn_tmp))
            self.smis.imdcGR(to_imdc_fn_tmp, gr)
            del gr.d
            del data
            gr.d = None
            del self.datas_tile[fn]["datas"]

        gdal_merge_main(["gdal_merge_main",
                         "-of", "GTiff",
                         "-n", "0",
                         "-ot", "Byte",
                         "-co", "COMPRESS=PACKBITS",
                         "-o", to_imdc_fn,
                         *to_fn_tmps, ])
        tiffAddColorTable(to_imdc_fn, code_colors=self.smis.color_table)

    def _initFitKeysName(self, fit_keys, name):
        if fit_keys is None:
            fit_keys = self.fit_keys
        if name is None:
            name = self.name
        return fit_keys, name

    def saveCSVFile(self):
        to_csv_fn = self.slog.kw("to_csv_fn", os.path.join(self.model_dirname, "{}_train_data.csv".format(self.name)))
        self.df.to_csv(to_csv_fn, index=False)

    def saveCodeFile(self):
        to_code_fn = self.slog.kw("to_code_fn", changefiledirname(__file__, self.model_dirname))
        self.smis.saveCodeFile(code_fn=__file__, to_code_fn=to_code_fn)


def splitKeys():
    keys = [
        "LTB1",
        "LTB1_01_mean", "LTB1_01_var", "LTB1_01_hom", "LTB1_01_con", "LTB1_01_dis", "LTB1_01_ent", "LTB1_01_asm",
        "LTB1_01_cor",
        "LTB1_1_1_mean", "LTB1_1_1_var", "LTB1_1_1_hom", "LTB1_1_1_con", "LTB1_1_1_dis", "LTB1_1_1_ent",
        "LTB1_1_1_asm", "LTB1_1_1_cor",
        "LTB1_10_mean", "LTB1_10_var", "LTB1_10_hom", "LTB1_10_con", "LTB1_10_dis", "LTB1_10_ent", "LTB1_10_asm",
        "LTB1_10_cor",
        "LTB1_11_mean", "LTB1_11_var", "LTB1_11_hom", "LTB1_11_con", "LTB1_11_dis", "LTB1_11_ent", "LTB1_11_asm",
        "LTB1_11_cor",
        "LTB2",
        "LTB2_01_mean", "LTB2_01_var", "LTB2_01_hom", "LTB2_01_con", "LTB2_01_dis", "LTB2_01_ent", "LTB2_01_asm",
        "LTB2_01_cor",
        "LTB2_1_1_mean", "LTB2_1_1_var", "LTB2_1_1_hom", "LTB2_1_1_con", "LTB2_1_1_dis", "LTB2_1_1_ent",
        "LTB2_1_1_asm", "LTB2_1_1_cor",
        "LTB2_10_mean", "LTB2_10_var", "LTB2_10_hom", "LTB2_10_con", "LTB2_10_dis", "LTB2_10_ent", "LTB2_10_asm",
        "LTB2_10_cor",
        "LTB2_11_mean", "LTB2_11_var", "LTB2_11_hom", "LTB2_11_con", "LTB2_11_dis", "LTB2_11_ent", "LTB2_11_asm",
        "LTB2_11_cor",
        "LTB3",
        "LTB3_01_mean", "LTB3_01_var", "LTB3_01_hom", "LTB3_01_con", "LTB3_01_dis", "LTB3_01_ent", "LTB3_01_asm",
        "LTB3_01_cor",
        "LTB3_1_1_mean", "LTB3_1_1_var", "LTB3_1_1_hom", "LTB3_1_1_con", "LTB3_1_1_dis", "LTB3_1_1_ent",
        "LTB3_1_1_asm", "LTB3_1_1_cor",
        "LTB3_10_mean", "LTB3_10_var", "LTB3_10_hom", "LTB3_10_con", "LTB3_10_dis", "LTB3_10_ent", "LTB3_10_asm",
        "LTB3_10_cor",
        "LTB3_11_mean", "LTB3_11_var", "LTB3_11_hom", "LTB3_11_con", "LTB3_11_dis", "LTB3_11_ent", "LTB3_11_asm",
        "LTB3_11_cor",
        "LTB4",
        "LTB4_01_mean", "LTB4_01_var", "LTB4_01_hom", "LTB4_01_con", "LTB4_01_dis", "LTB4_01_ent", "LTB4_01_asm",
        "LTB4_01_cor",
        "LTB4_1_1_mean", "LTB4_1_1_var", "LTB4_1_1_hom", "LTB4_1_1_con", "LTB4_1_1_dis", "LTB4_1_1_ent",
        "LTB4_1_1_asm", "LTB4_1_1_cor",
        "LTB4_10_mean", "LTB4_10_var", "LTB4_10_hom", "LTB4_10_con", "LTB4_10_dis", "LTB4_10_ent", "LTB4_10_asm",
        "LTB4_10_cor",
        "LTB4_11_mean", "LTB4_11_var", "LTB4_11_hom", "LTB4_11_con", "LTB4_11_dis", "LTB4_11_ent", "LTB4_11_asm",
        "LTB4_11_cor",
    ]

    to_dict = {"KEYS": keys}
    sar_list = ["SAR", "LTB1", "LTB2", "LTB3", "LTB4"]
    to_dict["SAR"] = sar_list
    for k in sar_list:
        to_list = []
        for name in keys:
            if k == name:
                continue
            if k in name:
                to_list.append(name)
        to_dict[k] = to_list

    glcm_list = ["mean", "var", "hom", "con", "dis", "ent", "asm", "cor", ]
    for k in glcm_list:
        to_list = []
        for name in keys:
            if k in name:
                to_list.append(name)
        to_dict["{}".format(k)] = to_list

    FRW(TEMP_DFN.fn("temp.json")).saveJson(to_dict)

    feat_iter = []
    for i in range(len(sar_list)):
        feat_iter.extend(itertools.combinations(sar_list, i + 1))

    return to_dict, feat_iter


def RF():
    def rf(raster_fn):
        to_imdc_fn = changefiledirname(changext(raster_fn, "_imdc.tif"), r"H:\ChengJiXiangMu\Imdc")
        to_csv_fn = None
        to_imdc_fn = numberfilename(to_imdc_fn)
        print("to_imdc_fn", to_imdc_fn)

        sample_dict, gr = samplingPoint(raster_fn, point_dict)
        df = pd.DataFrame(sample_dict)
        if to_csv_fn is not None:
            df.to_csv(to_csv_fn, index=False)
        print(df.keys())

        fit_keys = [
            'FEATURE_1', 'FEATURE_2', 'FEATURE_3', 'FEATURE_4',
            'FEATURE_5', 'FEATURE_6', 'FEATURE_7', 'FEATURE_8', 'FEATURE_9',
            'FEATURE_10', 'FEATURE_11', 'FEATURE_12', 'FEATURE_13', 'FEATURE_14',
            'FEATURE_15', 'FEATURE_16', 'FEATURE_17', 'FEATURE_18', 'FEATURE_19',
            'FEATURE_20', 'FEATURE_21', 'FEATURE_22', 'FEATURE_23', 'FEATURE_24',
            'FEATURE_25', 'FEATURE_26', 'FEATURE_27', 'FEATURE_28', 'FEATURE_29',
            'FEATURE_30', 'FEATURE_31', 'FEATURE_32', 'FEATURE_33'
        ]

        x = df[fit_keys].values
        y = df["Id"].values + 1

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=60)
        clf.fit(x_train, y_train)
        print(clf.score(x_test, y_test))

        imdc = dataModelPredict(gr.d, data_deal=None, is_jdt=True, model=clf)
        gr.save(imdc.astype("int8"), to_imdc_fn, fmt="GTiff", dtype=gdal.GDT_Byte, options=["COMPRESS=PACKBITS"])
        tiffAddColorTable(to_imdc_fn, 1, {1: (0, 0, 0), 2: (255, 255, 255)})

    sample_shapefile = r"H:\ChengJiXiangMu\Samples\Sample\Sample.shp"
    point_dict = readShape(sample_shapefile)
    # r"H:\ChengJiXiangMu\LTB_202402_TFXQ_SARGLCM_tif.tif"
    rf(r"H:\ChengJiXiangMu\LTB_202402_TFXQ_SARGLCM_tif.tif")

    r"""
    python -c "import sys; sys.path.append(r'F:\PyCodes'); from Temp.ChengJiXiangMu import main; main()" 
        """


def method_name():
    fns = [
        r"H:\ChengJiXiangMu\LTB_202404_TFXQ_tmp\LTB_202404_TFXQ",
        r"H:\ChengJiXiangMu\LTB_202404_TFXQ_tmp\LTB_202404_TFXQ_01",
        r"H:\ChengJiXiangMu\LTB_202404_TFXQ_tmp\LTB_202404_TFXQ_1_1",
        r"H:\ChengJiXiangMu\LTB_202404_TFXQ_tmp\LTB_202404_TFXQ_10",
        r"H:\ChengJiXiangMu\LTB_202404_TFXQ_tmp\LTB_202404_TFXQ_11",
    ]
    fns = [
        r"H:\ChengJiXiangMu\LTB_202402_TFXQ_tmp\LTB_202402_TFXQ",
        r"H:\ChengJiXiangMu\LTB_202402_TFXQ_tmp\LTB_202402_TFXQ_01",
        r"H:\ChengJiXiangMu\LTB_202402_TFXQ_tmp\LTB_202402_TFXQ_1_1",
        r"H:\ChengJiXiangMu\LTB_202402_TFXQ_tmp\LTB_202402_TFXQ_10",
        r"H:\ChengJiXiangMu\LTB_202402_TFXQ_tmp\LTB_202402_TFXQ_11",
    ]
    fns = [
        r"H:\ChengJiXiangMu\LTB_20240115_TFXQ_tmp\LTB_20240115_TFXQ",
        r"H:\ChengJiXiangMu\LTB_20240115_TFXQ_tmp\LTB_20240115_TFXQ_01",
        r"H:\ChengJiXiangMu\LTB_20240115_TFXQ_tmp\LTB_20240115_TFXQ_1_1",
        r"H:\ChengJiXiangMu\LTB_20240115_TFXQ_tmp\LTB_20240115_TFXQ_10",
        r"H:\ChengJiXiangMu\LTB_20240115_TFXQ_tmp\LTB_20240115_TFXQ_11",
    ]
    fns = [
        r"H:\ChengJiXiangMu\LTB_20240319_TFXQ_tmp\LTB_20240319_TFXQ",
        r"H:\ChengJiXiangMu\LTB_20240319_TFXQ_tmp\LTB_20240319_TFXQ_01",
        r"H:\ChengJiXiangMu\LTB_20240319_TFXQ_tmp\LTB_20240319_TFXQ_1_1",
        r"H:\ChengJiXiangMu\LTB_20240319_TFXQ_tmp\LTB_20240319_TFXQ_10",
        r"H:\ChengJiXiangMu\LTB_20240319_TFXQ_tmp\LTB_20240319_TFXQ_11",
    ]
    to_vrt_fn = r"H:\ChengJiXiangMu\LTB_20240319_TFXQ_SARGLCM.vrt"
    to_dirname = os.path.splitext(to_vrt_fn)[0] + "_VRTS"
    to_cmd_fn = changext(to_vrt_fn, "_cmd.txt")
    if not os.path.isdir(to_dirname):
        os.mkdir(to_dirname)
    writeTextLine(to_cmd_fn, "", mode="w")
    vrt_fns = []
    for fn in fns:
        gr = GDALRaster(fn)
        for n in range(gr.n_channels):
            to_fn = changext(fn, "_n{}.vrt".format(n + 1))
            to_fn = changefiledirname(to_fn, to_dirname)
            cmd_line = "gdalbuildvrt -b {0} \"{1}\" \"{2}\"".format(n + 1, to_fn, fn)
            print(cmd_line)
            writeTextLine(to_cmd_fn, cmd_line)
            vrt_fns.append(to_fn)
    filelist_fn = os.path.join(to_dirname, "filelist.txt")
    writeLines(filelist_fn, vrt_fns)
    cmd_line = "gdalbuildvrt -separate -input_file_list {0} {1}".format(filelist_fn, to_vrt_fn)
    print(cmd_line)
    writeTextLine(to_cmd_fn, cmd_line)
    cmd_line = "gdal_translate -of GTiff -ot Float32  \"{0}\" \"{1}\"".format(to_vrt_fn,
                                                                              changext(to_vrt_fn, "_tif.tif"))
    print(cmd_line)
    writeTextLine(to_cmd_fn, cmd_line)


def cjMkTu():
    def func1():
        gdic = GDALDrawImagesColumns((120, 120))
        gdic.fontdict["size"] = 16
        dfn = DirFileName(r"H:\ChengJiXiangMu")

        gdic.addRCC("RGB", dfn.fn(r"clip_510110天府新区0304.tif"), channel_list=[2, 1, 0], win_size=(600, 600))
        gdic.addRCC("LTB_202402", dfn.fn(r"LTB_202402_TFXQ.tif"), channel_list=[0], )
        gdic.addRCC("LTB_202404", dfn.fn(r"LTB_202404_TFXQ.tif"), channel_list=[0], )
        gdic.addRCC("LTB_202401", dfn.fn(r"LTB_20240115_TFXQ.tif"), channel_list=[0], )
        gdic.addRCC("LTB_202403", dfn.fn(r"LTB_20240319_TFXQ.tif"), channel_list=[0], )

        column_names = ["RGB", "LTB_202401", "LTB_202402", "LTB_202403", "LTB_202404", ]
        row_names = []

        def add1(name, x, y):
            n_row = len(row_names)
            row_names.append(name)
            gdic.addAxisDataXY(n_row, 0, "RGB", x, y, min_list=[0, 0, 0], max_list=[255, 255, 255], is_trans=True)
            gdic.addAxisDataXY(n_row, 1, "LTB_202401", x, y, min_list=[0], max_list=[255])
            gdic.addAxisDataXY(n_row, 2, "LTB_202402", x, y, min_list=[0], max_list=[255])
            gdic.addAxisDataXY(n_row, 3, "LTB_202403", x, y, min_list=[0], max_list=[255])
            gdic.addAxisDataXY(n_row, 4, "LTB_202404", x, y, min_list=[0], max_list=[255])

        add1("(1)    ", 104.122449, 30.439997)
        add1("(2)    ", 104.070022, 30.420251)
        add1("(3)    ", 104.064845, 30.429182)
        add1("(4)    ", 104.0861267, 30.3857188)

        gdic.draw(n_columns_ex=1.6, n_rows_ex=1.6, row_names=row_names, column_names=column_names)

        plt.show()

    def func2():
        gdic = GDALDrawImagesColumns((120, 120))
        gdic.fontdict["size"] = 16
        dfn = DirFileName(r"H:\ChengJiXiangMu")
        rf1_dfn = DirFileName(r"H:\ChengJiXiangMu\Imdc\RF1")

        gdic.addRCC("RGB", dfn.fn(r"clip_510110天府新区0304.tif"), channel_list=[2, 1, 0], win_size=(600, 600))
        gdic.addRCC("LTB_202402", dfn.fn(r"LTB_202402_TFXQ.tif"), channel_list=[0], )
        gdic.addRCC("LTB_202402_IMDC", rf1_dfn.fn(r"LTB_202402_TFXQ_SARGLCM_tif_imdc1.tif"), channel_list=[0], )
        gdic.addRCC("LTB_202404", dfn.fn(r"LTB_202404_TFXQ.tif"), channel_list=[0], )
        gdic.addRCC("LTB_202404_IMDC", rf1_dfn.fn(r"LTB_202404_TFXQ_SARGLCM_tif_imdc1.tif"), channel_list=[0], )
        gdic.addRCC("LTB_202401", dfn.fn(r"LTB_20240115_TFXQ.tif"), channel_list=[0], )
        gdic.addRCC("LTB_202401_IMDC", rf1_dfn.fn(r"LTB_20240115_TFXQ_SARGLCM_tif_imdc1.tif"), channel_list=[0], )
        gdic.addRCC("LTB_202403", dfn.fn(r"LTB_20240319_TFXQ.tif"), channel_list=[0], )
        gdic.addRCC("LTB_202403_IMDC", rf1_dfn.fn(r"LTB_20240319_TFXQ_SARGLCM_tif_imdc1.tif"), channel_list=[0], )

        column_names = [
            "RGB",
            "202401", "IMDC",
            "202402", "IMDC",
            "202403", "IMDC",
            "202404", "IMDC",
        ]
        row_names = []

        def add1(name, x, y):
            n_row = len(row_names)
            row_names.append(name)
            gdic.addAxisDataXY(n_row, 0, "RGB", x, y, min_list=[0, 0, 0], max_list=[255, 255, 255], is_trans=True)
            gdic.addAxisDataXY(n_row, 1, "LTB_202401", x, y, min_list=[0], max_list=[255])
            gdic.addAxisDataXY(n_row, 2, "LTB_202401_IMDC", x, y, min_list=[1], max_list=[2])
            gdic.addAxisDataXY(n_row, 3, "LTB_202402", x, y, min_list=[0], max_list=[255])
            gdic.addAxisDataXY(n_row, 4, "LTB_202402_IMDC", x, y, min_list=[1], max_list=[2])
            gdic.addAxisDataXY(n_row, 5, "LTB_202403", x, y, min_list=[0], max_list=[255])
            gdic.addAxisDataXY(n_row, 6, "LTB_202403_IMDC", x, y, min_list=[1], max_list=[2])
            gdic.addAxisDataXY(n_row, 7, "LTB_202404", x, y, min_list=[0], max_list=[255])
            gdic.addAxisDataXY(n_row, 8, "LTB_202404_IMDC", x, y, min_list=[1], max_list=[2])

        add1("(1)    ", 104.082024, 30.409161)
        add1("(2)    ", 104.120780, 30.379780)

        gdic.draw(n_columns_ex=1.6, n_rows_ex=1.6, row_names=row_names, column_names=column_names)

        plt.show()

    def func3():
        gdic = GDALDrawImagesColumns((200, 200))
        gdic.fontdict["size"] = 16
        dfn = DirFileName(r"H:\ChengJiXiangMu")
        rf_dfn = DirFileName(r"H:\ChengJiXiangMu\Models")

        gdic.addRCC("RGB", dfn.fn(r"clip_510110天府新区0304.tif"), channel_list=[2, 1, 0], win_size=(1000, 1000))
        gdic.addRCC("LTB_202402", dfn.fn(r"LTB_202402_TFXQ.tif"), channel_list=[0], )
        gdic.addRCC("LTB_202404", dfn.fn(r"LTB_202404_TFXQ.tif"), channel_list=[0], )
        gdic.addRCC("LTB_202401", dfn.fn(r"LTB_20240115_TFXQ.tif"), channel_list=[0], )
        gdic.addRCC("LTB_202403", dfn.fn(r"LTB_20240319_TFXQ.tif"), channel_list=[0], )

        gdic.addRCC("SAR", rf_dfn.fn(r"20240602H174134\DSSAR_imdc.tif"), channel_list=[0], )
        gdic.addRCC("SAR_GLCM", rf_dfn.fn(r"20240602H191022\DSSARGLCM_imdc.tif"), channel_list=[0], )

        column_names = ["RGB", "202401", "202402", "202403", "202404", "SAR", "SAR_GLCM", ]
        row_names = []

        def add1(name, x, y):
            n_row = len(row_names)
            row_names.append(name)
            gdic.addAxisDataXY(n_row, 0, "RGB", x, y, min_list=[0, 0, 0], max_list=[255, 255, 255], is_trans=True)
            gdic.addAxisDataXY(n_row, 1, "LTB_202401", x, y, min_list=[0], max_list=[255])
            gdic.addAxisDataXY(n_row, 2, "LTB_202402", x, y, min_list=[0], max_list=[255])
            gdic.addAxisDataXY(n_row, 3, "LTB_202403", x, y, min_list=[0], max_list=[255])
            gdic.addAxisDataXY(n_row, 4, "LTB_202404", x, y, min_list=[0], max_list=[255])
            gdic.addAxisDataXY(n_row, 5, "SAR", x, y, min_list=[1], max_list=[2])
            gdic.addAxisDataXY(n_row, 6, "SAR_GLCM", x, y, min_list=[1], max_list=[2])

        add1("(1)    ", 104.082024, 30.409161)
        add1("(2)    ", 104.084352, 30.444132)
        add1("(3)    ", 104.080739, 30.423285)
        add1("(4)    ", 104.097766, 30.429402)
        add1("(5)    ", 104.051552, 30.397804)
        add1("(6)    ", 104.089080, 30.399231)

        gdic.draw(n_columns_ex=1.6, n_rows_ex=1.6, row_names=row_names, column_names=column_names)

        plt.show()

    def func4():
        gdic = GDALDrawImagesColumns((200, 200))
        gdic.fontdict["size"] = 16
        dfn = DirFileName(r"H:\ChengJiXiangMu")
        rf_dfn = DirFileName(r"H:\ChengJiXiangMu\Models")

        gdic.addRCC("RGB", dfn.fn(r"clip_510110天府新区0304.tif"), channel_list=[2, 1, 0], win_size=(1000, 1000))
        gdic.addRCC("LTB_202402", dfn.fn(r"LTB_202402_TFXQ.tif"), channel_list=[0], )
        gdic.addRCC("LTB_202404", dfn.fn(r"LTB_202404_TFXQ.tif"), channel_list=[0], )
        gdic.addRCC("LTB_202401", dfn.fn(r"LTB_20240115_TFXQ.tif"), channel_list=[0], )
        gdic.addRCC("LTB_202403", dfn.fn(r"LTB_20240319_TFXQ.tif"), channel_list=[0], )

        gdic.addRCC(
            "SAR",
            r"H:\ChengJiXiangMu\CJTrainImdc_Mods\20240602H181526\ChengJiModel_epoch26_imdc1.tif",
            channel_list=[0],
        )

        column_names = ["RGB", "202401", "202402", "202403", "202404", "Classification", ]
        row_names = []

        def add1(name, x, y):
            n_row = len(row_names)
            row_names.append(name)
            gdic.addAxisDataXY(n_row, 0, "RGB", x, y, min_list=[0, 0, 0], max_list=[255, 255, 255], is_trans=True)
            gdic.addAxisDataXY(n_row, 1, "LTB_202401", x, y, min_list=[0], max_list=[255])
            gdic.addAxisDataXY(n_row, 2, "LTB_202402", x, y, min_list=[0], max_list=[255])
            gdic.addAxisDataXY(n_row, 3, "LTB_202403", x, y, min_list=[0], max_list=[255])
            gdic.addAxisDataXY(n_row, 4, "LTB_202404", x, y, min_list=[0], max_list=[255])
            gdic.addAxisDataXY(n_row, 5, "SAR", x, y, min_list=[1], max_list=[2])

        add1("(1)    ", 104.082024, 30.409161)
        add1("(2)    ", 104.084352, 30.444132)
        add1("(3)    ", 104.080739, 30.423285)
        add1("(4)    ", 104.097766, 30.429402)
        add1("(5)    ", 104.051552, 30.397804)
        add1("(6)    ", 104.089080, 30.399231)

        gdic.draw(n_columns_ex=1.6, n_rows_ex=1.6, row_names=row_names, column_names=column_names)

        plt.show()

    func4()


def rfRun():
    cj_rf = CJ_RFMain(
        r"H:\ChengJiXiangMu\Samples\2\zyh_spl2_1.csv",
        # r"H:\ChengJiXiangMu\LTB_1_4_GLCM.vrt",
        raster_tile_fns=[
            r"J:\data\tiles\cdtf_sar_glcm_envi_1_1.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_1_2.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_1_3.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_1_4.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_1_5.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_1_6.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_2_1.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_2_2.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_2_3.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_2_4.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_2_5.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_2_6.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_3_1.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_3_2.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_3_3.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_3_4.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_3_5.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_3_6.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_4_1.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_4_2.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_4_3.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_4_4.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_4_5.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_4_6.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_5_1.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_5_2.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_5_3.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_5_4.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_5_5.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_5_6.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_6_1.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_6_2.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_6_3.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_6_4.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_6_5.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_6_6.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_7_1.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_7_2.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_7_3.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_7_4.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_7_5.tif",
            r"J:\data\tiles\cdtf_sar_glcm_envi_7_6.tif",
        ]
    )
    json_dict = FRW(r"H:\ChengJiXiangMu\Samples\2\experiments.json").readJson()
    k = sys.argv[1]
    print(k, json_dict[k])
    cj_rf.train(k, json_dict[k])
    cj_rf.imdcTiles()


def cjtrainimdcRun():
    cjti = CJTrainImdc()
    cjti.main()
    # cjti.train()
    cjti.imdc()


def imdc():
    feats_type = sys.argv[1]
    mod_fn = sys.argv[2]
    im_fn = sys.argv[3]
    fn = getfilenamewithoutext(im_fn)
    to_fn = os.path.join(os.path.dirname(mod_fn), "{}_imdc.tif".format(fn))

    print("feats_type:", feats_type)
    print("mod_fn    :", mod_fn)
    print("im_fn     :", im_fn)
    print("to_fn     :", to_fn)

    json_dict = FRW(r"H:\ChengJiXiangMu\Samples\2\experiments.json").readJson()
    print(feats_type, json_dict[feats_type])
    clf = joblib.load(mod_fn)
    print("clf", clf)

    gi = GDALImdc(im_fn)
    gi.sfm.initCallBacks()
    gi.imdc1(clf, to_imdc_fn=to_fn, fit_names=json_dict[feats_type])


if __name__ == "__main__":
    # tiffAddColorTable(r"H:\ChengJiXiangMu\Models\20240615H100143\cdtf_sar_glcm_envi_imdc.tif", 1, {1: (0, 0, 0), 2: (255, 255, 255)})
    main()

    r"""
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Temp.ChengJiXiangMu import main; main()" 
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Temp.ChengJiXiangMu import rfRun; rfRun()"  DSSARGLCM
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Temp.ChengJiXiangMu import cjtrainimdcRun; cjtrainimdcRun()" 
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Temp.ChengJiXiangMu import imdc; imdc()" 
    """
