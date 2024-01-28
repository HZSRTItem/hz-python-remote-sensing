# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ZY5MNicDL.py
@Time    : 2024/1/11 16:35
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of ZY5MNIC
-----------------------------------------------------------------------------"""

import os.path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from SRTCodes.GDALRasterClassification import GDALRasterPrediction
from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.PytorchModelTraining import PytorchCategoryTraining
from SRTCodes.SRTData import SRTDataset
from SRTCodes.Utils import DirFileName, writeTexts, Jdt
from ZY5M.JPZ5MMain import conv_bn_act
from ZY5M.ZY5MWarp import ZY5MGDALRasterWarp

JPZ5M_XIANLI21_DFN = DirFileName(r"M:\jpz2m")
torch.cuda.set_device(0)

YEAR = int(2020)
print("-" * 22, end=" ")
print(YEAR, end=" ")
print("-" * 22)


def minmax01(d, d_min, d_max):
    d = np.clip(d, d_min, d_max)
    d = (d - d_min) / (d_max - d_min)
    return d


def dataDeal2(d):
    d[0] = minmax01(d[0], 74.71188917831876, 360.6762496700132)
    d[1] = minmax01(d[1], 44.21292113593984, 404.2524694067898)
    d[2] = minmax01(d[2], 0, 368.3457842153404)
    d[3] = minmax01(d[3], 0, 665.1536271227153)
    return d


dataDeal_dict = {2014: 550, 2016: 650, 2017: 650,
                 2018: 750, 2019: 700, 2020: 650,
                 2021: 1000, 2022: 1500, 2023: 650}


def dataDeal(d):
    # d = np.clip(d, 0, dataDeal_dict[YEAR])
    # d = d / dataDeal_dict[YEAR]
    d = d[:4, :, :]
    d = d / 1600
    return d


def net1(channels, channel_0=1, kernel_size=3, padding=1):
    cba_list = []
    out_channel = channels[-1] * channel_0
    for i in range(len(channels) - 1):
        cba_list.append(conv_bn_act(
            in_channels=channels[i] * channel_0,
            out_channels=channels[i + 1] * channel_0,
            kernel_size=kernel_size, padding=padding))
    return nn.Sequential(*tuple(cba_list)), out_channel


def pool(channels, kernel_size=2, stride=2):
    return nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride),
                         nn.Sigmoid())


class ZY2MAdmj_Network(nn.Module):

    def __init__(self, in_channel=3, win_size=16, n_category=1):
        super().__init__()

        self.cbas1, in_channel = net1([1, 2, 8, 16], in_channel)

        self.cbas2, in_channel = net1([1, 2, 1, 2], in_channel)

        self.conv_pool2 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=2, stride=2)
        self.conv_act = nn.Sigmoid()

        self.cbas3, in_channel = net1([1, 2, 1, 2], in_channel)

        self.fc = nn.Linear(int(1024), n_category)

    def forward(self, x):
        x = self.cbas1(x)
        x = self.cbas2(x)
        x = self.conv_pool2(x)
        x = self.conv_act(x)
        x = self.cbas3(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ZY2MAdmj_Network_res(nn.Module):

    def __init__(self, in_channel=3, win_size=16, n_category=1):
        super().__init__()

        self.cbas1, in_channel = net1([1, 2, 8, 16, 32], in_channel)
        self.cbas2, in_channel = net1([1, 1, 1, 1], in_channel)
        self.cbas3, in_channel = net1([1, 1, 1, 1], in_channel)
        self.pool1 = pool(in_channel)

        self.fc = nn.Linear(int(512), n_category)

    def forward(self, x):
        x = self.cbas1(x)
        x = x + self.cbas2(x)
        x = self.pool1(x)
        x = x + self.cbas3(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ZY2MAdmj_Network2(nn.Module):

    def __init__(self, in_channel=3, win_size=16, n_category=1):
        super().__init__()

        # self.conv3d_1 = nn.Conv3d(in_channels=in_channel, out_channels=in_channel, kernel_size=(3, 3, 3), stride=1,
        #                           padding=(0, 1, 1))
        # self.bn3d_1 = nn.BatchNorm3d(in_channel)
        # self.relu3d_1 = nn.ReLU()
        # self.flatten_1 = nn.Flatten(start_dim=1, end_dim=2)

        self.conv_front1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel * 2, kernel_size=3, padding=1)
        self.bn_front1 = nn.BatchNorm2d(in_channel * 2)
        self.relu_front1 = nn.ReLU()

        in_channel = in_channel * 2

        self.conv_front2 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel * 2, kernel_size=3, padding=1)
        self.bn_front2 = nn.BatchNorm2d(in_channel * 2)
        self.relu_front2 = nn.ReLU()

        in_channel = in_channel * 2

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel * 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channel * 2)
        self.relu1 = nn.ReLU()
        self.max_pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)

        win_size = win_size / 2
        in_channel = in_channel * 2

        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channel * 2)
        self.relu2 = nn.ReLU()
        self.max_pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)

        win_size = win_size / 2
        in_channel = in_channel * 2

        self.conv3 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel * 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(in_channel * 2)
        self.relu3 = nn.ReLU()
        self.max_pooling3 = nn.MaxPool2d(kernel_size=2, stride=2)

        win_size = int(win_size / 2)
        in_channel = in_channel * 2
        fc_size = in_channel * win_size * win_size

        self.fc1 = nn.Linear(int(fc_size), int(fc_size / 2))
        self.fc1_relu = nn.ReLU()
        self.fc2 = nn.Linear(int(fc_size / 2), int(fc_size / 4))
        self.fc2_relu = nn.ReLU()
        self.fc3 = nn.Linear(int(fc_size / 4), n_category)

    def forward(self, x):
        x = self.conv_front1(x)
        x = self.bn_front1(x)
        x = self.relu_front1(x)

        x = self.conv_front2(x)
        x = self.bn_front2(x)
        x = self.relu_front2(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.max_pooling1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.max_pooling2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.max_pooling3(x)

        x = x.view((x.size(0), -1))

        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.fc2(x)
        x = self.fc2_relu(x)
        x = self.fc3(x)

        return x


class ZY2MAdmj_Network3(nn.Module):

    def __init__(self, in_channel=3, win_size=16, n_category=1):
        super().__init__()

        # self.conv3d_1 = nn.Conv3d(in_channels=in_channel, out_channels=in_channel, kernel_size=(3, 3, 3), stride=1,
        #                           padding=(0, 1, 1))
        # self.bn3d_1 = nn.BatchNorm3d(in_channel)
        # self.relu3d_1 = nn.ReLU()
        # self.flatten_1 = nn.Flatten(start_dim=1, end_dim=2)

        self.cbas1, in_channel = net1([1, 2, 8, 16], in_channel)
        self.max_pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.cbas2, in_channel = net1([1, 2, 1, 2], in_channel)
        self.max_pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        # self.cbas3, in_channel = net1([1, 2, 1, 2], in_channel, kernel_size=1, padding=0)
        # self.max_pooling3 = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        # self.cbas4, in_channel = net1([2, 1], in_channel)

        self.fc = nn.Linear(int(512), n_category)

    def forward(self, x):
        x = self.cbas1(x)
        x = self.max_pooling1(x)
        x = self.cbas2(x)
        x = self.max_pooling2(x)
        # x = self.cbas3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ZY5MNIC_Network(nn.Module):

    def __init__(self, in_channel=4, win_size=16, n_category=1):
        super(ZY5MNIC_Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=8, )

    def forward(self, x):
        return x


class ZY2MAdmj_Loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.sigmoid_act = nn.Sigmoid()

    def forward(self, x, y, lamd=1):
        p = self.sigmoid_act(x)
        y = y.view((y.size(0), 1))
        loss = -torch.mean(lamd * (y * torch.log(p) + (1 - y) * torch.log(1 - p)))
        return loss


class ZY2MAdmj_DataSet(SRTDataset, Dataset):

    def __init__(self, is_test=False, spl_fn="", train_d_fn="", cname="CATEGORY"):
        super().__init__()

        self.spl_fn = spl_fn
        self.train_d_fn = train_d_fn
        self.df = pd.read_csv(self.spl_fn)
        is_tests = self.df["TEST"].values
        categorys = self.df[cname].values.tolist()
        self.addCategory("NOIS")
        self.addCategory("IS")
        self.addNPY(categorys, self.train_d_fn)
        self._isTest(is_test, is_tests)

    def _isTest(self, is_test, is_tests):
        datalist = []
        category_list = []
        for i in range(len(self.datalist)):
            if is_test:
                if is_tests[i] == 1:
                    datalist.append(self.datalist[i])
                    category_list.append(self.category_list[i])
            else:
                if is_tests[i] == 0:
                    datalist.append(self.datalist[i])
                    category_list.append(self.category_list[i])
        self.datalist = datalist
        self.category_list = category_list

    def __getitem__(self, index) -> T_co:
        return self.get(index)

    def get(self, index):
        x = self.datalist[index]
        x = dataDeal(x)

        y = self.category_list[index]
        return x, y


class ZY2MAdmj_PytorchTraining(PytorchCategoryTraining):

    def __init__(self, n_category, model_dir=None, category_names=None, epochs=10, device=None, n_test=100):
        super().__init__(n_category=n_category, model_dir=model_dir, category_names=category_names, epochs=epochs,
                         device=device, n_test=n_test)

    def logisticToCategory(self, logts):
        logts = torch.sigmoid(logts)
        logts = (logts.cpu().numpy().T[0] > 0.5) * 1
        return logts


class ZY2MAdmj_GDALRasterPrediction(GDALRasterPrediction):

    def __init__(self, geo_fn):
        super(ZY2MAdmj_GDALRasterPrediction, self).__init__(geo_fn)
        self.device = "cuda:0"
        self.is_category = False
        self.number_pred = 15000

    def predict(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        # y = np.ones(x.shape[0])
        n = x.shape[0]

        x = x.astype("float32")
        x = torch.from_numpy(x)
        x = x.to(self.device)
        y = torch.zeros((n, 1), dtype=torch.float)
        y = y.to(self.device)

        self.model.eval()
        with torch.no_grad():
            for i in range(0, n, self.number_pred):
                y_temp = self.model(x[i:i + self.number_pred, :])
                y[i:i + self.number_pred, :] = y_temp
            y = functional.sigmoid(y)
        y = y.cpu().numpy()
        y = y.T[0]
        if self.is_category:
            y = (y > 0.5) * 1

        return y

    def preDeal(self, row, column_start, column_end):
        d_row = self.d[:, row, column_start:column_end]
        return np.ones(d_row.shape[1], dtype="bool")


class ZY2MAdmj_Main:

    def __init__(self):
        self.this_dirname = self.mkdir(r"F:\ProjectSet\Huo\jpz5m4nian\imdcTest\1")
        self.model_dir = self.mkdir(os.path.join(self.this_dirname, "Mods"))
        self.n_category = 2
        self.category_names = ["NOIS", "IS"]
        self.epochs = 100
        self.device = "cuda:0"
        self.n_test = 10
        # X Y CATEGORY TEST 需要这四列 训练是1
        self.csv_fn = r"F:\ProjectSet\Huo\jpz5m4nian\imdcTest\1\jpz5m4_t17_0_catto_df.csv"
        self.npy_fn = r"F:\ProjectSet\Huo\jpz5m4nian\imdcTest\1\jpz5m4_t17_0_catto_df.csv.npy"
        self.test_ds = None
        self.train_ds = None
        self.win_size = 5
        self.mod = ZY2MAdmj_Network_res(in_channel=4, win_size=self.win_size, n_category=1)
        self.loss = ZY2MAdmj_Loss()
        self.geo_raster = r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_17_im1.tif"
        # self.geo_raster = r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_17_im0.tif"

        print("geo_raster", self.geo_raster)

        ...

    def mkdir(self, dirname):
        dirname = os.path.abspath(dirname)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        return dirname

    def train(self):

        print("spl_fn    :", self.csv_fn)
        print("raster_fn :", self.geo_raster)
        print("train_d_fn:", self.npy_fn)
        print("spl_size  :", self.win_size)

        self.train_ds = ZY2MAdmj_DataSet(is_test=False, spl_fn=self.csv_fn, train_d_fn=self.npy_fn, cname="category")
        self.test_ds = ZY2MAdmj_DataSet(is_test=True, spl_fn=self.csv_fn, train_d_fn=self.npy_fn, cname="category")

        pytorch_training = ZY2MAdmj_PytorchTraining(
            n_category=self.n_category,
            model_dir=self.model_dir,
            category_names=self.category_names,
            epochs=self.epochs,
            device=self.device,
            n_test=self.n_test
        )

        pytorch_training.trainLoader(self.train_ds, batch_size=128, shuffle=True)
        pytorch_training.testLoader(self.test_ds, batch_size=128, shuffle=False)
        pytorch_training.addModel(self.mod)
        pytorch_training.addCriterion(self.loss)
        pytorch_training.addOptimizer(lr=0.001, eps=0.00001)
        print("model_dir", pytorch_training.model_dir)
        save_fn = os.path.join(pytorch_training.model_dir, "save.txt")
        writeTexts(save_fn, "spl_fn    :", self.csv_fn, mode="a", end="\n")
        writeTexts(save_fn, "raster_fn :", self.geo_raster, mode="a", end="\n")
        writeTexts(save_fn, "train_d_fn:", self.npy_fn, mode="a", end="\n")
        writeTexts(save_fn, "spl_size  :", self.win_size, mode="a", end="\n")
        pytorch_training.train()

    def imdc(self, raster_dirname):
        mod_dirname = "20231119H185046"
        imdc_dirname = os.path.join(self.model_dir, mod_dirname, "model_20_imdc1")
        if not os.path.isdir(imdc_dirname):
            os.mkdir(imdc_dirname)
        imdc_fn = os.path.join(self.model_dir, mod_dirname, "model_20_imdc1.tif")
        mod_fn = os.path.join(self.model_dir, mod_dirname, "model_20.pth")
        np_type = "int8"
        self.mod.load_state_dict(torch.load(mod_fn))
        self.mod.to(self.device)
        for f in os.listdir(raster_dirname):
            if os.path.splitext(f)[1] == '.tif':
                print(f)
                geo_raster = os.path.join(raster_dirname, f)
                imdc_fn = os.path.join(imdc_dirname, f)
                if os.path.isfile(imdc_fn):
                    print("RasterClassification: 100%")
                    continue
                grp = ZY2MAdmj_GDALRasterPrediction(geo_raster)
                grp.is_category = True
                grp.run(imdc_fn=imdc_fn, np_type=np_type, mod=self.mod,
                        spl_size=[self.win_size, self.win_size],
                        row_start=self.win_size + 6, row_end=-(self.win_size + 6),
                        column_start=self.win_size + 6, column_end=-(self.win_size + 6),
                        n_one_t=20000, data_deal=dataDeal)

    def imdcOne(self, mod_fn=None):
        grp = ZY2MAdmj_GDALRasterPrediction(self.geo_raster)
        if mod_fn is not None:
            mod_dirname = mod_fn.split("\\")[4]
            imdc_fn = mod_fn + "_imdc.tif"
        else:
            # "H:\JPZ\ZY2MAdmj\Mods\20231119H191912\model_56.pth"
            mod_dirname = "20231119H191912"
            imdc_fn = os.path.join(self.model_dir, mod_dirname, "imdc1.tif")
            mod_fn = os.path.join(self.model_dir, mod_dirname, "model_56.pth")

        print("mod_dirname:", mod_dirname)
        print("imdc_fn    :", imdc_fn)
        print("mod_fn     :", mod_fn)

        self.imdcOneRun(grp, imdc_fn, mod_fn)

    def imdcOneRun(self, grp, imdc_fn, mod_fn):
        grp.is_category = True
        np_type = "int8"
        self.mod.load_state_dict(torch.load(mod_fn))
        self.mod.to(self.device)
        grp.run(imdc_fn=imdc_fn, np_type=np_type, mod=self.mod,
                spl_size=[self.win_size, self.win_size],
                row_start=10, row_end=-10,
                column_start=10, column_end=-10,
                n_one_t=15000, data_deal=dataDeal)

    def warp(self):
        zy5m_grw = ZY5MGDALRasterWarp()
        # "K:\zhongdianyanfa\jpz_5\good5m2017-2018\xianli\GF1_PMS2_E103.8_N13.4_20170216_L1A0002188979\spl1.csv"
        # zy5m_grw.warpImage(
        #     JPZ5M_XIANLI21_DFN.fn(r"GF1_PMS2_E103.8_N13.4_20170216_L1A0002188979\spl1.csv"),
        #     JPZ5M_XIANLI21_DFN.fn(r"GF1_PMS2_E103.8_N13.4_20170216_L1A0002188979\GF1_xianli_1.tif"),
        #     JPZ5M_XIANLI21_DFN.fn(r"GF1_PMS2_E103.8_N13.4_20170216_L1A0002188979\GF1_xianli_2.tif"),
        # )
        # GF2_PMS1_E103.8_N13.3_20170328_L1A0002272453
        zy5m_grw.warpImage(
            JPZ5M_XIANLI21_DFN.fn(r"GF1C_PMS_E104.0_N13.5_20221019_L1A1021971399\spl1.csv"),
            JPZ5M_XIANLI21_DFN.fn(r"GF1C_PMS_E104.0_N13.5_20221019_L1A1021971399\GF1C_xianli_1.tif"),
            JPZ5M_XIANLI21_DFN.fn(r"GF1C_PMS_E104.0_N13.5_20221019_L1A1021971399\GF1C_xianli_2.tif"),
        )

    def sampleNPY(self):
        # 使用CSV文件在影像中提取样本的数据
        spl_fn = self.csv_fn
        raster_fn = self.geo_raster
        train_d_fn = self.npy_fn
        spl_size = [self.win_size, self.win_size]

        print("spl_fn    :", spl_fn)
        print("raster_fn :", raster_fn)
        print("train_d_fn:", train_d_fn)
        print("spl_size  :", spl_size)

        df = pd.read_csv(spl_fn)
        gr = GDALRaster(raster_fn)
        d = np.zeros([len(df), gr.n_channels, spl_size[0], spl_size[1]])
        print(d.shape)
        for i in range(len(df)):
            x = df["X"][i]
            y = df["Y"][i]
            d[i, :] = gr.readAsArrayCenter(x, y, win_row_size=spl_size[0], win_column_size=spl_size[1],
                                           interleave="band",
                                           is_geo=True, is_trans=True)
            if i % 500 == 0:
                print(i, end=" ")
        np.save(train_d_fn, d)

    def func1(self):
        gr = GDALRaster(self.geo_raster)
        d = gr.readAsArray()
        n = 4
        for i in range(len(d)):
            mean = np.mean(d[i])
            std = np.std(d[i])
            print("d[{0}] = minmax01(d[{0}], {1}, {2})".format(i, mean - std * n, mean + std * n))

    def samplingCSVRasters(self):
        raster_fns = [
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_17_im0.tif",
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_17_im1.tif",
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_17_im2.tif",
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_17_im3.tif",
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_17_im4.tif",
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_17_im5.tif",
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_17_im6.tif",
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_17_im7.tif",
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_17_im8.tif",
        ]

        spl_fn = self.csv_fn
        raster_fn = self.geo_raster
        train_d_fn = self.npy_fn
        spl_size = [self.win_size, self.win_size]

        print("spl_fn    :", spl_fn)
        print("raster_fn :", raster_fn)
        print("train_d_fn:", train_d_fn)
        print("spl_size  :", spl_size)

        def func1():
            csv_fn = r"F:\ProjectSet\Huo\jpz5m4nian\imdcTest\1\jpz5m4_t17_0_cat.csv"

            gr_list = [GDALRaster(raster_fn) for raster_fn in raster_fns]
            df = pd.read_csv(csv_fn)
            jdt = Jdt(len(df), "Sampling CSV Rasters")
            jdt.start()
            gr = gr_list[0]
            to_df = []

            for i in range(len(df)):
                line = df.loc[i]
                x, y = float(line["X"]), float(line["Y"])
                if gr.isGeoIn(x, y):
                    to_df.append(i)
                else:
                    for gr in gr_list:
                        if gr.isGeoIn(x, y):
                            to_df.append(i)
                            break
                jdt.add()
            jdt.end()

            to_df = df.loc[to_df]
            to_df.to_csv(csv_fn + "to_df.csv", index=False)

        def func2():
            csv_fn = r"F:\ProjectSet\Huo\jpz5m4nian\imdcTest\1\jpz5m4_t17_0_catto_df.csv"

            gr_list = [GDALRaster(raster_fn) for raster_fn in raster_fns]
            df = pd.read_csv(csv_fn)
            jdt = Jdt(len(df), "Sampling CSV Rasters")
            jdt.start()
            gr = gr_list[0]
            # d = np.zeros([len(df), gr.n_channels, spl_size[0], spl_size[1]])
            d_list = []
            to_list = []

            for i in range(len(df)):
                line = df.loc[i]
                x, y = float(line["X"]), float(line["Y"])

                d0 = gr.readAsArrayCenter(x, y, win_row_size=spl_size[0], win_column_size=spl_size[1],
                                               interleave="band",
                                               is_geo=True, is_trans=True)
                if d0 is not None:
                    to_list.append(i)
                    d_list.append([d0])
                else:
                    for gr in gr_list:
                        d0 = gr.readAsArrayCenter(x, y, win_row_size=spl_size[0], win_column_size=spl_size[1],
                                                  interleave="band",
                                                  is_geo=True, is_trans=True)
                        if d0 is not None:
                            to_list.append(i)
                            d_list.append([d0])
                            break

                jdt.add()
            jdt.end()

            to_df = df.loc[to_list]
            to_df.to_csv(csv_fn + "to_df.csv", index=False)
            d = np.concatenate(d_list)
            np.save(csv_fn + ".npy", d)

        func2()

    def imdcs(self, mod_fn):

        raster_fns = [
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_17_im0.tif",
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_17_im1.tif",
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_17_im2.tif",
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_17_im3.tif",
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_17_im4.tif",
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_17_im5.tif",
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_17_im6.tif",
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_17_im7.tif",
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_17_im8.tif",
        ]

        for raster_fn in raster_fns:
            grp = ZY2MAdmj_GDALRasterPrediction(raster_fn)
            imdc_fn = mod_fn + os.path.splitext(os.path.split(raster_fn)[1])[0] + ".tif"
            print(imdc_fn)
            self.imdcOneRun(grp, imdc_fn, mod_fn)


def imdcOne(mod_fn):
    zy2m_admj = ZY2MAdmj_Main()
    zy2m_admj.imdcOne(mod_fn)


def imdcOne2():
    time1 = time.time()
    zy2m_admj = ZY2MAdmj_Main()
    if len(sys.argv) == 2:
        zy2m_admj.imdcOne(sys.argv[1])
    if len(sys.argv) == 3:
        zy2m_admj.imdcOne(sys.argv[2])
    time2 = time.time()
    print("Time: {0}s".format(time2 - time1))


def sampleNPY():
    zy2m_admj = ZY2MAdmj_Main()
    zy2m_admj.sampleNPY()


def train():
    zy2m_admj = ZY2MAdmj_Main()
    zy2m_admj.train()


def main():
    zy2m_admj = ZY2MAdmj_Main()
    # zy2m_admj.sampleNPY()
    # zy2m_admj.train()
    # zy2m_admj.imdcOne(r"F:\ProjectSet\Huo\jpz5m4nian\imdcTest\1\Mods\20240111H174525\model_epoch_90.pth")
    # zy2m_admj.func1()
    # zy2m_admj.samplingCSVRasters()
    zy2m_admj.imdcs(r"F:\ProjectSet\Huo\jpz5m4nian\imdcTest\1\Mods\20240111H180036\model_epoch_30.pth")
    zy2m_admj.imdcs(r"F:\ProjectSet\Huo\jpz5m4nian\imdcTest\1\Mods\20240111H180036\model_epoch_66.pth")
    zy2m_admj.imdcs(r"F:\ProjectSet\Huo\jpz5m4nian\imdcTest\1\Mods\20240111H180036\model_epoch_88.pth")

    pass


if __name__ == "__main__":
    main()
