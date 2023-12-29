# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : JPZ5MXianLi17.py
@Time    : 2023/10/4 9:49
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of JPZ3MXianLi17
-----------------------------------------------------------------------------"""
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from osgeo import gdal
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from torch.nn import functional
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from SRTCodes.GDALRasterClassification import GDALRasterPrediction
from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.PytorchModelTraining import PytorchCategoryTraining
from SRTCodes.SRTData import SRTDataset
from SRTCodes.Utils import DirFileName, Jdt, filterFileExt, changext
from ZY5M.ZY5MWarp import ZY5MGDALRasterWarp

JPZ5M_XIANLI21_DFN = DirFileName(r"K:\zhongdianyanfa\jpz_5\good5m2021-2022\xianli")


class JPZ5MXL21_Network(nn.Module):

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


class JPZ5MXL21_Loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.sigmoid_act = nn.Sigmoid()

    def forward(self, x, y, lamd=1):
        p = self.sigmoid_act(x)
        y = y.view((y.size(0), 1))
        loss = -torch.mean(lamd * (y * torch.log(p) + (1 - y) * torch.log(1 - p)))
        return loss


class JPZ5MXL21_DataSet(SRTDataset, Dataset):

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
        x = (np.clip(x, 200, 1600) - 200) / (1600 - 200)
        y = self.category_list[index]
        return x, y


class JPZ5MXL21_PytorchTraining(PytorchCategoryTraining):

    def __init__(self, n_category, model_dir=None, category_names=None, epochs=10, device=None, n_test=100):
        super().__init__(n_category=n_category, model_dir=model_dir, category_names=category_names, epochs=epochs,
                         device=device, n_test=n_test)

    def logisticToCategory(self, logts):
        logts = torch.sigmoid(logts)
        logts = (logts.cpu().numpy().T[0] > 0.5) * 1
        return logts


class JPZ5MXL21_GDALRasterPrediction(GDALRasterPrediction):

    def __init__(self, geo_fn):
        super(JPZ5MXL21_GDALRasterPrediction, self).__init__(geo_fn)
        self.device = "cuda:0"
        self.is_category = False
        self.number_pred = 15000

    def predict(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        # y = np.ones(x.shape[0])
        n = x.shape[0]

        x = x.astype("float32")
        # x = np.clip(x, 0, 1000) / 1000.0
        x = (np.clip(x, 200, 1600) - 200) / (1600 - 200)
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


class JPZ5MXL21_Main:

    def __init__(self):
        self.this_dirname = self.mkdir(r"H:\JPZ\JPZ5MXL21")
        self.model_dir = self.mkdir(os.path.join(self.this_dirname, "Mods"))
        self.n_category = 2
        self.category_names = ["NOIS", "IS"]
        self.epochs = 30
        self.device = "cuda:0"
        self.n_test = 10
        self.csv_fn = JPZ5M_XIANLI21_DFN.fn(r"samples\jpz5m_xianli_spl6_2_1.csv")
        self.npy_fn = JPZ5M_XIANLI21_DFN.fn(r"samples\jpz5m_xianli_spl6_2_1_sum.npy")
        self.test_ds = None
        self.train_ds = None
        self.win_size = 9
        self.mod = JPZ5MXL21_Network(in_channel=4, win_size=self.win_size, n_category=1)
        self.loss = JPZ5MXL21_Loss()
        self.geo_raster = JPZ5M_XIANLI21_DFN.fn("GF1C_xianli_2.tif")
        ...

    def mkdir(self, dirname):
        dirname = os.path.abspath(dirname)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        return dirname

    def train(self):
        self.train_ds = JPZ5MXL21_DataSet(is_test=False, spl_fn=self.csv_fn, train_d_fn=self.npy_fn, cname="CATEGORY")
        self.test_ds = JPZ5MXL21_DataSet(is_test=True, spl_fn=self.csv_fn, train_d_fn=self.npy_fn, cname="CATEGORY")

        pytorch_training = JPZ5MXL21_PytorchTraining(
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
        pytorch_training.addOptimizer(lr=0.0005)
        pytorch_training.train()

    def imdc(self, raster_dirname):
        # "H:\JPZ\JPZ5MXL21\Mods\20231006H202852\model_360.pth"
        # "H:\JPZ\JPZ5MXL21\Mods\20231008H213229\model_100.pth"
        # "H:\JPZ\JPZ5MXL21\Mods\20231009H083734\model_20.pth"
        mod_dirname = "20231009H083734"
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
                grp = JPZ5MXL21_GDALRasterPrediction(geo_raster)
                grp.is_category = True
                grp.run(imdc_fn=imdc_fn, np_type=np_type, mod=self.mod,
                        spl_size=[self.win_size, self.win_size],
                        row_start=self.win_size + 6, row_end=-(self.win_size + 6),
                        column_start=self.win_size + 6, column_end=-(self.win_size + 6),
                        n_one_t=20000)

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
                print(i)
        np.save(train_d_fn, d)


def tmpFuncs():
    # 提取NDVI和NDWI然后改一下范围
    def featExt(raster_name, to_raster_name):
        ds = GDALRaster(raster_name)
        d = ds.readAsArray()
        ndvi = (d[3] - d[2]) / (d[3] + d[2] + 0.00000001)
        plt.imshow(ndvi)
        plt.show()
        ndwi = (d[1] - d[3]) / (d[1] + d[3] + 0.00000001)
        d = np.concatenate([d, [ndvi], [ndwi]])

        def changeRange(channel, x_min, x_max):
            d[channel] = (np.clip(d[channel], x_min, x_max) - x_min) / (x_max - x_min)

        changeRange(0, 500, 1500)
        changeRange(1, 500, 1800)
        changeRange(2, 200, 1800)
        changeRange(3, 200, 1800)
        # changeRange(4, -0.6, 0.56)
        # changeRange(5, -0.4, 0.67)

        ds.save(d.astype("float32"), to_raster_name, dtype=gdal.GDT_Float32)

    filelist = filterFileExt(r"K:\zhongdianyanfa\jpz_5\good5m2021-2022\xianli\GF1C_xianli_2_retiles", ".tif")
    for f in filelist:
        to_f = os.path.join(r"K:\zhongdianyanfa\jpz_5\good5m2021-2022\xianli\GF1C_xianli_2_retiles2",
                            changext(os.path.split(f)[1], ext=".dat"))
        print(f, to_f)
        featExt(f, to_f)


def method_name2():
    # 使用随机森林分类一次看看结果
    df = pd.read_excel(r"K:\zhongdianyanfa\jpz_5\good5m2021-2022\xianli\samples\2021年暹粒样本筛选.xlsx",
                       sheet_name="Sheet1")
    field_list = []

    def rerange(field, x_min, x_max):
        df[field] = (np.clip(df[field], x_min, x_max) - x_min) / (x_max - x_min)
        field_list.append(field)

    rerange("Blue", 500, 1500)
    rerange("Green", 500, 1800)
    rerange("Red", 200, 1800)
    rerange("NIR", 200, 1800)
    rerange("NDVI", -0.6, 0.56)
    rerange("NDWI", -0.4, 0.67)
    rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=90)
    x = df[field_list].values
    y = df["CATEGORY"].values
    rfc.fit(x, y)
    print(rfc.score(x, y))

    # "K:\zhongdianyanfa\jpz_5\good5m2021-2022\xianli\GF1C_xianli_2_retiles\GF1C_xianli_2_2_8.tif"
    def imdcOne(raster_name, to_raster_name):
        ds = GDALRaster(raster_name)
        d = ds.readAsArray()
        ndvi = (d[3] - d[2]) / (d[3] + d[2])
        ndwi = (d[1] - d[3]) / (d[1] + d[3])
        d = np.concatenate([d, [ndvi], [ndwi]])

        def changeRange(channel, x_min, x_max):
            d[channel] = (np.clip(d[channel], x_min, x_max) - x_min) / (x_max - x_min)

        changeRange(0, 500, 1500)
        changeRange(1, 500, 1800)
        changeRange(2, 200, 1800)
        changeRange(3, 200, 1800)
        changeRange(4, -0.6, 0.56)
        changeRange(5, -0.4, 0.67)

        imdc = np.zeros((d.shape[1], d.shape[2]))
        print(to_raster_name)
        jdt = Jdt(d.shape[1], "Imdc One")
        jdt.start()
        for i in range(d.shape[1]):
            y1 = rfc.predict(d[:, i, :].T)
            imdc[i, :] = y1
            jdt.add()
        jdt.end()

        ds.save(imdc, to_raster_name)

    imdcOne(r"K:\zhongdianyanfa\jpz_5\good5m2021-2022\xianli\GF1C_xianli_2_retiles\GF1C_xianli_2_2_8.tif",
            r"K:\zhongdianyanfa\jpz_5\good5m2021-2022\xianli\temp\GF1C_xianli_2_2_8_imdc1.dat")


def method_name1():
    # 画出直方图
    df = pd.read_excel(r"K:\zhongdianyanfa\jpz_5\good5m2021-2022\xianli\samples\2021年暹粒样本筛选.xlsx",
                       sheet_name="Sheet1")
    # 'X', 'Y', 'category_o', 'CATEGORY', 'TEST', 'Blue', 'Green', 'Red', 'NIR', 'NDVI', 'NDWI'
    print(df.keys())

    def plot_hist(field):
        df[field][df["CATEGORY"] == 0].hist(color="g", bins=256, alpha=0.5)
        df[field][df["CATEGORY"] == 1].hist(color="r", bins=256, alpha=0.5)
        plt.title(field)

    plot_hist("Blue")
    plt.show()
    plot_hist("Green")
    plt.show()
    plot_hist("Red")
    plt.show()
    plot_hist("NIR")
    plt.show()
    plot_hist("NDVI")
    plt.show()
    plot_hist("NDWI")
    plt.show()


def main():
    jzp5m_xl21 = JPZ5MXL21_Main()

    """
    1. 需要修改样本文件
    2. 修改DataSet中的数据的范围
    3. 修改模型的文件夹
    4. 修改分类的tif文件
    5. 修改win_size
    6. Network(in_channel=8, win_size=9, n_category=1)中的channel的数量，窗口的大小和类别的数量
    """

    # jzp5m_xl21.warp()
    # jzp5m_xl21.sampleNPY()
    jzp5m_xl21.train()
    # jzp5m_xl21.imdc(r"K:\zhongdianyanfa\jpz_5\good5m2021-2022\xianli\GF1C_xianli_2_retiles")
    pass


if __name__ == "__main__":
    # main()
    tmpFuncs()
