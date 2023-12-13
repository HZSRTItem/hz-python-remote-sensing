# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : JPZ5MXianLi17.py
@Time    : 2023/10/4 9:49
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of JPZ3MXianLi17
-----------------------------------------------------------------------------"""
import os.path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from SRTCodes.GDALRasterClassification import GDALRasterPrediction
from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.OGRUtils import SRTGeoJson, SRTOGRSampleSpaceUniform, initFromGeoJson
from SRTCodes.PytorchModelTraining import PytorchCategoryTraining
from SRTCodes.SRTData import SRTDataset
from SRTCodes.Utils import DirFileName, readJson, Jdt, saveJson
from ZY5M.ZY5MWarp import ZY5MGDALRasterWarp

JPZ5M_XIANLI17_DFN = DirFileName(r"K:\zhongdianyanfa\jpz_5\good5m2017-2018\xianli")

from osgeo_utils import gdal_retile

class JPZ5MXL17_Network(nn.Module):

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


class JPZ5MXL17_Loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.sigmoid_act = nn.Sigmoid()

    def forward(self, x, y, lamd=1):
        p = self.sigmoid_act(x)
        y = y.view((y.size(0), 1))
        loss = -torch.mean(lamd * (y * torch.log(p) + (1 - y) * torch.log(1 - p)))
        return loss


class JPZ5MXL17_DataSet(SRTDataset, Dataset):

    def __init__(self, is_test=False, spl_fn="", train_d_fn=""):
        super().__init__()

        self.win_size = 9
        self.win_size_2 = int(self.win_size / 2)
        self.spl_fn = spl_fn
        self.train_d_fn = train_d_fn
        self.df = pd.read_csv(self.spl_fn)
        is_tests = self.df["TEST"].values
        categorys = self.df["CATEGORY"].values.tolist()
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
        n = int(x.shape[1] / 2)
        x = x[:, n - self.win_size_2:n + self.win_size_2 + 1, n - self.win_size_2:n + self.win_size_2 + 1]
        x = np.clip(x, 0, 3500) / 3500.0
        y = self.category_list[index]
        return x, y


class JPZ5MXL17_PytorchTraining(PytorchCategoryTraining):

    def __init__(self, n_category, model_dir=None, category_names=None, epochs=10, device=None, n_test=100):
        super().__init__(n_category=n_category, model_dir=model_dir, category_names=category_names, epochs=epochs,
                         device=device,
                         n_test=n_test)

    def logisticToCategory(self, logts):
        logts = torch.sigmoid(logts)
        logts = (logts.cpu().numpy().T[0] > 0.5) * 1
        return logts


class JPZ5MXL17_GDALRasterPrediction(GDALRasterPrediction):

    def __init__(self, geo_fn):
        super(JPZ5MXL17_GDALRasterPrediction, self).__init__(geo_fn)
        self.device = "cuda:0"
        self.is_category = False
        self.number_pred = 15000

    def predict(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        # y = np.ones(x.shape[0])
        n = x.shape[0]

        x = x.astype("float32")
        x = np.clip(x, 0, 3500) / 3500.0
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


class JPZ5MXL17_Main:

    def __init__(self):

        self.this_dirname = self.mkdir(r"H:\JPZ\JPZ5MXL17")
        self.model_dir = self.mkdir(os.path.join(self.this_dirname, "Mods"))
        self.n_category = 2
        self.category_names = ["NOIS", "IS"]
        self.epochs = 30
        self.device = "cuda:0"
        self.n_test = 10
        self.csv_fn = JPZ5M_XIANLI17_DFN.fn(r"sample\xinli17_spl1.csv")
        self.npy_fn = JPZ5M_XIANLI17_DFN.fn(r"sample\xinli17_spl1_sum.npy")
        self.test_ds = None
        self.train_ds = None
        self.win_size = 49
        self.win_size_2 = 9
        self.mod = JPZ5MXL17_Network(in_channel=4, win_size=9, n_category=1)
        self.loss = JPZ5MXL17_Loss()
        self.geo_raster = JPZ5M_XIANLI17_DFN.fn(r"GF1_PMS2_E103.8_N13.4_20170216_L1A0002188979\GF1_xianli_1.tif")
        ...

    def mkdir(self, dirname):
        dirname = os.path.abspath(dirname)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        return dirname

    def train(self):
        self.train_ds = JPZ5MXL17_DataSet(is_test=False, spl_fn=self.csv_fn, train_d_fn=self.npy_fn)
        self.test_ds = JPZ5MXL17_DataSet(is_test=True, spl_fn=self.csv_fn, train_d_fn=self.npy_fn)

        pytorch_training = JPZ5MXL17_PytorchTraining(
            n_category=self.n_category,
            model_dir=self.model_dir,
            category_names=self.category_names,
            epochs=self.epochs,
            device=self.device,
            n_test=self.n_test
        )

        pytorch_training.trainLoader(self.train_ds, batch_size=32, shuffle=True)
        pytorch_training.testLoader(self.test_ds, batch_size=32, shuffle=False)
        pytorch_training.addModel(self.mod)
        pytorch_training.addCriterion(self.loss)
        pytorch_training.addOptimizer()
        pytorch_training.train()

    def imdc(self):
        grp = JPZ5MXL17_GDALRasterPrediction(self.geo_raster)
        mod_dirname = "20231118H111752"
        imdc_fn = os.path.join(self.model_dir, mod_dirname, "imdc2.tif")
        mod_fn = os.path.join(self.model_dir, mod_dirname, "model_1500.pth")
        grp.is_category = True
        np_type = "int8"
        self.mod.load_state_dict(torch.load(mod_fn))
        self.mod.to(self.device)
        grp.run(imdc_fn=imdc_fn, np_type=np_type, mod=self.mod,
                spl_size=[self.win_size_2, self.win_size_2],
                row_start=10, row_end=-10,
                column_start=10, column_end=-10,
                n_one_t=15000)

    def warp(self):
        zy5m_grw = ZY5MGDALRasterWarp()
        # "K:\zhongdianyanfa\jpz_5\good5m2017-2018\xianli\GF1_PMS2_E103.8_N13.4_20170216_L1A0002188979\spl1.csv"
        # zy5m_grw.warpImage(
        #     JPZ5M_XIANLI17_DFN.fn(r"GF1_PMS2_E103.8_N13.4_20170216_L1A0002188979\spl1.csv"),
        #     JPZ5M_XIANLI17_DFN.fn(r"GF1_PMS2_E103.8_N13.4_20170216_L1A0002188979\GF1_xianli_1.tif"),
        #     JPZ5M_XIANLI17_DFN.fn(r"GF1_PMS2_E103.8_N13.4_20170216_L1A0002188979\GF1_xianli_2.tif"),
        # )
        # GF2_PMS1_E103.8_N13.3_20170328_L1A0002272453
        zy5m_grw.warpImage(
            JPZ5M_XIANLI17_DFN.fn(r"GF2_PMS1_E103.8_N13.3_20170328_L1A0002272453\spl1.csv"),
            JPZ5M_XIANLI17_DFN.fn(r"GF2_PMS1_E103.8_N13.3_20170328_L1A0002272453\GF2_xianli_1.tif"),
            JPZ5M_XIANLI17_DFN.fn(r"GF2_PMS1_E103.8_N13.3_20170328_L1A0002272453\GF2_xianli_2.tif"),
        )

    def sampleGet(self):

        def spl_get_1():
            # 将道路线转为点
            roads_d = readJson(r"K:\zhongdianyanfa\jpz_5\good5m2017-2018\xianli\sample\jpz5m_xianli_osm_roads2.geojson")
            roads_coors = {"type": roads_d["type"], "name": roads_d["name"] + "_tp1", "crs": roads_d["crs"],
                           "features": []}

            jdt = Jdt(len(roads_d["features"]), "Roads Sampling")
            jdt.start()
            for feat in roads_d["features"]:
                if feat["geometry"]["type"] == "MultiLineString":
                    continue
                for i in range(len(feat["geometry"]["coordinates"]) - 1):
                    x = (feat["geometry"]["coordinates"][i][0] + feat["geometry"]["coordinates"][i + 1][0]) / 2
                    y = (feat["geometry"]["coordinates"][i][1] + feat["geometry"]["coordinates"][i + 1][1]) / 2
                    coor_d = {"type": feat["type"], "properties": feat["properties"],
                              "geometry": {"type": "Point", "coordinates": [x, y]}}
                    roads_coors["features"].append(coor_d)
                jdt.add()
            jdt.end()

            saveJson(roads_coors,
                     r"K:\zhongdianyanfa\jpz_5\good5m2017-2018\xianli\sample\jpz5m_xianli_osm_roads1_tp1.geojson")

        def spl_get_2():
            # 建筑物的点和道路的点合并到一起
            buildings_d = readJson(
                r"K:\zhongdianyanfa\jpz_5\good5m2017-2018\xianli\sample\jpz5m_xianli_osm_building1_tp2.geojson")
            roads_d = readJson(
                r"K:\zhongdianyanfa\jpz_5\good5m2017-2018\xianli\sample\jpz5m_xianli_osm_roads1_tp1.geojson")
            to_coors = {"type": roads_d["type"], "name": "jpz5m_xianli_osm_building_roads_1", "crs": roads_d["crs"],
                        "features": []}
            not_eq = []
            jdt = Jdt(len(roads_d["features"]) + len(buildings_d["features"]), "Roads Building Sampling")
            jdt.start()
            for feat in roads_d["features"]:
                jdt.add()
                if feat["geometry"]["type"] != "Point":
                    if feat["geometry"]["type"] not in not_eq:
                        not_eq.append(feat["geometry"]["type"])
                    continue
                coor_d = {"type": feat["type"], "properties": {"category_osm": "road"}, "geometry": feat["geometry"]}
                to_coors["features"].append(coor_d)
            for feat in buildings_d["features"]:
                jdt.add()
                if feat["geometry"]["type"] != "Point":
                    if feat["geometry"]["type"] not in not_eq:
                        not_eq.append(feat["geometry"]["type"])
                    continue
                coor_d = {"type": feat["type"], "properties": {"category_osm": "building"},
                          "geometry": feat["geometry"]}
                to_coors["features"].append(coor_d)
            jdt.end()
            print(not_eq)
            saveJson(to_coors,
                     r"K:\zhongdianyanfa\jpz_5\good5m2017-2018\xianli\sample\jpz5m_xianli_osm_building_roads_1.geojson")

        # spl_get_2()

        def spl_get_3():
            geojson1 = SRTGeoJson(r"K:\zhongdianyanfa\jpz_5\good5m2021-2022\xianli\samples\jpz5m_xianli_spl5_2.geojson")
            geojson2 = initFromGeoJson(geojson1).geojson
            sossu = SRTOGRSampleSpaceUniform(x_len=100, y_len=100, is_trans_jiaodu=True)
            jdt = Jdt(len(geojson1["features"]), "Sample Space Uniform")
            jdt.start()
            for feat in geojson1["features"]:
                jdt.add()
                if sossu.coor(x=feat["geometry"]["coordinates"][0], y=feat["geometry"]["coordinates"][1]):
                    geojson2["features"].append(feat)
            jdt.end()
            saveJson(geojson2, r"K:\zhongdianyanfa\jpz_5\good5m2021-2022\xianli\samples\jpz5m_xianli_spl5_3.geojson")

        spl_get_3()

    def sampleNPY(self):
        # 使用CSV文件在影像中提取样本的数据
        spl_fn = self.csv_fn
        raster_fn = self.geo_raster
        train_d_fn = self.npy_fn
        spl_size = [self.win_size, self.win_size]
        print(spl_fn)
        df = pd.read_csv(spl_fn)
        print(raster_fn)
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


def main():
    jzp5m_xl17 = JPZ5MXL17_Main()
    # jzp5m_xl17.sampleNPY()
    # jzp5m_xl17.warp()
    # jzp5m_xl17.train()
    jzp5m_xl17.imdc()
    # jzp5m_xl17.sampleGet()
    pass


if __name__ == "__main__":
    main()
