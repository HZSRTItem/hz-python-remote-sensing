# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowDeepLearningFusion.py
@Time    : 2023/11/29 10:05
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of ShadowDeepLearningFusion
-----------------------------------------------------------------------------"""
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from SRTCodes.GDALRasterClassification import GDALRasterPrediction
from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.PytorchModelTraining import PytorchCategoryTraining
from SRTCodes.Utils import DirFileName, writeTexts
from Shadow.DeepLearning.SHDLData import SHDLDataSampleCollection, SHDLDataSample
from Shadow.DeepLearning.SHDLNetThreeBranches import SHDLNet_SimpleThreeBranches

SHDL_DFN = DirFileName(r"F:\ProjectSet\Shadow\DeepLearn")


def minmax01(d, d_min, d_max):
    d = np.clip(d, d_min, d_max)
    d = (d - d_min) / (d_max - d_min)
    return d


def dataDeal(x):
    x[:4] = minmax01(x[:4], 0, 5000)
    x[4:] = 10 * np.log10(x[4:])
    #
    # x[4] = minmax01(x[4], -24.609674, 15.9092603)
    # x[5] = minmax01(x[5], -31.865038, 5.2615275)
    # x[6] = minmax01(x[6], -27.851603, 15.094706)
    # x[7] = minmax01(x[7], -35.427082, 5.4092093)

    return x


class SHDLDataSet(Dataset):

    def __init__(self):
        super(SHDLDataSet, self).__init__()
        self.samples = SHDLDataSampleCollection()
        self.category_y = {11: 1, 12: 1, 21: 0, 22: 0, 31: 0, 32: 0, 41: 0, 42: 0}

        self.deal()

    def deal(self):
        for spl in self.samples:
            spl.data = dataDeal(spl.data)
            spl.category = self.category_y[spl.category]

    def __getitem__(self, index):
        return self.get(index)

    def get(self, index):
        spl = self.samples[index]
        x = spl.data
        y = spl.category
        return x, y

    def initCTAG(self, samples, c_tag=None):
        if c_tag is not None:
            samples = samples.filterCTAG(c_tag)
        self.samples = samples
        self.deal()

    def justISNOIS(self):
        spl_is, spl_nois = [], []
        for spl in self.samples:
            if spl.category == 0:
                spl_nois.append(spl)
            elif spl.category == 1:
                spl_is.append(spl)
        if len(spl_is) > len(spl_nois):
            spl_is = random.sample(spl_is, len(spl_nois))
        else:
            spl_nois = random.sample(spl_nois, len(spl_is))
        samples = self.samples.newFromSamples(spl_is + spl_nois)
        self.samples = samples

    def __len__(self):
        return len(self.samples)


class SDLF_PytorchTraining(PytorchCategoryTraining):

    def __init__(self, n_category, model_dir=None, category_names=None, epochs=10, device=None, n_test=100):
        super().__init__(n_category=n_category, model_dir=model_dir, category_names=category_names, epochs=epochs,
                         device=device, n_test=n_test)

    def train(self, batch_save=False, epoch_save=True, *args, **kwargs):
        self._initTrain()
        self._printModel()
        self._log.saveHeader()

        for epoch in range(self.epochs):
            self.model.train()
            for batchix, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                # x, y = x.float(), y.int()

                logts = self.model(x)  # 模型训练
                self.loss = self.criterion(logts, y)  # 损失函数
                self.loss = self.lossDeal(self.loss)  # loss处理
                self.optimizer.zero_grad()  # 梯度清零
                self.loss.backward()  # 反向传播
                self.optimizer.step()  # 优化迭代

                # 测试 ------------------------------------------------------------------
                if self.test_loader is not None:
                    if batchix % self.n_test == 0:
                        self.testAccuracy()
                        modname = self.log(batchix, epoch)
                        if batch_save:
                            self.saveModel(modname)

            print("-" * 73)
            self.testAccuracy()
            modname = self.log(-1, epoch)
            modname = self.model_name + "_epoch_{0}.pth".format(epoch)
            print("*" * 73)

            if epoch_save:
                self.saveModel(modname)

    def logisticToCategory(self, logts):
        logts = torch.sigmoid(logts)
        logts = (logts.cpu().numpy().T[0] > 0.5) * 1 + 1
        return logts


class SDLF_GDALRasterPrediction(GDALRasterPrediction):

    def __init__(self, geo_fn, n_categorys=2):
        super(SDLF_GDALRasterPrediction, self).__init__(geo_fn)
        self.device = "cuda:0"
        self.is_category = False
        self.number_pred = 15000
        self.n_categorys = n_categorys

    def predict(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        # y = np.ones(x.shape[0])
        n = x.shape[0]

        x = x.astype("float32")
        x = torch.from_numpy(x)
        x = x.to(self.device)
        x = x[:, [0, 1, 2, 3, 14, 15, 45, 46], :, :]
        y = torch.zeros((n, self.n_categorys), dtype=torch.float)
        y = y.to(self.device)

        self.model.eval()
        with torch.no_grad():
            for i in range(0, n, self.number_pred):
                y_temp = self.model(x[i:i + self.number_pred, :])
                y[i:i + self.number_pred, :] = y_temp
            y = torch.sigmoid(y)
        y = torch.argmax(y, dim=1)
        y = y.cpu().numpy()
        # y = y.T[0]
        # if self.is_category:
        #     y = (y > 0.5) * 1

        return y

    def preDeal(self, row, column_start, column_end):
        d_row = self.d[:, row, column_start:column_end]
        return np.ones(d_row.shape[1], dtype="bool")


def saveSampleToCSV(ds: SHDLDataSet, csv_fn):
    dfs = []
    for i in range(len(ds)):
        spl: SHDLDataSample = ds.samples[i]
        to_dict = spl.toDict()
        n = int(spl.data.shape[1] / 2)
        d = spl.data[:, n, n]
        for j in range(len(d)):
            to_dict["_DATA_{0}".format(j + 1)] = d[j]
        dfs.append(to_dict)
    df = pd.DataFrame(dfs)
    df.to_csv(csv_fn)


def plotCategory(csv_fn):
    df = pd.read_csv(csv_fn)
    plt.scatter(x=df["_DATA_3"].values, y=df["_DATA_4"].values, c=df["CATEGORY"].values)
    plt.show()


class ShadowDeepLearningFusion_Main:

    def __init__(self):
        # Init
        self.this_dirname = self.mkdir(SHDL_DFN.fn(""))
        self.model_dir = self.mkdir(os.path.join(self.this_dirname, "Mods"))
        self.n_category = 2
        self.category_names = ["NOIS", "IS"]
        self.epochs = 200
        self.device = "cuda:0"
        self.n_test = 10
        self.batch_size = 128

        # Samples
        self.csv_fn = SHDL_DFN.fn(r"Samples\20231201H094911\SHDL_THREE.csv")
        self.npy_fn = SHDL_DFN.fn(r"Samples\20231201H094911\SHDL_THREE.npy")
        self.samples = SHDLDataSampleCollection()
        self.samples.addCSV(csv_fn=self.csv_fn, data_fn=self.npy_fn)
        self.train_ds = SHDLDataSet()
        self.train_ds.initCTAG(self.samples, "Train")
        # self.train_ds.justISNOIS()
        self.test_ds = SHDLDataSet()
        self.test_ds.initCTAG(self.samples, "Test")
        self.test_sh_ds = SHDLDataSet()
        self.test_sh_ds.initCTAG(self.samples, "ShadowTest")

        # saveSampleToCSV(self.train_ds, r"F:\ProjectSet\Shadow\DeepLearn\Temp\tmp2.csv")

        # Model
        self.mod = SHDLNet_SimpleThreeBranches()
        # self.mod = SHDLNet_Test()
        self.loss = nn.CrossEntropyLoss()

        self.win_size = 9

    def mkdir(self, dirname):
        dirname = os.path.abspath(dirname)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        return dirname

    def train(self):

        pytorch_training = SDLF_PytorchTraining(
            n_category=self.n_category,
            model_dir=self.model_dir,
            category_names=self.category_names,
            epochs=self.epochs,
            device=self.device,
            n_test=self.n_test
        )

        pytorch_training.trainLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        pytorch_training.testLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)
        pytorch_training.addModel(self.mod)
        pytorch_training.addCriterion(self.loss)
        pytorch_training.addOptimizer(lr=0.0005, eps=0.00001)

        print("model_dir", pytorch_training.model_dir)
        save_fn = os.path.join(pytorch_training.model_dir, "save.txt")
        writeTexts(save_fn, "spl_fn           :", self.csv_fn, mode="a", end="\n")
        writeTexts(save_fn, "train_d_fn       :", self.npy_fn, mode="a", end="\n")
        writeTexts(save_fn, "spl_size         :", self.win_size, mode="a", end="\n")
        writeTexts(save_fn, "mod_code_filename:", self.mod.this_file, mode="a", end="\n")
        pytorch_training.saveModelCodeFile(self.mod.this_file)

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
                grp = SDLF_GDALRasterPrediction(geo_raster)
                grp.is_category = True
                grp.run(imdc_fn=imdc_fn, np_type=np_type, mod=self.mod,
                        spl_size=[self.win_size, self.win_size],
                        row_start=self.win_size + 6, row_end=-(self.win_size + 6),
                        column_start=self.win_size + 6, column_end=-(self.win_size + 6),
                        n_one_t=20000, data_deal=dataDeal)

    def imdcOne(self, geo_raster, mod_fn=None):
        # r"K:\zhongdianyanfa\jpz_5\good5m2017-2018\madewang\Images\mdw_im1.tif"
        # grp = ZY2MMdw_GDALRasterPrediction( r"K:\zhongdianyanfa\jpz_5\good5m2017-2018\madewang\Images\mdw_im1.tif")
        grp = SDLF_GDALRasterPrediction(geo_raster)
        if mod_fn is not None:
            imdc_fn = mod_fn + "_imdc.tif"
        else:
            # "H:\JPZ\ZY2MMdw\Mods\20231119H191912\model_56.pth"
            mod_dirname = "20231119H191912"
            imdc_fn = os.path.join(self.model_dir, mod_dirname, "imdc1.tif")
            mod_fn = os.path.join(self.model_dir, mod_dirname, "model_56.pth")

        print("imdc_fn    :", imdc_fn)
        print("mod_fn     :", mod_fn)

        grp.is_category = True
        np_type = "int8"
        self.mod.load_state_dict(torch.load(mod_fn))
        self.mod.to(self.device)
        grp.run(imdc_fn=imdc_fn, np_type=np_type, mod=self.mod,
                spl_size=[self.win_size, self.win_size],
                row_start=10, row_end=-10,
                column_start=10, column_end=-10,
                n_one_t=15000, data_deal=dataDeal)

    def func1(self):
        gr = GDALRaster(self.geo_raster)
        d = gr.readAsArray()
        n = 4
        for i in range(len(d)):
            mean = np.mean(d[i])
            std = np.std(d[i])
            print("d[{0}] = minmax01(d[{0}], {1}, {2})".format(i, mean - std * n, mean + std * n))


def main():
    sdlf_main = ShadowDeepLearningFusion_Main()
    # sdlf_main.train()
    # sdlf_main.imdcOne(
    #     geo_raster=r"F:\ProjectSet\Shadow\Release\QingDaoImages\SH_QD_envi.dat",
    #     mod_fn=r"F:\ProjectSet\Shadow\DeepLearn\Mods\20231202H201511\model_epoch_162.pth"
    # )
    pass


if __name__ == "__main__":
    # print(__file__)
    main()
    # plotCategory(r"F:\ProjectSet\Shadow\DeepLearn\Temp\tmp1.csv")
