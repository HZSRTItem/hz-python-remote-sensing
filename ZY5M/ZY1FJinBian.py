# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ZY1FJinBian.py
@Time    : 2023/9/20 20:23
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of ZY1FJinBian
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
from SRTCodes.PytorchModelTraining import PytorchCategoryTraining
from SRTCodes.SRTData import SRTDataset
from SRTCodes.Utils import DirFileName

ZY1F_JINBIAN_DFN = DirFileName(r"K:\zhongdianyanfa\jpz_5\good5m2021-2022\ZY1F_VNIC_E105.2_N11.6_20221231_L1B0000322465")


class ZY1FJB21_Network(nn.Module):

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


class ZY1FJB21_Loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.sigmoid_act = nn.Sigmoid()

    def forward(self, x, y, lamd=1):
        p = self.sigmoid_act(x)
        y = y.view((y.size(0), 1))
        loss = -torch.mean(lamd * (y * torch.log(p) + (1 - y) * torch.log(1 - p)))
        return loss


class ZY1FJB21_DataSet(SRTDataset, Dataset):

    def __init__(self, is_test=False):
        super().__init__()

        self.spl_fn = ZY1F_JINBIAN_DFN.fn(r"samples\jinbian2021_spl3_3.csv")
        self.train_d_fn = ZY1F_JINBIAN_DFN.fn(r"samples\jinbian2021_spl3_3\sum.npy")
        self.df = pd.read_csv(self.spl_fn)
        is_tests = self.df["TEST"].values
        categorys = self.df["WX_V2"].values.tolist()
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
        x = np.clip(x, 0, 3500) / 3500.0
        y = self.category_list[index]
        return x, y


class ZY1FJB21_PytorchTraining(PytorchCategoryTraining):

    def __init__(self, n_category, model_dir=None, category_names=None, epochs=10, device=None, n_test=100):
        super().__init__(n_category=n_category, model_dir=model_dir, category_names=category_names, epochs=epochs,
                         device=device,
                         n_test=n_test)

    def logisticToCategory(self, logts):
        logts = torch.sigmoid(logts)
        logts = (logts.cpu().numpy().T[0] > 0.5) * 1
        return logts


class ZY1FJB21_GDALRasterPrediction(GDALRasterPrediction):

    def __init__(self, geo_fn):
        super(ZY1FJB21_GDALRasterPrediction, self).__init__(geo_fn)
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


class ZY1FJB21_Main:

    def __init__(self):
        self.this_dirname = self.mkdir(r"H:\JPZ\ZY1FJB21")
        self.model_dir = self.mkdir(os.path.join(self.this_dirname, "Mods"))
        self.n_category = 2
        self.category_names = ["NOIS", "IS"]
        self.epochs = 30
        self.device = "cuda:0"
        self.n_test = 10
        self.csv_fn = ZY1F_JINBIAN_DFN.fn(r"samples\jinbian2021_spl3_3.csv")
        self.npy_fn = ZY1F_JINBIAN_DFN.fn(r"samples\jinbian2021_spl3_3\sum.npy")
        self.train_ds = ZY1FJB21_DataSet(is_test=False)
        self.test_ds = ZY1FJB21_DataSet(is_test=True)
        self.win_size = 9
        self.mod = ZY1FJB21_Network(8, 9, 1)
        self.loss = ZY1FJB21_Loss()
        self.geo_raster = ZY1F_JINBIAN_DFN.fn("ZY1F_JinBian_2.tif")

    def mkdir(self, dirname):
        dirname = os.path.abspath(dirname)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        return dirname

    def train(self):
        pytorch_training = ZY1FJB21_PytorchTraining(
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
        grp = ZY1FJB21_GDALRasterPrediction(self.geo_raster)
        mod_dirname = "20230921H162901"
        imdc_fn = os.path.join(self.model_dir, mod_dirname, "imdc1.tif")
        mod_fn = os.path.join(self.model_dir, mod_dirname, "model_396.pth")
        grp.is_category = True
        np_type = "int8"
        self.mod.load_state_dict(torch.load(mod_fn))
        self.mod.to(self.device)
        grp.run(imdc_fn=imdc_fn, np_type=np_type, mod=self.mod,
                spl_size=[self.win_size, self.win_size],
                row_start=10, row_end=-10,
                column_start=10, column_end=-10,
                n_one_t=15000)


def main():
    zy1fjb21 = ZY1FJB21_Main()
    # zy1fjb21.train()
    zy1fjb21.imdc()
    pass


if __name__ == "__main__":
    main()
