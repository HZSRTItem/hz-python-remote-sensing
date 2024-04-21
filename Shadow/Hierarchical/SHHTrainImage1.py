# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHHTrainImage1.py
@Time    : 2024/3/29 16:34
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHHTrainImage1
-----------------------------------------------------------------------------"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from SRTCodes.GDALRasterIO import GDALRasterChannel
from SRTCodes.NumpyUtils import categoryMap, NumpyDataCenter
from SRTCodes.SRTModelImage import SRTModImPytorch
from SRTCodes.Utils import savecsv
from Shadow.Hierarchical import SHHConfig
from Shadow.Hierarchical.SHHRunMain import SHHModImPytorch
from Shadow.Hierarchical.SHHTransformer import VIT_WS_T1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(10, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Flatten(),
            nn.Linear(1600, 8),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


def data_deal(x, y=None):
    # is_expand_dims = False
    # if len(x.shape) == 1:
    #     is_expand_dims = True
    #     x = np.expand_dims(x, axis=0)
    x[0:2] = x[0:2] / 30 + 1
    x[2:4] = x[3:5] / 30 + 1
    x[4:10] = x[6:] / 3000
    x = x[:10]
    # if is_expand_dims:
    #     x = x[0]
    if y is not None:
        y = y - 1
        return x, y
    return x


class NetDataset(Dataset):

    def __init__(self):
        super(NetDataset, self).__init__()
        self.df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\SampleData\shh2_spl\shh2_spl.csv")
        self.data = np.load(r"F:\ProjectSet\Shadow\Hierarchical\Samples\SampleData\shh2_spl\shh2_spl.npy")
        self.y = np.array(categoryMap(self.df["CATEGORY"].values, SHHConfig.CATE_MAP_SH881))
        self.ndc = NumpyDataCenter()

    def get(self, index):
        x, y = self.ndc.fit(self.data[index]), self.y[index]
        x, y = data_deal(x, y)
        return x, y, self.df.loc[index].to_dict()

    def __getitem__(self, index):
        x, y = self.ndc.fit(self.data[index]), self.y[index]
        x, y = data_deal(x, y)
        return x, y

    def __len__(self):
        return len(self.y)


class ThisMain:

    def __init__(self):
        self.smip = SHHModImPytorch()

        def func_predict(model: Net, x: torch.Tensor):
            logit = model(x)
            y = torch.argmax(logit, dim=1) + 1
            return y

        self.func_predict = func_predict

    def main(self):
        self.smip.model_dirname = r"F:\ProjectSet\Shadow\Hierarchical\WSModels"
        self.smip.model_name = "VIT_WS_T1"
        self.smip.epochs = 100
        self.smip.device = "cuda:0"
        self.smip.n_test = 10
        self.smip.batch_size = 32
        self.smip.n_class = 8
        self.smip.class_names = SHHConfig.SHH_CNAMES8
        self.smip.win_size = (3, 3)
        self.smip.model = VIT_WS_T1().to(self.smip.device)
        self.smip.func_predict = self.func_predict
        self.smip.func_y_deal = lambda y: y + 1
        self.smip.init8ColorTable()

        self.imdc()

    def train(self):
        ds = NetDataset()
        ds.ndc.__init__(3, self.smip.win_size, (21, 21))
        self.smip.train_ds, self.smip.test_ds = torch.utils.data.random_split(dataset=ds, lengths=[0.8, 0.2], )
        self.smip.timeDirName()
        self.smip.initTrainLog()
        self.smip.initPytorchTraining()
        self.smip.pt.func_logit_category = self.func_predict
        self.smip.pt.func_y_deal = lambda y: y + 1
        self.smip.initModel()
        self.smip.initDataLoader()
        self.smip.initCriterion(nn.CrossEntropyLoss())
        self.smip.initOptimizer(torch.optim.Adam, lr=0.0001, eps=0.00001)
        self.smip.copyFile(__file__)
        print(self.smip.model_dirname)
        self.smip.train()

    def imdc(self):
        self.smip.loadPTH(r"F:\ProjectSet\Shadow\Hierarchical\WSModels\20240330H114435\VIT_WS_T1_epoch90.pth")
        grc: GDALRasterChannel = GDALRasterChannel()
        grc.addGDALDatas(SHHConfig.SHH2_QD1_FNS[0])
        self.smip.imdc(grc=grc, is_jdt=True, data_deal=data_deal)
        # self.smip.imdcTiles(
        #     # to_fn=r"F:\Week\20240331\Data\20240329H185618\Net2_epoch2_imdc5.tif",
        #     tiles_dirname=r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\qd_sh2_1_retile",
        #     data_deal=data_deal,
        # )
        pass


def main():
    def func1():
        ds = NetDataset()
        ds.ndc.__init__(3, (3, 3), (21, 21))
        csv_fn = r"F:\Week\20240331\Data\t1.csv"
        to_dict = {"N": [], "X": [], "Y": [], **{"Channel{0}".format(i + 1): [] for i in range(12)}}
        ks = list(to_dict.keys())

        def add_to_dict_line(line):
            for j, k in enumerate(ks):
                to_dict[k].append(line[j])

        for i in range(len(ds)):
            x, y, line_dict = ds.get(i)
            x = x[:, 1, 1]
            add_to_dict_line([i + 1, line_dict["X"], line_dict["Y"], *x.tolist()])

        savecsv(csv_fn, to_dict)
        return

    # func1()
    ThisMain().main()
    pass


if __name__ == "__main__":
    r"""
    python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHTrainImage1 import ThisMain; ThisMain().main()"
    """
    main()
