# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ZY5MImage.py
@Time    : 2023/5/9 17:38
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PytorchModel of ZY5MImage
-----------------------------------------------------------------------------"""

import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset

from SRTCodes.ENVIRasterClassification import ENVIRasterClassification
from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.ModelTraining import TrainLog, ConfusionMatrix
from SRTCodes.Utils import RumTime, getRandom, CoorInPoly, DirFileName
from ZY5M.ZY5MModel import ZY5MLoss

np.set_printoptions(suppress=True, precision=3, linewidth=600)


class ZY5MNet(nn.Module):

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


class Test1EnviCate(ENVIRasterClassification):

    def __init__(self, dat_fn=None, hdr_fn=None):
        super(Test1EnviCate, self).__init__(dat_fn, hdr_fn)

    def predict(self, x, *args, **kwargs) -> np.ndarray:
        x = x.astype("float32")
        x = torch.from_numpy(x)
        x = x.to("cuda:0")
        x = x / 255
        self.model.eval()
        with torch.no_grad():
            y = self.model(x)
        y = 1 / (1 + torch.exp(-y))
        y = y.T[0] > 0.5
        y = y + 1
        y = y.cpu().numpy()
        return y


class ZY5MDataset(Dataset):

    def __init__(self, df, category, d, select=None):
        super().__init__()
        self.df = df
        self.d = d
        self.category = category
        if select is not None:
            self.df = self.df.loc[select]
            self.d = self.d[select, :]
            self.category = self.category[select]
        self.length = self.category.shape[0]

    def __getitem__(self, idx):
        x_out = self.d[idx, :]
        y_out = self.category[idx]
        return x_out, y_out

    def __len__(self):
        return self.length


def loadZY5MDS(csv_fn, npy_fn):
    df = pd.read_csv(csv_fn)
    d = np.load(npy_fn).astype("float32") / 255
    category = df["Category"].values
    category[category == 3] = 1
    category = category - 1

    select = np.random.choice([0, 1], size=len(category), p=[.3, .7])
    train_ds = ZY5MDataset(df, category, d, select == 1)
    test_ds = ZY5MDataset(df, category, d, select == 0)

    return train_ds, test_ds


def imdcZY5M():
    mod_fn = r"E:\ImageData\CambodianMNR\fenleit1\mods\20230509H162633\model_50.pth"
    """
    "E:\ImageData\CambodianMNR\fenleit1\mods\20230509H162633\model_50.pth"
    "E:\ImageData\CambodianMNR\fenleit1\mods\20230509H162856\model_22.pth"
    :return: 
    "E:\ImageData\CambodianMNR\testcate\zy5m_testcate_1.dat"
    "E:\ImageData\CambodianMNR\test2\t.dat"
    """
    mod = ZY5MNet(in_channel=3, win_size=9, n_category=1)
    mod.load_state_dict(torch.load(mod_fn))
    tec = Test1EnviCate(r"E:\ImageData\CambodianMNR\testcate\zy5m_testcate_1.dat")
    tec.run(r"E:\ImageData\CambodianMNR\fenleit1\imdc\t2_3_c11.dat", mod,
            spl_size=[9, 9], row_start=10, row_end=-10,
            column_start=10, column_end=-10)


def getdirfiles(dir_name, ext=None):
    """ 获得文件夹下的拓展名为ext的文件
    """
    files = []
    for f0 in os.listdir(dir_name):
        ff = os.path.join(dir_name, f0)
        if os.path.isfile(ff):
            if ext is None:
                files.append(ff)
            if os.path.splitext(f0)[1] == ext:
                files.append(ff)
    return files


def imdcZY5MAll():
    o_dir = r"F:\ProjectSet\FiveMeter\ZY5M\retiles1"
    c_dir = r"F:\ProjectSet\FiveMeter\ZY5M\retiles1_imdc\model_50_2"
    mod_fn = r"F:\ProjectSet\FiveMeter\FMIPytorchModel\Mods\zy5m\model_50.pth"
    device = "cuda:0"

    mod = ZY5MNet(in_channel=3, win_size=9, n_category=1)
    mod.load_state_dict(torch.load(mod_fn))
    mod.to(device)

    o_files = getdirfiles(o_dir, ".dat")
    run_time = RumTime(len(o_files))
    run_time.strat()

    for i, o_fn in enumerate(o_files):
        fn = os.path.split(o_fn)[1]
        to_fn = os.path.join(c_dir, "Imdc_" + fn)
        print(i + 1, to_fn)

        if not os.path.isfile(to_fn):
            tec = Test1EnviCate(o_fn)
            tec.run(to_fn, mod, spl_size=[9, 9], row_start=10, row_end=-10, column_start=10, column_end=-10)

            run_time.add()
            run_time.printInfo()

        print()


def trainZY5M():
    pytorch_training = TestPytorchTraining(
        2,
        model_dir=r"E:\ImageData\CambodianMNR\fenleit1\mods",
        epochs=30,
        device="cuda:0",
        n_test=1000
    )
    train_ds, test_ds = loadZY5MDS(r"E:\ImageData\CambodianMNR\fenleit1\sample\zy5mfnelei1_sample12_spl.csv",
                                   r"E:\ImageData\CambodianMNR\fenleit1\sample\zy5mfnelei1_sample12_spl.npy")
    pytorch_training.trainLoader(train_ds, batch_size=32, shuffle=True)
    pytorch_training.testLoader(test_ds, batch_size=32, shuffle=True)
    pytorch_training.addModel(ZY5MNet(in_channel=3, win_size=9, n_category=1))
    pytorch_training.addCriterion(ZY5MLoss())
    pytorch_training.addOptimizer()
    pytorch_training.train()


def test1():
    zy5m_net = ZY5MNet()
    zy5m_loss = ZY5MLoss()
    print(zy5m_net)
    x = torch.rand((32, 3, 16, 16))
    logits = zy5m_net(x)
    y = torch.zeros(32)
    loss = zy5m_loss(logits, y)


def train1():
    pytorch_training = TestPytorchTraining(
        2,
        model_dir=r"E:\ImageData\CambodianMNR\fenleit1\mods",
        epochs=10,
        device="cuda:0",
        n_test=100
    )

    train_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = torchvision.datasets.MNIST('./data/', train=True, download=True, transform=train_transform)
    pytorch_training.trainLoader(train_ds, batch_size=32, shuffle=True)
    test_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    test_ds = torchvision.datasets.MNIST('./data/', train=False, download=True, transform=test_transform)
    pytorch_training.testLoader(test_ds, batch_size=128, shuffle=True)

    pytorch_training.addModel(ZY5MNet())
    pytorch_training.addCriterion(nn.CrossEntropyLoss())
    pytorch_training.addOptimizer()
    pytorch_training.train()


def method_name():
    train_log = TrainLog(r"D:\FiveMeterImper\PytorchModel\Temp\train_log.csv")
    train_log.addField("epoch", field_type="int")
    train_log.addField("train_accuracy", field_type="float")
    train_log.addField("test_accuracy", field_type="float")
    train_log.addField("model_filename", field_type="string")
    train_log.printOptions(print_float_decimal=6)
    train_log.printFirstLine(is_to_file=True)
    train_log.saveHeader()
    cm = ConfusionMatrix(3, ["IS", "NOIS", "WATER"])
    for i in range(1000):
        train_log.updateField("epoch", i + 1)
        train_log.updateField("train_accuracy", random.random() * 100)
        train_log.updateField("test_accuracy", random.random() * 100)
        train_log.updateField("model_filename", "mod_name" + str(i))
        train_log.print(is_to_file=True)
        train_log.saveLine()
        train_log.newLine()

        cm.addData(y_true=np.random.randint(1, 4, size=10), y_pred=np.random.randint(1, 4, size=10))
    cm.printCM()


class ZY5MFrontTrainDataLoader:

    def __init__(self, raster_fn=r"F:\BaiduNetdiskDownload\jpz5m\Mosaic-5m-origin.tif"):
        self.gr = GDALRaster(raster_fn)
        self.jpz_isin = CoorInPoly()
        self.jpz_isin.readCoors(r"F:\ProjectSet\FiveMeter\Region\jpz_region.txt")

    def get(self, win_row_size=1, win_column_size=1):
        i = 0
        while True:
            x = getRandom(self.gr.x_min, self.gr.x_max)
            y = getRandom(self.gr.y_min, self.gr.y_max)
            x0, y0 = self.gr.toWgs84(x, y)
            if self.jpz_isin.t(x0, y0):
                # print(x, y, x0, y0, "1", sep=",")
                break
            i += 1
            if i == 10000:
                raise Exception("ZY5MFrontTrainDataLoader can not find coor.")
        return self.gr.readAsArrayCenter(x, y, is_geo=True, is_trans=False, win_row_size=win_row_size,
                                         win_column_size=win_column_size)

    def gets(self, n=1, win_row_size=1, win_column_size=1, no_data=0):
        d = np.ones([n, self.gr.n_channels, win_row_size, win_column_size]) * no_data
        for i in range(n):
            d[i, :] = self.get(win_row_size=win_column_size, win_column_size=win_column_size)
            if i % 1000 == 0:
                print(i)
        i0, j0 = int(win_row_size / 2), int(win_column_size / 2)
        return d, np.mean(d[:, :, i0, j0], axis=1)


if __name__ == "__main__":
    save_dfn = DirFileName(r"F:\ProjectSet\FiveMeter\ZY5M\fenleit2\train\1")
    dataloader = ZY5MFrontTrainDataLoader()
    x, y = dataloader.gets(10000, 15, 15, 0)
    print(x.shape)
    print(y.shape)
    np.save(save_dfn.fn("zy5m_testspl_d1.npy"), x.astype("int8"))
    np.save(save_dfn.fn("zy5m_testspl_y1.npy"), y.astype("int8"))

# {
# gdal.GDT_Unknown   : "Unknown",
# gdal.GDT_Byte      : "int8",
# gdal.GDT_UInt16    : "uint16",
# gdal.GDT_Int16     : "int16",
# gdal.GDT_UInt32    : "uint32",
# gdal.GDT_Int32     : "int32",
# gdal.GDT_Float32   : "float32",
# gdal.GDT_Float64   : "float64"
# }
# {
# "Unknown": gdal.GDT_Unknown,
# "int8": gdal.GDT_Byte      ,
# "uint16": gdal.GDT_UInt16  ,
# "int16": gdal.GDT_Int16    ,
# "uint32": gdal.GDT_UInt32  ,
# "int32": gdal.GDT_Int32    ,
# "float32": gdal.GDT_Float32,
# "float64": gdal.GDT_Float64
# }
