import gc
import sys

import pandas as pd
import torch
from osgeo import gdal
from torch import nn
from torch.utils.data import Dataset

from SRTCodes.GDALUtils import GDALSamplingInit, GDALSamplingFast, GDALSampling
from SRTCodes.NumpyUtils import connectedComponent, categoryMap, NumpyDataCenter
from SRTCodes.SRTModelImage import SRTModImPytorch
from SRTCodes.Utils import readText, printList
from Shadow.Hierarchical import SHHConfig

sys.path.append(r"F:\PyCodes")
# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SRTFuncs.py
@Time    : 2023/11/3 9:59
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of SRTFuncs
-----------------------------------------------------------------------------"""

import numpy as np
from scipy.ndimage import uniform_filter

from SRTCodes.GDALRasterIO import readGEORaster, saveGEORaster, GDALRaster, GDALRasterChannel
from Shadow.ShadowGeoDraw import _10log10


class TNet(nn.Module):

    def __init__(self):
        super(TNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.act = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 2, 3, 1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


def main():
    gsf = GDALSampling()
    gsf.initNPYRaster(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\2\SHH2_QD2_data.npy")
    data = gsf.sampling2DF(
        df=pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\sh2_spl25_2_spl2_2.csv"),
        win_row=21, win_column=21
    )
    print(data[0][:3])
    np.save(r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\sh2_spl25_2_spl2_2_data.npy", data.astype("float32"))
    # input(">")
    # del gsf.gr.data
    # gsf.gr.data = None
    # gc.collect()
    # # data1 = np.load(r"F:\Week\20240609\Data\spl1_data.npy")
    # # data2 = np.load(r"F:\Week\20240609\Data\spl1_data2.npy")
    # # print(np.sum(data1-data2))
    # input(">")
    pass


def method_name4():
    bib2csv(r"F:\Articles\ConnectedPapers\Derivative-Works-for-"
            r"Improving-the-impervious-surface-estimation-with-combined-use-o"
            r"f-optical-and-SAR-remote-sensing-images.bib",
            r"F:\Articles\ConnectedPapers\tmp.csv")


def bib2csv(bib_fn, csv_fn):
    bib_fns = [
        r"F:\Articles\ConnectedPapers\ConnectedPapers-for-Improving-the-impervious-surface-estimation-with-combined-use-of-optical-and-SAR-remote-sensing-images.bib"
        ,r"F:\Articles\ConnectedPapers\Derivative-Works-for-Improving-the-impervious-surface-estimation-with-combined-use-of-optical-and-SAR-remote-sensing-images.bib"
        ,r"F:\Articles\ConnectedPapers\Prior-Works-for-Improving-the-impervious-surface-estimation-with-combined-use-of-optical-and-SAR-remote-sensing-images.bib"
    ]

    bib_text = ""
    for bib_fn in bib_fns:
        bib_text += "\n\n"
        bib_text += readText(bib_fn)

    def getch(_text):
        n_kh = 0
        lines = []
        line = None
        for i, ch in enumerate(_text):
            if ch == "{":
                if n_kh == 0:
                    line = ""
                n_kh += 1
            elif ch == "}":
                n_kh -= 1
            if line is not None:
                line += ch
            if n_kh == 0:
                if line is not None:
                    lines.append(line)
                    line = None
        return lines

    list1 = getch(bib_text)

    def getkeys():
        for line in bib_text.split('\n'):
            if "=" in line:
                k = line.split("=")[0]
                k = k.strip()
                if k not in keys:
                    keys.append(k)
    keys = []
    getkeys()
    print(keys)

    to_list = []
    for line in list1:
        to_dict = {}
        for line1 in line.split("\n"):
            for k in keys:
                if line1.startswith(k):
                    tmp_ch = getch(line1)
                    if len(tmp_ch) ==1 :
                        to_dict[k] = tmp_ch[0].strip("{}")

        to_list.append(to_dict)
    print(pd.DataFrame(to_list))
    pd.DataFrame(to_list).to_csv(csv_fn)


def method_name3():
    import torch.nn.functional as F
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
        out_x = np.zeros((10, x.shape[1], x.shape[2]))
        out_x[0:2] = x[0:2] / 30 + 1
        out_x[2:4] = x[3:5] / 30 + 1
        out_x[4:] = x[6:] / 3000
        x = out_x
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

    smip = SRTModImPytorch()
    smip.model_dirname = r"F:\Week\20240331\Data"
    smip.model_name = "Net2"
    smip.epochs = 100
    smip.device = "cuda"
    smip.n_test = 10
    smip.batch_size = 32
    smip.n_class = 8
    smip.class_names = SHHConfig.SHH_CNAMES8
    smip.win_size = (5, 5)
    smip.model = Net2().to(smip.device)

    def func_predict(model: Net, x: torch.Tensor):
        logit = model(x)
        y = torch.argmax(logit, dim=1) + 1
        return y

    smip.func_predict = func_predict

    def train():
        # shh_spl = SHH2Samples()
        # shh_spl.addCSV(r"F:\ProjectSet\Shadow\Hierarchical\Samples\11\sh2_spl11_21.csv")
        # shh_spl.loadNpy(r"F:\ProjectSet\Shadow\Hierarchical\Samples\11\sh2_spl11_21_data.npy")
        # shh_spl.ndc.__init__(3, smip.win_size, (21, 21))
        # shh_spl = copySHH2Samples(shh_spl).addSamples(shh_spl.filterNotEQ("CATEGORY", 0))
        # shh_spl.initCategory("CATEGORY", map_dict=SHHConfig.CATE_MAP_SH881, others=0)
        # shh_spl = copySHH2Samples(shh_spl).addSamples(shh_spl.filterEQ("SH_IMDC", 2))
        ds = NetDataset()
        ds.ndc.__init__(3, smip.win_size, (21, 21))
        smip.train_ds, smip.test_ds = torch.utils.data.random_split(dataset=ds, lengths=[0.8, 0.2], )

        smip.timeDirName()
        smip.initTrainLog()
        smip.initPytorchTraining()
        smip.pt.func_logit_category = func_predict
        smip.pt.func_y_deal = lambda y: y + 1
        smip.initModel()
        smip.initDataLoader()
        smip.initCriterion(nn.CrossEntropyLoss())
        smip.initOptimizer(torch.optim.Adam, lr=0.0001, eps=0.00001)
        smip.copyFile(__file__)
        print(smip.model_dirname)
        smip.train()

    def imdc():
        smip.loadPTH()
        grc: GDALRasterChannel = GDALRasterChannel()
        smip.imdc(grc=grc, is_jdt=True, data_deal=data_deal)
        pass

    train()


def method_name2():
    # x = torch.randn((1, 3, 128, 128))
    # mod = TNet()
    # out_x = mod(x)
    gr = GDALRaster(r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\shadowimdc\cd_shadowimdc1.tif")
    data = gr.readAsArray() - 1
    print(np.unique(data, return_counts=True))
    out_d = connectedComponent(data, is_jdt=True)
    gr.save(out_d.astype("float32"),
            r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\shadowimdc\cd_shadowimdc1_4.tif",
            fmt="GTiff", dtype=gdal.GDT_Float32)


def method_name1():
    # Define
    def lee_filter(image, window_size):
        """ Lee filter function """
        # Calculate the local mean using a convolution
        local_mean = uniform_filter(image, (window_size, window_size))

        # Calculate the local variance
        local_variance = uniform_filter(image ** 2, (window_size, window_size)) - local_mean ** 2

        # Estimate the noise variance
        noise_variance = local_variance.mean()

        # Calculate the filtered image
        filtered_image = local_mean + (image - local_mean) * np.minimum(
            noise_variance / (local_variance + noise_variance),
            1)

        return filtered_image

    # Load your SAR image as a NumPy array
    # Replace 'your_image_array' with your SAR image array
    # Example:
    # your_image_array = np.array([[...]])
    your_image_array = readGEORaster(r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_envi.dat", band_list=["AS_C11"])
    print(your_image_array.shape)
    # Set the window size for the filter (you can adjust this parameter)
    window_size = 7
    # Apply the Lee filter to the SAR image
    filtered_sar_image = lee_filter(your_image_array, window_size)
    print(filtered_sar_image.shape)
    saveGEORaster(_10log10(filtered_sar_image), r"F:\ProjectSet\Shadow\BeiJing\Image\Temp\tmp35")


if __name__ == "__main__":
    main()
