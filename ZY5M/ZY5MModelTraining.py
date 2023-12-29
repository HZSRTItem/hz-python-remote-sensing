# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ZY5MModelTraining.py
@Time    : 2023/6/22 17:15
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PytorchGeo of ZY5MModelTraining
-----------------------------------------------------------------------------"""
import os.path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchsummary import summary

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.PytorchModelTraining import PytorchRegressionTraining, PytorchCategoryTraining
from SRTCodes.Utils import CoorInPoly, getRandom, DirFileName
from ZY5M.ZY5MModel import ZY5MDenseNet, ZY5MLoss

np.set_printoptions(suppress=True, precision=3, linewidth=600)


def xDeal(x, y):
    x = x.astype("float32")
    x = x / 255
    y = y - 1
    return x, y


class ZY5MFrontTrainDataset(Dataset):

    def __init__(self, raster_fn=r"F:\BaiduNetdiskDownload\jpz5m\Mosaic-5m-origin.tif",
                 win_row_size=7, win_column_size=7, n_spl=10000):
        super(ZY5MFrontTrainDataset, self).__init__()
        self.gr = GDALRaster(raster_fn)
        self.jpz_isin = CoorInPoly()
        self.jpz_isin.readCoors(r"F:\ProjectSet\FiveMeter\Region\jpz_region.txt")
        self.win_row_size = win_row_size
        self.win_column_size = win_column_size
        self.row_center = int(win_row_size / 2)
        self.column_center = int(win_column_size / 2)
        self.n_spl = n_spl

    def __getitem__(self, idx):
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
        d = self.gr.readAsArrayCenter(x, y, is_geo=True, is_trans=False, win_row_size=self.win_row_size,
                                      win_column_size=self.win_column_size)
        y = np.mean(d[:, self.row_center, self.column_center])
        return xDeal(d, y)

    def __len__(self):
        return self.n_spl


class ZY5MFrontTestDataset(Dataset):

    def __init__(self, win_row_size=7, win_column_size=7):
        super(ZY5MFrontTestDataset, self).__init__()
        fenlei2_train1_dfn = DirFileName(r"F:\ProjectSet\FiveMeter\ZY5M\fenleit2\train\1")
        self.x = np.load(fenlei2_train1_dfn.fn("zy5m_testspl_d1.npy"))
        self.x = self.x[:, :, :win_row_size, :win_column_size]
        self.y = np.mean(self.x[:, :, int(win_row_size / 2), int(win_column_size / 2)], axis=0)

    def __getitem__(self, idx):
        x_out = self.x[idx, :]
        y_out = self.y[idx]
        return xDeal(x_out, y_out)

    def __len__(self):
        return len(self.y)


class ZY5MPytorchRegressionTraining(PytorchRegressionTraining):

    def __init__(self, model_dir, model_name, epochs, device, n_test):
        super(ZY5MPytorchRegressionTraining, self).__init__(model_dir=model_dir, model_name=model_name, epochs=epochs,
                                                            device=device, n_test=n_test)

    def _printModel(self):
        summary(self.model, (3, 7, 7), device=self.device)


def zy5mTrainOrigin():
    pytorch_training = ZY5MPytorchRegressionTraining(
        model_dir=r"F:\ProjectSet\FiveMeter\ZY5M\fenleit2\mod",
        model_name="ZY5M_mod",
        epochs=10,
        device="cuda",
        n_test=5
    )
    win_row_size = 7
    win_column_size = 7
    train_ds = ZY5MFrontTrainDataset(win_row_size=win_row_size, win_column_size=win_column_size)
    test_ds = ZY5MFrontTestDataset(win_row_size=win_row_size, win_column_size=win_column_size)
    pytorch_training.trainLoader(train_ds, batch_size=128, shuffle=False)
    pytorch_training.testLoader(test_ds, batch_size=128, shuffle=False)
    pytorch_training.addModel(ZY5MDenseNet(growth_rate=32,
                                           block_config=(6, 12),
                                           num_init_features=64,
                                           num_classes=1))
    pytorch_training.addCriterion(ZY5MLoss())
    pytorch_training.addOptimizer(lr=0.0001)
    pytorch_training.train()


class ZY5MDataset(Dataset):

    def __init__(self, spl_xlsx_fn, data_fn=None, cname="CATEGORY"):
        super(ZY5MDataset, self).__init__()
        self.spl_xlsx_fn = spl_xlsx_fn
        self.data_fn = data_fn
        self.df = None
        self.d = None
        self.cname = cname
        self.train_d_fn = os.path.splitext(self.spl_xlsx_fn)[0] + "_train.npy"
        self.test_d_fn = os.path.splitext(self.spl_xlsx_fn)[0] + "_test.npy"

    def readTrainDF(self):
        self.df = pd.read_excel(self.spl_xlsx_fn, sheet_name="train")
        for k in self.df:
            if k.lower() == self.cname.lower():
                self.cname = k
        self.d = np.load(self.train_d_fn)

    def readTestDF(self):
        self.df = pd.read_excel(self.spl_xlsx_fn, sheet_name="test")
        for k in self.df:
            if k.lower() == self.cname.lower():
                self.cname = k
        self.d = np.load(self.test_d_fn)

    def readData(self):
        self.d = np.load(self.data_fn)

    def get(self, idx):
        x = self.d[idx, :]
        y = self.df[self.cname][idx]
        return xDeal(x, y)

    def __getitem__(self, index) -> T_co:
        return self.get(index)

    @classmethod
    def sampleRaster(cls, raster_fn, xlsx_fn, spl_size, train_d_fn, test_d_fn):
        ds = ZY5MDataset(xlsx_fn)
        gr = GDALRaster(raster_fn)

        ds.readTrainDF()
        d = np.zeros([len(ds.df), gr.n_channels, spl_size[0], spl_size[1]])
        print(d.shape)
        for i in range(len(ds.df)):
            x = ds.df["X"][i]
            y = ds.df["Y"][i]
            d[i, :] = gr.readAsArrayCenter(x, y, win_row_size=spl_size[0], win_column_size=spl_size[1],
                                           interleave="band",
                                           is_geo=True, is_trans=True)
        np.save(train_d_fn, d)

        ds.readTestDF()
        d = np.zeros([len(ds.df), gr.n_channels, spl_size[0], spl_size[1]])
        print(d.shape)
        for i in range(len(ds.df)):
            x = ds.df["X"][i]
            y = ds.df["Y"][i]
            d[i, :] = gr.readAsArrayCenter(x, y, win_row_size=spl_size[0], win_column_size=spl_size[1],
                                           interleave="band",
                                           is_geo=True, is_trans=True)
        np.save(test_d_fn, d)


class ZY5MTrainDataset(ZY5MDataset):

    def __init__(self, spl_xlsx_fn, data_fn=None, cname="CATEGORY"):
        super(ZY5MTrainDataset, self).__init__(spl_xlsx_fn=spl_xlsx_fn, data_fn=data_fn, cname=cname)
        self.readTrainDF()

    def __getitem__(self, idx):
        return self.get(idx)

    def __len__(self):
        return len(self.df)


class ZY5MTestDataset(ZY5MDataset):

    def __init__(self, spl_xlsx_fn, data_fn=None, cname="CATEGORY"):
        super(ZY5MTestDataset, self).__init__(spl_xlsx_fn=spl_xlsx_fn, data_fn=data_fn, cname=cname)
        self.readTestDF()

    def __getitem__(self, idx):
        return self.get(idx)

    def __len__(self):
        return len(self.df)


class ZY5MPytorchCategoryTraining(PytorchCategoryTraining):

    def __init__(self, model_dir=None, model_name="model", n_category=2, category_names=None,
                 epochs=10, device=None, n_test=100):
        super(ZY5MPytorchCategoryTraining, self).__init__(
            model_dir=model_dir, model_name=model_name, n_category=n_category, category_names=category_names,
            epochs=epochs, device=device, n_test=n_test)

    def _printModel(self):
        summary(self.model, (3, 7, 7), device=self.device)

    def logisticToCategory(self, logts):
        y = logts.sigmoid()
        y = (y > 0.5) * 1
        y = y.cpu().numpy()
        return y[:, 0]


def zy5mTrain():
    # 20230625H075716
    sample_fn = r"F:\ProjectSet\FiveMeter\ZY5M\fenleit2\train\2\zy5m_spl1_1.xlsx"
    mod_fn = r"F:\ProjectSet\FiveMeter\ZY5M\fenleit2\mod\20230625H075716\ZY5M_mod_51.pth"

    pytorch_training = ZY5MPytorchCategoryTraining(
        model_dir=r"F:\ProjectSet\FiveMeter\ZY5M\fenleit2\mod",
        model_name="ZY5M_mod2",
        n_category=2,
        category_names=["NOIS", "IS"],
        epochs=10,
        device="cuda",
        n_test=5
    )

    win_row_size = 7
    win_column_size = 7

    mod = ZY5MDenseNet(growth_rate=32, block_config=(6, 12), num_init_features=64, num_classes=1)
    # mod.load_state_dict(torch.load(mod_fn))

    train_ds = ZY5MTrainDataset(sample_fn)
    test_ds = ZY5MTestDataset(sample_fn)
    pytorch_training.trainLoader(train_ds, batch_size=64, shuffle=True)
    pytorch_training.testLoader(test_ds, batch_size=64, shuffle=False)
    pytorch_training.addModel(mod)
    pytorch_training.addCriterion(ZY5MLoss())
    pytorch_training.addOptimizer(lr=0.001)
    pytorch_training.train()


def main():
    # jpz_image_fn = r"F:\BaiduNetdiskDownload\jpz5m\Mosaic-5m-Cambodia.tif"
    # temp_dir = DirectoryFileName(r"F:\ProjectSet\FiveMeter\ZY5M\temp")
    # jpz_isin = JPZCoorIsIn()
    # # print(jpz_isin.isin([104.2811479, 11.6580675]))
    # gr = GDALRaster(jpz_image_fn)
    # f = open(temp_dir.fn("t2.csv"), "w", encoding="utf-8")
    # print("id", "X", "Y", "IN", sep=",", file=f)
    # for i in range(10000):
    #     x = getRandom(gr.x_min, gr.x_max)
    #     y = getRandom(gr.y_min, gr.y_max)
    #     x, y = gr.toWgs84(x, y)
    #     print(i + 1, x, y, jpz_isin.t(x, y), sep=",", file=f)
    # f.close()

    # t = ZY5MFrontTrainDataLoader()
    # t.gr.readAsArrayCenter(107.42701790601555, 12.305601868958993, is_geo=True, is_trans=True)
    # x, y = t.gets(128, 13, 13)

    zy5mTrain()
    # sampleRaster(r"F:\ProjectSet\FiveMeter\ZY5M\fenleit2\train\2\zy5m_spl1_1.xlsx",
    #              raster_fn=r"F:\BaiduNetdiskDownload\jpz5m\Mosaic-5m-origin.tif",
    #              spl_size=[7, 7])

    pass


if __name__ == "__main__":
    main()
