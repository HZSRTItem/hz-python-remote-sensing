# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ZY5MImageC.py
@Time    : 2023/6/24 13:41
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PytorchGeo of ZY5MImageC
-----------------------------------------------------------------------------"""
import os.path
import time

import numpy as np
import torch
from torch.nn import functional

from SRTCodes.GDALRasterClassification import GDALRasterPrediction
from ZY5M.ZY5MModel import ZY5MDenseNet


def zy5mDeal(d_row):
    d_row = np.sum(d_row, axis=0)
    d_select1 = np.ones(d_row.shape, dtype="bool")
    d_select2 = d_row == 765
    column = 0
    while column < d_row.shape[0]:
        if d_select2[column]:
            d_select1[column] = False
        else:
            break
        column += 1
    if column == d_row.shape[0]:
        return d_select1
    column = d_row.shape[0] - 1
    while column >= 0:
        if d_select2[column]:
            d_select1[column] = False
        else:
            break
        column += 1
    return d_select1


class ZY5MGDALRasterPrediction(GDALRasterPrediction):

    def __init__(self, geo_fn):
        super(ZY5MGDALRasterPrediction, self).__init__(geo_fn)
        self.device = "cuda"
        self.is_category = False
        self.number_pred = 3000

    def predict(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        # y = np.ones(x.shape[0])
        n = x.shape[0]

        x = x.astype("float32")
        x = x / 255
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


def changeext(fn, t, dirname):
    fn = os.path.split(fn)[1]
    fn1 = os.path.splitext(fn)
    return os.path.join(dirname, fn1[0] + str(t) + ".tif")


def readSelectFiles(txt_fn):
    with open(txt_fn, "r", encoding="utf-8") as fr:
        lines = fr.readlines()
    files = []
    for f in lines:
        f = f.strip()
        if f == "":
            continue
        if f.startswith("#"):
            continue
        if os.path.isfile(f):
            files.append(f)
        else:
            print("Warning: file is not existed. " + f)
    return files


def main():
    raster_fns = readSelectFiles(r"F:\ProjectSet\FiveMeter\ZY5M\TestImages\SelectRetiles1.txt")
    #     [
    #     r"F:\ProjectSet\FiveMeter\ZY5M\retiles1\Mosaic-5m-Cambodia_18_14.dat",
    #     r"F:\ProjectSet\FiveMeter\ZY5M\TestImages\zy5m_testim_1.tif",
    #     r"F:\ProjectSet\FiveMeter\ZY5M\TestImages\zy5m_testim_2.tif",
    #     r"F:\ProjectSet\FiveMeter\ZY5M\TestImages\zy5m_testim_3.tif",
    #     r"F:\ProjectSet\FiveMeter\ZY5M\TestImages\zy5m_testim_4.tif",
    #     r"F:\ProjectSet\FiveMeter\ZY5M\TestImages\zy5m_testim_5.tif",
    #     r"F:\ProjectSet\FiveMeter\ZY5M\TestImages\zy5m_testim_6.tif"
    # ]
    # "F:\ProjectSet\FiveMeter\ZY5M\fenleit2\mod\20230625H150354\ZY5M_mod2_66.pth"
    # "F:\ProjectSet\FiveMeter\ZY5M\fenleit2\mod\20230625H150244\ZY5M_mod2_110.pth"
    mod_fn = r"F:\ProjectSet\FiveMeter\ZY5M\fenleit2\mod\20230625H150354\ZY5M_mod2_66.pth"
    dirname = os.path.dirname(mod_fn)
    to_name = "_imdc1"
    device = "cuda"
    win_row_size = 7
    win_column_size = 7
    np_type = "int8"
    is_category = True

    mod = ZY5MDenseNet(growth_rate=32,
                       block_config=(6, 12),
                       num_init_features=64,
                       num_classes=1)
    mod.load_state_dict(torch.load(mod_fn))
    mod.to(device)

    for filename in raster_fns:
        imdc_fn = changeext(filename, to_name, dirname)
        print(filename, imdc_fn)

        t1 = time.time()
        grp = ZY5MGDALRasterPrediction(filename)
        grp.device = device
        grp.is_category = is_category
        grp.run(imdc_fn=imdc_fn, np_type=np_type, mod=mod,
                spl_size=[win_row_size, win_column_size],
                row_start=10, row_end=-10,
                column_start=10, column_end=-10)
        t2 = time.time()
        print(t2 - t1)
    pass


if __name__ == "__main__":
    main()
