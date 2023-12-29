# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : JPZ5MFunc.py
@Time    : 2023/9/21 9:55
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of JPZ5MFunc
-----------------------------------------------------------------------------"""
import numpy as np
import pandas as pd

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.SRTData import SRTDataset
from ZY5M.ZY1FJinBian import ZY1F_JINBIAN_DFN


def main():
    spl_fn = ZY1F_JINBIAN_DFN.fn(r"samples\jinbian2021_spl3_3.csv")
    train_d_fn = ZY1F_JINBIAN_DFN.fn(r"samples\jinbian2021_spl3_3\sum.npy")
    df = pd.read_csv(spl_fn)
    categorys = df["WX_V2"].values.tolist()
    sds = SRTDataset()
    sds.addCategory("NOIS")
    sds.addCategory("IS")
    sds.addNPY(categorys, train_d_fn)
    pass


def method_name():
    # 使用CSV文件在影像中提取样本的数据
    spl_fn = ZY1F_JINBIAN_DFN.fn(r"samples\jinbian2021_spl3_3.csv")
    raster_fn = ZY1F_JINBIAN_DFN.fn("ZY1F_JinBian_2.tif")
    train_d_fn = ZY1F_JINBIAN_DFN.fn(r"samples\jinbian2021_spl3_3\sum.npy")
    spl_size = [9, 9]
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


if __name__ == "__main__":
    main()
