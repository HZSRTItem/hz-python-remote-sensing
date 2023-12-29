# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowMkTu.py
@Time    : 2023/11/10 10:22
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of ShadowMkTu
-----------------------------------------------------------------------------"""
import numpy as np

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.Utils import DirFileName

SH_RELE_DFN = DirFileName(r"F:\ProjectSet\Shadow\Release")
SH_MKTU_DFN = DirFileName(r"F:\ProjectSet\Shadow\MkTu")
QD_raster_fn = SH_RELE_DFN.fn("QD_SH.vrt")
BJ_raster_fn = SH_RELE_DFN.fn("BJ_SH.vrt")
CD_raster_fn = SH_RELE_DFN.fn("CD_SH.vrt")


def main():
    raster_fn = QD_raster_fn
    channel_name = "Red"
    func_forward = lambda x: 10*np.log10(x)
    save_fn

    gr = GDALRaster(raster_fn)
    d = gr.readGDALBand(channel_name)
    d = func_forward(d)

    pass


if __name__ == "__main__":
    main()
