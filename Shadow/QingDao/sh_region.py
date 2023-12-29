# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : sh_region.py
@Time    : 2023/6/12 10:03
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : BaseCodes of sh_region_ana
-----------------------------------------------------------------------------"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SRTCodes.RasterIO import ENVIRaster
from SRTCodes.Utils import DirectoryFileName

dir_QDS2RGBN_gee_2_deal = DirectoryFileName(r"G:\ImageData\QingDao\20211023\qd20211023\QDS2RGBN_gee_2_deal")


def main():
    # import image data
    envi_raster = ENVIRaster(dir_QDS2RGBN_gee_2_deal.fullname("shadow_clip"))
    d_shadow_clip = envi_raster.readAsArray()
    sh_select = d_shadow_clip[:, :, 1] == 1
    imd = d_shadow_clip[sh_select, 2:]
    df = pd.DataFrame(imd, columns=envi_raster.names[2:])
    print(df)
    df.to_csv(dir_QDS2RGBN_gee_2_deal.fullname("shadow_d1.csv"))
    pass


if __name__ == "__main__":
    main()
