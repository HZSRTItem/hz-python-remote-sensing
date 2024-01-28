# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : FMFuncs.py
@Time    : 2024/1/23 14:18
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of FMFuncs
-----------------------------------------------------------------------------"""

import numpy as np
from osgeo import gdal

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.Utils import filterFileExt, changext, changefiledirname


def main():
    fns = [] # filterFileExt(r"G:\GraduationProject\QingDaoImages", ".tif")
    for fn in fns:
        to_fn = changefiledirname(changext(fn, "_6.dat"), r"G:\GraduationProject\QingDaoImages\RGB_VV_VH")
        changext(fn, "_6.dat")
        print(fn, to_fn)
        gr = GDALRaster(fn)
        d = gr.readAsArray()
        out_d = np.zeros((6, d.shape[1], d.shape[2]))
        out_d[:4] = d[4:8] * 10000
        out_d[4] = (((d[0] + d[2]) / 2.0) + 60) * 500
        out_d[5] = (((d[1] + d[3]) / 2.0) + 60) * 500
        gr.save(out_d.astype("uint16"), to_fn, dtype=gdal.GDT_UInt16,
                descriptions=["Blue", "Green", "Red", "NIR", "VV", "VH"])
    gr = GDALRaster(r"G:\GraduationProject\QingDaoImages\RGB_VV_VH\qdgrid_im_6_tif.tif")
    d = gr.readAsArray(100,100,  20, 10)
    print(d)


if __name__ == "__main__":
    main()
