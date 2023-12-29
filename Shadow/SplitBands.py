# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SplitBands.py
@Time    : 2023/6/29 11:00
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : GEOCodes of SplitBands
G:\ImageData\QingDao\20211023\qd20211023\QDS2RGBN_gee_2
-----------------------------------------------------------------------------"""
import os.path

from osgeo import gdal
import numpy as np

from SRTCodes.GDALRasterIO import GDALRaster


def main():
    images = [
        r"G:\ImageData\QingDao\20211023\qd20211023\Temp\temp2_2.dat",
        r"G:\ImageData\QingDao\20211023\qd20211023\Temp\C11_AS_1.dat",
        r"G:\ImageData\QingDao\20211023\qd20211023\Temp\C22_AS_1.dat",
        r"G:\ImageData\QingDao\20211023\qd20211023\Temp\C11_DE_1.dat",
        r"G:\ImageData\QingDao\20211023\qd20211023\Temp\C22_DE_1.dat",
    ]
    save_fn = r"G:\ImageData\QingDao\20211023\qd20211023\Temp\temp3_2.dat"

    grs = []
    d = []

    for im in images:
        print(im)
        grs.append(GDALRaster(im))
        d0 = grs[-1].readAsArray()
        if len(d0.shape) == 2:
            d0 = np.array([d0])
        d.append(d0)
        print(d[-1].shape)

    d = np.concatenate(d)

    grs[0].save(d, save_fn, dtype=gdal.GDT_Float32)
    pass


def method_name():
    dir_raster = r"G:\ImageData\QingDao\20211023\qd20211023\QDS2RGBN_gee_2"
    filename = dir_raster + ".dat"
    gr = GDALRaster(filename)
    d = gr.readAsArray()
    print(d.shape)
    for i in range(len(d)):
        to_f = os.path.join(dir_raster, str(i + 1) + ".dat")
        print(to_f)
        if i in [4, 5, 7, 8]:
            d[i] = np.power(10, d[i] / 10)
        gr.save(d[i], save_geo_raster_fn=to_f)


if __name__ == "__main__":
    main()
