# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SunHuo.py
@Time    : 2023/12/19 10:57
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of SunHuo
-----------------------------------------------------------------------------"""
import math
import os

import numpy as np
from matplotlib import pyplot as plt
from osgeo import gdal

from SRTCodes.GDALRasterIO import GDALRasterWarp, GDALRaster
from Shadow.ShadowGeoDraw import GDALResterCenterDraw


def drawShadowImage_Optical(name, x, y, rows, columns, raster_fn, to_fn, d_min=0, d_max=3500, channel_list=None,
                            width=0.0, height=0.0, is_expand=False):
    grcd = GDALResterCenterDraw(name=name, raster_fn=raster_fn, channel_list=channel_list)
    grcd.read(x, y, rows, columns)
    grcd.scaleMinMax(d_min, d_max)
    to_fn = os.path.join(to_fn, name + ".png")
    grcd.toImage(to_fn, width=width, height=height, is_expand=is_expand)


def drawShadowImage_SAR(name, x, y, rows, columns, raster_fn, to_fn, d_min=-60.0, d_max=60.0, channel_list=None,
                        width=0.0, height=0.0, is_expand=False):
    grcd = GDALResterCenterDraw(name=name, raster_fn=raster_fn, channel_list=channel_list)
    grcd.read(x, y, rows, columns)
    grcd.callBackFunc(_10log10)
    grcd.scaleMinMax(d_min, d_max)
    to_fn = os.path.join(to_fn, name + ".png")
    grcd.toImage(to_fn, width=width, height=height, is_expand=is_expand)


def main():
    def func1(fn):
        to_fn = fn + "_2"
        gr = GDALRaster(fn)
        d = gr.readAsArray()
        d = np.abs(d)
        gr.save(d,to_fn, dtype=gdal.GDT_Float32)

    func1(r"G:\ImageData\2598957_Paris\2598957_Paris\SLC\s11.bin")
    func1(r"G:\ImageData\2598957_Paris\2598957_Paris\SLC\s12.bin")
    func1(r"G:\ImageData\2598957_Paris\2598957_Paris\SLC\s21.bin")
    func1(r"G:\ImageData\2598957_Paris\2598957_Paris\SLC\s22.bin")


def method_name2():

    filename = r"F:\ProjectSet\Huo\huatu\images\ht_619.tif"
    gr = GDALRaster(filename)
    d = gr.readAsArray()

    d_out = np.zeros((d.shape[0], d.shape[1], d.shape[2] * 2))
    alpha1 = -1
    for i in range(d.shape[2]):
        alpha0 = int(math.sin(math.pi / 2 * i / d.shape[2]) * i * 1.5)
        if alpha1 == -1:
            alpha1 = alpha0
        if alpha1 < alpha0:
            for j in range(alpha1, alpha0):
                d_out[:, :, alpha1] = d[:, :, i]
                alpha1 += 1
        print(i, alpha0)
        alpha1 = alpha0 + 1
        d_out[:, :, alpha0] = d[:, :, i]
    gr.save(d_out, r"F:\ProjectSet\Huo\huatu\images\ht_619_2.dat")

    print(d.shape)

    pass


def method_name3():
    coors = [[110.89118319545658, 34.518519233545675, 0, ]]
    zy5m_grw = GDALRasterWarp()
    zy5m_grw.initGDALRaster(r"G:\ImageData\S1A_IW_GRDH_1SDV_20180106T103634_20180106T103659_020033_02221D_2345\S"
                            r"1A_IW_GRDH_1SDV_20180106T103634_20180106T103659_020033_02221D_2345.data\Amplitude_VV.img")
    zy5m_grw.addGCP(110.89118319545658, 34.518519233545675, 0, 15793, 13871)
    zy5m_grw.addGCP(112.09395752320125, 34.14031113159616, 0, 9860, 24040)
    zy5m_grw.addGCP(111.46232893017176, 33.306915766902854, 0, 1822, 16734)
    zy5m_grw.addGCP(110.42148375662285, 32.981702141025536, 0, 14, 6559)
    zy5m_grw.warp(r"G:\ImageData\S1A_IW_GRDH_1SDV_20180106T103634_20180106T103659_020033_02221D_2345\S"
                  r"1A_IW_GRDH_1SDV_20180106T103634_20180106T103659_020033_02221D_2345.data\Amplitude_VV_2.img"
                  , xres=zy5m_grw.RESOLUTION_ANGLE, yres=zy5m_grw.RESOLUTION_ANGLE, geo_fmt="ENVI",
                  dtype="uint16")


def method_name():
    x, y = 120.374037, 36.071475
    rows, columns = 200, 200
    to_dn = r"F:\ProjectSet\Huo\huatu\images"
    name = "JC6_1"
    width, height = 6, 6
    is_expand = True
    raster_fn = r"F:\ProjectSet\Shadow\Release\QingDaoImages\SH_QD_look_envi.dat"

    def func_optical():
        to_name = name + "_OPT"
        grcd = GDALResterCenterDraw(name=to_name, raster_fn=raster_fn, channel_list=["Red", "Green", "Blue", ])
        grcd.read(x, y, rows, columns)
        grcd.scaleMinMax(0, 2600)
        to_fn = os.path.join(to_dn, to_name + ".png")
        print(to_fn)
        grcd.toImage(to_fn, width=width, height=height, is_expand=is_expand)
        return grcd.d

    def func_SAR():
        to_name = name + "_SAR"
        grcd = GDALResterCenterDraw(name=to_name, raster_fn=raster_fn, channel_list=["AS_VV", "AS_VH", "AS_VHDVV"])
        grcd.read(x, y, rows, columns)
        grcd.scaleMinMax(-30, 6)
        to_fn = os.path.join(to_dn, to_name + ".png")
        print(to_fn)
        grcd.toImage(to_fn, width=width, height=height, is_expand=is_expand)
        return grcd.d

    d_opt = func_optical()
    d_sar = func_SAR()
    d_sar = np.sum(d_sar, axis=2)
    print(d_sar.shape)
    d_sar_show = np.zeros((10, 10))
    # for i in range(10):
    #     for j in range(10):
    #         d = np.mean(d_sar[i:i + 20, j:j + 20])
    #         d_sar_show[i, j] = d
    # d_sar_show = (d_sar_show - np.min(d_sar_show)) / (np.max(d_sar_show) - np.min(d_sar_show))
    # print(d_sar_show)
    for i in range(10):
        for j in range(10):
            d_sar_show[i, j] += 0.5
            # random.random()
    plt.imshow(d_sar_show, cmap="Greys")
    plt.xticks([i - 0.5 for i in range(10)])
    plt.yticks([i - 0.5 for i in range(10)])
    plt.grid(axis="both", linewidth=1.5, color="black")
    plt.savefig(r"F:\ProjectSet\Huo\huatu\images\im_6.26_2.jpg", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
