# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2FC.py
@Time    : 2024/6/7 19:46
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2FC
-----------------------------------------------------------------------------"""
import numpy as np
from osgeo import gdal

from SRTCodes.GDALRasterIO import saveGTIFFImdc, GDALRaster


def fc1(to_fn, vhl_fn, is_fn, ws_fn, vhl_dict=None, is_dict=None, ws_dict=None):
    if ws_dict is None:
        ws_dict = {"WAT": 1, "IS_SH": 2, "VEG_SH": 3, "SOIL_SH": 4, "WAT_SH": 5}
    if is_dict is None:
        is_dict = {"IS": 2, "SOIL": 1}
    if vhl_dict is None:
        vhl_dict = {"HIGH": 2, "VEG": 1, "LOW": 3}
    vhl_imdc, is_imdc, ws_imdc = readImdc1(is_fn, vhl_fn, ws_fn)
    to_imdc = np.zeros_like(vhl_imdc)
    to_imdc[vhl_imdc == vhl_dict["VEG"]] = 2

    is_imdc_tmp = np.zeros_like(is_imdc)
    is_imdc_tmp[is_imdc == is_dict["IS"]] = 1
    is_imdc_tmp[is_imdc == is_dict["SOIL"]] = 3
    to_imdc[vhl_imdc == vhl_dict["HIGH"]] = is_imdc_tmp[vhl_imdc == vhl_dict["HIGH"]]

    ws_imdc_tmp = np.zeros_like(ws_imdc)
    ws_imdc_tmp[ws_imdc == ws_dict["WAT"]] = 4
    ws_imdc_tmp[ws_imdc == ws_dict["IS_SH"]] = 1
    ws_imdc_tmp[ws_imdc == ws_dict["VEG_SH"]] = 2
    ws_imdc_tmp[ws_imdc == ws_dict["SOIL_SH"]] = 3
    ws_imdc_tmp[ws_imdc == ws_dict["WAT_SH"]] = 4
    to_imdc[vhl_imdc == vhl_dict["LOW"]] = ws_imdc_tmp[vhl_imdc == vhl_dict["LOW"]]

    saveGTIFFImdc(GDALRaster(vhl_fn), to_imdc.astype("int8"), to_fn,
                  color_table={1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), })


def fc2(to_fn, vhl_fn, is_fn, ws_fn, vhl_dict=None, is_dict=None, ws_dict=None):
    if ws_dict is None:
        ws_dict = {"IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 3, "WAT_SH": 4}
    if is_dict is None:
        is_dict = {"IS": 2, "SOIL": 1}
    if vhl_dict is None:
        vhl_dict = {"HIGH": 2, "VEG": 1, "LOW": 4, "WAT":3}
    vhl_imdc, is_imdc, ws_imdc = readImdc1(is_fn, vhl_fn, ws_fn)
    to_imdc = np.zeros_like(vhl_imdc)
    to_imdc[vhl_imdc == vhl_dict["VEG"]] = 2
    to_imdc[vhl_imdc == vhl_dict["WAT"]] = 4

    is_imdc_tmp = np.zeros_like(is_imdc)
    is_imdc_tmp[is_imdc == is_dict["IS"]] = 1
    is_imdc_tmp[is_imdc == is_dict["SOIL"]] = 3
    to_imdc[vhl_imdc == vhl_dict["HIGH"]] = is_imdc_tmp[vhl_imdc == vhl_dict["HIGH"]]

    ws_imdc_tmp = np.zeros_like(ws_imdc)
    ws_imdc_tmp[ws_imdc == ws_dict["IS_SH"]] = 1
    ws_imdc_tmp[ws_imdc == ws_dict["VEG_SH"]] = 2
    ws_imdc_tmp[ws_imdc == ws_dict["SOIL_SH"]] = 3
    ws_imdc_tmp[ws_imdc == ws_dict["WAT_SH"]] = 4
    to_imdc[vhl_imdc == vhl_dict["LOW"]] = ws_imdc_tmp[vhl_imdc == vhl_dict["LOW"]]

    saveGTIFFImdc(GDALRaster(vhl_fn), to_imdc.astype("int8"), to_fn,
                  color_table={1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), })


def readImdc1(is_fn, vhl_fn, ws_fn):
    vhl_imdc = gdal.Open(vhl_fn, gdal.GA_ReadOnly).ReadAsArray()
    is_imdc = gdal.Open(is_fn, gdal.GA_ReadOnly).ReadAsArray()
    ws_imdc = gdal.Open(ws_fn, gdal.GA_ReadOnly).ReadAsArray()
    print(vhl_imdc.shape)
    print(is_imdc.shape)
    print(ws_imdc.shape)
    return vhl_imdc, is_imdc, ws_imdc


def main():
    fc2(
        to_fn=r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240607H191529\FCDL.tif",
        vhl_fn=r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240607H191529\OPTASDE_epoch36_imdc1.tif",
        is_fn=r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240607H194544\OPTASDE_epoch98_imdc1.tif",
        ws_fn=r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240607H200709\OPTASDE_epoch78_imdc1.tif",
    )
    pass


if __name__ == "__main__":
    main()
