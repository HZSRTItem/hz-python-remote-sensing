# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHHFuncs.py
@Time    : 2024/3/4 13:14
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHHFuncs
-----------------------------------------------------------------------------"""
import numpy as np

from SRTCodes.GDALRasterIO import GDALRaster, tiffAddColorTable
from SRTCodes.GDALUtils import GDALRastersSampling


def rastersHist(*raster_fns, bins=256):
    to_dict = {}
    for raster_fn in raster_fns:
        if raster_fn in to_dict:
            continue
        to_dict[raster_fn] = {}
        gr = GDALRaster(raster_fn)
        print(raster_fn)
        for i in range(gr.n_channels):
            print(gr.names[i])
            to_dict[raster_fn][gr.names[i]] = {}
            d = gr.readGDALBand(i + 1)
            data, data_edge = np.histogram(d, bins=bins)
            to_dict[raster_fn]["DATA"] = data.tolist()
            to_dict[raster_fn]["DATA_EDGE"] = data_edge[:-1].tolist()
    return to_dict


def initSHHGRS(grs_type, raster_fns=None):
    if raster_fns is None:
        if grs_type == "qd_sh1":
            rasters_sampling = GDALRastersSampling(
                r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\qd_sh2_1.tif")
        elif grs_type == "qd_esa21":
            rasters_sampling = GDALRastersSampling(
                r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\qd_sh2_esa.tif")
        else:
            raise Exception("Can not find grs type \"{0}\"".format(grs_type))
    else:
        rasters_sampling = GDALRastersSampling(*tuple(raster_fns))
    return rasters_sampling


def main():
    tiffAddColorTable(
        r"F:\ProjectSet\Shadow\Hierarchical\Mods\20240304H201631\model_epoch_86.pth_imdc1.tif",
        code_colors={0:(0, 0, 0, 0), 1: (0, 255, 0), 2: (220, 220, 220), 3: (60, 60, 60)}
    )
    pass


if __name__ == "__main__":
    main()
