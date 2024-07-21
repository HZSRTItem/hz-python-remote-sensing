# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : GDALTorch.py
@Time    : 2024/7/6 19:30
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of GDALTorch
-----------------------------------------------------------------------------"""
import os

import numpy as np
import torch
from osgeo import gdal
from osgeo_utils.gdal_merge import main as gdal_merge_main

from SRTCodes.GDALRasterClassification import tilesRasterImdc
from SRTCodes.GDALRasterIO import tiffAddColorTable, GDALRaster
from SRTCodes.PytorchModelTraining import torchDataPredict
from SRTCodes.Utils import Jdt, changext, DirFileName


def torchImdc3(
        raster_fn, func_predict, win_size, to_geo_fn, read_size=(5, -1),
        fit_names=None, data_deal=None, is_jdt=True, color_table=None, device="cuda",
        is_save_tiles=False,
):
    def tilesRasterImdc_predict_func(data):
        return torchDataPredict(data, win_size, func_predict, data_deal, device, is_jdt)

    tiles_dirname = None
    if is_save_tiles:
        tiles_dirname = ""

    tilesRasterImdc(
        raster_fn, to_geo_fn, tilesRasterImdc_predict_func, read_size=(-1, -1), interval_size=read_size,
        channels=fit_names, tiles_dirname=tiles_dirname, dtype="float32", color_table=color_table, is_jdt=is_jdt
    )


class GDALTorchImdc:

    def __init__(self, raster_fns):
        if isinstance(raster_fns, str):
            raster_fns = [raster_fns]
        self.raster_fns = raster_fns

    def readRaster(self, data, fit_names, gr, is_jdt):
        jdt = Jdt(len(fit_names), "Read Raster").start(is_jdt)
        for i, name in enumerate(fit_names):
            data_i = gr.readGDALBand(name)
            data_i[np.isnan(data_i)] = 0
            data[i] = torch.from_numpy(data_i)
            jdt.add(is_jdt)
        jdt.end(is_jdt)

    def imdc2(self, func_predict, win_size, to_imdc_fn,
              fit_names=None, data_deal=None, color_table=None,
              n=-1, is_jdt=True, device="cuda", fun_print=print):

        if len(self.raster_fns) == 0:
            raise Exception("Can not find raster")

        if len(self.raster_fns) == 1:
            raster_fn = self.raster_fns[0]
            if fun_print is not None:
                fun_print("Raster:", raster_fn, end="\n")
                fun_print("Imdc:", to_imdc_fn, end="\n")
            self._imdc2(
                func_predict, raster_fn, win_size, to_imdc_fn,
                fit_names, data_deal, is_jdt, color_table, device, n=n
            )
        else:
            to_imdc_dirname = changext(to_imdc_fn, "_tiles")
            to_imdc_dfn = DirFileName(to_imdc_dirname)
            to_imdc_dfn.mkdir()

            to_fn_tmps = []
            for raster_fn in self.raster_fns:
                to_imdc_fn_tmp = to_imdc_dfn.fn(changext(os.path.split(raster_fn)[1], "_imdc.tif"))
                to_fn_tmps.append(to_imdc_fn_tmp)
                fun_print("Raster:", raster_fn, end="\n")
                fun_print("Imdc:", to_imdc_fn_tmp, end="\n")
                self._imdc2(
                    func_predict, raster_fn, win_size, to_imdc_fn_tmp,
                    fit_names, data_deal, is_jdt, color_table, device, n=n
                )

            fun_print("Imdc:", to_imdc_fn, end="\n")
            gdal_merge_main([
                "gdal_merge_main", "-of", "GTiff", "-n", "0", "-ot", "Byte",
                "-co", "COMPRESS=PACKBITS", "-o", to_imdc_fn, *to_fn_tmps,
            ])

            if color_table is not None:
                tiffAddColorTable(to_imdc_fn, 1, code_colors=color_table)

    def _imdc2(self, func_predict, raster_fn, win_size, to_geo_fn, fit_names, data_deal,
               is_jdt, color_table, device, n=1000):

        gr = GDALRaster(raster_fn)
        n_rows, n_columns = gr.n_rows, gr.n_columns

        if fit_names is None:
            fit_names = gr.names

        data = torch.zeros(len(fit_names), n_rows, n_columns, dtype=torch.float32)
        self.readRaster(data, fit_names, gr, is_jdt)

        imdc = torchDataPredict(data, win_size, func_predict, data_deal, device, is_jdt)

        gr.save(imdc.astype("int8"), to_geo_fn, fmt="GTiff", dtype=gdal.GDT_Byte, options=["COMPRESS=PACKBITS"])
        if color_table is not None:
            tiffAddColorTable(to_geo_fn, 1, color_table)

        data = None

    def imdc3(self, func_predict, win_size, to_imdc_fn,
              fit_names=None, data_deal=None, color_table=None,
              is_jdt=True, device="cuda", fun_print=print,
              read_size=(1000, -1), is_save_tiles=False, ):

        def tilesRasterImdc_predict_func(data):
            data = torch.from_numpy(data)
            return torchDataPredict(data, win_size, func_predict, data_deal, device, is_jdt)

        tiles_dirname = None
        if is_save_tiles:
            tiles_dirname = ""

        raster_fn = self.raster_fns[0]

        fun_print("Raster:", raster_fn, end="\n")
        fun_print("Imdc:", to_imdc_fn, end="\n")

        interval_size = -(win_size[0] + 2), -(win_size[1] + 2)

        tilesRasterImdc(
            raster_fn, to_imdc_fn, tilesRasterImdc_predict_func, read_size=read_size, interval_size=interval_size,
            channels=fit_names, tiles_dirname=tiles_dirname, dtype="float32", color_table=color_table, is_jdt=is_jdt
        )


def main():
    pass


if __name__ == "__main__":
    main()
