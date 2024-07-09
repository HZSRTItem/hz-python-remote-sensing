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
import time

import numpy as np
import torch
from osgeo import gdal
from osgeo_utils.gdal_merge import main as gdal_merge_main

from SRTCodes.GDALRasterIO import tiffAddColorTable, GDALRaster
from SRTCodes.Utils import Jdt, changext, DirFileName, TimeDeltaRecord


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
        row_start, column_start = int(win_size[0] / 2), int(win_size[1] / 2)

        gr = GDALRaster(raster_fn)
        if fit_names is None:
            fit_names = gr.names

        data = torch.zeros(len(fit_names), gr.n_rows, gr.n_columns, dtype=torch.float32)
        self.readRaster(data, fit_names, gr, is_jdt)

        # input("data = data.unfold(1, win_size[0], 1).unfold(2, win_size[1], 1) >")
        data = data.to(device)
        if data_deal is not None:
            data = data_deal(data)
        data = data.unfold(1, win_size[0], 1).unfold(2, win_size[1], 1)
        imdc = torch.zeros(gr.n_rows, gr.n_columns, device=device)
        # input("imdc = torch.zeros(gr.n_rows, gr.n_columns, device=device) >")

        tdr = TimeDeltaRecord(r"F:\ProjectSet\Shadow\Hierarchical\Temp\time.txt")
        jdt = Jdt(data.shape[1], "Raster Torch Predict").start(is_jdt=is_jdt)
        for i in range(data.shape[1]):
            tdr.update(1)
            x_data = data[:, i]

            tdr.update(2)

            x_data = torch.transpose(x_data, 0, 1)

            tdr.update(3)
            y = func_predict(x_data)

            tdr.update(4)
            imdc[row_start + i, column_start: column_start + data.shape[2]] = y

            tdr.update(5)
            jdt.add(is_jdt=is_jdt)
            tdr.add()
        jdt.end(is_jdt=is_jdt)

        imdc = imdc.cpu().numpy()
        gr.save(imdc.astype("int8"), to_geo_fn, fmt="GTiff", dtype=gdal.GDT_Byte, options=["COMPRESS=PACKBITS"])
        if color_table is not None:
            tiffAddColorTable(to_geo_fn, 1, color_table)

        data = None


def main():
    pass


if __name__ == "__main__":
    main()
