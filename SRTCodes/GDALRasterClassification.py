# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : GDALRasterClassification.py
@Time    : 2023/6/23 15:53
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PytorchGeo of GDALRasterClassification
-----------------------------------------------------------------------------"""
import csv
import os.path
import random
import time
import warnings

import numpy as np
import pandas as pd
from osgeo import gdal

from SRTCodes.GDALRasterIO import GDALRaster, saveGTIFFImdc, tiffAddColorTable
from SRTCodes.ModelTraining import ConfusionMatrix, ModelDataCategory, dataModelPredict
from SRTCodes.NumpyUtils import categoryMap
from SRTCodes.RasterClassification import RasterPrediction
from SRTCodes.Utils import readcsv, changext, Jdt, getfilenamewithoutext


class GDALRasterPrediction(GDALRaster, RasterPrediction):
    """ GDALRasterPrediction Usage:

        Extends GDALRasterPrediction. You Will overwrite follow functions:

            Func::predict
                args
                    x:[n, channels, rows, columns])
                return
                    Y:[n]
            Func::preDeal
                args
                    row, column_start, column_end
                return
                    True or False
    """

    def __init__(self, gdal_raster_fn=""):
        GDALRaster.__init__(self, gdal_raster_fn=gdal_raster_fn)
        RasterPrediction.__init__(self)

    def saveToGTIFF(self, imdc_fn, np_type):
        self.save(
            self.imdc.astype(np_type),
            save_geo_raster_fn=imdc_fn,
            fmt="GTIFF",
            dtype=GDALRaster.NpType2GDALType[np_type],
            options=[
                # "COMPRESS=JPEG", "PHOTOMETRIC=YCBCR",
                # "COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=9",
                "COMPRESS=PACKBITS"
            ]
        )

    def run(self, imdc_fn, np_type, mod=None, spl_size=None, row_start=0, row_end=-1, column_start=0, column_end=-1,
            n_one_t=2000, data_deal=None):
        if self.d is None:
            self.readAsArray(interleave="band")
            if data_deal is not None:
                self.d = data_deal(self.d)
        self.addModel(mod)
        self.fit(spl_size=spl_size,
                 row_start=row_start,
                 row_end=row_end,
                 column_start=column_start,
                 column_end=column_end,
                 n_one_t=n_one_t)
        self.saveToGTIFF(imdc_fn, np_type)


class GDALImdcAcc:

    def __init__(self, gr_geo_fn=None):
        self.gr = None
        self.initGR(gr_geo_fn)

        self.x_column_name = "X"
        self.y_column_name = "Y"
        self.c_column_name = "CATEGORY"
        self.is_geo = True

        self.x_list = []
        self.y_list = []
        self.category_list = []
        self.to_category_list = []
        self.map_category = None
        self.to_map_category = None

        self.cm = ConfusionMatrix()

        self.category_func = lambda category1, category2: category1
        self.to_category_func = lambda category1, category2: category2

    def initGR(self, gr_geo_fn):
        if gr_geo_fn is None:
            return
        if isinstance(gr_geo_fn, str):
            self.gr = GDALRaster(gr_geo_fn)
        elif isinstance(gr_geo_fn, GDALRaster):
            self.gr = gr_geo_fn

    def _args_addDataFrame(self, x_column_name=None, y_column_name=None, c_column_name=None, is_geo=None):
        if x_column_name is None:
            x_column_name = self.x_column_name
        else:
            self.x_column_name = x_column_name
        if y_column_name is None:
            y_column_name = self.y_column_name
        else:
            self.y_column_name = y_column_name
        if c_column_name is None:
            c_column_name = self.c_column_name
        else:
            self.c_column_name = c_column_name
        if is_geo is None:
            is_geo = self.is_geo
        else:
            self.is_geo = is_geo
        return x_column_name, y_column_name, c_column_name, is_geo

    def addDataFrame(self, df, x_column_name=None, y_column_name=None, c_column_name=None, is_geo=None):
        x_column_name, y_column_name, c_column_name, is_geo = self._args_addDataFrame(
            x_column_name, y_column_name, c_column_name, is_geo)
        df_dict = {
            x_column_name: df[x_column_name].tolist(),
            y_column_name: df[y_column_name].tolist(),
            c_column_name: df[c_column_name].tolist(),
        }
        for i in range(len(df)):
            x = float(df_dict[x_column_name][i])
            y = float(df_dict[y_column_name][i])
            category = int(df_dict[c_column_name][i])
            self._addXYC(category, x, y)

    def _addXYC(self, category, x, y):
        self.x_list.append(float(x))
        self.y_list.append(float(y))
        self.category_list.append(int(category))

    def sampling(self):
        to_category_list = []
        jdt = Jdt(len(self.category_list), "GDALImdcAcc::sampling").start()
        for i in range(len(self.category_list)):
            x, y = self.x_list[i], self.y_list[i]
            y10 = self.gr.readAsArray(x, y, is_geo=self.is_geo, win_row_size=1, win_column_size=1)
            y10 = y10.ravel()
            to_category_list.append(y10[0])
            jdt.add()
        jdt.end()
        self.to_category_list = to_category_list
        return to_category_list

    def calCM(self, cnames):
        self.cm = ConfusionMatrix(len(cnames), cnames)
        if self.map_category is not None:
            category_list = categoryMap(self.category_list, self.map_category)
        else:
            category_list = self.category_list
        self.sampling()
        if self.to_map_category is not None:
            to_category_list = categoryMap(self.to_category_list, self.to_map_category)
        else:
            to_category_list = self.to_category_list
        category_list = self.category_func(category_list, to_category_list)
        to_category_list = self.to_category_func(category_list, to_category_list)
        self.cm.addData(category_list, to_category_list)
        return self.cm


class GDALRasterClassificationAccuracy:
    GDAL_RASTER = {}  # have open gdal raster object

    def __init__(self):
        self.f_cm = None
        self.cw = None
        self.f_csv = None
        self.x = []
        self.y = []
        self.category = []
        self.category_code = {}
        self.is_geo = True
        self.gr = GDALRaster()
        self.cm = None
        self.save_line = {}
        self.spl_fn = ""

        self.newCSVLine()

    def clear(self):
        self.x.clear()
        self.y.clear()
        self.category.clear()
        self.category_code.clear()
        self.is_geo = True

    def addSampleCSV(self, csv_fn, x_column_name="X", y_column_name="Y", c_column_name="CATEGORY", is_geo=True):
        self.spl_fn = csv_fn
        self.is_geo = is_geo
        d = readcsv(csv_fn)
        for i in range(len(d[x_column_name])):
            x = float(d[x_column_name][i])
            y = float(d[y_column_name][i])
            category = str(d[c_column_name][i])
            self._addXYC(category, x, y)

    def _addXYC(self, category, x, y):
        self.x.append(x)
        self.y.append(y)
        if category not in self.category_code:
            raise Exception(
                "Can not find category:\"{0}\" in this category_code:{1}.".format(category, self.category_code))
        self.category.append(self.category_code[category])

    def addSampleExcel(self, excel_fn, x_column_name="X", y_column_name="Y", c_column_name="CATEGORY",
                       sheet_name=0, is_geo=True):
        self.spl_fn = excel_fn
        self.is_geo = is_geo
        df = pd.read_excel(excel_fn, sheet_name=sheet_name)
        self.addDataFrame(df, x_column_name, y_column_name, c_column_name)

    def addDataFrame(self, df, x_column_name="X", y_column_name="Y", c_column_name="CATEGORY", is_geo=True):
        self.is_geo = is_geo
        for i in range(len(df)):
            x = float(df[x_column_name][i])
            y = float(df[y_column_name][i])
            category = str(df[c_column_name][i])
            self._addXYC(category, x, y)

    def addCategoryCode(self, **category_code):
        for k in category_code:
            if k in self.category_code:
                warnings.warn(Warning("category:\"{0}\" have in category_code, will update code "
                                      "from {1} to {2}".format(k, self.category_code[k], category_code[k])))
            self.category_code[k] = category_code[k]
            self.category_code[category_code[k]] = k
        self.newCSVLine()

    def fit(self, raster_fn):
        self.gr = self.getGDALRaster(raster_fn)
        cnames = self.getCNames()
        self.cm = ConfusionMatrix(len(cnames), cnames)
        self.calCM()

    def calCM(self):
        y1 = []
        for i in range(len(self.category)):
            y10 = self.gr.readAsArray(self.x[i], self.y[i], is_geo=self.is_geo, win_row_size=1, win_column_size=1)
            y10 = y10.ravel()
            y1.append(y10[0])
        self.cm.addData(self.category, y1)

    def getCNames(self):
        cnames = []
        for k in self.category_code:
            if isinstance(k, str):
                cnames.append(k)
        return cnames

    @classmethod
    def getGDALRaster(cls, raster_fn) -> GDALRaster:
        raster_fn = os.path.abspath(raster_fn)
        if raster_fn not in cls.GDAL_RASTER:
            cls.GDAL_RASTER[raster_fn] = GDALRaster(raster_fn)
        return cls.GDAL_RASTER[raster_fn]

    def fitModelDirectory(self, mod_dirname):
        cnames = self.getCNames()
        dir_name = os.path.split(mod_dirname)[1]
        fn = os.path.join(mod_dirname, "train_save_" + dir_name + ".csv")
        if not os.path.isfile(fn):
            raise Exception("Can not find {0}".format(fn))
        self.cm = ConfusionMatrix(len(cnames), cnames)
        self.newCSVLine()
        df = pd.read_csv(fn)
        for mod_name in df["ModelName"].values:
            mod_name = mod_name.strip()
            raster_fn = os.path.join(mod_dirname, mod_name + "_imdc.dat")
            if os.path.isfile(raster_fn):
                self.gr = self.getGDALRaster(raster_fn)
                self.calCM()
                self.updateSaveCSVLine(raster_fn)
                self.saveCSVLine()
                self.saveCM(raster_fn)
                self.printSaveCSVLine()
                self.newCSVLine()
                self.cm.clear()
            else:
                print("Warning: can not find raster {0}".format(raster_fn))

    def newCSVLine(self):
        self.save_line["NAME"] = ""
        cnames = self.getCNames()
        self.save_line["OA"] = 0
        for name in cnames:
            self.save_line["{0} UA".format(name)] = 0
            self.save_line["{0} PA".format(name)] = 0
        self.save_line["RASTER_FN"] = ""
        self.save_line["SPL_FN"] = self.spl_fn

    def updateSaveCSVLine(self, raster_fn):
        ff = os.path.split(raster_fn)[1]
        self.save_line["NAME"] = os.path.splitext(ff)[0]
        cnames = self.getCNames()
        self.save_line["OA"] = self.cm.OA()
        for name in cnames:
            self.save_line["{0} UA".format(name)] = self.cm.UA(name)
            self.save_line["{0} PA".format(name)] = self.cm.PA(name)
        self.save_line["RASTER_FN"] = raster_fn
        self.save_line["SPL_FN"] = self.spl_fn

    def openSaveCSVFileName(self, csv_fn, open_mode="w"):
        self.closeSaveCSVFileName()
        self.f_csv = open(csv_fn, open_mode, encoding="utf-8", newline="")
        self.cw = csv.writer(self.f_csv)
        self.cw.writerow(list(self.save_line.keys()))

    def saveCSVLine(self):
        self.cw.writerow(list(self.save_line.values()))

    def closeSaveCSVFileName(self):
        if self.f_csv is not None:
            self.f_csv.close()
            self.f_csv = None
        if self.cw is not None:
            self.cw = None

    def openSaveCMFileName(self, cm_fn, open_mode="w"):
        self.closeSaveCMFileName()
        self.f_cm = open(cm_fn, open_mode, encoding="utf-8")

    def closeSaveCMFileName(self):
        if self.f_cm is not None:
            self.f_cm.close()
            self.f_cm = None

    def saveCM(self, raster_fn):
        self.f_cm.write("> {0}\n".format(self.save_line["NAME"], raster_fn))
        self.f_cm.write(self.cm.fmtCM())
        self.f_cm.write("\n")
        pass

    def printSaveCSVLine(self):
        print("{0} OA:{1:.3f}".format(self.save_line["NAME"], self.save_line["OA"]))


class GDALModelDataCategory(ModelDataCategory):

    def __init__(self, *raster_fns):
        super(GDALModelDataCategory, self).__init__()
        self.gr = GDALRaster()
        self.addRasters(*raster_fns)

    def addRasters(self, *raster_fns):
        for raster_fn in raster_fns:
            raster_fn = os.path.abspath(raster_fn)
            gr = GDALRaster(raster_fn)
            if self.gr.raster_ds is None:
                self.gr = gr
            self.addData(gr.readAsArray())

    def fit(self, to_imdc_fns, model, data_deal=None, is_jdt=False, color_table=None, description="Category",
            *args, **kwargs):
        if color_table is None:
            color_table = {}
        if isinstance(to_imdc_fns, str):
            if not to_imdc_fns.endswith(".tif"):
                to_imdc_fns = to_imdc_fns + ".tif"
            to_imdc_fns = [changext(to_imdc_fns, "_{0}.tif".format(i + 1)) for i in range(len(self.data_list))]
        for i, data in enumerate(self.data_list):
            imdc = dataModelPredict(data, data_deal, is_jdt, model)
            saveGTIFFImdc(self.gr, imdc, to_imdc_fns[i], color_table=color_table, description=description)
        return to_imdc_fns


def tilesRasterImdc(
        raster_fn, to_imdc_fn, predict_func, read_size=(-1, -1), interval_size=(0, 0),
        channels=None, tiles_dirname=None, dtype="float32", color_table=None, is_jdt=True, to_dtype="int8"
):
    read_size = list(read_size)
    interval_size = list(interval_size)

    gr = GDALRaster(raster_fn)
    imdc = np.zeros((gr.n_rows, gr.n_columns), dtype=to_dtype)

    init_geo_transform = gr.geo_transform

    if channels is None:
        channels = gr.names
    if read_size[0] <= 0:
        read_size[0] = gr.n_rows
        interval_size[0] = 0
    if read_size[1] <= 0:
        read_size[1] = gr.n_columns
        interval_size[1] = 0

    n_rows, n_columns = read_size
    interval_row_size, interval_column_size = interval_size
    step_row, step_column = n_rows + interval_row_size, n_columns + interval_column_size

    def get_read_n(read_n, n, n_max):
        if n + read_n > n_max:
            read_n = n_max - n
        return read_n

    def get_to_n(interval_n_size, n, read_n):
        if interval_n_size >= 0:
            _start1, _end1 = n, n + read_n
            _start2, _end2 = 0, read_n
        else:
            _start1, _end1 = n - int(interval_n_size / 2), n + read_n + int(interval_n_size / 2)
            _start2, _end2 = 0 - int(interval_n_size / 2), 0 + read_n + int(interval_n_size / 2)

        return _start1, _end1, _start2, _end2

    def save(_data, _fn, _geo_transform=None):
        if os.path.isfile(_fn):
            os.remove(_fn)
        to_gdal_dtype = GDALRaster.NpType2GDALType[to_dtype]
        gr.save(
            d=_data, save_geo_raster_fn=_fn, fmt="GTiff", dtype=to_gdal_dtype,
            start_xy=None, descriptions=["Category"], geo_transform=_geo_transform,
            options=["COMPRESS=LZW"]
        )
        if color_table is not None:
            tiffAddColorTable(_fn, code_colors=color_table)

    def timestr(_time):

        tian = int(_time // (60 * 60 * 24))
        _time = _time % (60 * 60 * 24)
        shi = int(_time // (60 * 60))
        _time = _time % (60 * 60)
        fen = int(_time // 60)
        miao = int(_time % 60)

        if tian >= 1:
            return "{}D {:02d}:{:02d}:{:02d}".format(tian, shi, fen, miao)
        if shi >= 1:
            return "{:02d}:{:02d}:{:02d}".format(shi, fen, miao)
        return "{:02d}:{:02d}".format(fen, miao)

    n_run = 1
    for i, row in enumerate(range(0, gr.n_rows, step_row)):
        for j, column in enumerate(range(0, gr.n_columns, step_column)):
            n_run += 1

    class _Jdt:

        def __init__(self):
            self.n = n_run

            self.time = 0.0
            self.n_current = 0
            self.time_mean = 0

        def start(self):
            if not is_jdt:
                return
            self.time = time.time()
            self.n_current = 0
            self.time_mean = 0
            return self

        def add(self):
            if not is_jdt:
                return
            time_current = time.time()
            time_sum = self.n_current * self.time_mean + time_current - self.time
            self.n_current += 1
            self.time_mean = time_sum / self.n_current

            print("{:>3d}% {:>2d}/{} [{}<{} {}/it]".format(
                int(self.n_current / self.n * 100), self.n_current, self.n,
                timestr(time_sum), timestr(self.time_mean * self.n), timestr(self.time_mean)))
            self.time = time_current

    jdt = _Jdt().start()
    n_run = 1

    for i, row in enumerate(range(0, gr.n_rows, step_row)):
        for j, column in enumerate(range(0, gr.n_columns, step_column)):

            if is_jdt:
                print("{:>2}. ".format(n_run), end="")
            n_run += 1

            read_n_rows = get_read_n(n_rows, row, gr.n_rows)
            read_n_columns = get_read_n(n_columns, column, gr.n_columns)

            data = np.zeros((len(channels), read_n_rows, read_n_columns), dtype=dtype)

            for i_channel, channel in enumerate(channels):
                data[i_channel] = gr.getGDALBand(channel).ReadAsArray(column, row, read_n_columns, read_n_rows)

            imdc_tmp = predict_func(data).astype(to_dtype)

            r_start1, r_end1, r_start2, r_end2 = get_to_n(interval_row_size, row, read_n_rows)
            c_start1, c_end1, c_start2, c_end2 = get_to_n(interval_column_size, column, read_n_columns)

            imdc[r_start1:r_end1, c_start1:c_end1] = imdc_tmp[r_start2:r_end2, c_start2:c_end2]

            if tiles_dirname is not None:
                if tiles_dirname == "":
                    tiles_dirname = changext(to_imdc_fn, "_imdctiles")
                if not os.path.isdir(tiles_dirname):
                    os.mkdir(tiles_dirname)
                to_name = getfilenamewithoutext(to_imdc_fn)
                to_fn = os.path.join(tiles_dirname, "{}_{}_{}.tif".format(to_name, i + 1, j + 1))

                x0, y0 = gr.coorRaster2Geo(row, column)
                x1, y1 = gr.coorRaster2Geo(row + read_n_rows, column + read_n_columns)
                if x0 > x1:
                    x0, x1 = x1, x0
                if y0 > y1:
                    y0, y1 = y1, y0
                geo_transform = (x0, gr.geo_transform[1], 0, y1, 0, gr.geo_transform[5])

                save(imdc_tmp, to_fn, geo_transform)

            jdt.add()

    save(imdc, to_imdc_fn, init_geo_transform)

    return imdc


def main():
    def predict_func(data):
        ndvi = (data[1] - data[0]) / (data[1] + data[0])
        time.sleep(random.random())
        return (ndvi < 0) + 1

    def tilesRasterImdc_predict_func(data):
            data = torch.from_numpy(data)
            return torchDataPredict(data, win_size, func_predict, data_deal, device, is_jdt)

    tilesRasterImdc(
        raster_fn=r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\SHH2_QD2_envi.dat",
        to_imdc_fn=r"F:\Week\20240721\Data\imdc.tif", predict_func=predict_func,
        read_size=(1600, 1600), interval_size=(-60, -60),
        channels=["Red", "NIR"], tiles_dirname="", dtype="float32",
        color_table={1: (255, 255, 255), 2: (0, 255, 0)},
    )


if __name__ == "__main__":
    main()
