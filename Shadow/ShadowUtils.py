# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowUtils.py
@Time    : 2023/12/21 20:45
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of ShadowUtils
-----------------------------------------------------------------------------"""
import os.path

import numpy as np
import pandas as pd

from SRTCodes.GDALRasterClassification import GDALRasterClassificationAccuracy
from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.Utils import listUnique


class _DFSamples:

    def __init__(self):
        self.df = pd.DataFrame()

    def initCSV(self, csv_fn):
        if csv_fn is not None:
            self.df = pd.read_csv(csv_fn)

    def initDataFrame(self, df: pd.DataFrame):
        self.df = df.copy()

    def initExcel(self, excel_fn, sheet_name=0):
        if excel_fn is not None:
            self.df = pd.read_excel(excel_fn, sheet_name=sheet_name)


class _GDALRasters:

    def __init__(self):
        self.grs = {}
        self.datas = {}

    def add(self, raster_fn) -> GDALRaster:
        raster_fn = os.path.abspath(raster_fn)
        if raster_fn not in self.grs:
            self.grs[raster_fn] = GDALRaster(raster_fn)
            self.datas[raster_fn] = None
        return self.grs[raster_fn]

    def get(self, raster_fn=None, n=0) -> GDALRaster:
        if raster_fn is None:
            raster_fn = list(self.grs.keys())[n]
        raster_fn = os.path.abspath(raster_fn)
        return self.grs[raster_fn]

    def __getitem__(self, item) -> GDALRaster:
        return self.add(item)

    def data(self, raster_fn=None, n=0) -> np.ndarray:
        if raster_fn is None:
            raster_fn = list(self.grs.keys())[n]
        raster_fn = os.path.abspath(raster_fn)
        if self.datas[raster_fn] is None:
            self.datas[raster_fn] = self.get(raster_fn).readAsArray()
        return self.datas[raster_fn]


class ShadowSampleAdjustNumber(_DFSamples):

    def __init__(self, csv_fn=None, c_k_name="CNAME"):
        super().__init__()

        self.df = None
        self.c_k_name = c_k_name
        self.datas = {}
        self.initCSV(csv_fn)

    def initCKName(self, c_k_name=None):
        if c_k_name is not None:
            self.c_k_name = c_k_name
        d = np.unique(self.df[self.c_k_name].values)
        for k in d:
            self.datas[k] = self.df.loc[self.df[self.c_k_name] == k]

    def numbers(self):
        return {k: len(self.datas[k]) for k in self.datas}

    def adjustNumber(self, n_dict: dict, **kwargs):
        for k in n_dict:
            if len(self.datas[k]) != n_dict[k]:
                self.adjust(k, n_dict[k])

    def adjust(self, k, n):
        d: pd.DataFrame = self.datas[k]
        self.datas[k] = d.sample(n=n)

    def printNumber(self):
        n_dict = self.numbers()
        n = max([len(k) for k in n_dict])
        fmt = "{0:" + str(n) + "}"
        for k in n_dict:
            print(fmt.format(k), ":", n_dict[k])

    def saveToCSV(self, csv_fn, *dfs):
        dfs = list(dfs)
        dfs += [self.datas[k] for k in self.datas]
        df = pd.concat(dfs)
        df.to_csv(csv_fn, index=False)


class ShadowFindErrorSamples(_DFSamples, GDALRasterClassificationAccuracy):

    def __init__(self):
        _DFSamples.__init__(self)
        GDALRasterClassificationAccuracy.__init__(self)
        self.grs = _GDALRasters()
        self.imdc_fn = None
        self.x_column_name = ""
        self.y_column_name = ""
        self.c_column_name = ""
        self.t_cname_name = "T_CNAME"
        self.o_cname_name = "O_CNAME"
        self.t_f_name = "T_F"
        self.t_f_cname = "T_F_C"

    def addDataFrame(self, df=None, x_column_name="X", y_column_name="Y", c_column_name="CNAME", is_geo=True):
        if df is None:
            df = self.df
        self.x_column_name, self.y_column_name, self.c_column_name = x_column_name, y_column_name, c_column_name
        self.is_geo = is_geo
        for line in df.iterrows():
            line = line[1]
            x = float(line[x_column_name])
            y = float(line[y_column_name])
            category = str(line[c_column_name])
            self._addXYC(category, x, y)

    def imdcFN(self, imdc_fn=None):
        if imdc_fn is not None:
            self.imdc_fn = os.path.abspath(imdc_fn)
        return self.imdc_fn

    def calCMImdc(self, imdc_fn=None):
        imdc_fn = self.imdcFN(imdc_fn)
        c_list = self.sampleImdc(imdc_fn)
        cnames = self.getCNames()
        cm = ConfusionMatrix(len(cnames), cnames)
        cm.addData(self.category, c_list)
        return cm

    def sampleImdc(self, imdc_fn=None):
        imdc_fn = self.imdcFN(imdc_fn)
        d = self.grs[imdc_fn].readGDALBand(1)
        c_list = []
        for i in range(len(self.x)):
            r, c = self.grs[imdc_fn].coorGeo2Raster(self.x[i], self.y[i], is_int=True)
            cate = d[r, c]
            # if cate in self.c_convert:
            #     cate = self.c_convert[cate]
            c_list.append(cate)
        return c_list

    def fitImdc(self, imdc_fn=None):
        imdc_fn = self.imdcFN(imdc_fn)
        cname_true_list = [self.category_code[k] for k in self.category]
        self.df[self.o_cname_name] = cname_true_list
        c_list = self.sampleImdc(imdc_fn)
        # for i in range(len(c_list)):
        #     if self.category[i] in self.c_convert:
        #         c_list[i] = self.c_convert[c_list[i]]
        cname_pred_list = [self.category_code[k] for k in c_list]
        self.df[self.t_cname_name] = cname_pred_list
        t_f_list = []
        c_qgis_list = []
        for i in range(len(cname_true_list)):
            if cname_pred_list[i] in cname_true_list[i]:
                t_f_list.append("TRUE_C")
                c_qgis_list.append(2)
            else:
                t_f_list.append("FALSE_C")
                c_qgis_list.append(1)
        self.df[self.t_f_name] = t_f_list
        self.df[self.t_f_cname] = c_qgis_list
        return self.df

    def toCSV(self, to_fn=None, keys=None, sort_column=None):
        if keys is None:
            keys = list(self.df.keys())
        init_keys = [self.x_column_name, self.y_column_name, self.c_column_name]
        if self.t_cname_name in self.df:
            init_keys.append(self.t_cname_name)
        if self.o_cname_name in self.df:
            init_keys.append(self.o_cname_name)
        if self.t_f_name in self.df:
            init_keys.append(self.t_f_name)
        if self.t_f_cname in self.df:
            init_keys.append(self.t_f_cname)
        keys = listUnique(init_keys + keys)
        if to_fn is None:
            to_fn = self.imdc_fn + "_test.csv"
        df = self.df[keys]
        if sort_column is not None:
            df = df.sort_values(by=sort_column)
        df.to_csv(to_fn, index=False)


class ShadowTestAll(ShadowFindErrorSamples):

    def __init__(self):
        ShadowFindErrorSamples.__init__(self)

    def calCMImdc(self, imdc_fn=None):
        imdc_fn = self.imdcFN(imdc_fn)
        c_list = self.sampleImdc(imdc_fn)
        cnames = self.getCNames()
        category = np.array(self.category.copy())
        category[category >= 5] = category[category >= 5] - 4
        cm = ConfusionMatrix(len(cnames), cnames)
        cm.addData(category, c_list)
        return cm

    def fitDirName(self, dirname):
        fn_list = []
        fn_len = []
        for fn in os.listdir(dirname):
            if fn.endswith(".dat"):
                fn_list.append((fn, os.path.join(dirname, fn)))
                fn_len.append(len(fn))
        fmt = "{:<" + str(max(fn_len)) + "}"
        d_dict = {"NAME": [], "OA": [], "KAPPA": []}
        for fn, fn2 in fn_list:
            if "SVM" in fn:
            # if ".dat" in fn:
                self.imdcFN(fn)
                cm = self.calCMImdc(fn2)
                # print(cm.fmtCM())
                d_dict["NAME"].append(fn)
                d_dict["OA"].append(cm.OA())
                d_dict["KAPPA"].append(cm.getKappa())
        df = pd.DataFrame(d_dict)
        df = df.sort_values(by="OA", ascending=False)
        print(df)



def main():

    pass


if __name__ == "__main__":
    main()
