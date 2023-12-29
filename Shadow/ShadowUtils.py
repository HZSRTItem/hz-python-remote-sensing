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
from SRTCodes.Utils import listUnique, filterFileContain

SH_C_DICT = {0: 'NOT_KNOW', 'NOT_KNOW': 0, 11: 'IS', 'IS': 11, 12: 'IS_SH', 'IS_SH': 12, 21: 'VEG', 'VEG': 21,
             22: 'VEG_SH', 'VEG_SH': 22, 31: 'SOIL', 'SOIL': 31, 32: 'SOIL_SH',
             'SOIL_SH': 32, 41: 'WAT', 'WAT': 41, 42: 'WAT_SH', 'WAT_SH': 42}


def updateShadowSamplesCategory(chang_csv_fn, o_csv_fn, to_csv_fn=None, srt_field_name="SRT", cname_field_name="CNAME",
                                category_field_name="CATEGORY", change_category_field_name="CATEGORY_CODE",
                                is_change_fields=False):
    if to_csv_fn is None:
        to_csv_fn = o_csv_fn
    df_change = pd.read_csv(chang_csv_fn, index_col=srt_field_name)
    df_o_csv_fn = pd.read_csv(o_csv_fn, index_col=srt_field_name)
    keys = list(df_o_csv_fn)
    if is_change_fields:
        if "IS_CHANGE" not in keys:
            df_o_csv_fn["IS_CHANGE"] = np.zeros(len(df_o_csv_fn))
    df_add = {"X": [], "Y": [], "CNAME": [], "CATEGORY": [], "SRT": []}
    if is_change_fields:
        df_add[ "IS_CHANGE"] = []

    for i, item in df_change.iterrows():
        c_code = int(item[change_category_field_name])
        c_name = SH_C_DICT[c_code]

        if pd.isna(i):
            df_add["X"].append(float(item["X"]))
            df_add["Y"].append(float(item["Y"]))
            df_add["CNAME"].append(c_name)
            df_add["CATEGORY"].append(c_code)
            if is_change_fields:
                df_add["IS_CHANGE"].append(1)
            df_add["SRT"].append(-1)
            continue

        if is_change_fields:
            if int(df_o_csv_fn.loc[i, "IS_CHANGE"]) != 0:
                continue

        df_o_csv_fn.loc[i, category_field_name] = c_code
        df_o_csv_fn.loc[i, cname_field_name] = c_name
        if is_change_fields:
            df_o_csv_fn.loc[i, "IS_CHANGE"] = 1
    df_o_csv_fn = pd.concat([df_o_csv_fn, pd.DataFrame(df_add).set_index("SRT")])
    df_o_csv_fn.to_csv(to_csv_fn)
    return



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

    def fitImdcCSVS(self, mod_dirname, to_csv_fn="fit_imdc_csvs.csv", filter_list=None):
        if filter_list is None:
            filter_list = []
        filter_list.append("_test.csv")
        names1 = ['T_CNAME',
                  # 'O_CNAME',
                  # 'T_F',
                  'T_F_C',
                  ]
        sort_columns = []
        fns = filterFileContain(mod_dirname, filter_list)
        print(fns)
        df_out = None
        n_err = None
        for fn in fns:
            df = pd.read_csv(fn, index_col="SRT")
            if df_out is None:
                df_out = df.copy()
                n_err = np.zeros(len(df_out))
            names2 = ["{0} {1}".format(name, os.path.split(fn)[1]) for name in names1]
            df = df.rename(columns={names1[i]: names2[i] for i in range(len(names1))})
            df_out = pd.merge(df_out, df[names2], left_index=True, right_index=True)
            n_err += df.sort_values(by="SRT")[names2[-1]].values == 1
            print(df_out.keys(), names2[-1])
            df_out = df_out.drop(columns=[names2[-1]])
            sort_columns.append(names2[0])

        print(df_out)
        df_out = df_out.sort_values(by="SRT")
        df_out["O_CNAME_T"] = df_out["O_CNAME"]
        df_out["N_ERROR"] = n_err
        df_out = df_out.sort_values(by=["N_ERROR", "CNAME"] + sort_columns, ascending=False)
        to_csv_fn = os.path.join(mod_dirname, to_csv_fn)
        df_out.to_csv(to_csv_fn)
        return to_csv_fn


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
