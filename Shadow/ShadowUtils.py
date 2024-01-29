# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowUtils.py
@Time    : 2023/12/21 20:45
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of ShadowUtils
-----------------------------------------------------------------------------"""
import csv
import os.path
import shutil

import numpy as np
import pandas as pd

from RUN.RUNFucs import QJYTxt_main
from SRTCodes.GDALRasterClassification import GDALRasterClassificationAccuracy
from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.PandasUtils import filterDF
from SRTCodes.SRTReadWrite import SRTInfoFileRW
from SRTCodes.Utils import listUnique, filterFileContain, Jdt, saveJson, readJson, timeDirName, changefiledirname, \
    changext

SH_C_DICT = {0: 'NOT_KNOW', 'NOT_KNOW': 0, 11: 'IS', 'IS': 11, 12: 'IS_SH', 'IS_SH': 12, 21: 'VEG', 'VEG': 21,
             22: 'VEG_SH', 'VEG_SH': 22, 31: 'SOIL', 'SOIL': 31, 32: 'SOIL_SH',
             'SOIL_SH': 32, 41: 'WAT', 'WAT': 41, 42: 'WAT_SH', 'WAT_SH': 42}
pd.set_option('expand_frame_repr', False)



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
        df_add["IS_CHANGE"] = []

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


def _isTo01(_is):
    if _is:
        return 1
    else:
        return 0


def _listToInt(_list: list):
    return list(map(int, _list))


class ShadowTiaoTestAcc:

    def __init__(self, init_dirname=None, init_name=None, ):
        self.init_dirname = init_dirname
        self.init_name = init_name
        if self.init_dirname is not None:
            if not os.path.isdir(self.init_dirname):
                os.mkdir(self.init_dirname)
            if init_name is None:
                self.init_name = os.path.split(self.init_dirname)[1]

        self.mod_dirname = None
        self.imdc_fns = {}
        self.imdc_grs = {}
        self.category_map = {}
        self.c_field_name = "CNAME"
        self.cm_names = []

        self.spl_df = pd.DataFrame()

    def buildNew(self, mod_dirname, csv_fn=None, *filters, x_field_name="X", y_field_name="Y", is_jdt=True):
        self.mod_dirname = mod_dirname
        self._getImdcFNS()
        self.addCSV(csv_fn=csv_fn, *filters)
        self.samplingImdc(x_field_name=x_field_name, y_field_name=y_field_name, is_jdt=is_jdt)
        self.saveDFToCSV()
        self.save()

    def saveDFToCSV(self):
        to_csv_fn = os.path.join(self.init_dirname, self.init_name + "_data.csv")
        self.spl_df.to_csv(to_csv_fn, index=False)

    def categoryMap(self, _map_dict=None, *args, **kwargs):
        if _map_dict is None:
            _map_dict = {}
        for k, v in args:
            _map_dict[k] = v
        for k, v in kwargs.items():
            _map_dict[k] = v
        self.category_map = _map_dict.copy()

    def save(self, save_json_fn=None):
        if save_json_fn is None:
            save_json_fn = os.path.join(self.init_dirname, self.init_name + "_STTA.json")
        save_dict = {
            "NAME": self.init_name,
            "MOD_DIRNAME": self.mod_dirname,
            "IMDC_FNS": self.imdc_fns,
            "CATEGORY_MAP": self.category_map,
            "CATEGORY_FIELD_NAME": self.c_field_name,
            "CM_NAMES": self.cm_names,
        }
        saveJson(save_dict, save_json_fn)

    def load(self, load_json_fn=None):
        if load_json_fn is None:
            load_json_fn = os.path.join(self.init_dirname, self.init_name + "_STTA.json")
        load_dict = readJson(load_json_fn)
        self.init_name = load_dict["NAME"]
        self.mod_dirname = load_dict["MOD_DIRNAME"]
        self.imdc_fns = load_dict["IMDC_FNS"]
        self.category_map = load_dict["CATEGORY_MAP"]
        self.c_field_name = load_dict["CATEGORY_FIELD_NAME"]
        self.cm_names = load_dict["CM_NAMES"]

        self.imdc_grs = {k: GDALRaster(fn) for k, fn in self.imdc_fns.items()}
        self.readCSVSplDF()

    def readCSVSplDF(self):
        to_csv_fn = os.path.join(self.init_dirname, self.init_name + "_data.csv")
        self.spl_df = pd.read_csv(to_csv_fn)

    def _getImdcFNS(self):
        for fn in os.listdir(self.mod_dirname):
            if fn.endswith("_imdc.dat"):
                fn2 = os.path.join(self.mod_dirname, fn)
                name = fn[:fn.find("_imdc.dat")]
                self.imdc_fns[name] = fn2
                self.imdc_grs[name] = GDALRaster(fn2)

    def addCSV(self, csv_fn=None, *filters, **kwargs):
        if csv_fn is None:
            csv_fn = os.path.join(self.mod_dirname, "train_data.csv")
        df = pd.read_csv(csv_fn)
        df = filterDF(df, *filters, **kwargs)
        self.spl_df = df
        return df

    def samplingImdc(self, x_field_name="X", y_field_name="Y", is_jdt=True):
        x_list, y_list = self.spl_df[x_field_name].tolist(), self.spl_df[y_field_name].tolist()
        imdc_dict = {k: [] for k in self.imdc_fns}
        jdt = Jdt(len(self.spl_df), "ShadowTiaoTestAcc samplingImdc")
        if is_jdt:
            jdt.start()
        for i in range(len(self.spl_df)):
            x, y = x_list[i], y_list[i]
            for k, gr in self.imdc_grs.items():
                gr: GDALRaster
                category = gr.readAsArray(x, y, 1, 1, is_geo=True)
                if category is not None:
                    category = category.ravel()
                    imdc_dict[k].append(int(category[0]))
                else:
                    imdc_dict[k].append(0)
            if is_jdt:
                jdt.add()
        if is_jdt:
            jdt.end()
        for k in imdc_dict:
            self.spl_df[k] = imdc_dict[k]
        imdc_dict[x_field_name] = x_list
        imdc_dict[y_field_name] = y_list
        return pd.DataFrame(imdc_dict)

    def getCategoryList(self):
        cname_list = self.spl_df[self.c_field_name].to_list()
        c_list = [self.category_map[cname] for cname in cname_list]
        return c_list

    def updateTrueFalseColumn(self, is_to_csv=False):
        to_dict = {}
        init_c_list = np.array(self.getCategoryList())
        for k in self.imdc_fns:
            to_k = "TF_{0}".format(k)
            c_list = self.spl_df[k].values
            to_dict[to_k] = (init_c_list == c_list) * 1
        for k in to_dict:
            self.spl_df[k] = to_dict[k]
        to_dict[self.c_field_name] = init_c_list
        if is_to_csv:
            self.saveDFToCSV()
        return pd.DataFrame(to_dict)

    def sortColumn(self, column_names, ascending=True, is_to_csv=False):
        self.spl_df = self.spl_df.sort_values(column_names, ascending=ascending)
        if is_to_csv:
            self.saveDFToCSV()

    def sumColumns(self, name, *filters, column_names=None, is_to_csv=False):
        if column_names is None:
            column_names = []
        column_names = list(column_names)
        for k in self.spl_df:
            if k not in column_names:
                if all(f in k for f in filters):
                    column_names.append(k)
        print("FUNC:sumColumns --COLUMN_NAMES=", column_names)
        self.spl_df[name] = self.spl_df[column_names].sum(axis=1)
        if is_to_csv:
            self.saveDFToCSV()
        return self.spl_df[name]

    def toCSV(self, to_csv_fn, field_names=None, *filters, **filter_maps):
        if field_names is None:
            field_names = list(self.spl_df.keys())
        to_df = filterDF(self.spl_df, *filters, **filter_maps)
        to_df[field_names].to_csv(to_csv_fn, index=False)

    def addQJY(self, name, field_names=None, *filters, **filter_maps):
        to_dirname = os.path.join(self.init_dirname, "QJY_{0}".format(name))
        if not os.path.isdir(to_dirname):
            os.mkdir(to_dirname)
        to_csv_fn = os.path.join(to_dirname, "{0}_qjy_{1}.csv".format(self.init_name, name))
        to_txt_fn = os.path.join(to_dirname, "{0}_qjy_{1}.txt".format(self.init_name, name))
        field_names += list(self.imdc_fns.keys())
        self.toCSV(to_csv_fn, field_names=field_names, *filters, **filter_maps)
        qjy_main = QJYTxt_main()
        qjy_main.run(["", to_csv_fn, "-o", to_txt_fn])

    def accQJY(self, name, is_save=False):
        to_dirname = os.path.join(self.init_dirname, "QJY_{0}".format(name))
        csv_fn = os.path.join(to_dirname, "{0}_qjy_{1}.csv".format(self.init_name, name))
        txt_fn = os.path.join(to_dirname, "{0}_qjy_{1}.txt".format(self.init_name, name))
        srt_fr = SRTInfoFileRW(txt_fn)
        d = srt_fr.readAsDict()
        df_dict = {"__X": [], "__Y": [], "__CNAME": [], "__IS_TAG": []}
        for k in d["FIELDS"]:
            df_dict[k] = []

        fields = d["FIELDS"].copy()
        cr = csv.reader(d["DATA"])
        for line in cr:
            for k in df_dict:
                df_dict[k].append(None)

            x = float(line[3])
            y = float(line[4])
            c_name = line[1].strip()
            is_tag = eval(line[2])
            df_dict["__X"][-1] = x
            df_dict["__Y"][-1] = y
            df_dict["__CNAME"][-1] = c_name
            df_dict["__IS_TAG"][-1] = is_tag

            for i in range(5, len(line)):
                df_dict[fields[i - 5]][-1] = line[i]

        c_list = [self.category_map[cname] for cname in df_dict["__CNAME"]]

        df_acc = []
        cm_dict = {}
        for k in self.imdc_fns:
            print(k)
            imdc_list = _listToInt(df_dict[k])
            cm = ConfusionMatrix(4, self.cm_names)
            cm.addData(c_list, imdc_list)
            print(cm.fmtCM())
            cm_dict[k] = cm.fmtCM()
            to_dict = {"NAME": k, "OA": cm.OA(), "Kappa": cm.getKappa(), }
            for cm_name in cm:
                to_dict[cm_name + " UATest"] = cm.UA(cm_name)
                to_dict[cm_name + " PATest"] = cm.PA(cm_name)
            df_acc.append(to_dict)

        df_acc = pd.DataFrame(df_acc)
        df_acc = df_acc.sort_values("OA", ascending=False)
        pd.options.display.precision = 2
        print(df_acc)
        pd.options.display.precision = 6

        if is_save:
            to_dirname_save = timeDirName(to_dirname, True)
            time_name = os.path.split(to_dirname_save)[1]

            to_csv_fn = changefiledirname(csv_fn, to_dirname_save)
            to_csv_fn = changext(to_csv_fn, "_{0}.csv".format(time_name))
            shutil.copyfile(csv_fn, to_csv_fn)

            to_txt_fn = changefiledirname(txt_fn, to_dirname_save)
            to_txt_fn = changext(to_txt_fn, "_{0}.txt".format(time_name))
            shutil.copyfile(csv_fn, to_txt_fn)

            df_acc.to_csv(os.path.join(to_dirname_save, "{0}_acc.csv".format(time_name)))
            to_cm_fn = os.path.join(to_dirname_save, "{0}_cm.txt".format(time_name))
            with open(to_cm_fn, "w", encoding="utf-8") as fw:
                for k, cm in cm_dict.items():
                    print(">", k, file=fw)
                    print(cm, file=fw)
                    print(file=fw)

        return


def main():
    # 'SRT', 'X', 'Y', 'CNAME', 'CATEGORY', 'TAG', 'TEST', 'Blue', 'Green', 'Red', 'NIR', 'NDVI', 'NDWI', 'OPT_asm',
    # 'OPT_con', 'OPT_cor', 'OPT_dis', 'OPT_ent', 'OPT_hom', 'OPT_mean', 'OPT_var', 'AS_VV', 'AS_VH', 'AS_VHDVV',
    # 'AS_C11', 'AS_C12_imag', 'AS_C12_real', 'AS_C22', 'AS_Lambda1', 'AS_Lambda2', 'AS_SPAN', 'AS_Epsilon', 'AS_Mu',
    # 'AS_RVI', 'AS_m', 'AS_Beta', 'AS_VH_asm', 'AS_VH_con', 'AS_VH_cor', 'AS_VH_dis', 'AS_VH_ent', 'AS_VH_hom',
    # 'AS_VH_mean', 'AS_VH_var', 'AS_VV_asm', 'AS_VV_con', 'AS_VV_cor', 'AS_VV_dis', 'AS_VV_ent', 'AS_VV_hom',
    # 'AS_VV_mean', 'AS_VV_var', 'DE_VV', 'DE_VH', 'DE_VHDVV', 'DE_C11', 'DE_C12_imag', 'DE_C12_real', 'DE_C22',
    # 'DE_SPAN', 'DE_Lambda1', 'DE_Lambda2', 'DE_Epsilon', 'DE_Mu', 'DE_RVI', 'DE_m', 'DE_Beta', 'DE_VH_asm',
    # 'DE_VH_con', 'DE_VH_cor', 'DE_VH_dis', 'DE_VH_ent', 'DE_VH_hom', 'DE_VH_mean', 'DE_VH_var', 'DE_VV_asm',
    # 'DE_VV_con', 'DE_VV_cor', 'DE_VV_dis', 'DE_VV_ent', 'DE_VV_hom', 'DE_VV_mean', 'DE_VV_var',
    # 'SPL_NOSH-RF-TAG-AS-DE', 'SPL_NOSH-RF-TAG-AS', 'SPL_NOSH-RF-TAG-DE', 'SPL_NOSH-RF-TAG-OPTICS-AS-DE',
    # 'SPL_NOSH-RF-TAG-OPTICS-AS', 'SPL_NOSH-RF-TAG-OPTICS-DE', 'SPL_NOSH-RF-TAG-OPTICS', 'SPL_NOSH-SVM-TAG-AS-DE',
    # 'SPL_NOSH-SVM-TAG-AS', 'SPL_NOSH-SVM-TAG-DE', 'SPL_NOSH-SVM-TAG-OPTICS-AS-DE', 'SPL_NOSH-SVM-TAG-OPTICS-AS',
    # 'SPL_NOSH-SVM-TAG-OPTICS-DE', 'SPL_NOSH-SVM-TAG-OPTICS', 'SPL_SH-RF-TAG-AS-DE', 'SPL_SH-RF-TAG-AS',
    # 'SPL_SH-RF-TAG-DE', 'SPL_SH-RF-TAG-OPTICS-AS-DE', 'SPL_SH-RF-TAG-OPTICS-AS', 'SPL_SH-RF-TAG-OPTICS-DE',
    # 'SPL_SH-RF-TAG-OPTICS', 'SPL_SH-SVM-TAG-AS-DE', 'SPL_SH-SVM-TAG-AS', 'SPL_SH-SVM-TAG-DE',
    # 'SPL_SH-SVM-TAG-OPTICS-AS-DE', 'SPL_SH-SVM-TAG-OPTICS-AS', 'SPL_SH-SVM-TAG-OPTICS-DE', 'SPL_SH-SVM-TAG-OPTICS',
    # 'TF_SPL_NOSH-RF-TAG-AS-DE', 'TF_SPL_NOSH-RF-TAG-AS', 'TF_SPL_NOSH-RF-TAG-DE', 'TF_SPL_NOSH-RF-TAG-OPTICS-AS-DE',
    # 'TF_SPL_NOSH-RF-TAG-OPTICS-AS', 'TF_SPL_NOSH-RF-TAG-OPTICS-DE', 'TF_SPL_NOSH-RF-TAG-OPTICS',
    # 'TF_SPL_NOSH-SVM-TAG-AS-DE', 'TF_SPL_NOSH-SVM-TAG-AS', 'TF_SPL_NOSH-SVM-TAG-DE',
    # 'TF_SPL_NOSH-SVM-TAG-OPTICS-AS-DE', 'TF_SPL_NOSH-SVM-TAG-OPTICS-AS', 'TF_SPL_NOSH-SVM-TAG-OPTICS-DE',
    # 'TF_SPL_NOSH-SVM-TAG-OPTICS', 'TF_SPL_SH-RF-TAG-AS-DE', 'TF_SPL_SH-RF-TAG-AS', 'TF_SPL_SH-RF-TAG-DE',
    # 'TF_SPL_SH-RF-TAG-OPTICS-AS-DE', 'TF_SPL_SH-RF-TAG-OPTICS-AS', 'TF_SPL_SH-RF-TAG-OPTICS-DE',
    # 'TF_SPL_SH-RF-TAG-OPTICS', 'TF_SPL_SH-SVM-TAG-AS-DE', 'TF_SPL_SH-SVM-TAG-AS', 'TF_SPL_SH-SVM-TAG-DE',
    # 'TF_SPL_SH-SVM-TAG-OPTICS-AS-DE', 'TF_SPL_SH-SVM-TAG-OPTICS-AS', 'TF_SPL_SH-SVM-TAG-OPTICS-DE',
    # 'TF_SPL_SH-SVM-TAG-OPTICS', 'SUM1'

    stta = ShadowTiaoTestAcc(r"F:\ProjectSet\Shadow\Analysis\10\bj")
    # stta.buildNew(r"F:\ProjectSet\Shadow\BeiJing\Mods\20231225H110303")
    stta.load()
    # stta.categoryMap(NOT_KNOW=0, IS=1, VEG=2, SOIL=3, WAT=4, IS_SH=1, VEG_SH=2, SOIL_SH=3, WAT_SH=4)
    # stta.cm_names = ["IS", "VEG", "SOIL", "WAT"]
    # stta.updateTrueFalseColumn(True)
    # stta.sumColumns("SUM1", "TF_", "OPTICS")
    # stta.sortColumn(["SUM1", "CATEGORY"], ascending=True)
    # stta.addQJY("2", field_names=[
    #     'SRT', 'X', 'Y', 'CNAME', 'CATEGORY', 'TAG', 'TEST', 'NDVI', 'NDWI', 'SUM1'
    # ], TEST=0)
    stta.accQJY("2", is_save=False)
    stta.saveDFToCSV()
    stta.save()

    pass


if __name__ == "__main__":
    main()
