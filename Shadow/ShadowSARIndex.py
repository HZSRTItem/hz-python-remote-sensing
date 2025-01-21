# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowSARIndex.py
@Time    : 2024/8/14 10:05
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of ShadowSARIndex
-----------------------------------------------------------------------------"""
import csv
import math
import os
from shutil import copyfile

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from onedal.svm import SVC
from osgeo import gdal
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import RandomForestClassifier
from tabulate import tabulate

from RUN.RUNFucs import splTxt2Dict
from SRTCodes.GDALDraw import GDALDrawImage
from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.GDALUtils import GDALSamplingFast, getImageDataFromGeoJson
from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.NumpyUtils import update10EDivide10, eig2, update10Log10, reHist
from SRTCodes.PandasUtils import vLookUpCount
from SRTCodes.SRTModel import SamplesData, RF_RGS, SVM_RGS, MLModel
from SRTCodes.SRTSample import SRTSampleSelect
from SRTCodes.SRTTimeDirectory import TimeDirectory
from SRTCodes.SampleUtils import SamplesUtil
from SRTCodes.Utils import DirFileName, FRW, changext, getfilenamewithoutext, RumTime, savecsv, readJson, \
    TableLinePrint, printDict, printList, readLines, saveJson, filterFileEndWith
from Shadow.Hierarchical import SHH2Config

_RASTER_FN_1 = r"F:\ProjectSet\Shadow\ASDEIndex\Images\QD_SI_BS_1.tif"
_RANGE_FN_1 = changext(_RASTER_FN_1, "_range.json")
_BJ_RASTER_FN = r"F:\ProjectSet\Shadow\ASDEIndex\Images\BJ_SI_BS_1.tif"
_BJ_RANGE_FN = changext(_BJ_RASTER_FN, "_range.json")
_CD_RASTER_FN = r"F:\ProjectSet\Shadow\ASDEIndex\Images\CD_SI_BS_1.tif"
_CD_RANGE_FN = changext(_CD_RASTER_FN, "_range.json")

_QD_RASTER_FN_2 = r"F:\ProjectSet\Shadow\ASDEIndex\Images\QD_SI_BS_2.dat"
_BJ_RASTER_FN_2 = r"F:\ProjectSet\Shadow\ASDEIndex\Images\BJ_SI_BS_2.dat"
_CD_RASTER_FN_2 = r"F:\ProjectSet\Shadow\ASDEIndex\Images\CD_SI_BS_2.dat"
_QD_RANGE_FN_2 = changext(_QD_RASTER_FN_2, "_range.json")
_BJ_RANGE_FN_2 = changext(_BJ_RASTER_FN_2, "_range.json")
_CD_RANGE_FN_2 = changext(_CD_RASTER_FN_2, "_range.json")

_QD_RASTER_FN_3 = r"G:\SHImages\QD_SHImages2.vrt"
_BJ_RASTER_FN_3 = r"G:\SHImages\BJ_SHImages2.vrt"
_CD_RASTER_FN_3 = r"G:\SHImages\CD_SHImages2.vrt"
_QD_RANGE_FN_3 = changext(_QD_RASTER_FN_3, "_range2.json")
_BJ_RANGE_FN_3 = changext(_BJ_RASTER_FN_3, "_range2.json")
_CD_RANGE_FN_3 = changext(_CD_RASTER_FN_3, "_range2.json")

_QD_ENVI_FN = r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\SHH2_QD2_envi.dat"
_BJ_ENVI_FN = r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\SHH2_BJ2_envi.dat"
_CD_ENVI_FN = r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\SHH2_CD2_envi.dat"

_RASTER_NAMES_1 = [
    "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "NDVI", "NDWI", "MNDWI",
    "AS1", "AS2",
    "DE1", "DE2",
    "E1", "E2"
]
_MODEL_DIRNAME = r"F:\ProjectSet\Shadow\ASDEIndex\Models"

_MAP_DICT = {"IS": 1, "VEG": 2, "SOIL": 3, "WAT": 4, "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 3, "WAT_SH": 4}
_COLOR_TABLE_4 = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), }

_QD_SPL_FN = r"F:\ProjectSet\Shadow\ASDEIndex\Samples\sh2_spl30_qd6.csv"
_BJ_SPL_FN = r"F:\ProjectSet\Shadow\ASDEIndex\Samples\sh2_spl30_bj1.csv"
_CD_SPL_FN = r"F:\ProjectSet\Shadow\ASDEIndex\Samples\sh2_spl30_cd6.csv"


def _GET_CITY_NAME(city_name, _qd, _bj, _cd):
    if city_name == "qd":
        return _qd
    if city_name == "bj":
        return _bj
    if city_name == "cd":
        return _cd
    return None


def _SAMPLING(raster_fn, csv_fn, to_csv_fn=None):
    if to_csv_fn is None:
        to_csv_fn = csv_fn
    GDALSamplingFast(raster_fn).csvfile(csv_fn=csv_fn, to_csv_fn=to_csv_fn)


def _GET_MODEL(name):
    name = name.upper()

    def func1():
        if name == "RF":
            return RF_RGS()
        if name == "SVM":
            return SVM_RGS()
        return None

    def func2():
        if name == "RF":
            return RandomForestClassifier()
        if name == "SVM":
            return SVC()
        return None

    return func1()


def show1(_name, _data):
    print("{:>10} {:>15.3f}  {:>15.3f}".format(_name, _data.min(), _data.max()))
    return _data


def calData():
    dfn = DirFileName(r"F:\ProjectSet\Shadow\ASDEIndex\Images\4").mkdir()

    gr = GDALRaster(SHH2Config.CD_ENVI_FN)
    to_fn = dfn.fn("CD_SI_BS_1.tif")

    def read(_name):
        _data = show1(_name, gr.readGDALBand(_name))
        _data = show1(_name, update10EDivide10(_data))
        return _data

    as1 = read("AS_VV")
    as2 = read("AS_VH")
    de1 = read("DE_VV")
    de2 = read("DE_VH")

    # gr.save(as1, dfn.fn("as1.dat"))
    # gr.save(as2, dfn.fn("as2.dat"))
    # gr.save(de1, dfn.fn("de1.dat"))
    # gr.save(de2, dfn.fn("de2.dat"))
    # gr.save(update10Log10(as1), dfn.fn("as1_10log10.dat"))
    # gr.save(update10Log10(as2), dfn.fn("as2_10log10.dat"))
    # gr.save(update10Log10(de1), dfn.fn("de1_10log10.dat"))
    # gr.save(update10Log10(de2), dfn.fn("de2_10log10.dat"))

    print("-" * 60)
    e1, e2, v11, v12, v21, v22 = eig2(as1 * as1, as1 * de2, as2 * de1, de1 * de1)
    e21 = e2
    show1("e1", e1)
    show1("e2", e2)
    # gr.save(e2, dfn.fn("e21.dat"))
    e2 = update10Log10(e2)
    show1("e1", e1)
    show1("e2", e2)
    # gr.save(e2, dfn.fn("e21_10log10.dat"))

    print("-" * 60)
    e1, e2, v11, v12, v21, v22 = eig2(as2 * as2, as1 * de2, as2 * de1, de2 * de2)
    e22 = e2
    show1("e1", e1)
    show1("e2", e2)
    # gr.save(e2, dfn.fn("e22.dat"))
    e2 = update10Log10(e2)
    show1("e1", e1)
    show1("e2", e2)

    # gr.save(e2, dfn.fn("e22_10log10.dat"))

    def read2(_name):
        _data = show1(_name, gr.readGDALBand(_name))
        return _data

    print("-" * 60)
    # b g r n s1 s2 ndvi ndwi mndwi as1 as2 de1 de2 e1 e2
    # 1 2 3 4  5  6    7    8     9  10  11  12  13 14 15
    to_data = np.zeros((15, gr.n_rows, gr.n_columns))
    opt_names = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "NDVI", "NDWI", ]
    i = 0
    for i, name in enumerate(opt_names):
        to_data[i] = read2(name)
    i += 1
    to_data[i] = (to_data[1] - to_data[5]) / (to_data[5] + to_data[1] + 0.0000001)
    i += 1
    for data in [as1, as2, de1, de2, e21, e22]:
        to_data[i] = update10Log10(data)
        i += 1

    sar_names = ["AS1", "AS2", "DE1", "DE2", "E1", "E2"]
    print(to_fn)
    if os.path.isfile(to_fn):
        os.remove(to_fn)
    gr.save(to_data.astype("float32"), to_fn, fmt="GTiff",
            dtype=gdal.GDT_Float32, descriptions=[*opt_names, "MNDWI", *sar_names])
    return


def getRange(raster_fn=_CD_RASTER_FN):
    def x_range(data):
        data = np.expand_dims(data, axis=0)
        print(data.shape)
        x = reHist(data, ratio=0.005)
        return float(x[0][0]), float(x[0][1])

    gr = GDALRaster(raster_fn)
    to_dict = {}

    for fn in gr.names:
        x_min, x_max = x_range(gr.readGDALBand(fn))
        to_dict[fn] = {"min": x_min, "max": x_max}
        show1(fn, np.array([x_min, x_max]))

    FRW(changext(raster_fn, "_range.json")).saveJson(to_dict)


class SH1SI_TIC:

    def __init__(self, city_name, csv_fn, td: TimeDirectory = None, range_fn=_RANGE_FN_1, raster_fn=_RASTER_FN_1):
        self._dfn = DirFileName(os.path.join(_MODEL_DIRNAME, "TEMP"))
        self._dfn.mkdir()

        self.city_name = city_name
        self.range_fn = range_fn
        self.raster_fn = raster_fn

        self.td = td
        self.initTD(td)

        self.csv_fn = csv_fn
        self.sd = None
        self.initSD(csv_fn)

        self.model_name = None
        self.model = None
        self.models = {}
        self.models_records = {}
        self.accuracy_dict = {}

        self.log("#", "-" * 36, self.city_name.upper(), "SHH2FC", "-" * 36, "#\n")
        self.kw("CITY_NAME", self.city_name)
        self.kw("CSV_FN", self.csv_fn)
        self.kw("RASTER_FN", self.raster_fn)
        self.kw("RANGE_FN", self.range_fn)

    def initTD(self, td: TimeDirectory = None):
        if td is None:
            return
        self.td = td
        self._dfn = DirFileName(self.td.time_dirname())
        self.log(self.td.time_dfn.dirname)
        self.td.copyfile(__file__)

    def initSD(self, csv_fn):
        csv_fn = os.path.abspath(csv_fn)
        self.csv_fn = csv_fn
        to_spl_csv_fn = changext(csv_fn, "_spl.csv")
        if not os.path.isfile(to_spl_csv_fn):
            _SAMPLING(self.raster_fn, csv_fn, to_spl_csv_fn)
        self.sd = SamplesData()
        self.sd.addCSV(to_spl_csv_fn)
        if self.td is not None:
            self.td.copyfile(to_spl_csv_fn)

    def addModel(self, name, clf_name="rf", x_keys=None, map_dict=None, color_table=None,
                 train_filters=None, test_filters=None, cm_names=None):
        if test_filters is None:
            test_filters = []
        if train_filters is None:
            train_filters = []
        if x_keys is None:
            x_keys = SHH2Config.FEAT_NAMES.ALL
        if map_dict is None:
            map_dict = _MAP_DICT
        if color_table is None:
            color_table = _COLOR_TABLE_4
        if cm_names is None:
            cm_names = ["IS", "VEG", "SOIL", "WAT"]

        self.log("\n#", "-" * 30, self.city_name.upper(), "NOFC", name.upper(), clf_name.upper(), "-" * 30, "#\n")

        ml_mod = MLModel()
        ml_mod.filename = self._dfn.fn("{}-{}-{}.shh2mod".format(self.city_name, name, clf_name))
        ml_mod.data_scale.readJson(self.range_fn)

        ml_mod.x_keys = x_keys
        ml_mod.map_dict = map_dict
        ml_mod.color_table = color_table
        ml_mod.clf = _GET_MODEL(clf_name)
        ml_mod.train_filters = train_filters
        ml_mod.test_filters = test_filters
        ml_mod.cm_names = cm_names

        ml_mod.sampleData(self.sd)

        self.kw("NAME", getfilenamewithoutext(ml_mod.filename))
        self.kw("MLMOD.X_KEYS", ml_mod.x_keys)
        self.kw("MLMOD.MAP_DICT", ml_mod.map_dict)
        self.kw("MLMOD.COLOR_TABLE", ml_mod.color_table)
        self.kw("MLMOD.CLF", ml_mod.clf)
        self.kw("MLMOD.TRAIN_FILTERS", ml_mod.train_filters)
        self.kw("MLMOD.TEST_FILTERS", ml_mod.test_filters)
        self.kw("MLMOD.CM_NAMES", ml_mod.cm_names)

        self.models[name] = ml_mod
        self.model = ml_mod
        self.model_name = name
        self.models_records[name] = {"TYPE": self.city_name, "CLF": clf_name, }
        return ml_mod

    def train(self, name=None, mod=None, is_imdc=False):
        if name is None:
            name = self.model_name
        self.log("\n# Training ------")
        if name not in self.models:
            self.models[name] = mod
            self.models_records[name] = {"TYPE": "None", "CLF": "None", }
        if mod is not None:
            self.models[name] = mod
        self.model = self.models[name]

        self.model.samples.showCounts(self.log)
        self.model.train()
        acc_dict = self.accuracy()
        self.accuracy_dict[name] = acc_dict
        self.log("\n# Accuracy ------")
        for k in acc_dict:
            if "CM" in k:
                self.kw(k, acc_dict[k].fmtCM(), sep=":\n", end="")
        for k in acc_dict:
            if "CM" not in k:
                self.kw(k, acc_dict[k])
                self.models_records[name][k] = acc_dict[k]
        if is_imdc:
            self.model.save(is_samples=False)
            self.imdc()
        return mod

    def trains(self, is_imdc=False):
        for name in self.models:
            self.train(name, is_imdc=is_imdc)

    def imdc(self, mod=None, to_imdc_fn=None):
        if mod is None:
            mod = self.model
        self.log("\n# Image Classification ------")
        mod.imdc(self.raster_fn, to_imdc_fn=to_imdc_fn)
        self.kw("TO_IMDC_FN", to_imdc_fn)

    def accuracyOA(self, mod=None):
        if mod is None:
            mod = self.model
        y1 = mod.accuracy_dict["y1"]
        y2 = mod.accuracy_dict["y2"]
        cm = ConfusionMatrix(class_names=["IS", "VEG", "SOIL", "WAT"])
        cm.addData(y1, y2)
        return cm

    def accuracy(self, mod=None):
        to_dict = {}
        cm_oa = self.accuracyOA(mod)
        to_dict["IS_CM"] = cm_oa
        to_dict["IS_OA"] = cm_oa.accuracyCategory("IS").OA()
        to_dict["IS_Kappa"] = cm_oa.accuracyCategory("IS").getKappa()
        return to_dict

    def showAccuracy(self):
        df = pd.DataFrame(self.accuracy_dict).T
        to_ks = []
        for k in df:
            if "CM" not in k:
                to_ks.append(k)
        df = df[to_ks]
        self.log(tabulate(df, headers="keys", tablefmt="simple"))
        if self.td is not None:
            self.td.saveDF("accuracy", df)

    def kw(self, key, value, sep=": ", end="\n", is_print=None):
        if self.td is None:
            print(key, value, sep=sep, end=end, )
            return value
        else:
            return self.td.kw(key, value, sep=sep, end=end, is_print=is_print)

    def log(self, *text, sep=" ", end="\n", is_print=None):
        if self.td is None:
            print(*text, sep=sep, end=end, )
        else:
            self.td.log(*text, sep=sep, end=end, is_print=is_print)


def main():
    csv_fn = r"F:\ProjectSet\Shadow\ASDEIndex\Samples\sh2_spl30_qd6.csv"
    td = TimeDirectory(_MODEL_DIRNAME).initLog()
    tic = SH1SI_TIC("qd", csv_fn, td=td)
    is_imdc = True

    run_list = []

    f_opt = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "NDVI", "NDWI", "MNDWI", ]
    f_as = ["AS1", "AS2", ]
    f_de = ["DE1", "DE2", ]
    f_e = ["E1", "E2"]

    def model_func(*args, **kwargs):
        run_list.append((args, kwargs))

    for clf_name in ["RF", "SVM"]:
        model_func("{}_O".format(clf_name), clf_name, x_keys=f_opt)
        model_func("{}_OA".format(clf_name), clf_name, x_keys=f_opt + f_as)
        model_func("{}_OD".format(clf_name), clf_name, x_keys=f_opt + f_de)
        model_func("{}_OAD".format(clf_name), clf_name, x_keys=f_opt + f_as + f_de)
        model_func("{}_OE".format(clf_name), clf_name, x_keys=f_opt + f_e)

    run_time = RumTime(len(run_list), tic.log).strat()
    for _args, _kwargs in run_list:
        tic.addModel(*_args, **_kwargs)
        tic.train(is_imdc=is_imdc)
        run_time.add().printInfo()
    run_time.end()

    tic.showAccuracy()

    return


def run(city_name):
    # csv_fn = _GET_CITY_NAME(city_name, _QD_SPL_FN, _BJ_SPL_FN, _CD_SPL_FN)
    # raster_fn = _GET_CITY_NAME(city_name, _RASTER_FN_1, _BJ_RASTER_FN, _CD_RASTER_FN)
    # range_fn = _GET_CITY_NAME(city_name, _RANGE_FN_1, _BJ_RANGE_FN, _CD_RANGE_FN)

    # csv_fn = _GET_CITY_NAME(city_name, _QD_SPL_FN, _BJ_SPL_FN, _CD_SPL_FN)
    csv_fn = _GET_CITY_NAME(
        city_name,
        r"F:\GraduationDesign\Result\run\Samples\1\qd_spl2.csv",
        r"F:\GraduationDesign\Result\run\Samples\1\bj_spl2.csv",
        r"F:\GraduationDesign\Result\run\Samples\1\cd_spl2.csv",
    )
    # raster_fn = _GET_CITY_NAME(city_name, _QD_RASTER_FN_2, _BJ_RASTER_FN_2, _CD_RASTER_FN_2)
    # range_fn = _GET_CITY_NAME(city_name, _QD_RANGE_FN_2, _BJ_RANGE_FN_2, _CD_RANGE_FN_2)

    raster_fn = _GET_CITY_NAME(city_name, _QD_RASTER_FN_3, _BJ_RASTER_FN_3, _CD_RASTER_FN_3)
    range_fn = _GET_CITY_NAME(city_name, _QD_RANGE_FN_3, _BJ_RANGE_FN_3, _CD_RANGE_FN_3)

    td = TimeDirectory(_MODEL_DIRNAME).initLog()
    tic = SH1SI_TIC(city_name, csv_fn, td=td, raster_fn=raster_fn, range_fn=range_fn)
    is_imdc = True

    run_list = []

    f_opt = [
        "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "NDVI", "NDWI", "MNDWI",
        "OPT_asm", "OPT_con", "OPT_cor", "OPT_dis", "OPT_ent", "OPT_hom", "OPT_mean", "OPT_var",
    ]
    f_as = [
        "AS_VV", "AS_VH",
        "AS_VH_asm", "AS_VH_con", "AS_VH_cor", "AS_VH_dis", "AS_VH_ent", "AS_VH_hom", "AS_VH_mean", "AS_VH_var",
        "AS_VV_asm", "AS_VV_con", "AS_VV_cor", "AS_VV_dis", "AS_VV_ent", "AS_VV_hom", "AS_VV_mean", "AS_VV_var",
    ]
    f_de = [
        "DE_VV", "DE_VH",
        "DE_VH_asm", "DE_VH_con", "DE_VH_cor", "DE_VH_dis", "DE_VH_ent", "DE_VH_hom", "DE_VH_mean", "DE_VH_var",
        "DE_VV_asm", "DE_VV_con", "DE_VV_cor", "DE_VV_dis", "DE_VV_ent", "DE_VV_hom", "DE_VV_mean", "DE_VV_var",
    ]
    f_e = [
        "E1", "E2",
        "E1_mean", "E1_var", "E1_hom", "E1_con", "E1_dis", "E1_ent", "E1_asm", "E1_cor",
        "E2_mean", "E2_var", "E2_hom", "E2_con", "E2_dis", "E2_ent", "E2_asm", "E2_cor",
    ]

    f = [
        "AS_A", "AS_Alpha", "AS_angle", "AS_Beta", "AS_C11", "AS_C12_imag", "AS_C12_real", "AS_C22", "AS_Epsilon",
        "AS_H", "AS_Lambda1", "AS_Lambda2", "AS_m", "AS_Mu", "AS_RVI", "AS_SPAN", "AS_VH", "AS_VHDVV", "AS_VH_asm",
        "AS_VH_con", "AS_VH_cor", "AS_VH_dis", "AS_VH_ent", "AS_VH_hom", "AS_VH_mean", "AS_VH_var", "AS_VV",
        "AS_VV_asm", "AS_VV_con", "AS_VV_cor", "AS_VV_dis", "AS_VV_ent", "AS_VV_hom", "AS_VV_mean", "AS_VV_var", "Blue",
        "DE_A", "DE_Alpha", "DE_angle", "DE_Beta", "DE_C11", "DE_C12_imag", "DE_C12_real", "DE_C22", "DE_Epsilon",
        "DE_H", "DE_Lambda1", "DE_Lambda2", "DE_m", "DE_Mu", "DE_RVI", "DE_SPAN", "DE_VH", "DE_VHDVV", "DE_VH_asm",
        "DE_VH_con", "DE_VH_cor", "DE_VH_dis", "DE_VH_ent", "DE_VH_hom", "DE_VH_mean", "DE_VH_var", "DE_VV",
        "DE_VV_asm", "DE_VV_con", "DE_VV_cor", "DE_VV_dis", "DE_VV_ent", "DE_VV_hom", "DE_VV_mean", "DE_VV_var", "E1",
        "E1_asm", "E1_con", "E1_cor", "E1_dis", "E1_ent", "E1_hom", "E1_mean", "E1_var", "E2", "E2_asm", "E2_con",
        "E2_cor", "E2_dis", "E2_ent", "E2_hom", "E2_mean", "E2_var", "Green", "MNDWI", "NDVI", "NDWI", "NIR", "OPT_asm",
        "OPT_con", "OPT_cor", "OPT_dis", "OPT_ent", "OPT_hom", "OPT_mean", "OPT_var", "Red", "SWIR1", "SWIR2",
    ]

    def model_func(*args, **kwargs):
        run_list.append((args, kwargs))

    for clf_name in ["RF"]:
        # model_func("{}_1AS".format(clf_name), clf_name, x_keys=f_as)
        # model_func("{}_2DE".format(clf_name), clf_name, x_keys=f_as)
        # model_func("{}_3ASDE".format(clf_name), clf_name, x_keys=f_as + f_de)
        # model_func("{}_4O".format(clf_name), clf_name, x_keys=f_opt)
        # model_func("{}_5OA".format(clf_name), clf_name, x_keys=f_opt + f_as)
        # model_func("{}_6OD".format(clf_name), clf_name, x_keys=f_opt + f_de)
        # model_func("{}_7OAD".format(clf_name), clf_name, x_keys=f_opt + f_as + f_de)
        # model_func("{}_10E".format(clf_name), clf_name, x_keys=f_e)
        # model_func("{}_11E".format(clf_name), clf_name, x_keys=f_opt + f_e)

        # model_func("{}_E2".format(clf_name), clf_name, x_keys=f_e)
        # model_func("{}_OD2".format(clf_name), clf_name, x_keys=f_de)
        # model_func("{}_OE".format(clf_name), clf_name, x_keys=f_opt + f_e)
        # model_func("{}_OAD".format(clf_name), clf_name, x_keys=f_opt + f_as + f_de)
        # model_func("{}_OA".format(clf_name), clf_name, x_keys=f_opt + f_as)
        # model_func("{}_OD".format(clf_name), clf_name, x_keys=f_opt + f_de)

        # model_func("{}_E1".format(clf_name), clf_name, x_keys=["E1", "E2", ])
        # model_func("{}_AD1".format(clf_name), clf_name, x_keys=["AS_VV", "AS_VH", "DE_VV", "DE_VH", ])
        # model_func("{}_OA1".format(clf_name), clf_name, x_keys=["AS_VV", "AS_VH", ])
        # model_func("{}_OD1".format(clf_name), clf_name, x_keys=["DE_VV", "DE_VH", ])

        # model_func("{}_E2".format(clf_name), clf_name, x_keys=f_e)
        # model_func("{}_AD2".format(clf_name), clf_name, x_keys=f_as + f_de)
        # model_func("{}_OA2".format(clf_name), clf_name, x_keys=f_as)
        # model_func("{}_OD2".format(clf_name), clf_name, x_keys=f_de)

        # model_func("{}_1ADESI".format(clf_name), clf_name, x_keys=["E1", "E2", ])
        #
        # model_func("{}_2AS".format(clf_name), clf_name, x_keys=["AS_VV", "AS_VH", ])
        # model_func("{}_3DE".format(clf_name), clf_name, x_keys=["DE_VV", "DE_VH", ])
        #
        # model_func("{}_4ASC2".format(clf_name), clf_name, x_keys=["AS_C11", "AS_C22", ])
        # model_func("{}_5DEC2".format(clf_name), clf_name, x_keys=["DE_C11", "DE_C22", ])

        model_func("{}_6ASH".format(clf_name), clf_name, x_keys=["AS_H", "AS_A", ])
        model_func("{}_7DEH".format(clf_name), clf_name, x_keys=["DE_H", "DE_A", ])

        # model_func("{}_8ADESI_GLCM".format(clf_name), clf_name, x_keys=f_e)
        # model_func("{}_9ADESI_GLCM_OPT".format(clf_name), clf_name, x_keys=f_e + f_opt)

    run_time = RumTime(len(run_list), tic.log).strat()
    for _args, _kwargs in run_list:
        tic.addModel(*_args, **_kwargs)
        tic.train(is_imdc=is_imdc)
        run_time.add().printInfo()
    run_time.end()

    tic.showAccuracy()


def draw():
    def func1():
        gr = GDALRaster(_RASTER_FN_1)
        print(gr.names)

        to_dfn = DirFileName(r"F:\SARIndex\Images\41")

        def read(_name):
            _data = show1(_name, gr.readGDALBand(_name))
            return _data

        def data_range(_name, x_min, x_max):
            _data = read(_name)
            _data = np.clip(_data, x_min, x_max)
            _data = (_data - x_min) / (x_max - x_min)
            _data = _data * 255
            _data = _data.astype("int8")
            to_fn = to_dfn.fn("{}_show.tif".format(_name))
            print(to_fn)
            gr.save(_data, to_fn, fmt="GTiff", dtype=gdal.GDT_Byte, descriptions=[_name])
            return _name, _data

        data_range("AS1", -20, 2)
        data_range("AS2", -28, -7)

        data_range("DE1", -26, 2)
        data_range("DE2", -33, -7)

        data_range("E1", -35, 10)
        data_range("E2", -35, -7)

    def func2():
        def remove_white_border(input_image_path, output_image_path=None):
            if output_image_path is None:
                output_image_path = input_image_path
            img = Image.open(input_image_path)
            img = img.convert("RGB")
            datas = img.getdata()
            white = (255, 255, 255)

            left, top, right, bottom = img.width, img.height, 0, 0
            for y in range(img.height):
                for x in range(img.width):
                    if datas[y * img.width + x] != white:
                        if x < left:
                            left = x
                        if x > right:
                            right = x
                        if y < top:
                            top = y
                        if y > bottom:
                            bottom = y

            # left, top, right, bottom = left + 1, top + 1, right - 1, bottom - 1
            bbox = left, top, right, bottom
            img_cropped = img.crop(bbox)
            img_cropped.save(output_image_path, format="JPEG", dpi=(300, 300))

        fns = [
            r"F:\SARIndex\Images\41\AS1.jpg",
            r"F:\SARIndex\Images\41\AS2.jpg",
            r"F:\SARIndex\Images\41\DE1.jpg",
            r"F:\SARIndex\Images\41\DE2.jpg",
            r"F:\SARIndex\Images\41\E1.jpg",
            r"F:\SARIndex\Images\41\E2.jpg",
        ]

        for fn in fns:
            to_fn = changext(fn, "_rk.jpg")
            print(to_fn)
            remove_white_border(fn, to_fn)

    def func3():
        FONT_SIZE = 16

        win_size = (131, 131)
        gi_win_size = int(win_size[0] * 2640.0 / 121.0), int(win_size[1] * 2026.0 / 121.0)
        print("win_size", win_size)
        print("gi_win_size", gi_win_size)
        gdi = GDALDrawImage(win_size)

        qd = gdi.addGR(raster_fn=_RASTER_FN_1, geo_range=_RANGE_FN_1)
        imdc_oe = gdi.addGR(r"F:\SARIndex\Result\qd-SVM_OE-SVM_imdc.tif")
        imdc_oad = gdi.addGR(r"F:\SARIndex\Result\qd-RF_OAD-RF_imdc.tif")
        imdc_oa = gdi.addGR(r"F:\SARIndex\Result\qd-RF_OA-RF_imdc.tif")
        imdc_od = gdi.addGR(r"F:\SARIndex\Result\qd-RF_OD-RF_imdc.tif")
        imdc_o = gdi.addGR(r"F:\SARIndex\Result\qd-RF_O-RF_imdc.tif")

        gdi.addRCC("RGB", qd, channel_list=["Red", "Green", "Blue"])
        gdi.addRCC("NRG", qd, channel_list=["NIR", "Red", "Green"])

        gdi.addRCC("ADEI", imdc_oe, channel_list=[0], is_01=False, is_min_max=False, )
        gdi.addRCC("AD", imdc_oad, channel_list=[0], is_01=False, is_min_max=False, )
        gdi.addRCC("AS", imdc_oa, channel_list=[0], is_01=False, is_min_max=False, )
        gdi.addRCC("DE", imdc_od, channel_list=[0], is_01=False, is_min_max=False, )
        gdi.addRCC("Opt", imdc_o, channel_list=[0], is_01=False, is_min_max=False, )

        n_rows, n_columns = 4, 6
        n = 13
        fig = plt.figure(figsize=(n_columns / n_columns * n, n_rows / n_columns * n,), )
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, hspace=0.03, wspace=0.03)

        column_names = ["S2", "Opt-ADEI", "Opt-AD", "Opt-AS", "Opt-DE", "Opt", ]

        from matplotlib import colors
        color_dict = {
            1: colors.to_hex((1, 0, 0)),
            2: colors.to_hex((0, 1, 0)),
            3: colors.to_hex((1, 1, 0)),
            4: colors.to_hex((0, 0, 1)),
        }
        cname_dict = {1: "IS", 2: "VEG", 3: "SO", 4: "WAT"}
        sh_type_dict = {
            1: "Optical Shadow",
            2: "AS SAR Shadow",
            3: "DE SAR Shadow",
        }

        imdc_color_dict = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255)}

        class draw_column:

            def __init__(self):
                self.row = 1
                self.column = 1
                self.name = ""
                self.n = 1

            def fit(self, name, x, y, ):
                self.column = 1
                self.name = name
                self.n = 1

                self._readDraw("NRG", x, y, min_list=[300, 300, 300], max_list=[2900, 1500, 1500])
                self._readDraw("ADEI", x, y, color_dict=imdc_color_dict)
                self._readDraw("AD", x, y, color_dict=imdc_color_dict)
                self._readDraw("AS", x, y, color_dict=imdc_color_dict)
                self._readDraw("DE", x, y, color_dict=imdc_color_dict)
                self._readDraw("Opt", x, y, color_dict=imdc_color_dict)

                self.row += 1

            def _readDraw(self, *args, **kwargs):
                plt.subplot(n_rows, n_columns, (self.row - 1) * n_columns + self.n)
                gdi.readDraw(*args, **kwargs)
                self.column_draw()
                self.n += 1

            def column_draw(self):
                ax = plt.gca()
                if self.row == 1:
                    ax.set_title(column_names[self.column - 1], fontdict={"size": FONT_SIZE})
                if self.column == 1:
                    ax.set_ylabel(self.name, rotation=0, labelpad=10, fontdict={"size": FONT_SIZE})
                self.column += 1

        column = draw_column()
        column.fit(r"(a)", 120.099385, 36.200576)
        column.fit(r"(b)", 120.075381, 36.228891)
        column.fit(r"(c)", 120.228361, 36.293254)
        column.fit(r"(d)", 120.37508, 36.06523)

        plt.savefig(r"F:\SARIndex\Images\result2.jpg", dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.show()

    def add_imdc_grs(gdi: GDALDrawImage, mod_name="RF"):
        json_dict = readJson(r"F:\SARIndex\Draw\result_fns.json")

        data_dict = json_dict[mod_name]
        to_dict = {name: [] for name in [
            "O", "OA", "OAD", "OD", "OE", "OG", "OAG",
            "OADG", "ODG", "OEG", "AD", "ADG", "E", "EG", "A", "AG", "D", "DG",
        ]}
        for name in to_dict:
            for city_name in data_dict:
                fn = data_dict[city_name][name]
                to_dict[name].append(gdi.addGR(fn))
        for name in to_dict:
            gdi.addRCC(name, *to_dict[name], channel_list=[0], is_01=False, is_min_max=False, )
        return gdi

    def func4():
        FONT_SIZE = 16

        win_size = (131, 131)
        gi_win_size = int(win_size[0] * 2640.0 / 121.0), int(win_size[1] * 2026.0 / 121.0)
        print("win_size", win_size)
        print("gi_win_size", gi_win_size)
        gdi = GDALDrawImage(win_size)

        qd = gdi.addGR(raster_fn=_RASTER_FN_1, geo_range=_RANGE_FN_1)
        bj = gdi.addGR(raster_fn=_BJ_RASTER_FN, geo_range=_BJ_RANGE_FN)
        cd = gdi.addGR(raster_fn=_CD_RASTER_FN, geo_range=_CD_RANGE_FN)
        gdi.addRCC("RGB", qd, bj, cd, channel_list=["Red", "Green", "Blue"])
        gdi.addRCC("NRG", qd, bj, cd, channel_list=["NIR", "Red", "Green"])

        qd_imdc_oe = gdi.addGR(r"F:\SARIndex\Result\qd-SVM_OE-SVM_imdc.tif")
        qd_imdc_oad = gdi.addGR(r"F:\SARIndex\Result\qd-RF_OAD-RF_imdc.tif")
        qd_imdc_oa = gdi.addGR(r"F:\SARIndex\Result\qd-RF_OA-RF_imdc.tif")
        qd_imdc_od = gdi.addGR(r"F:\SARIndex\Result\qd-RF_OD-RF_imdc.tif")
        qd_imdc_o = gdi.addGR(r"F:\SARIndex\Result\qd-RF_O-RF_imdc.tif")

        bj_imdc_oe = gdi.addGR(r"F:\SARIndex\Result2\Beijing\bj-SVM_OE-SVM_imdc.tif")
        bj_imdc_oad = gdi.addGR(r"F:\SARIndex\Result2\Beijing\bj-RF_OAD-RF_imdc.tif")
        bj_imdc_oa = gdi.addGR(r"F:\SARIndex\Result2\Beijing\bj-RF_OA-RF_imdc.tif")
        bj_imdc_od = gdi.addGR(r"F:\SARIndex\Result2\Beijing\bj-RF_OD-RF_imdc.tif")
        bj_imdc_o = gdi.addGR(r"F:\SARIndex\Result2\Beijing\bj-RF_O-RF_imdc.tif")

        cd_imdc_oe = gdi.addGR(r"F:\SARIndex\Result2\Chengdu\cd-SVM_OE-SVM_imdc.tif")
        cd_imdc_oad = gdi.addGR(r"F:\SARIndex\Result2\Chengdu\cd-RF_OAD-RF_imdc.tif")
        cd_imdc_oa = gdi.addGR(r"F:\SARIndex\Result2\Chengdu\cd-RF_OA-RF_imdc.tif")
        cd_imdc_od = gdi.addGR(r"F:\SARIndex\Result2\Chengdu\cd-RF_OD-RF_imdc.tif")
        cd_imdc_o = gdi.addGR(r"F:\SARIndex\Result2\Chengdu\cd-RF_O-RF_imdc.tif")

        gdi.addRCC("ADEI", qd_imdc_oe, bj_imdc_oe, cd_imdc_oe, channel_list=[0], is_01=False, is_min_max=False, )
        gdi.addRCC("AD", qd_imdc_oad, bj_imdc_oad, cd_imdc_oad, channel_list=[0], is_01=False, is_min_max=False, )
        gdi.addRCC("AS", qd_imdc_oa, bj_imdc_oa, cd_imdc_oa, channel_list=[0], is_01=False, is_min_max=False, )
        gdi.addRCC("DE", qd_imdc_od, bj_imdc_od, cd_imdc_od, channel_list=[0], is_01=False, is_min_max=False, )
        gdi.addRCC("Opt", qd_imdc_o, bj_imdc_o, cd_imdc_o, channel_list=[0], is_01=False, is_min_max=False, )

        n_rows, n_columns = 4, 6
        n = 13
        fig = plt.figure(figsize=(n_columns / n_columns * n, n_rows / n_columns * n,), )
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.90, hspace=0.03, wspace=0.03)

        column_names = ["S2", "Opt-ADEI", "Opt-AD", "Opt-AS", "Opt-DE", "Opt", ]

        from matplotlib import colors
        color_dict = {
            1: colors.to_hex((1, 0, 0)),
            2: colors.to_hex((0, 1, 0)),
            3: colors.to_hex((1, 1, 0)),
            4: colors.to_hex((0, 0, 1)),
        }
        cname_dict = {1: "IS", 2: "VEG", 3: "SO", 4: "WAT"}
        sh_type_dict = {
            1: "Optical Shadow",
            2: "AS SAR Shadow",
            3: "DE SAR Shadow",
        }

        imdc_color_dict = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255)}

        class draw_column:

            def __init__(self):
                self.row = 1
                self.column = 1
                self.name = ""
                self.n = 1

            def fit(self, name, x, y, ):
                self.column = 1
                self.name = name
                self.n = 1

                self._readDraw("NRG", x, y, min_list=[300, 300, 300], max_list=[2900, 1500, 1500])
                self._readDraw("ADEI", x, y, color_dict=imdc_color_dict)
                self._readDraw("AD", x, y, color_dict=imdc_color_dict)
                self._readDraw("AS", x, y, color_dict=imdc_color_dict)
                self._readDraw("DE", x, y, color_dict=imdc_color_dict)
                self._readDraw("Opt", x, y, color_dict=imdc_color_dict)

                self.row += 1

            def _readDraw(self, *args, **kwargs):
                plt.subplot(n_rows, n_columns, (self.row - 1) * n_columns + self.n)
                gdi.readDraw(*args, **kwargs)
                self.column_draw()
                self.n += 1

            def column_draw(self):
                ax = plt.gca()
                if self.row == 1:
                    ax.set_title(column_names[self.column - 1], fontdict={"size": FONT_SIZE})
                if self.column == 1:
                    ax.set_ylabel(self.name, rotation=0, labelpad=40, fontdict={"size": FONT_SIZE})
                self.column += 1

        column = draw_column()

        column.fit(r"(a) Beijing", 116.504023, 39.886551)
        column.fit(r"(b) Beijing", 116.348373, 39.782519)
        column.fit(r"(c) Chengdu", 104.101211, 30.788077)
        column.fit(r"(d) Chengdu", 104.065650, 30.696051)

        plt.savefig(r"F:\SARIndex\Images\result2.jpg", dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.show()

    def func5():
        fn_tifs = r"F:\SARIndex\Draw\result_tifs.txt"
        fns = readLines(fn_tifs)
        to_dict = {"RF": {"qd": {}, "bj": {}, "cd": {}}, "SVM": {"qd": {}, "bj": {}, "cd": {}}}
        names = [
            "O", "OA", "OAD", "OD", "OE", "OG", "OAG",
            "OADG", "ODG", "OEG", "AD", "ADG", "E", "EG", "A", "AG", "D", "DG",
        ]
        for fn in fns:
            for rf_svm_name in to_dict:
                for city_name in to_dict[rf_svm_name]:
                    if (rf_svm_name in fn) and (city_name in fn):
                        to_dict[rf_svm_name][city_name][names[len(to_dict[rf_svm_name][city_name])]] = fn
        tlp = TableLinePrint().firstLine("MODEL", "CITY", "N", "FILENAME")
        for rf_svm_name in to_dict:
            for city_name in to_dict[rf_svm_name]:
                for fn in to_dict[rf_svm_name][city_name]:
                    tlp.print(rf_svm_name, city_name, fn, to_dict[rf_svm_name][city_name][fn])
                tlp.separationLine()
        saveJson(to_dict, r"F:\SARIndex\Draw\result_fns.txt")

    def func6():
        FONT_SIZE = 16

        win_size = (331, 331)
        gi_win_size = int(win_size[0] * 2640.0 / 121.0), int(win_size[1] * 2026.0 / 121.0)
        print("win_size", win_size)
        print("gi_win_size", gi_win_size)

        gdi = GDALDrawImage(win_size)

        qd = gdi.addGR(raster_fn=_QD_RASTER_FN_2, geo_range=_QD_RANGE_FN_2)
        bj = gdi.addGR(raster_fn=_BJ_RASTER_FN_2, geo_range=_BJ_RANGE_FN_2)
        cd = gdi.addGR(raster_fn=_CD_RASTER_FN_2, geo_range=_CD_RANGE_FN_2)
        gdi.addRCC("RGB", qd, bj, cd, channel_list=["Red", "Green", "Blue"])
        gdi.addRCC("NRG", qd, bj, cd, channel_list=["NIR", "Red", "Green"])

        add_imdc_grs(gdi)

        n_rows, n_columns = 5, 6
        n = 13
        fig = plt.figure(figsize=(n_columns / n_columns * n, n_rows / n_columns * n,), )
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.90, hspace=0.03, wspace=0.03)

        column_names = ["S2", "Opt-ADEI", "Opt-AD", "Opt-AS", "Opt-DE", "Opt", ]

        imdc_color_dict = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255)}

        names = [
            "O", "OA", "OAD", "OD", "OE", "OG", "OAG",
            "OADG", "ODG", "OEG", "AD", "ADG", "E", "EG", "A", "AG", "D", "DG",
        ]

        class draw_column:

            def __init__(self):
                self.row = 1
                self.column = 1
                self.name = ""
                self.n = 1

            def fit(self, name, x, y, ):
                self.column = 1
                self.name = name
                self.n = 1

                self._readDraw("NRG", x, y, min_list=[300, 300, 300], max_list=[2900, 1500, 1500])

                self._readDraw("EG", x, y, color_dict=imdc_color_dict)
                self._readDraw("ADG", x, y, color_dict=imdc_color_dict)
                self._readDraw("AG", x, y, color_dict=imdc_color_dict)
                self._readDraw("DG", x, y, color_dict=imdc_color_dict)

                self.row += 1

            def _readDraw(self, *args, **kwargs):
                plt.subplot(n_rows, n_columns, (self.row - 1) * n_columns + self.n)
                gdi.readDraw(*args, **kwargs)
                self.column_draw()
                self.n += 1

            def column_draw(self):
                ax = plt.gca()
                if self.row == 1:
                    ax.set_title(column_names[self.column - 1], fontdict={"size": FONT_SIZE})
                if self.column == 1:
                    ax.set_ylabel(self.name, rotation=0, labelpad=40, fontdict={"size": FONT_SIZE})
                self.column += 1

        column = draw_column()

        column.fit(r"  ", 120.32396, 36.35739)
        column.fit(r"(b) Beijing", 116.348373, 39.782519)
        column.fit(r"(c) Chengdu", 104.101211, 30.788077)
        column.fit(r"(d) Chengdu", 104.065650, 30.696051)

        plt.show()

    def func7():
        FONT_SIZE = 16

        win_size = (331, 331)
        gi_win_size = int(win_size[0] * 2640.0 / 121.0), int(win_size[1] * 2026.0 / 121.0)
        print("win_size", win_size)
        print("gi_win_size", gi_win_size)

        gdi = GDALDrawImage(win_size)

        qd = gdi.addGR(raster_fn=_QD_RASTER_FN_2, geo_range=_QD_RANGE_FN_2)
        bj = gdi.addGR(raster_fn=_BJ_RASTER_FN_2, geo_range=_BJ_RANGE_FN_2)
        cd = gdi.addGR(raster_fn=_CD_RASTER_FN_2, geo_range=_CD_RANGE_FN_2)

        gdi.addRCC("RGB", qd, bj, cd, channel_list=["Red", "Green", "Blue"])
        gdi.addRCC("NRG", qd, bj, cd, channel_list=["NIR", "Red", "Green"])

        add_imdc_grs(gdi)

        n_rows, n_columns = 1, 6
        n = 13
        fig = plt.figure(figsize=(n_columns / n_columns * n, n_rows / n_columns * n,), )
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.90, hspace=0.03, wspace=0.03)

        gdis = gdi.draws()

        gdis.add()

    def func8():

        def get_tif(*_dirnames):
            _fn_dict = {}
            for _dirname in _dirnames:
                _fns = filterFileEndWith(_dirname, ".tif")
                for _fn in _fns:
                    _name = getfilenamewithoutext(_fn)
                    _name = _name.split("-")[1][3:]
                    if _name not in _fn_dict:
                        _fn_dict[_name] = _fn
            return _fn_dict

        to_dict = {
            "qd": get_tif(
                r"F:\GraduationDesign\Result\QingDao\20250120H183522",
                r"F:\GraduationDesign\Result\QingDao\20250120H202054",
            ),
            "bj": get_tif(
                r"F:\GraduationDesign\Result\BeiJing\20250120H190546",
                r"F:\GraduationDesign\Result\BeiJing\20250120H202608",
            ),
            "cd": get_tif(
                r"F:\GraduationDesign\Result\ChengDu\20250120H193443",
                r"F:\GraduationDesign\Result\ChengDu\20250120H203155",
            ),
        }

        printDict("Qingdao", to_dict["qd"])
        printDict("Beijing", to_dict["bj"])
        printDict("Chengdu", to_dict["cd"])

        # saveJson(to_dict, r"F:\GraduationDesign\Result\results.json")

        to_dict1 = pd.DataFrame(to_dict).T.to_dict()
        print(to_dict1)

        return

    def func9():

        # plt.rcParams.update({'font.size': 16})
        # # plt.rcParams['font.family'] = ['SimSun', "Times New Roman", ] + plt.rcParams['font.family']
        # plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]
        # plt.rcParams['mathtext.fontset'] = 'stix'
        #
        # fig = plt.figure()
        #
        # axes = fig.add_subplot(1, 1, 1)
        #
        # # axes.text(0.2, 0.2, r"1 -> $ \mathrm{ \mu \alpha \tau \pi \lambda \omega  \iota \beta } $")
        # # axes.text(0.2, 0.4, r"$ \lim_{x \rightarrow \infty} \frac{1}{x} $")
        # # axes.text(0.2, 0.8, r"$ a \leq  b  \leq  c  \Rightarrow  a  \leq  c$")
        # # axes.text(0.4, 0.2, r"$ \sum_{i=1}^{\infty}\ x_i^3$")
        # # axes.text(0.4, 0.4, r"$ \sin(\frac{3\pi}{2}) = \cos(\pi)$")
        # # axes.text(0.4, 0.6, r"$ \sqrt[3]{x} = \sqrt{y}$")
        # # axes.text(0.6, 0.6, r"$ \neg (a \wedge b) \Leftrightarrow \neg a \ \vee \neg b$")
        # axes.text(0.6, 0.6, r" 这是一段中文 Italic  $ \rm  Italic $ $\neg (a \wedge b) \Leftrightarrow \neg a \ \vee \neg b$"
        #           , family=["SimSun", "Times New Roman", ])
        # # axes.text(0.6, 0.2, r"$ \int_a^b f(x)dx$")
        # # axes.text(0.6, 0.1, r'$ s(t) = \mathrm{A}\/\sin(2 \omega t) $')
        #
        # plt.text(
        #     0.5,
        #     0.8,
        #     r"中英文混编：This is English. 这是中文",
        #     fontdict=getFont(family=["Times New Roman", "SimSun"]),
        #     size=20,
        #     ha="center",
        #     va="center",
        # )

        FONT_SIZE = 14

        plt.rcParams.update({'font.size': FONT_SIZE})
        plt.rcParams['font.family'] = ["Times New Roman", 'SimSun', ] + plt.rcParams['font.family']

        win_size = (231, 231)
        gi_win_size = int(win_size[0] * 2640.0 / 121.0), int(win_size[1] * 2026.0 / 121.0)
        print("win_size", win_size)
        print("gi_win_size", gi_win_size)

        gdi = GDALDrawImage(win_size)

        qd = gdi.addGR(raster_fn=_QD_RASTER_FN_2, geo_range=_QD_RANGE_FN_2)
        bj = gdi.addGR(raster_fn=_BJ_RASTER_FN_2, geo_range=_BJ_RANGE_FN_2)
        cd = gdi.addGR(raster_fn=_CD_RASTER_FN_2, geo_range=_CD_RANGE_FN_2)
        gdi.addRCC("RGB", qd, bj, cd, channel_list=["Red", "Green", "Blue"])
        gdi.addRCC("NRG", qd, bj, cd, channel_list=["NIR", "Red", "Green"])
        to_dict = pd.DataFrame(readJson(r"F:\GraduationDesign\Result\results.json")).T.to_dict("list")
        for name in to_dict:
            gdi.addRCC(name, *to_dict[name], channel_list=[0], is_01=False, is_min_max=False, )
        print(gdi.keys())

        names = [
            'RGB', 'NRG',
            '1ADESI', '2AS', '3DE', '4ASC2', '5DEC2', '6ASH', '7DEH', '8ADESI_GLCM', '9ADESI_GLCM_OPT'
        ]

        column_names = ["S2", "ADESI-IS", "σ-AS-IS", "σ-DE-IS", "C2-AS-IS", "C2-DE-IS",
                        "H/α-AS-IS", "H/α-AS-IS", "ADESI-G-IS", "ADESI-GO-IS", ]

        imdc_color_dict = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255)}

        n_rows, n_columns = 6, len(column_names)
        n = 16
        n2 = 0.06
        fig = plt.figure(figsize=(n, n_rows / n_columns * n,), )
        fig.subplots_adjust(top=1 - n2, bottom=n2, left=n2, right=1 - n2, hspace=0.03, wspace=0.03)

        class draw_column:

            def __init__(self):
                self.row = 1
                self.column = 1
                self.name = ""
                self.n = 1

            def fit(self, name, x, y, ):
                self.column = 1
                self.name = name
                self.n = 1

                self._readDraw("NRG", x, y, min_list=[300, 300, 300], max_list=[2900, 1500, 1500])

                for _name in [
                    '1ADESI', '2AS', '3DE', '4ASC2', '5DEC2',
                    '6ASH', '7DEH', '8ADESI_GLCM', '9ADESI_GLCM_OPT'
                ]:
                    self._readDraw(_name, x, y, color_dict=imdc_color_dict)

                self.row += 1

            def _readDraw(self, *args, **kwargs):
                plt.subplot(n_rows, n_columns, (self.row - 1) * n_columns + self.n)
                gdi.readDraw(*args, **kwargs)
                self.column_draw()
                self.n += 1

            def column_draw(self):
                ax = plt.gca()
                if self.row == 1:
                    ax.set_title(column_names[self.column - 1], fontdict={"size": FONT_SIZE})
                if self.column == 1:
                    ax.set_ylabel(self.name, rotation=0, labelpad=10, fontdict={"size": FONT_SIZE})
                self.column += 1

        column = draw_column()

        column.fit(r"(1)", 120.32396, 36.35739)
        column.fit(r"(2)", 120.194134,36.301147)
        column.fit(r"(3)", 116.348373, 39.782519)
        column.fit(r"(4)", 116.701712,39.721282)
        column.fit(r"(5)", 104.101211, 30.788077)
        column.fit(r"(6)", 104.065650, 30.696051)

        plt.show()

    return func9()


class _SampleUpdate:

    def __init__(self, name, to_dirname):
        self.name = name
        self.to_dirname = to_dirname
        self.dfn = DirFileName(to_dirname)
        self.samples = []
        self.fit_cnames = []

    def readTxt(self, txt_fn):
        copyfile(txt_fn, txt_fn + "-back")
        df = splTxt2Dict(txt_fn)
        to_csv_fn = self.dfn.fn("{}.csv".format(self.name))
        savecsv(to_csv_fn, df)
        df = pd.read_csv(to_csv_fn)
        self.samples.extend([{
            "_CNAME": spl["CATEGORY_NAME"],
            **spl,
        } for spl in df.to_dict("records")])

    def sampleImdc(self, imdc_fns):
        x, y = [spl["X"] for spl in self.samples], [spl["Y"] for spl in self.samples]
        for name in imdc_fns:
            self.fit_cnames.append(name)
            fn = imdc_fns[name]
            gsf = GDALSamplingFast(fn)
            df_imdc = gsf.sampling(x, y)
            imdc_list = df_imdc[list(df_imdc.keys())[0]]
            for i in range(len(self.samples)):
                self.samples[i][name] = imdc_list[i]

    def get(self, name, map_dict=None):
        if map_dict is not None:
            return [map_dict[spl[name]] for spl in self.samples]
        else:
            return [spl[name] for spl in self.samples]

    def accuracy(self):
        y1 = self.get("_CNAME", _MAP_DICT)
        cm_names = ["IS", "VEG", "SOIL", "WAT"]
        to_list = []

        def kw(k, data, _dict):
            print("{}: {}".format(k, data))
            _dict[k] = data

        for name in self.fit_cnames:
            cm = ConfusionMatrix(class_names=cm_names)
            y2 = self.get(name)
            cm.addData(y1, y2)
            to_dict = {"NAME": name}
            oa_cm = cm.accuracyCategory("IS")
            print("# {} ------".format(name))
            print(cm.fmtCM(), )
            print(oa_cm.fmtCM(), )
            kw("IS_OA", oa_cm.OA(), to_dict)
            kw("IS_Kappa", oa_cm.getKappa(), to_dict)
            kw("IS_PA", oa_cm.PA("IS"), to_dict)
            kw("IS_UA", oa_cm.UA("IS"), to_dict)
            for cm_name in cm_names:
                kw("{}_PA".format(cm_name), cm.PA(cm_name), to_dict)
            for cm_name in cm_names:
                kw("{}_UA".format(cm_name), cm.UA(cm_name), to_dict)

            to_list.append(to_dict)
            print()

        to_df = pd.DataFrame(to_list).sort_values("IS_OA", ascending=False)
        print(tabulate(to_df, headers="keys"))

    def toCSV(self, to_csv_fn):
        pd.DataFrame(self.samples).to_csv(to_csv_fn, index=False)


def accuracy():
    def func1():
        csv_fn = r"F:\ProjectSet\Shadow\ASDEIndex\Samples\sh2_spl30_qd6.csv"
        df = pd.read_csv(csv_fn)
        df = pd.DataFrame(df[df["TEST"] == 0].to_dict("records"))
        print(df.head(6).T)
        df.to_csv(r"F:\SARIndex\Result\AccuracyUpdate\test_samples.csv", index=False)

    def func2():
        sample_update = _SampleUpdate("update1", r"F:\SARIndex\Result\AccuracyUpdate")
        sample_update.readTxt(r"F:\SARIndex\Result\AccuracyUpdate\test_samples2.txt")
        sample_update.sampleImdc({
            "OAD": r"F:\SARIndex\Result\qd-RF_OAD-RF_imdc.tif",
            "OA": r"F:\SARIndex\Result\qd-RF_OA-RF_imdc.tif",
            "OD": r"F:\SARIndex\Result\qd-RF_OD-RF_imdc.tif",
            "OE": r"F:\SARIndex\Result\qd-SVM_OE-SVM_imdc.tif",
            "O": r"F:\SARIndex\Result\qd-RF_O-RF_imdc.tif",
        })
        sample_update.accuracy()
        sample_update.toCSV(r"F:\SARIndex\Result\AccuracyUpdate\test_samples3.csv")

    return func2()


class _SI_AngleDiagram:

    def __init__(self):
        self.dfn = DirFileName(r"F:\ProjectSet\Shadow\ASDEIndex\AngleDiagram")
        self.angle_fn = self.dfn.fn("adsi_angle_qd.geojson")
        self.angle_spls_fn = self.dfn.fn("adsi_angle_spls_qd.geojson")
        self.data_fn = self.dfn.fn("adsi_data_qd.csv")
        self.data = None

    def read(self, angle_fn=None, angle_spls_fn=None, data_fn=None):
        if angle_fn is not None:
            self.angle_fn = angle_fn
        if angle_spls_fn is not None:
            self.angle_spls_fn = angle_spls_fn
        if data_fn is not None:
            self.data_fn = data_fn

        def _read_geojson(_fn):
            _data = readJson(_fn)
            _list = []
            for feat in _data["features"]:
                _list.append([feat["properties"]["CATEGORY"], feat["geometry"]["coordinates"]])
            return _list

        tlp = TableLinePrint()
        tlp.print("CATEGORY", "DX", "DY", "atan2", "angle")
        tlp.separationLine()

        angle_data = _read_geojson(self.angle_fn)
        angle_data_dict = {}
        for data in angle_data:
            dx = data[1][1][0] - data[1][0][0]
            dy = data[1][1][1] - data[1][0][1]
            if dy > 0:
                angle = math.atan2(abs(dx), abs(dy))
                angle_degrees = math.degrees(angle)
            else:
                angle = math.atan2(abs(dy), abs(dx))
                angle_degrees = math.degrees(angle)
                angle_degrees += 90
            tlp.print(data[0], dx, dy, angle, angle_degrees)
            angle_data_dict[data[0]] = angle_degrees
        printDict("angle_data_dict", angle_data_dict)

        spl_data_list = [
            *getImageDataFromGeoJson(self.angle_spls_fn, _RASTER_FN_1),
            *getImageDataFromGeoJson(self.angle_spls_fn, _BJ_RASTER_FN),
            *getImageDataFromGeoJson(self.angle_spls_fn, _CD_RASTER_FN),
        ]
        df = pd.DataFrame(spl_data_list)
        # df["E1"] = df["E1"] / (df["E1"].abs() ** 0.5)
        # df["E2"] = df["E2"] / (df["E2"].abs() ** 0.5)
        df["E1"] = df["E1"] / 2
        df["E2"] = df["E2"] / 2

        def _data_range(_field_name, _min, _max):
            df[_field_name] = (df[_field_name] - _min) / (_max - _min)

        # _data_range("AS1",  -39.886482, 24.9383)
        # _data_range("AS2", -36.196, 24.9383)

        # _data_range("DE1", -34.1167, 21.6045)
        # _data_range("DE2", -34.1167, 21.6045)

        # _data_range("E1", -39.9284/2, 49.9383/2)
        # _data_range("E2", -39.9284/2, 49.9383/2)

        def _data_range_2(_field_name, _mean, _std):
            df[_field_name] = (df[_field_name] - _mean) / _std

        _data_range("AS1", -10.030696, 6.388037)
        _data_range("AS2", -18.444008, 5.848238)

        _data_range("DE1", -12.106682, 7.100646)
        _data_range("DE2", -19.716860, 6.667820)

        _data_range("E1", -17.420959, 12.526972)
        _data_range("E2", -26.863699, 9.042142)

        to_df = df.groupby("CATEGORY").mean()
        to_df.to_csv(self.data_fn)
        to_data_dict = to_df.to_dict("index")
        for n in to_data_dict:
            to_data_dict[n]["ANGLE"] = angle_data_dict[n]
        to_df = pd.DataFrame(to_data_dict).T
        print(tabulate(to_df, headers="keys"))

        if self.data is None:
            self.data = to_df
        else:
            self.data = pd.DataFrame(self.data.to_dict("records") + to_df.to_dict("records"))

    def plot(self):

        plt.rc('font', family='Times New Roman')
        plt.rcParams.update({'font.size': 16})

        fig = plt.figure(figsize=(12, 9))
        fig.subplots_adjust(top=0.983, bottom=0.143, left=0.087, right=0.648, hspace=0.396, wspace=0.203)

        angle_split = 0.5
        is_show_1 = True
        is_line = False
        is_fft = False
        is_gauss = True

        def _plot(_field_name, marker, label, color):
            _data = self.data[["ANGLE", _field_name]]
            _x = [i * angle_split for i in range(0, int(180 / angle_split))]
            _y = []
            for i in range(len(_x) - 1):
                _y_list = []
                for j in _data.index:
                    if (_data["ANGLE"][j] < 55) or (_data["ANGLE"][j] > 115):
                        continue
                    if _x[i] < _data["ANGLE"][j] <= _x[i + 1]:
                        _y_list.append(float(_data[_field_name][j]))
                if _y_list:
                    _y.append(np.mean(_y_list))
                else:
                    _y.append(None)

            _x_show = [_x[i] + angle_split / 2.0 for i in range(len(_y)) if _y[i] is not None]
            _y_show = [_y[i] for i in range(len(_y)) if _y[i] is not None]

            if is_show_1:
                plt.scatter(_x_show, _y_show, marker=marker,
                            edgecolors="black", color=color, s=50, label=label)

            if is_fft:
                y_fft = np.fft.fft(_y_show)
                threshold = np.sort(np.abs(np.real(y_fft)))[-5]
                y_filtered = np.where(np.abs(np.real(y_fft)) > threshold, y_fft, 0)
                y_reconstructed = np.real(np.fft.ifft(y_filtered))
                plt.plot(_x_show, y_reconstructed, color=color, label=label + " Line")

            if is_gauss:
                y_gauss = gaussian_filter1d(_y_show, 2.5)
                plt.plot(_x_show, y_gauss, color=color, label=label + " Line")

            if is_line:
                an = np.polyfit(_x_show, _y_show, 2)  # 用3次多项式拟合
                # 如果源数据点不够要自己扩充，否则直接使用源数据点即可
                x1 = np.arange(np.min(_x_show), np.max(_x_show), 0.1)  # 画曲线用的数据点
                yvals = np.polyval(an, x1)  # 根据多项式系数计算拟合后的值

                plt.plot(x1, yvals, color=color, label=label + " Line")

        def _as_de_line():
            plt.plot([90 - 8.18, 90 - 8.18], [-0.1, 1.1], "--", color="black")
            plt.plot([90 + 8.18, 90 + 8.18], [-0.1, 1.1], "--", color="black")

        plt.subplot(211)
        _plot("AS1", "o", "Ascending-VV", "red")
        _plot("DE1", "^", "Descending-VV", "lightgreen")
        _plot("E1", "s", "ADESI-1", "blue")
        plt.xlim([60, 120])
        plt.ylim([0, 0.8])
        plt.xlabel("Angle (°)")
        plt.ylabel("Value")
        plt.xticks([60, 70, 80, 90 - 8.18, 90, 90 + 8.18, 100, 110, 120], rotation=45)
        plt.legend(
            # facecolor="lightgray", edgecolor="black",
            frameon=False,
            # loc='upper center', bbox_to_anchor=(-0.1, -0.15), ncol=6,
            loc='center left', bbox_to_anchor=(1.05, 0.2), ncol=1,
        )
        _as_de_line()

        plt.subplot(212)
        _plot("AS2", "o", "Ascending-VH", "red")
        _plot("DE2", "^", "Descending-VH", "lightgreen")
        _plot("E2", "s", "ADESI-2", "blue")
        plt.xlim([60, 120])
        plt.ylim([0, 0.7])
        plt.xlabel("Angle (°)")
        plt.ylabel("Value")
        plt.xticks([60, 70, 80, 90 - 8.18, 90, 90 + 8.18, 100, 110, 120], rotation=45)
        plt.legend(
            # facecolor="lightgray", edgecolor="black",
            frameon=False,
            # loc='upper center', bbox_to_anchor=(-0.1, -0.15), ncol=6,
            loc='center left', bbox_to_anchor=(1.05, 0.2), ncol=1,
        )
        _as_de_line()

    def showData(self):
        print(tabulate(self.data, headers="keys"))


def angleDiagram_Funcs():
    def func1():
        si_ad = _SI_AngleDiagram()
        si_ad.read(
            angle_fn=si_ad.dfn.fn("adsi_angle_qd.geojson"),
            angle_spls_fn=si_ad.dfn.fn("adsi_angle_spls_qd.geojson"),
            data_fn=si_ad.dfn.fn("adsi_data_qd.csv"),
        )
        si_ad.read(
            angle_fn=si_ad.dfn.fn("adsi_angle_bj.geojson"),
            angle_spls_fn=si_ad.dfn.fn("adsi_angle_spls_bj.geojson"),
            data_fn=si_ad.dfn.fn("adsi_data_bj.csv"),
        )
        si_ad.read(
            angle_fn=si_ad.dfn.fn("adsi_angle_cd.geojson"),
            angle_spls_fn=si_ad.dfn.fn("adsi_angle_spls_cd.geojson"),
            data_fn=si_ad.dfn.fn("adsi_data_cd.csv"),
        )
        si_ad.showData()
        si_ad.plot()
        plt.show()

        return

    return func1()


def funcs():
    def func1():
        gr = GDALRaster(_CD_RASTER_FN)
        range_dict = readJson(_CD_RANGE_FN)

        def _range_func(_data, _x_min, _x_max):
            _data = np.clip(_data, _x_min, _x_max)
            return (_data - _x_min) / (_x_max - _x_min)

        data_e1 = _range_func(gr.readGDALBand("E1"), range_dict["E1"]["min"], range_dict["E1"]["max"], )
        print(range_dict["E1"])

        data_e2 = _range_func(gr.readGDALBand("E2"), range_dict["E2"]["min"], range_dict["E2"]["max"], )
        print(range_dict["E2"])

        dfn = DirFileName(r"F:\ProjectSet\Shadow\ASDEIndex\Images\glcm")

        gr.save(data_e1.astype("float32"), dfn.fn("cd_e1"), dtype=gdal.GDT_Float32)
        gr.save(data_e2.astype("float32"), dfn.fn("cd_e2"), dtype=gdal.GDT_Float32)

    def func2():

        def func21(
                raster_fn=_BJ_ENVI_FN,
                raster_fn_1=_BJ_RASTER_FN,
                e1_glcm_fn=r"H:\SARIndex\GLCM\bj_e1_glcm",
                e2_glcm_fn=r"H:\SARIndex\GLCM\bj_e2_glcm",
                to_raster_fn=r"F:\ProjectSet\Shadow\ASDEIndex\Images\BJ_SI_BS_2.dat",
        ):
            gr = GDALRaster(raster_fn)
            gr1 = GDALRaster(raster_fn_1)
            gr_e1_glcm = GDALRaster(e1_glcm_fn)
            gr_e2_glcm = GDALRaster(e2_glcm_fn)

            to_names = []
            data = np.zeros((71, gr.n_rows, gr.n_columns), dtype="float32")

            tlp = TableLinePrint()
            tlp.print("No.", "NAME", "MIN", "MAX")
            tlp.separationLine()

            def read_data(_gr, *_names):
                for _name in _names:
                    to_names.append(_name)
                    _n = len(to_names) - 1
                    data[_n] = _gr.readGDALBand(_name)
                    tlp.print(_n, _name, float(data[_n].min()), float(data[_n].max()), )

            read_data(gr, "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "NDVI", "NDWI", )

            to_names.append("MNDWI")
            n = len(to_names) - 1
            data[n] = (data[5] - data[1]) / (data[5] + data[1] + 0.000001)
            tlp.print(n, "MNDWI", float(data[n].min()), float(data[n].max()), )

            read_data(
                gr,
                "OPT_asm", "OPT_con", "OPT_cor", "OPT_dis", "OPT_ent", "OPT_hom", "OPT_mean", "OPT_var",
                "AS_VV", "AS_VH",
                "AS_VH_asm", "AS_VH_con", "AS_VH_cor", "AS_VH_dis", "AS_VH_ent", "AS_VH_hom", "AS_VH_mean", "AS_VH_var",
                "AS_VV_asm", "AS_VV_con", "AS_VV_cor", "AS_VV_dis", "AS_VV_ent", "AS_VV_hom", "AS_VV_mean", "AS_VV_var",
                "DE_VV", "DE_VH",
                "DE_VH_asm", "DE_VH_con", "DE_VH_cor", "DE_VH_dis", "DE_VH_ent", "DE_VH_hom", "DE_VH_mean", "DE_VH_var",
                "DE_VV_asm", "DE_VV_con", "DE_VV_cor", "DE_VV_dis", "DE_VV_ent", "DE_VV_hom", "DE_VV_mean", "DE_VV_var",
            )

            read_data(gr1, "E1", "E2")
            read_data(gr_e1_glcm, "E1_mean", "E1_var", "E1_hom", "E1_con", "E1_dis", "E1_ent", "E1_asm", "E1_cor", )
            read_data(gr_e2_glcm, "E2_mean", "E2_var", "E2_hom", "E2_con", "E2_dis", "E2_ent", "E2_asm", "E2_cor", )

            print(len(to_names))
            printList("to_names", to_names)

            gr.save(data, to_raster_fn, dtype=gdal.GDT_Float32, descriptions=to_names)

        func21(
            raster_fn=_QD_ENVI_FN,
            raster_fn_1=_RASTER_FN_1,
            e1_glcm_fn=r"H:\SARIndex\GLCM\qd_e1_glcm",
            e2_glcm_fn=r"H:\SARIndex\GLCM\qd_e2_glcm",
            to_raster_fn=r"F:\ProjectSet\Shadow\ASDEIndex\Images\QD_SI_BS_2.dat",
        )

        func21(
            raster_fn=_CD_ENVI_FN,
            raster_fn_1=_CD_RASTER_FN,
            e1_glcm_fn=r"H:\SARIndex\GLCM\cd_e1_glcm",
            e2_glcm_fn=r"H:\SARIndex\GLCM\cd_e2_glcm",
            to_raster_fn=r"F:\ProjectSet\Shadow\ASDEIndex\Images\CD_SI_BS_2.dat",
        )

    def func3():
        dfn = DirFileName(r"H:\SARIndex\GLCM")
        glcm_list = ["mean", "var", "hom", "con", "dis", "ent", "asm", "cor", ]

        tlp = TableLinePrint()
        tlp.print("NAME", "N", "MIN", "MAX")

        def cat_fn(_fn, _name):
            _fn = dfn.fn(_fn)
            to_data = None
            grs = [GDALRaster(_fn.format(_data)) for _data in ["01", "10", "11", "1_1"]]
            tlp.separationLine()
            for i in range(1, 9):
                datas = np.concatenate([[gr.readGDALBand(i)] for gr in grs])
                datas = np.mean(datas, axis=0)
                if to_data is None:
                    to_data = np.zeros((8, datas.shape[0], datas.shape[1]), dtype="float32")
                to_data[i - 1] = datas
                tlp.print(os.path.basename(_fn).format(_name), i, float(np.min(datas)), float(np.max(datas)))
            _to_fn = _fn.format("glcm")
            grs[0].save(
                to_data, _to_fn, dtype=gdal.GDT_Float32,
                descriptions=["{}_{}".format(_name, glcm_name) for glcm_name in glcm_list]
            )

        # cat_fn("bj_e1_{}", "E1")
        # cat_fn("bj_e2_{}", "E2")
        cat_fn("cd_e1_{}", "E1")
        cat_fn("cd_e2_{}", "E2")
        # cat_fn("qd_e1_{}", "E1")
        # cat_fn("qd_e2_{}", "E2")

    def func4():
        getRange(_QD_RASTER_FN_2)
        getRange(_BJ_RASTER_FN_2)
        getRange(_CD_RASTER_FN_2)

    def func5():

        class _accuracy:

            def __init__(self, _city_name):
                self.city_name = _city_name
                self.acc = None
                self.cm = {}

            def add(self, _dirname):
                _log_fn = os.path.join(_dirname, "log.txt")
                with open(_log_fn, "r", encoding="utf-8") as f:
                    is_cm = False
                    name = None
                    for line in f:
                        if line.startswith("#") and line.endswith("#\n"):
                            if name is not None:
                                if self.cm[name] == "":
                                    self.cm.pop(name)
                            name = " ".join(line.split(" ")[2:-3])
                            self.cm[name] = ""

                        if is_cm:
                            if not line.startswith(" "):
                                is_cm = False
                            else:
                                self.cm[name] += line
                        else:
                            if line.strip() == "IS_CM:":
                                is_cm = True

                acc = pd.read_csv(os.path.join(_dirname, "accuracy"), index_col=0).T.to_dict("dict")
                if self.acc is None:
                    self.acc = acc
                    return
                for name in acc:
                    self.acc[name] = acc[name]

            def show_acc(self):
                print(self.city_name)
                print(pd.DataFrame(self.acc).T)
                print()

            def show_cm(self):
                for name in self.cm:
                    print(name)
                    print(self.cm[name])
                    print()

        qd = _accuracy("qd")
        qd.add(r"F:\GraduationDesign\Result\QingDao\20250120H183522")
        qd.add(r"F:\GraduationDesign\Result\QingDao\20250120H202054")
        qd.show_cm()

        bj = _accuracy("bj")
        bj.add(r"F:\GraduationDesign\Result\BeiJing\20250120H190546")
        bj.add(r"F:\GraduationDesign\Result\BeiJing\20250120H202608")
        bj.show_cm()

        cd = _accuracy("cd")
        cd.add(r"F:\GraduationDesign\Result\ChengDu\20250120H193443")
        cd.add(r"F:\GraduationDesign\Result\ChengDu\20250120H203155")
        cd.show_cm()

    def func6():
        csv_fn = r"F:\GraduationDesign\Result\cm.csv"
        cms = {}

        def is_string_identical(text_string):
            i = 0
            while i < len(text_string):
                if text_string[i] == text_string[i + 1]:
                    i += 1
                    return True
                else:
                    return False

        with open(csv_fn, "r", encoding="utf-8") as f:
            is_cm = False
            for line in f:
                line = line.strip()

                if is_cm:
                    if not is_string_identical(line):
                        lines = line.split(",")[1:]
                        print(lines)
                        cms[city_name][name].append(lines)

                if not line.startswith(","):
                    city_name, _, name = tuple(line.split(",")[:3])
                    if city_name not in cms:
                        cms[city_name] = {}
                    cms[city_name][name] = []
                    print("#", city_name, name, "-" * 60)
                    is_cm = True

                if is_string_identical(line):
                    is_cm = False

        cms.pop('\ufeff')

        for city_name in cms:
            for name in cms[city_name]:
                lines = cms[city_name][name]
                for i in range(1, len(lines) - 1):
                    for j in range(1, len(lines[i]) - 1):
                        lines[i][j] = int(lines[i][j])

        for city_name in cms:
            for name in cms[city_name]:
                lines = cms[city_name][name]
                for i in range(1, len(lines)):
                    lines[i][-1] = "{:.2f}%".format(float(lines[i][-1]))
                for i in range(1, len(lines[-1]) - 1):
                    lines[-1][i] = "{:.2f}%".format(float(lines[-1][i]))

        for city_name in cms:
            for name in cms[city_name]:
                print("#", city_name, name, "-" * 60)
                lines = cms[city_name][name]
                print(*lines, sep="\n")

        with open(r"F:\GraduationDesign\Result\cm2.csv", "w", encoding="utf-8", newline="") as fr:
            cw = csv.writer(fr)

            for city_name in cms:
                to_lines = [[] for i in range(7)]
                for name in cms[city_name]:
                    print(name)
                    to_lines[0].append(name)
                    to_lines[0].extend([" " for i in range(5)])
                    for i in range(1, 7):
                        to_lines[i].extend(cms[city_name][name][i - 1])

                for line in to_lines:
                    cw.writerow(line)
                fr.write("\n")

    return func6()


class _MakeADSI(SamplesUtil):

    def __init__(self):
        super(_MakeADSI, self).__init__()
        self.images_fns = readJson(r"G:\SHImages\image.json")

    def samplingName(self, name, _func=None, _filters_and=None, _filters_or=None,
                     x_field_name="X", y_field_name="Y", is_jdt=True):
        return self.sampling1(name, self.images_fns[name], _func=_func,
                              _filters_and=_filters_and, _filters_or=_filters_or,
                              x_field_name=x_field_name, y_field_name=y_field_name, is_jdt=is_jdt)

    def samplingNames(self, *names, _func=None, _filters_and=None, _filters_or=None,
                      x_field_name="X", y_field_name="Y", is_jdt=True):
        for name in names:
            self.samplingName(name, _func=_func,
                              _filters_and=_filters_and, _filters_or=_filters_or,
                              x_field_name=x_field_name, y_field_name=y_field_name, is_jdt=is_jdt)


def makeADSI():
    def func1():
        plt.rc('font', family='Times New Roman')
        plt.rcParams.update({'font.size': 9})

        dfn = DirFileName(r"F:\ProjectSet\Shadow\ASDEIndex\MakeIndex")
        gsu = _MakeADSI()
        gsu.addCSV(dfn.fn(r"adsi_makeindex_spls4.csv"), field_datas={"CNAME": "IS", "CITY": "qd"})
        gsu.addQJY(dfn.fn(r"adsi_makeindex_spls_qd1.txt"), field_datas={"CITY": "qd"})
        gsu.addQJY(dfn.fn(r"adsi_makeindex_spls_bj1.txt"), field_datas={"CITY": "bj"})
        gsu.addCSV(dfn.fn(r"adsi_spl_cd1.csv"), field_datas={"CITY": "cd"})

        gsu.samplingNames(
            "E1", "E2", "AS_H", "AS_Alpha", "DE_H", "DE_Alpha",
            "AS_VV", "AS_VH", "DE_VV", "DE_VH",
            "AS_C11", "AS_C22", "DE_C11", "DE_C22",
        )

        df = gsu.toDF()
        print(df.value_counts("CNAME"))
        print(df.keys())

        def to_01():
            for name in ["E1", "E2", "AS_H", "AS_Alpha", "DE_H", "DE_Alpha",
                         "AS_VV", "AS_VH", "DE_VV", "DE_VH",
                         "AS_C11", "AS_C22", "DE_C11", "DE_C22", ]:
                df[name] = (df[name] - df[name].mean()) / df[name].std()
                df[name] = (df[name] - df[name].min()) / (df[name].max() - df[name].min())

        to_01()

        # df["E1"] = df["E1"] / 2
        # df["E2"] = df["E2"] / 2

        def _get_data_1(_cname, _fields=None):
            _df = df[df["CNAME"] == _cname]
            if _fields is not None:
                return _df[_fields]
            else:
                return _df

        def _hist(_cname, _field_name, _color, _name=None):
            _data = _get_data_1(_cname, _field_name).values
            y, x = np.histogram(_data, density=True, bins=50)
            for i in range(1, len(y)):
                if y[i] == 0:
                    y[i] = y[i - 1]
            if _name is None:
                _name = "{}".format(_cname, _field_name)
            plt.plot(x[:-1], y, color=_color, label=_name)

        def _hist_1(_field_name, title_fmt="{}"):
            _hist("IS", _field_name, "darkred")
            _hist("VEG", _field_name, "green")
            _hist("SOIL", _field_name, "yellow")
            _hist("WAT", _field_name, "blue")
            plt.title(title_fmt.format(_field_name))
            # if _field_name == "E2":
            #     plt.ylim([0, 0.30])

        def _hist_2(_field_name1, _field_name2):
            _field_names = [
                "E1", "AS_{}".format(_field_name1), "DE_{}".format(_field_name1),
                "E2", "AS_{}".format(_field_name2), "DE_{}".format(_field_name2),
            ]
            plt.figure(figsize=(10, 6))
            for i in range(1, 7):
                plt.subplot(230 + i)
                plt.subplots_adjust(top=0.95, bottom=0.055, left=0.046, right=0.988, hspace=0.208, wspace=0.194)
                _hist_1(_field_names[i - 1])
            plt.legend()
            plt.show()

        def _hist_3():
            def _get_field_names(_field_name1, _field_name2):
                return [
                    "AS_{}".format(_field_name1), "DE_{}".format(_field_name1),
                    "AS_{}".format(_field_name2), "DE_{}".format(_field_name2),
                ]

            _field_names = [
                "E1", "E2",
                *_get_field_names("VV", "VH"),
                *_get_field_names("C11", "C22"),
                *_get_field_names("H", "Alpha"),
            ]
            plt.figure(figsize=(9, 9))
            plt.subplots_adjust(top=0.964, bottom=0.059, left=0.059, right=0.982, hspace=0.482, wspace=0.339)

            xuhao = "abcdefghijklmnopqrstuvwxyz"

            for i in range(1, len(_field_names) + 1):
                plt.subplot(4, 4, i)
                _hist_1(_field_names[i - 1], title_fmt="({})".format(xuhao[i - 1]) + " {}")
                plt.xlabel("Value")
                plt.ylabel("Frequency(%)")

            plt.legend(
                frameon=False,
                # loc='upper center', bbox_to_anchor=(-0.1, -0.15), ncol=6,
                loc='center left', bbox_to_anchor=(1.1, 0.2), ncol=1,
            )
            plt.savefig(dfn.fn("sar_index.jpg", is_show=True), dpi=300)
            plt.show()

        # _hist_2("VV", "VH")
        # _hist_2("C11", "C22")
        # _hist_2("H", "Alpha")

        _hist_3()

        df = df[[
            "SRT", "X", "Y", "CITY", "CNAME",
            "E1", "E2", "AS_H", "AS_Alpha", "DE_H", "DE_Alpha",
            "AS_VV", "AS_VH", "DE_VV", "DE_VH",
            "AS_C11", "AS_C22", "DE_C11", "DE_C22",
        ]]

        df_spls = df.to_dict("records")
        to_list = []
        for spl in df_spls:
            if spl["CNAME"] in ["IS", "VEG", "SOIL", "WAT"]:
                to_list.append(spl)

        df = pd.DataFrame(to_list)
        df.to_csv(r"F:\ProjectSet\Shadow\ASDEIndex\MakeIndex\save_samples3.csv", index=False)

        vLookUpCount(df, "CITY", "CNAME", is_print=True)

        return

    def func2():
        tlp = TableLinePrint().firstLine("CITY", "FIELD", "MIN", "MAX")

        def _read(_raster_fn, to_name):
            gr = GDALRaster(_raster_fn)
            for name in gr.names:

                fn = r"G:\SHImages\{}_{}.tif".format(to_name, name)
                if not os.path.isfile(fn):
                    # print("gdaladdo ", fn)
                    data = gr.readGDALBand(name)
                    tlp.print(to_name, name, data.min(), data.max())
                    # gr.save(data, fn, fmt="GTiff", dtype=gdal.GDT_Float32, options=["COMPRESS=LZW"])
            tlp.separationLine()

        # _read(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\SHH2_QD2_envi.dat", "QD")
        # _read(r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\SHH2_BJ2_envi.dat", "BJ")
        # _read(r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\SHH2_CD2_envi.dat", "CD")

        _read(r"F:\ProjectSet\Shadow\ASDEIndex\Images\QD_SI_BS_2.dat", "QD")
        _read(r"F:\ProjectSet\Shadow\ASDEIndex\Images\BJ_SI_BS_2.dat", "BJ")
        _read(r"F:\ProjectSet\Shadow\ASDEIndex\Images\CD_SI_BS_2.dat", "CD")

    def func3():
        dirname = r"G:\SHImages"
        fns = {}
        for fn in os.listdir(dirname):
            if not fn.endswith(".tif"):
                continue
            name = fn.split("_", 1)[1].split(".")[0]
            if name not in fns:
                fns[name] = []
            fns[name].append(os.path.join(dirname, fn))
        saveJson(fns, os.path.join(dirname, "image.json"))

    def func4():
        df = pd.read_csv(r"F:\ProjectSet\Shadow\ASDEIndex\MakeIndex\adsi_makeindex_spls_cd2.csv")
        x, y = SRTSampleSelect(x=df, y=df["CATEGORY"].tolist()).get({1: 253, 2: 223, 3: 150, 4: 153})
        vLookUpCount(x, "CNAME", is_print=True)
        x.to_csv(r"F:\ProjectSet\Shadow\ASDEIndex\MakeIndex\adsi_spl_cd1.csv", index=False)

    def func5():
        df = pd.read_csv(r"F:\ProjectSet\Shadow\ASDEIndex\MakeIndex\save_samples.csv")

        def _show(_city_name, _city_name_2):
            _df = df[df["CITY"] == _city_name]

            def _show_1(_cname, _color):
                _df_1 = _df[_df["CNAME"] == _cname]
                plt.scatter(_df_1["X"], _df_1["Y"], c=_color, label=_cname, edgecolors="black", s=20)

            plt.gca().set_aspect('equal', adjustable='box')
            plt.title(_city_name_2)
            _show_1("IS", "red")
            _show_1("VEG", "green")
            _show_1("SOIL", "yellow")
            _show_1("WAT", "blue")

        plt.figure(figsize=(15, 6))
        plt.subplots_adjust(top=0.941, bottom=0.06, left=0.043, right=0.965, hspace=0.2, wspace=0.15)
        plt.subplot(131)
        _show("qd", "Qingdao")
        plt.subplot(132)
        _show("bj", "Beijing")
        plt.subplot(133)
        _show("cd", "Chengdu")

        plt.legend(bbox_to_anchor=(1.25, 0), loc="lower right", frameon=False)
        plt.show()

        df_spls = df.to_dict("records")
        to_list = []
        for spl in df_spls:
            if spl["CNAME"] in ["IS", "VEG", "SOIL", "WAT"]:
                to_list.append(spl)

        df = pd.DataFrame(to_list)
        df.to_csv(r"F:\ProjectSet\Shadow\ASDEIndex\MakeIndex\save_samples2.csv", index=False)

        vLookUpCount(df, "CITY", "CNAME", is_print=True)

    return func1()


if __name__ == "__main__":
    draw()
    r"""
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.ShadowSARIndex import getRange; getRange()"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.ShadowSARIndex import run; run('qd')"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.ShadowSARIndex import run; run('bj')"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.ShadowSARIndex import run; run('cd')"


python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.ShadowSARIndex import funcs; funcs()"
    """
