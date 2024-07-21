# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowASDEHSamples.py
@Time    : 2024/7/15 10:45
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of ShadowASDEHSamples

六个实验

feature

opt


-----------------------------------------------------------------------------"""
import os
import random
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal
from tabulate import tabulate

from RUN.RUNFucs import df2SplTxt, splTxt2Dict
from SRTCodes.GDALDraw import GDALDrawImages
from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.GDALUtils import GDALSamplingFast, RasterToVRTS, dictRasterToVRT, RasterRandomCoors
from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.NumpyUtils import update10EDivide10, eig2
from SRTCodes.SRTModel import SamplesData, MLModel, DataScale, SVM_RGS, RF_RGS, FeatFuncs
from SRTCodes.SRTSample import samplesDescription
from SRTCodes.SRTTimeDirectory import TimeDirectory
from SRTCodes.Utils import saveJson, changext, mkdir, DirFileName, getfilenamewithoutext, printDict, \
    readJson, SRTWriteText, timeStringNow, timeDirName, readLines, savecsv, filterFileEndWith, printList
from Shadow.Hierarchical.SHH2Config import FEAT_NAMES_CLS
from Shadow.ShadowMainBeiJing import bjFeatureDeal
from Shadow.ShadowMainChengDu import cdFeatureDeal
from Shadow.ShadowMainQingDao import qdFeatureDeal

_MODEL_DIRNAME = r"F:\ProjectSet\Shadow\ASDEHSamples\Models"

_BJ_RASTER_FN = r"F:\ProjectSet\Shadow\ASDEHSamples\Images\BeiJing\HSPL_BJ_envi.dat"
_CD_RASTER_FN = r"F:\ProjectSet\Shadow\ASDEHSamples\Images\ChengDu\HSPL_CD_envi.dat"
_QD_RASTER_FN = r"F:\ProjectSet\Shadow\ASDEHSamples\Images\QingDao\HSPL_QD_envi.dat"

_BJ_RANGE_FN = r"F:\ProjectSet\Shadow\ASDEHSamples\Images\BeiJing\HSPL_BJ.range"
_CD_RANGE_FN = r"F:\ProjectSet\Shadow\ASDEHSamples\Images\ChengDu\HSPL_CD.range"
_QD_RANGE_FN = r"F:\ProjectSet\Shadow\ASDEHSamples\Images\QingDao\HSPL_QD.range"

_BJ_OSPL_FN = r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\BeiJing\sh_bj_sample.csv"
_CD_OSPL_FN = r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\ChengDu\sh_cd_sample.csv"
_QD_OSPL_FN = r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\QingDao\sh_qd_sample.csv"

_BJ_SPL_FN = r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\HSPL_BJ.csv"
_CD_SPL_FN = r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\HSPL_CD.csv"
_QD_SPL_FN = r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\HSPL_QD.csv"

_SAMPLES_DFN = DirFileName(r"F:\ProjectSet\Shadow\ASDEHSamples\Samples")
_QD_SAMPLES_DFN = DirFileName(r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\QingDao")
_BJ_SAMPLES_DFN = DirFileName(r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\BeiJing")
_CD_SAMPLES_DFN = DirFileName(r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\ChengDu")

_CATEGORY_NAMES = ["IS", "VEG", "SOIL", "WAT"]

_CITY_NAMES_FULL = ["QingDao", "BeiJing", "ChengDu"]

_MODEL_LOG_DIRNAME = r"F:\ProjectSet\Shadow\ASDEHSamples\Models\Logs"
_MODEL_DFN = DirFileName(_MODEL_DIRNAME)

_C_NAME_CODE_MAP_DICT = {
    "NOT_KNOW": 0,
    "IS": 11, "IS_SH": 12, "IS_AS_SH": 13, "IS_DE_SH": 14,
    "VEG": 21, "VEG_SH": 22, "VEG_AS_SH": 23, "VEG_DE_SH": 24,
    "SOIL": 31, "SOIL_SH": 32, "SOIL_AS_SH": 33, "SOIL_DE_SH": 34,
    "WAT": 41, "WAT_SH": 42, "WAT_AS_SH": 43, "WAT_DE_SH": 44,
}

_C_CODE_NAME_MAP_DICT = {
    0: "NOT_KNOW",
    11: "IS", 12: "IS_SH", 13: "IS_AS_SH", 14: "IS_DE_SH",
    21: "VEG", 22: "VEG_SH", 23: "VEG_AS_SH", 24: "VEG_DE_SH",
    31: "SOIL", 32: "SOIL_SH", 33: "SOIL_AS_SH", 34: "SOIL_DE_SH",
    41: "WAT", 42: "WAT_SH", 43: "WAT_AS_SH", 44: "WAT_DE_SH"
}

_CATEGORY_NAMES_4 = [
    'IS', 'IS_SH', 'IS_AS_SH', 'IS_DE_SH',
    'VEG', 'VEG_SH', 'VEG_AS_SH', 'VEG_DE_SH',
    'SOIL', 'SOIL_SH', 'SOIL_AS_SH', 'SOIL_DE_SH',
    'WAT', 'WAT_SH', 'WAT_AS_SH', 'WAT_DE_SH'
]


def _sampleCSV(raster_fn, csv_fn, is_spl, to_fn=None):
    if to_fn is None:
        to_fn = changext(csv_fn, "_spl.csv")
    if os.path.isfile(to_fn) and (not is_spl):
        return to_fn
    GDALSamplingFast(raster_fn).csvfile(csv_fn, to_fn)
    return to_fn


_BJ_SPLING_FN = lambda is_spl=False: _sampleCSV(_BJ_RASTER_FN, _BJ_SPL_FN, is_spl)
_CD_SPLING_FN = lambda is_spl=False: _sampleCSV(_CD_RASTER_FN, _CD_SPL_FN, is_spl)
_QD_SPLING_FN = lambda is_spl=False: _sampleCSV(_QD_RASTER_FN, _QD_SPL_FN, is_spl)


def _CITY_NAME_GET(city_name, qd, bj, cd):
    if city_name == "qd":
        return qd
    if city_name == "bj":
        return bj
    if city_name == "cd":
        return cd


def _SPLING_FN(city_name):
    if city_name == "qd":
        return _QD_SPLING_FN()
    if city_name == "bj":
        return _BJ_SPLING_FN()
    if city_name == "cd":
        return _CD_SPLING_FN()
    return None


def _RANGE_FN(city_name):
    return _CITY_NAME_GET(city_name, _QD_RANGE_FN, _BJ_RANGE_FN, _CD_RANGE_FN)


def _RASTER_FN(city_name):
    return _CITY_NAME_GET(city_name, _QD_RASTER_FN, _BJ_RASTER_FN, _CD_RASTER_FN)


_FEAT_NAMES = FEAT_NAMES_CLS()


def _10EDivide10(x):
    return np.power(10, x / 10)


def _10Log10(x):
    return 10 * np.log10(x + 0.000001)


def _COM(data1, f, data2):
    if f == "==":
        return data1 == data2
    if f == ">":
        return data1 > data2
    if f == "<":
        return data1 < data2
    if f == ">=":
        return data1 >= data2
    if f == "<=":
        return data1 <= data2
    raise Exception("{}".format(f))


def _DICT_FILTER_AND(_dict, *_filters):
    for name, f, data in _filters:
        if not _COM(_dict[name], f, data):
            return False
    return True


def featFuncs(_type="10EDivide10"):
    ff = FeatFuncs()
    if _type is None:
        return ff

    def func1(func):
        for name in [
            "AS_VV", "AS_VH", "AS_C11", "AS_VHDVV", "AS_C22", "AS_Lambda1", "AS_Lambda2", "AS_SPAN", "AS_Epsilon",
            "DE_VV", "DE_VH", "DE_C11", "DE_VHDVV", "DE_C22", "DE_Lambda1", "DE_Lambda2", "DE_SPAN", "DE_Epsilon",
        ]:
            ff.add(name, func)

    if _type == "10EDivide10":
        func1(_10EDivide10)
    elif _type == "10Log10":
        func1(_10Log10)

    return ff


def sampleTypes(df):
    def func1():
        df_list = df.to_dict("records")
        free_df_list = []
        opt_df_list = []
        sar_df_list = []
        for line in df_list:
            if line["TEST"] == 0:
                free_df_list.append(line.copy())
                opt_df_list.append(line.copy())
                sar_df_list.append(line.copy())
            else:
                cname = line["CNAME"]
                if cname in ["IS", "VEG", "SOIL", "WAT"]:
                    free_df_list.append(line.copy())
                    opt_df_list.append(line.copy())
                    sar_df_list.append(line.copy())
                elif cname in ["IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH"]:
                    opt_df_list.append(line.copy())
                    sar_df_list.append(line.copy())
                else:
                    sar_df_list.append(line.copy())
        return pd.DataFrame(free_df_list), pd.DataFrame(opt_df_list), pd.DataFrame(sar_df_list)

    def func2():
        test_df_list = df[df["TEST"] == 0].to_dict("records")
        free_df_list = df[df["NS"] == 1].to_dict("records") + test_df_list
        opt_df_list = df[df["OS"] == 1].to_dict("records") + test_df_list
        sar_df_list = df[df["HS"] == 1].to_dict("records") + test_df_list
        return pd.DataFrame(free_df_list), pd.DataFrame(opt_df_list), pd.DataFrame(sar_df_list)

    return func2()


def sampleTypesToDF(df, n_fn):
    class _samples:

        def __init__(self, _df):
            self.df_list = _df.to_dict("records")
            random.shuffle(self.df_list)

        def remove_find(self, _cname, n, find_func=None):
            _df_list = []
            ret_list = []
            n_select = 0
            for line in self.df_list:
                if (line["CNAME"] == _cname) and (n_select < n):
                    if find_func is not None:
                        if find_func(line):
                            ret_list.append(line)
                            n_select += 1
                        else:
                            _df_list.append(line)
                    else:
                        ret_list.append(line)
                        n_select += 1
                else:
                    _df_list.append(line)
            self.df_list = _df_list
            return ret_list

        def remove_finds(self, cname_n_dict, find_func=None):
            ret_dict = {}
            for _cname, n in cname_n_dict.items():
                ret_dict[_cname] = self.remove_find(_cname, n, find_func)
            return ret_dict

    def get_index(_spls):
        _to_list = []
        for datas in _spls.values():
            _to_list.extend([int(data["_OID"]) for data in datas])
        _list = np.array([0 for _ in range(len(df))])
        _list[_to_list] = 1
        return _list

    def show_count(_name, _dict, _to_counts):
        _n_dict = {}
        for k in _dict:
            to_n = -1
            if k in _to_counts:
                to_n = _to_counts[k]
            _n_dict[k] = "{} -> {}".format(len(_dict[k]), to_n)
        printDict(_name, _n_dict)
        _n_dict = {name: len(_dict[name]) for name in _dict}
        print("SUM:", sum(_n_dict.values()))

    df["_OID"] = [i for i in range(len(df))]
    spl = _samples(df[df["TEST"] == 1])

    df_n = [int(data) for data in readLines(n_fn)]
    df_n = {_CATEGORY_NAMES_4[i]: df_n[i] for i in range(len(df_n))}

    def _hs_func(_line):
        if _DICT_FILTER_AND(_line, ("CNAME", "==", "IS"), ("AS_VV", "<", -12)):
            return False
        if _DICT_FILTER_AND(_line, ("CNAME", "==", "IS"), ("DE_VV", "<", -12)):
            return False
        return True

    hs_spls = spl.remove_finds(df_n, _hs_func)
    show_count("hs_spls", hs_spls, df_n)
    hs_index = get_index(hs_spls)

    df_n_ns_add = {
        "IS": df_n['IS_SH'] + df_n['IS_AS_SH'] + df_n['IS_DE_SH'],
        "VEG": df_n['VEG_SH'] + df_n['VEG_AS_SH'] + df_n['VEG_DE_SH'],
        "SOIL": df_n['SOIL_SH'] + df_n['SOIL_AS_SH'] + df_n['SOIL_DE_SH'],
        "WAT": df_n['WAT_SH'] + df_n['WAT_AS_SH'] + df_n['WAT_DE_SH'],
    }
    ns_spls_add = spl.remove_finds(df_n_ns_add)
    ns_spls = {
        "IS": hs_spls["IS"] + ns_spls_add["IS"],
        "VEG": hs_spls["VEG"] + ns_spls_add["VEG"],
        "SOIL": hs_spls["SOIL"] + ns_spls_add["SOIL"],
        "WAT": hs_spls["WAT"] + ns_spls_add["WAT"],
    }
    show_count("ns_spls", ns_spls, {
        "IS": df_n['IS'] + df_n['IS_SH'] + df_n['IS_AS_SH'] + df_n['IS_DE_SH'],
        "VEG": df_n['VEG'] + df_n['VEG_SH'] + df_n['VEG_AS_SH'] + df_n['VEG_DE_SH'],
        "SOIL": df_n['SOIL'] + df_n['SOIL_SH'] + df_n['SOIL_AS_SH'] + df_n['SOIL_DE_SH'],
        "WAT": df_n['WAT'] + df_n['WAT_SH'] + df_n['WAT_AS_SH'] + df_n['WAT_DE_SH'],
    })
    ns_index = get_index(ns_spls)

    df_n_os_add = {
        "IS_SH": df_n['IS_AS_SH'] + df_n['IS_DE_SH'],
        "VEG_SH": df_n['VEG_AS_SH'] + df_n['VEG_DE_SH'],
        "SOIL_SH": df_n['SOIL_AS_SH'] + df_n['SOIL_DE_SH'],
        "WAT_SH": df_n['WAT_AS_SH'] + df_n['WAT_DE_SH'],
    }
    os_spls_add = spl.remove_finds(df_n_os_add)
    os_spls = {
        "IS": hs_spls["IS"],
        "VEG": hs_spls["VEG"],
        "SOIL": hs_spls["SOIL"],
        "WAT": hs_spls["WAT"],
        "IS_SH": hs_spls["IS_SH"] + os_spls_add["IS_SH"],
        "VEG_SH": hs_spls["VEG_SH"] + os_spls_add["VEG_SH"],
        "SOIL_SH": hs_spls["SOIL_SH"] + os_spls_add["SOIL_SH"],
        "WAT_SH": hs_spls["WAT_SH"] + os_spls_add["WAT_SH"],

    }
    show_count("os_spls", os_spls, {

        "IS_SH": df_n['IS_SH'] + df_n['IS_AS_SH'] + df_n['IS_DE_SH'],
        "VEG_SH": df_n['VEG_SH'] + df_n['VEG_AS_SH'] + df_n['VEG_DE_SH'],
        "SOIL_SH": df_n['SOIL_SH'] + df_n['SOIL_AS_SH'] + df_n['SOIL_DE_SH'],
        "WAT_SH": df_n['WAT_SH'] + df_n['WAT_AS_SH'] + df_n['WAT_DE_SH'],
    })
    os_index = get_index(os_spls)

    df["HS"] = hs_index
    df["OS"] = os_index
    df["NS"] = ns_index

    return df


class _DFFilter:

    def __init__(self, df=None, _list=None):
        self.df_list = []
        if df is None:
            if _list is not None:
                self.df_list = _list
            else:
                return
        else:
            self.df_list = df.to_dict("records")

    def remove(self, *filters):
        _df_list = []
        for line in self.df_list:
            is_remove = True
            for i, (name, f, data) in enumerate(filters):
                if not _COM(line[name], f, data):
                    is_remove = False
                    break
            if not is_remove:
                _df_list.append(line)
        self.df_list = _df_list
        return self

    def df(self):
        return pd.DataFrame(self.df_list)

    def all(self, *filters):
        _df_list = []
        for line in self.df_list:
            is_add = True
            for i, (name, f, data) in enumerate(filters):
                if not _COM(line[name], f, data):
                    is_add = False
                    break
            if is_add:
                _df_list.append(line)
        self.df_list = _df_list
        return self

    def any(self, *filters):
        _df_list = []
        for line in self.df_list:
            is_add = False
            for i, (name, f, data) in enumerate(filters):
                if _COM(line[name], f, data):
                    is_add = True
                    break
            if is_add:
                _df_list.append(line)
        self.df_list = _df_list
        return self

    def add(self, _df_filter):
        self.df_list += _df_filter.df_list
        return self


def _trainimdc(dirname, is_save, model, name, range_json, raster_fn, sd, x_keys,
               feat_funcs=None, map_dict=None, print_func=print):
    if model == "svm":
        model = SVM_RGS()
    elif model == "rf":
        model = RF_RGS()
    if feat_funcs is None:
        feat_funcs = FeatFuncs()
    data_scale = DataScale()
    if range_json is not None:
        data_scale = DataScale().readJson(range_json)
    if map_dict is None:
        map_dict = {
            "IS": 1, "VEG": 2, "SOIL": 3, "WAT": 4,
            "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 3, "WAT_SH": 4
        }
    ml_mod = MLModel()
    ml_mod.name = name
    ml_mod.filename = os.path.join(dirname, "{}.hm".format(name))
    ml_mod.x_keys = x_keys
    ml_mod.feat_funcs = feat_funcs
    ml_mod.map_dict = map_dict
    ml_mod.data_scale = data_scale
    ml_mod.color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255)}
    ml_mod.clf = model
    ml_mod.train_filters = [("TEST", "==", 1)]
    ml_mod.test_filter = [("TEST", "==", 0)]
    # Import samples
    ml_mod.sampleData(sd)
    ml_mod.samples.dataDescription(print_func=print_func, end="\n")
    # Training
    ml_mod.train()
    # Save
    if is_save:
        ml_mod.save()
    if raster_fn is not None:
        print_func("model file name:", ml_mod.filename)
        ml_mod.imdc(raster_fn)
    # Load
    # ml_mod = MLModel().load("*.shh2mod")
    return ml_mod


class _MLModel:

    def __init__(self, name, dirname, sd, x_keys, range_json, raster_fn=None, model="svm",
                 is_save=False, cm_name=None, feat_funcs=None, map_dict=None, print_func=print):
        if cm_name is None:
            cm_name = _CATEGORY_NAMES
        self.name = name
        self.dirname = dirname
        self.sd = sd
        self.x_keys = x_keys
        self.range_json = range_json
        self.raster_fn = raster_fn
        self.model = model
        self.is_save = is_save
        self.feat_funcs = feat_funcs
        self.map_dict = map_dict
        self.ml_mods = []
        self.cm_name = cm_name
        self.print_func = print_func

    def fit(self):
        ml_mod = _trainimdc(
            self.dirname, self.is_save,
            self.model, self.name, self.range_json,
            self.raster_fn, self.sd,
            self.x_keys, feat_funcs=self.feat_funcs,
            map_dict=self.map_dict,
            print_func=self.print_func,
        )
        self.ml_mods.append(ml_mod)
        return ml_mod.cm(self.cm_name)

    def accuracy(self, n_mod=-1):
        ml_mod: MLModel = self.ml_mods[n_mod]
        y2 = ml_mod.predict(ml_mod.samples.x_test).tolist()
        cnames = [spl.cname for spl in ml_mod.samples.spls_test]
        test_is = [spl["TEST_IS"] for spl in ml_mod.samples.spls_test]
        test_sh = [spl["TEST_SH"] for spl in ml_mod.samples.spls_test]
        to_dict = {}

        def cal_cm(test_list, y1_map_dict, y2_map_dict, cm_names):
            y1_list = []
            y2_list = []
            for i in range(len(cnames)):
                if test_list[i] == 1:
                    y1_list.append(y1_map_dict[cnames[i]])
                    y2_list.append(y2_map_dict[y2[i]])
            _cm = ConfusionMatrix(class_names=cm_names)
            _cm.addData(y1_list, y2_list)
            return _cm

        cm = cal_cm(
            test_list=test_is,
            y1_map_dict={
                "IS": 1, "VEG": 2, "SOIL": 3, "WAT": 4,
                "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 3, "WAT_SH": 4,
            },
            y2_map_dict={1: 1, 2: 2, 3: 3, 4: 4},
            cm_names=_CATEGORY_NAMES,
        )

        self.print_func("+ cm IS")
        self.print_func(cm.fmtCM())

        to_dict["IS_OA"] = cm.accuracyCategory("IS").OA()
        to_dict["IS_Kappa"] = cm.accuracyCategory("IS").getKappa()

        cm = cal_cm(
            test_list=test_sh,
            y1_map_dict={
                "IS": 0, "VEG": 0, "SOIL": 0, "WAT": 0,
                "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 3, "WAT_SH": 4,
            },
            y2_map_dict={1: 1, 2: 2, 3: 3, 4: 4},
            cm_names=_CATEGORY_NAMES,
        )

        self.print_func("+ cm IS_SH")
        self.print_func(cm.fmtCM())

        to_dict["IS_SH_OA"] = cm.accuracyCategory("IS").OA()
        to_dict["IS_SH_Kappa"] = cm.accuracyCategory("IS").getKappa()

        return to_dict, cm


def trainimdcMain():
    def func1():
        log_fn = os.path.join(_MODEL_LOG_DIRNAME, "log{}.txt".format(timeStringNow(str_fmt="%Y%m%d%H%M%S")))
        sw = SRTWriteText(log_fn, is_show=True)
        feat_funcs = featFuncs("")

        citys = {
            "qd": (_QD_SPLING_FN(), _QD_RANGE_FN),
            "bj": (_BJ_SPLING_FN(), _BJ_RANGE_FN),
            "cd": (_CD_SPLING_FN(), _CD_RANGE_FN),
        }
        models = ["rf", "svm"]

        for model in models:
            for city_name, (csv_fn, range_json) in citys.items():
                df = pd.read_csv(csv_fn)
                sd = SamplesData().addDF(df)
                to_dict = {}

                def func12(name, x_keys):
                    sw.write(name)
                    ti_mod = _MLModel(
                        name=name, dirname=_MODEL_DIRNAME, sd=sd,
                        x_keys=x_keys, range_json=range_json, raster_fn=None,
                        model=model, feat_funcs=feat_funcs,
                    )
                    cm = ti_mod.fit()

                    sw.write(cm.fmtCM())
                    to_dict[name] = ({
                        "IS_OA": cm.accuracyCategory("IS").OA(), "IS_Kappa": cm.accuracyCategory("IS").getKappa()
                    })
                    sw.write("IS OA:", to_dict[name]["IS_OA"])
                    sw.write("IS Kappa:", to_dict[name]["IS_Kappa"])
                    sw.write()

                func12("Opt", _FEAT_NAMES.f_opt())
                func12("Opt-AS", _FEAT_NAMES.f_opt_as())
                func12("Opt-DE", _FEAT_NAMES.f_opt_de())
                func12("Opt-AS-DE", _FEAT_NAMES.f_opt_as_de())

                sw.write("#", city_name, model, "-" * 6)
                sw.write(pd.DataFrame(to_dict).T.sort_values("IS_OA", ascending=False))

    def func2():

        feat_funcs = featFuncs("")

        citys = {
            "qd": (_QD_SPLING_FN(), _QD_RANGE_FN, _QD_RASTER_FN),
            "bj": (_BJ_SPLING_FN(), _BJ_RANGE_FN, _BJ_RASTER_FN),
            "cd": (_CD_SPLING_FN(), _CD_RANGE_FN, _CD_RASTER_FN),
        }
        for city_name, (csv_fn, range_json, raster_fn) in citys.items():
            df = pd.read_csv(csv_fn)
            sd = SamplesData().addDF(df)
            ti_mod = _MLModel(
                name="{}_shimdc".format(city_name),
                dirname=r"F:\ProjectSet\Shadow\ASDEHSamples\Models\ShadowImdcs",
                sd=sd, x_keys=_FEAT_NAMES.f_opt(), range_json=range_json, raster_fn=raster_fn,
                model="svm", feat_funcs=feat_funcs,
                map_dict={
                    "IS": 1, "VEG": 1, "SOIL": 1, "WAT": 1,
                    "IS_SH": 2, "VEG_SH": 2, "SOIL_SH": 2, "WAT_SH": 2,
                },
                cm_name=["NOSH", "SH"]
            )
            cm = ti_mod.fit()

            print(cm.fmtCM())
            print("OA:", cm.OA())
            print("Kappa:", cm.getKappa())
            print()

    def func3():
        csv_fn = _CD_SPLING_FN()
        range_json = _CD_RANGE_FN
        feat_funcs = featFuncs("")
        model = "svm"

        df = pd.read_csv(csv_fn)
        sd = SamplesData().addDF(df)
        to_dict = {}

        def func12(name, x_keys):
            ti_mod = _MLModel(
                name=name, dirname=_MODEL_DIRNAME, sd=sd,
                x_keys=x_keys, range_json=range_json, raster_fn=None,
                model=model, feat_funcs=feat_funcs,
            )
            ti_mod.fit()
            to_dict_tmp, cm = ti_mod.accuracy()
            to_dict[name] = to_dict_tmp
            printDict("# {} ------".format(name), to_dict_tmp)
            print(cm.fmtCM())

        func12("Opt", _FEAT_NAMES.f_opt())
        func12("Opt-AS", _FEAT_NAMES.f_opt_as())
        func12("Opt-DE", _FEAT_NAMES.f_opt_de())
        func12("Opt-AS-DE", _FEAT_NAMES.f_opt_as_de())

        print(pd.DataFrame(to_dict).T.sort_values("IS_OA", ascending=False))

    def func4():
        csv_fn = _QD_SAMPLES_DFN.fn("3", "qd_data.csv")
        csv_fn = _sampleCSV(_QD_RASTER_FN, csv_fn, True, changext(csv_fn, "_spl.csv"))
        _TrainImdc("qd", csv_fn, is_save=False).fit()

    func4()
    return


def training(city_name):
    # df = pd.read_csv(_SPLING_FN(city_name))
    df = pd.read_csv(_QD_SAMPLES_DFN.fn("3", r"spl_select3.csv"))
    # print(samplesDescription(df))
    # df = _DFFilter(df).remove(
    #     ("TEST", "==", 1), ("CNAME", "==", "IS"), ("AS_VV", "<", -10)).remove(
    #     ("TEST", "==", 1), ("CNAME", "==", "IS"), ("DE_VV", "<", -10)).remove(
    #     ("TEST", "==", 1), ("CNAME", "==", "IS_SH"), ("AS_VV", "<", -10)).remove(
    #     ("TEST", "==", 1), ("CNAME", "==", "IS_SH"), ("DE_VV", "<", -10)).df()

    samplesDescription(df)

    df_free, df_opt, df_sar = sampleTypes(df)
    df_dict = {"SAR": df_sar, "Opt": df_opt, "FREE": df_free, }
    range_json = _RANGE_FN(city_name)
    range_json = None
    feat_funcs = featFuncs()
    raster_fn = _RASTER_FN(city_name)
    # raster_fn = None
    model = "rf"
    is_save = False

    to_dict = {}
    dirname = timeDirName(r"F:\ProjectSet\Shadow\ASDEHSamples\Models", True)

    for df_name, df in df_dict.items():
        print("#", df_name, len(df), "-" * 6)
        samplesDescription(df)
        sd = SamplesData().addDF(df)

        def func12(name, x_keys):
            ti_mod = _MLModel(
                name="{}_{}".format(city_name, name),
                dirname=dirname,
                sd=sd, x_keys=x_keys,
                range_json=range_json, raster_fn=raster_fn,
                model=model, feat_funcs=feat_funcs,
                map_dict={
                    "NOT_KNOW": 0,
                    "IS": 1, "IS_SH": 1, "IS_AS_SH": 1, "IS_DE_SH": 1,
                    "VEG": 2, "VEG_SH": 2, "VEG_AS_SH": 2, "VEG_DE_SH": 2,
                    "SOIL": 3, "SOIL_SH": 3, "SOIL_AS_SH": 3, "SOIL_DE_SH": 3,
                    "WAT": 4, "WAT_SH": 4, "WAT_AS_SH": 4, "WAT_DE_SH": 4,
                },
                cm_name=["NOSH", "SH"],
                is_save=is_save
            )
            ti_mod.fit()
            to_dict_tmp, cm = ti_mod.accuracy()
            to_dict[name] = to_dict_tmp
            printDict("# {} ------".format(city_name.upper()), to_dict_tmp)
            print(cm.fmtCM())

        if (df_name == "Opt") or (df_name == "SAR"):
            func12("{}-Opt-AS-DE".format(df_name), _FEAT_NAMES.f_opt_as_de())
        else:
            func12("{}-Opt-AS-DE".format(df_name), _FEAT_NAMES.f_opt_as_de())
            func12("{}-Opt-AS".format(df_name), _FEAT_NAMES.f_opt_as())
            func12("{}-Opt-DE".format(df_name), _FEAT_NAMES.f_opt_de())
            func12("{}-Opt".format(df_name), _FEAT_NAMES.f_opt())

    # .sort_values("IS_OA", ascending=False)

    print()


class _TrainImdc:

    def __init__(self, city_name, csv_fn=None, model="rf", is_save=False,
                 is_imdc=False, feat_funcs_type=None):
        self.city_name = city_name
        self.csv_fn = None
        self.initCSV(csv_fn)
        self.model = model
        self.range_json = _RANGE_FN(city_name) if model == "svm" else None
        self.is_save = is_save
        self.is_imdc = is_imdc
        self.feat_funcs = featFuncs(feat_funcs_type)
        self.raster_fn = _RASTER_FN(city_name) if is_imdc else None

    def initCSV(self, csv_fn):
        if csv_fn is None:
            if self.city_name == "qd":
                csv_fn = _SAMPLES_DFN.fn("HSPL_QD_select.csv")
                df = pd.read_csv(_SPLING_FN(self.city_name))
                df = sampleTypesToDF(df, _SAMPLES_DFN.fn("HSPL_QD_n_select.txt"))
                df.to_csv(csv_fn, index=False)
                GDALSamplingFast(_RASTER_FN(self.city_name)).csvfile(csv_fn, csv_fn)
            elif self.city_name == "bj":
                csv_fn = _SAMPLES_DFN.fn("HSPL_BJ_select.csv")
                df = pd.read_csv(_SPLING_FN(self.city_name))
                df = sampleTypesToDF(df, _SAMPLES_DFN.fn("HSPL_BJ_n_select.txt"))
                df.to_csv(csv_fn, index=False)
                GDALSamplingFast(_RASTER_FN(self.city_name)).csvfile(csv_fn, csv_fn)
            elif self.city_name == "cd":
                csv_fn = _SAMPLES_DFN.fn("HSPL_CD_select.csv")
                df = pd.read_csv(_SPLING_FN(self.city_name))
                df = sampleTypesToDF(df, _SAMPLES_DFN.fn("HSPL_CD_n_select.txt"))
                df.to_csv(csv_fn, index=False)
                GDALSamplingFast(_RASTER_FN(self.city_name)).csvfile(csv_fn, csv_fn)
        self.csv_fn = _SPLING_FN(self.city_name) if csv_fn is None else csv_fn

    def fit(self, sw: SRTWriteText = None):
        def _sw(*text, sep=" ", end="\n"):
            if _sw is not None:
                sw.write(*text, sep=sep, end=end)

        td = TimeDirectory(_MODEL_DIRNAME)
        td.initLog()

        td.kw("city_name", self.city_name)
        td.kw("csv_fn", self.csv_fn)
        td.kw("model", self.model)
        td.kw("range_json", self.range_json)
        td.kw("is_save", self.is_save)
        td.kw("is_imdc", self.is_imdc)
        td.kw("feat_funcs", self.feat_funcs)
        td.kw("raster_fn", self.raster_fn)
        map_dict = td.kw("map_dict", {
            "NOT_KNOW": 0,
            "IS": 1, "IS_SH": 1, "IS_AS_SH": 1, "IS_DE_SH": 1,
            "VEG": 2, "VEG_SH": 2, "VEG_AS_SH": 2, "VEG_DE_SH": 2,
            "SOIL": 3, "SOIL_SH": 3, "SOIL_AS_SH": 3, "SOIL_DE_SH": 3,
            "WAT": 4, "WAT_SH": 4, "WAT_AS_SH": 4, "WAT_DE_SH": 4,
        })

        df = pd.read_csv(self.csv_fn)

        td.copyfile(__file__)
        td.copyfile(self.csv_fn)

        td.kw("Counts", tabulate(
            samplesDescription(df, is_print=False), headers="keys", tablefmt="simple"
        ), sep=":\n", end="\n\n")
        df_free, df_opt, df_sar = sampleTypes(df)
        df_dict = {"SAR": df_sar, "Opt": df_opt, "FREE": df_free, }

        to_dict = {}
        for df_name, df in df_dict.items():
            td.log("#", "-" * 36, "Sample Type", df_name, "-" * 36, "#", end="\n\n")
            td.kw("Counts", tabulate(
                samplesDescription(df, is_print=False), headers="keys", tablefmt="simple"
            ), sep=":\n", end="\n\n")
            sd = SamplesData().addDF(df)

            def func12(name, x_keys):
                td.log("# {} {}".format(name, "-" * 6), end="\n\n")
                ti_mod = _MLModel(
                    name="{}_{}".format(self.city_name, name),
                    dirname=td.time_dirname(),
                    sd=sd, x_keys=x_keys,
                    range_json=self.range_json, raster_fn=self.raster_fn,
                    model=self.model, feat_funcs=self.feat_funcs,
                    map_dict=map_dict,
                    cm_name=["NOSH", "SH"],
                    is_save=self.is_save,
                    print_func=td.log,
                )
                ti_mod.fit()
                to_dict_tmp, cm = ti_mod.accuracy()
                to_dict[name] = to_dict_tmp
                printDict("> Accuracy " + "-" * 6, to_dict_tmp, print_func=td.log, end="\n", )

            if (df_name == "Opt") or (df_name == "SAR"):
                func12("{}-Opt-AS-DE".format(df_name), _FEAT_NAMES.f_opt_as_de())
            else:
                func12("{}-Opt-AS-DE".format(df_name), _FEAT_NAMES.f_opt_as_de())
                func12("{}-Opt-AS".format(df_name), _FEAT_NAMES.f_opt_as())
                func12("{}-Opt-DE".format(df_name), _FEAT_NAMES.f_opt_de())
                func12("{}-Opt".format(df_name), _FEAT_NAMES.f_opt())

        df_acc = pd.DataFrame(to_dict).T
        td.kw("accuracy", tabulate(df_acc, headers="keys", tablefmt="simple"), sep=":\n")
        td.saveDF("accuracy.csv", df_acc)

        _sw("{} -> {} {:<3} {}\n".format(timeStringNow(), self.city_name, self.model, td.time_dirname()))

        return self


def checkOID(csv_fn):
    df = pd.read_csv(csv_fn)
    _list = df["SRT"].tolist()
    if len(_list) != len(set(_list)):
        warnings.warn("OID not unique. {}".format(csv_fn))


def _samplingImdc(df, dirname, is_ret_names=False, ):
    fns = filterFileEndWith(dirname, "_imdc.tif")
    names = []
    for fn in fns:
        name = getfilenamewithoutext(fn).replace("_imdc.tif", "")
        gsf = GDALSamplingFast(fn)
        df_imdc = gsf.sampling(df["X"], df["Y"], )
        df[name] = df_imdc[list(df_imdc.keys())[0]]
        names.append(name)
    if is_ret_names:
        return df, names
    return df


class _SampleUpdate:

    def __init__(self, name, city_name, update_type, to_dirname):
        self.name = name
        self.city_name = city_name
        self.csv_fn = _SPLING_FN(city_name)
        self.update_type = update_type
        self.to_dirname = to_dirname
        self.dfn = DirFileName(to_dirname)
        self.txt_fn = self.dfn.fn("{}.txt".format(self.name))

        self.checkOID()

    def checkOID(self):
        print(self.csv_fn)
        df = pd.read_csv(self.csv_fn)
        _list = df["SRT"].tolist()
        if len(_list) != len(set(_list)):
            warnings.warn("OID not unique. {}".format(self.csv_fn))
        samplesDescription(df)

    def to_txt(self):
        df = pd.read_csv(self.csv_fn)
        if self.update_type == "training":
            df = pd.DataFrame(df[df["TEST"] == 1].to_dict("records"))
            df2SplTxt(df, "shadow2", self.txt_fn)
        elif self.update_type == "test_is":
            df = pd.DataFrame(df[df["TEST_IS"] == 1].to_dict("records"))
            df2SplTxt(df, "shadow2", self.txt_fn)
        elif self.update_type == "test_sh":
            df = pd.DataFrame(df[df["TEST_SH"] == 1].to_dict("records"))
            df2SplTxt(df, "shadow2", self.txt_fn)

    def update(self):
        to_dfn = DirFileName(timeDirName(self.to_dirname, is_mk=True))
        shutil.copyfile(self.txt_fn, to_dfn.fn(os.path.basename(self.txt_fn)))
        df_txt = self.readTxt()
        df_list = df_txt.to_dict("records")

        to_df_list = pd.read_csv(self.csv_fn)
        srt_add = int(to_df_list["SRT"].max())
        to_df_list = to_df_list.to_dict("records")

        test_is, test_sh, test = 0, 0, 1
        if self.update_type == "training":
            test_is, test_sh, test = 0, 0, 1
        elif self.update_type == "test_is":
            test_is, test_sh, test = 1, 0, 0
        elif self.update_type == "test_sh":
            test_is, test_sh, test = 0, 1, 0

        for line in df_list:
            if not pd.isna(line["SRT"]):
                oid = int(line["SRT"])
                cname = str(line["CATEGORY_NAME"])
                is_find = False
                for line2 in to_df_list:
                    if int(line2["SRT"]) == oid:
                        if line2["CNAME"] != cname:
                            line2["CNAME"] = cname
                            line2["CATEGORY"] = int(line["CATEGORY_CODE"])
                        is_find = True
                        break
            else:
                is_find = False
            if not is_find:
                srt_add += 1
                to_dict = {
                    "SRT": srt_add, "X": line["X"], "Y": line["Y"], "CNAME": line["CATEGORY_NAME"],
                    "CATEGORY": int(line["CATEGORY_CODE"]), "TAG": "SELECT", "TEST2": 1,
                    "TEST_IS": test_is, "TEST_SH": test_sh, "TEST": test,
                }
                to_dict_keys = list(to_dict.keys())
                for k in line:
                    if k not in to_dict_keys:
                        to_dict[k] = line[k]
                to_df_list.append(to_dict)

        to_df = pd.DataFrame(to_df_list)
        return to_df

    def readTxt(self):
        df = splTxt2Dict(self.txt_fn)
        to_csv_fn = self.dfn.fn("{}.csv".format(self.name))
        savecsv(to_csv_fn, df)
        df = pd.read_csv(to_csv_fn)
        return df

    def counts(self):
        df = self.update()
        df = sampleTypesToDF(df, _QD_SAMPLES_DFN.fn("3", r"spl_select3.txt"))
        samplesDescription(df)
        return df

    def samplingImdc(self, dirname, is_ret_names=False, df=None):
        if df is None:
            df = pd.read_csv(self.csv_fn)
        return _samplingImdc(df, dirname, is_ret_names)

    def to_txt_testIS(self, mod_dirname):
        df, names = self.samplingImdc(_MODEL_DFN.fn(mod_dirname), is_ret_names=True)
        df: pd.DataFrame
        df["CATEGORY2"] = (df["CATEGORY"] / 10).round()
        df = df.sort_values(["CATEGORY2"] + names)
        select_names = [
            "SRT", "X", "Y", "CNAME", "CATEGORY", "TAG", "TEST2", "TEST_IS", "TEST_SH", "TEST",
            "HS", "OS", "NS",
            "Blue", "Green", "Red", "NIR", "NDVI", "NDWI", "CATEGORY2",
            *names
        ]
        df = df[select_names]
        to_is = None
        for name in names:
            if to_is is None:
                to_is = (df[name] == df["CATEGORY2"]) * 1
            else:
                to_is += (df[name] == df["CATEGORY2"]) * 1
        df["IS_TF"] = to_is
        df = df.sort_values(["IS_TF", "CATEGORY"])
        printList("df.keys", list(df.keys()))
        to_csv_fn = self.dfn.fn("imdc_spl.txt")
        df.to_csv(to_csv_fn, index=False)
        print(to_csv_fn)
        df = pd.read_csv(to_csv_fn)
        df = pd.DataFrame(df[df["TEST_IS"] == 1].to_dict("records"))
        df2SplTxt(df, "shadow2", self.txt_fn)

    def to_txt_testSH(self, mod_dirname):
        df, names = self.samplingImdc(_MODEL_DFN.fn(mod_dirname), is_ret_names=True)
        df: pd.DataFrame
        df["CATEGORY2"] = (df["CATEGORY"] / 10).round()
        df = df.sort_values(["CATEGORY2"] + names)
        select_names = [
            "SRT", "X", "Y", "CNAME", "CATEGORY", "TAG", "TEST2", "TEST_IS", "TEST_SH", "TEST", "HS", "OS", "NS",
            "Blue", "Green", "Red", "NIR", "NDVI", "NDWI", "CATEGORY2",
            *names
        ]
        df = df[select_names]
        to_is = None
        for name in names:
            if to_is is None:
                to_is = (df[name] == df["CATEGORY2"]) * 1
            else:
                to_is += (df[name] == df["CATEGORY2"]) * 1
        df["IS_TF"] = to_is
        df = df.sort_values(["IS_TF", "CATEGORY"])
        printList("df.keys", list(df.keys()))
        to_csv_fn = self.dfn.fn("imdc_spl.txt")
        df.to_csv(to_csv_fn, index=False)
        print(to_csv_fn)
        df = pd.read_csv(to_csv_fn)
        df = pd.DataFrame(df[df["TEST_SH"] == 1].to_dict("records"))
        df2SplTxt(df, "shadow2", self.txt_fn)

    def testIS(self, mod_dirname):
        df = self.readTxt()
        df, names = self.samplingImdc(_MODEL_DFN.fn(mod_dirname), True, df)
        df: pd.DataFrame
        df["CATEGORY2"] = (df["CATEGORY_CODE"] / 10).round()
        acc_dict = {}
        for name in names:
            cm = ConfusionMatrix(class_names=_CATEGORY_NAMES)
            cm.addData(df["CATEGORY2"].tolist(), df[name].tolist())
            print("#", name, "-" * 6)
            print(cm.fmtCM())
            acc_dict[name] = {
                "IS OA": cm.accuracyCategory("IS").OA(),
                "IS Kappa": cm.accuracyCategory("IS").getKappa(),
            }
            printDict("", acc_dict[name])
        print(pd.DataFrame(acc_dict).T.sort_values("IS OA", ascending=False))

    def testSH(self, mod_dirname):
        df = self.readTxt()
        df, names = self.samplingImdc(_MODEL_DFN.fn(mod_dirname), True, df)
        df: pd.DataFrame
        df["CATEGORY2"] = (df["CATEGORY_CODE"] / 10).round()
        acc_dict = {}
        for name in names:
            cm = ConfusionMatrix(class_names=_CATEGORY_NAMES)
            cm.addData(df["CATEGORY2"].tolist(), df[name].tolist())
            print("#", name, "-" * 6)
            print(cm.fmtCM())
            acc_dict[name] = {
                "IS SH OA": cm.accuracyCategory("IS").OA(),
                "IS SH Kappa": cm.accuracyCategory("IS").getKappa(),
            }
            printDict("", acc_dict[name])
        print(pd.DataFrame(acc_dict).T.sort_values("IS SH OA", ascending=False))

    def cat(self, test_is_fn, test_sh_fn, update_dict=None):
        df = pd.read_csv(self.csv_fn)
        oid_add = np.max(df["SRT"].values)
        to_list = []

        df_is_list = pd.read_csv(test_is_fn).to_dict("records")
        for line in df_is_list:
            oid = line["SRT"]
            if np.isnan(oid):
                oid_add += 1
                oid = oid_add
            to_dict = {
                'SRT': int(oid), 'X': line["X"], 'Y': line["Y"], 'CNAME': line["CATEGORY_NAME"],
                'CATEGORY': _C_NAME_CODE_MAP_DICT[line["CATEGORY_NAME"]],
                'TAG': "SELECT", 'TEST2': 0, 'TEST_IS': 1,
                'TEST_SH': 0, 'TEST': 0, 'HS': 0, 'OS': 0, 'NS': 0,
            }
            to_list.append(to_dict)

        df_sh_list = pd.read_csv(test_sh_fn).to_dict("records")
        for line in df_sh_list:
            oid = line["SRT"]
            if np.isnan(oid):
                oid_add += 1
                oid = oid_add
            to_dict = {
                'SRT': int(oid), 'X': line["X"], 'Y': line["Y"], 'CNAME': line["CATEGORY_NAME"],
                'CATEGORY': _C_NAME_CODE_MAP_DICT[line["CATEGORY_NAME"]],
                'TAG': "SELECT", 'TEST2': 0, 'TEST_IS': 0,
                'TEST_SH': 1, 'TEST': 0, 'HS': 0, 'OS': 0, 'NS': 0,
            }
            to_list.append(to_dict)

        to_list = pd.DataFrame(to_list).sort_values("SRT").to_dict("records")
        to_list2 = []
        oid = -1
        for line in to_list:
            if update_dict is not None:
                if line["SRT"] in update_dict:
                    line["CNAME"] = update_dict[line["SRT"]]
                    line["CATEGORY"] = _C_NAME_CODE_MAP_DICT[line["CNAME"]]
            if oid == -1:
                oid = line["SRT"]
                to_list2.append(line)
            else:
                if oid != line["SRT"]:
                    to_list2.append(line)
                    oid = line["SRT"]
                else:
                    to_list2[-1]["TEST_IS"] = 1
                    to_list2[-1]["TEST_SH"] = 1
                    print(oid)

        return pd.DataFrame(to_list2)


def featExtHA():
    init_dfn = DirFileName(r"F:\ProjectSet\Shadow\ASDEHSamples\Images\HA")

    def func3(city_name, raster_fn):
        gr = GDALRaster(raster_fn)

        def func1(name, dfn):
            if not os.path.isdir(dfn.fn()):
                os.mkdir(dfn.fn())

            print(dfn.fn("{}_H.dat".format(name)))

            c11_key = "{}_C11".format(name)
            c22_key = "{}_C22".format(name)
            c12_real_key = "{}_C12_real".format(name)
            c12_imag_key = "{}_C12_imag".format(name)

            d_c11 = update10EDivide10(gr.readGDALBand(c11_key))
            d_c22 = update10EDivide10(gr.readGDALBand(c22_key))
            d_c12_real = gr.readGDALBand(c12_real_key)
            d_c12_imag = gr.readGDALBand(c12_imag_key)

            e1, e2, v11, v12, v21, v22 = eig2(
                d_c11,
                2 * (d_c12_real + d_c12_imag * 1j),
                2 * (d_c12_real - d_c12_imag * 1j),
                4 * d_c22,
            )

            p1 = e1 / (e1 + e2)
            p2 = e2 / (e1 + e2)
            p1[p1 <= 0] = 0.0000001
            p2[p2 <= 0] = 0.0000001
            d_h = -(p1 * (np.log(p1) / np.log(2)) + p2 * (np.log(p2) / np.log(2)))
            a = p2 - p1

            d_v11 = np.clip(np.abs(v21), 0, 1)
            print("np.min(d_v11), np.max(d_v11)", np.min(d_v11), np.max(d_v11))
            alp1 = np.rad2deg(np.arccos(d_v11))

            d_v12 = np.clip(np.abs(v22), 0, 1)
            print("np.min(d_v11), np.max(d_v11)", np.min(d_v12), np.max(d_v12))
            alp2 = np.rad2deg(np.arccos(d_v12))

            alp = p1 * alp1 + p2 * alp2

            gr.save(d_h, dfn.fn("{}_H.dat".format(name)), descriptions=["{}_H".format(name)])
            gr.save(a, dfn.fn("{}_A.dat".format(name)), descriptions=["{}_A".format(name)])
            gr.save(alp, dfn.fn("{}_Alpha.dat".format(name)), descriptions=["{}_Alpha".format(name)])

        func1("AS", DirFileName(init_dfn.fn(city_name)))
        func1("DE", DirFileName(init_dfn.fn(city_name)))

    func3("QD", _QD_RASTER_FN)
    func3("BJ", _BJ_RASTER_FN)
    func3("CD", _CD_RASTER_FN)


def rasterFuncs():
    def func1():
        """ 统计范围
        :return:
        """

        class cls1:

            def __init__(self):
                self.dict = {}

            def featureCallBack(self, *args, **kwargs):
                return

            def featureScaleMinMax(self, name, x_min, x_max):
                self.dict[name] = {"min": x_min, "max": x_max}

            def toDict(self, fn):
                saveJson(self.dict, fn)

        data = cls1()
        cdFeatureDeal(data)
        data.toDict(r"F:\ProjectSet\Shadow\ASDEHSamples\Images\ChengDu\HSPL_CD.range")

    def func2():
        class cls1:

            def __init__(self, raster_fn):
                self.dict = {}
                self.gr = GDALRaster(raster_fn, gdal.GA_Update)
                print("*", raster_fn)
                print("-" * 60)

            def featureCallBack(self, name, _func):
                data = self.gr.readGDALBand(name)
                print("{:<10} {:15.2f} {:15.2f} {:15.2f}".format(name, data.min(), data.mean(), data.max()))
                data = _func(data)
                print("{:<10} {:15.2f} {:15.2f} {:15.2f}".format(" ", data.min(), data.mean(), data.max()))
                print("-" * 60)
                if is_update:
                    band = self.gr.getGDALBand(name)
                    band.WriteArray(data.astype("float32"))
                return

            def featureScaleMinMax(self, name, x_min, x_max):
                self.dict[name] = {"min": x_min, "max": x_max}

            def toDict(self, fn):
                saveJson(self.dict, fn)

        is_update = False
        qdFeatureDeal(cls1(_QD_RASTER_FN))
        bjFeatureDeal(cls1(_BJ_RASTER_FN))
        cdFeatureDeal(cls1(_CD_RASTER_FN))

    def func3():
        gr = GDALRaster(r"F:\ProjectSet\Shadow\ChengDu\Mods\20240222H170152\SPL_SH-RF-TAG-OPTICS-AS-DE_imdc.dat")
        data = gr.readGDALBand(1)
        d, n = np.unique(data, return_counts=True)
        print(n / np.sum(n))

    def func4():
        gr = GDALRaster(_QD_RASTER_FN)
        for name in ["AS_VV", "AS_VH", "DE_VV", "DE_VH"]:
            data = gr.readGDALBand(name)
            data = np.power(10, data / 10)
            gr.save(
                data.astype("float32"),
                os.path.join(r"F:\ProjectSet\Shadow\ASDEHSamples\Images\QingDao", "HSPL_QD_{}_envi.dat".format(name)),
                dtype=gdal.GDT_Float32
            )

    def func5():
        raster_fn = _CD_RASTER_FN
        gr = GDALRaster(raster_fn)
        to_fn = changext(raster_fn, "_vrt.vrt")
        ha_dfn = DirFileName(r"F:\ProjectSet\Shadow\ASDEHSamples\Images\HA")
        raster_dict = {}

        def func51(_fn):
            raster_dict[getfilenamewithoutext(_fn)] = _fn

        tmp_dirname = mkdir(r"F:\ProjectSet\Shadow\ASDEHSamples\Images\Temp")
        shutil.rmtree(tmp_dirname)
        mkdir(tmp_dirname)
        tmp_dfn = DirFileName(tmp_dirname)
        rtvrts = RasterToVRTS(raster_fn)
        rtvrts.save(tmp_dfn.fn())

        for name in gr.names:
            raster_dict[name] = tmp_dfn.fn("{}.vrt".format(name))

        func51(ha_dfn.fn("CD", "AS_H.dat"))
        func51(ha_dfn.fn("CD", "AS_A.dat"))
        func51(ha_dfn.fn("CD", "AS_Alpha.dat"))
        func51(ha_dfn.fn("CD", "DE_H.dat"))
        func51(ha_dfn.fn("CD", "DE_A.dat"))
        func51(ha_dfn.fn("CD", "DE_Alpha.dat"))

        printDict("raster_dict", raster_dict)
        print("to_fn", to_fn)
        dictRasterToVRT(to_fn, raster_dict)

    def func6():
        raster_fn = _QD_RASTER_FN
        gr = GDALRaster(raster_fn, gdal.GA_Update)
        is_update = False

        data = gr.readGDALBand("AS_VHDVV")
        print("{:<10} {:15.2f} {:15.2f} {:15.2f}".format("AS_VHDVV", data.min(), data.mean(), data.max()))
        data = _10Log10(data)
        print("{:<10} {:15.2f} {:15.2f} {:15.2f}".format(" ", data.min(), data.mean(), data.max()))
        if is_update:
            band = gr.getGDALBand("AS_VHDVV")
            band.WriteArray(data.astype("float32"))

        data = gr.readGDALBand("DE_VHDVV")
        print("{:<10} {:15.2f} {:15.2f} {:15.2f}".format("DE_VHDVV", data.min(), data.mean(), data.max()))
        data = _10Log10(data)
        print("{:<10} {:15.2f} {:15.2f} {:15.2f}".format(" ", data.min(), data.mean(), data.max()))
        if is_update:
            band = gr.getGDALBand("DE_VHDVV")
            band.WriteArray(data.astype("float32"))

    def func7():
        json_fn = _BJ_RANGE_FN
        json_dict = readJson(json_fn)
        # json_dict["AS_H"] = {"min": 0.0, "max": 1.0}
        # json_dict["AS_Alpha"] = {"min": 0.0, "max": 90.0}
        # json_dict["DE_H"] = {"min": 0.0, "max": 1.0}
        # json_dict["DE_Alpha"] = {"min": 0.0, "max": 90.0}
        printDict(json_fn, json_dict)
        saveJson(json_dict, json_fn)

    def func8():
        gr = GDALRaster(r"F:\ProjectSet\Shadow\ASDEHSamples\Models\ShadowImdcs\cd_shimdc_imdc.tif")
        data = gr.readGDALBand(1)
        d, n = np.unique(data, return_counts=True)
        print(n / np.sum(n))

    func8()

    return


def samplesFuncs():
    def filter_eq(_list, name, data):
        to_list = []
        for _line in _list:
            if _line[name] == data:
                to_list.append(_line)
        return to_list

    def get_data(test, cname, n_select, csv_fn):
        _df_list = pd.read_csv(csv_fn).to_dict("records")
        _df_list = filter_eq(_df_list, "TEST", test)
        _df_list = filter_eq(_df_list, "CNAME", cname)
        if len(_df_list) > n_select:
            _df_list = random.sample(_df_list, n_select)
        # print(len(_df_list))
        # if len(_df_list) != 0:
        #     _df = pd.DataFrame(_df_list)
        #     print(_df["CNAME"].value_counts())
        return _df_list

    def func1():
        to_dict = {}

        def cname_test_des(name, csv_fn):
            def filter_test(n):
                return df[df["TEST"] == n]

            df = pd.read_csv(csv_fn)
            to_dict["{}_1".format(name)] = filter_test(1)["CNAME"].value_counts()
            to_dict["{}_0".format(name)] = filter_test(0)["CNAME"].value_counts()

        cname_test_des("qd", _QD_OSPL_FN)
        cname_test_des("bj", _BJ_OSPL_FN)
        cname_test_des("cd", _CD_OSPL_FN)
        df_des = pd.DataFrame(to_dict)
        df_des.loc["SUM"] = df_des.sum()
        print(df_des)

    def func2():
        csv_fn = r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\HSPL_CD.csv"
        df_n_testing = pd.read_csv(
            r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\ChengDu\1\n_testing.txt", index_col="NAME")

        df_list = pd.read_csv(csv_fn).to_dict("records")
        test_numbers = {str(name): 0 for name in df_n_testing.index}
        print(test_numbers)
        for line in df_list:
            if line["TEST"] == 0:
                if line["CNAME"] in test_numbers:
                    test_numbers[line["CNAME"]] += 1

        data_list = []
        for name in test_numbers:
            to_n = int(df_n_testing["N"][name])
            n = test_numbers[name]
            print("{:>10} {:>6d} {:>6d}".format(name, to_n, n))
            if to_n > n:
                data_list.extend(get_data(0, name, 1000, csv_fn))
                data_list.extend(get_data(1, name, to_n - n, csv_fn))
            else:
                data_list.extend(get_data(0, name, to_n, csv_fn))

        to_df = pd.DataFrame(data_list)
        print(to_df["CNAME"].value_counts())
        to_df.to_csv(r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\ChengDu\1\cd_spl1.csv")

    def func3():
        df_qd = pd.read_csv(_QD_SPL_FN)
        df_bj = pd.read_csv(_BJ_SPL_FN)
        df_cd = pd.read_csv(_CD_SPL_FN)
        city_df = {"QingDao": df_qd, "BeiJing": df_bj, "ChengDu": df_cd}

        category_counts = {}
        for city_name in _CITY_NAMES_FULL:
            for n_test in [1, 0]:
                train_test_name = "Training" if n_test == 1 else "Testing"
                name = "{} {}".format(city_name, train_test_name)
                category_counts[name] = 0
        city_counts = category_counts.copy()

        to_list = []
        for cname in _CATEGORY_NAMES:
            category_counts = {category_counts_name: 0 for category_counts_name in category_counts}
            for spl_type, spl_name in [("FREE", ""), ("OPT", "_SH"), ("AS_SAR", "_AS_SH"), ("DE_SAR", "_DE_SH")]:
                to_dict = {"CNAME": cname, "TYPE": spl_type}
                spl_type_sum = 0
                for city_name in _CITY_NAMES_FULL:
                    df = city_df[city_name]
                    for n_test in [1, 0]:
                        train_test_name = "Training" if n_test == 1 else "Testing"
                        name = "{} {}".format(city_name, train_test_name)
                        df_count = df["CNAME"][df["TEST"] == n_test].value_counts()
                        df_count_name = "{}{}".format(cname, spl_name)
                        if df_count_name in df_count:
                            to_dict[name] = int(df_count[df_count_name])
                        else:
                            to_dict[name] = 0
                        spl_type_sum += to_dict[name]
                        category_counts[name] += to_dict[name]
                to_dict["SUM"] = spl_type_sum
                to_list.append(to_dict)
            to_list.append({"CNAME": cname, "TYPE": "SUM", **category_counts, "SUM": sum(category_counts.values())})
            for city_counts_name in city_counts:
                city_counts[city_counts_name] += category_counts[city_counts_name]
        to_list.append({"CNAME": " ", "TYPE": "CITY SUM", **city_counts, "SUM": sum(city_counts.values())})
        to_df = pd.DataFrame(to_list)
        to_df.to_csv(r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\counts.csv")
        print(to_df)

    def func4():
        csv_fn = _QD_SPLING_FN()
        to_csv_fn = r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\QingDao\2\qingdao4_de.csv"
        df = pd.read_csv(csv_fn)
        print(df.keys())
        select_names = [
            'SRT', 'X', 'Y', 'CNAME', 'CATEGORY', 'TAG', 'TEST2', 'TEST_IS', 'TEST_SH', 'TEST',
            'Blue', 'Green', 'Red', 'NIR', 'NDVI', 'NDWI', 'AS_VV', 'AS_VH', 'DE_VV', 'DE_VH'
        ]
        df = df[select_names]
        df["AS_BS_MEAN"] = (df["AS_VV"] + df["AS_VH"]) / 2
        df["DE_BS_MEAN"] = (df["DE_VV"] + df["DE_VH"]) / 2
        df["AS_VV2"] = _10EDivide10(df["AS_VV"])
        df["AS_VH2"] = _10EDivide10(df["AS_VH"])
        df["DE_VV2"] = _10EDivide10(df["DE_VV"])
        df["DE_VH2"] = _10EDivide10(df["DE_VH"])
        df["AS_BS2_MEAN"] = (df["AS_VV2"] + df["AS_VH2"]) / 2
        df["DE_BS2_MEAN"] = (df["DE_VV2"] + df["DE_VH2"]) / 2
        df = df.sort_values(["TEST", "CNAME", "DE_BS2_MEAN"], ascending=[False, True, True])
        df.to_csv(to_csv_fn, index=False)
        os.system("run qjy_txt {} -o {}  -ccn shadow2".format(to_csv_fn, changext(to_csv_fn, ".txt")))

    def func5():
        csv_fn = r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\QingDao\2\qingdao22_tiao.csv"
        txt_fn = r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\QingDao\2\qingdao22_tiao.txt"

        to_dirname = timeDirName(r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\QingDao\2\tiao", True)
        shutil.copyfile(csv_fn, os.path.join(to_dirname, os.path.basename(csv_fn)))
        shutil.copyfile(txt_fn, os.path.join(to_dirname, os.path.basename(txt_fn)))

        df = pd.read_csv(csv_fn)
        df["CNAME"] = df["CATEGORY_NAME"]
        df["CATEGORY"] = df["CATEGORY_CODE"]
        df["TEST"] = df["TEST"][pd.isna(df["TEST"])] = 1
        print(df.keys())
        print(df)
        select_names = [
            'X', 'Y', 'CNAME', "CATEGORY", 'IS_TAG', 'SRT',
            'TAG', 'TEST2', 'TEST_IS', 'TEST_SH', 'TEST', 'Blue', 'Green', 'Red',
            'NIR', 'NDVI', 'NDWI', 'AS_VV', 'AS_VH', 'DE_VV', 'DE_VH', 'AS_BS_MEAN',
            'DE_BS_MEAN', 'AS_VV2', 'AS_VH2', 'DE_VV2', 'DE_VH2', 'AS_BS2_MEAN', 'DE_BS2_MEAN'
        ]
        df = df[select_names]
        df = df.sort_values(["TEST", "CATEGORY", "AS_BS2_MEAN"], ascending=[False, True, True])
        df.to_csv(changext(csv_fn, "-back.csv"))
        print("run qjy_txt {} -o {} -ccn shadow2".format(changext(csv_fn, "-back.csv"), txt_fn))

    def func6():
        json_dict = readJson(r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\QingDao\2\HSPL_QD.geojson")
        numbers = {}
        datas = {"X": [], "Y": []}
        for feat in json_dict["features"]:
            if feat["properties"]["CATEGORY"] not in numbers:
                numbers[feat["properties"]["CATEGORY"]] = 0
            numbers[feat["properties"]["CATEGORY"]] += 1
            for name in feat["properties"]:
                if name not in datas:
                    datas[name] = []
                datas[name].append(feat["properties"][name])
            datas["X"].append(feat["geometry"]["coordinates"][0])
            datas["Y"].append(feat["geometry"]["coordinates"][1])
        names = list(numbers.keys())
        names.sort()
        show_numbers = {name: numbers[name] for name in names}
        printDict("numbers", show_numbers)
        pd.DataFrame(datas).to_csv(r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\QingDao\2\qingdao3.csv", index=False)

    def func7():
        csv_fn = r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\QingDao\2\qingdao4_de3.csv"
        df_list = pd.read_csv(csv_fn).to_dict("records")

        to_df_list = pd.read_csv(_QD_SPL_FN)
        srt_add = int(to_df_list["SRT"].max())
        to_df_list = to_df_list.to_dict("records")

        for line in df_list:
            if not pd.isna(line["SRT"]):
                oid = int(line["SRT"])
                cname = str(line["CATEGORY_NAME"])
                is_find = False
                for line2 in to_df_list:
                    if int(line2["SRT"]) == oid:
                        if line2["CNAME"] != cname:
                            line2["CNAME"] = cname
                            line2["CATEGORY"] = int(line["CATEGORY_CODE"])
                        is_find = True
                        break
            else:
                is_find = False
            if not is_find:
                srt_add += 1
                to_df_list.append({
                    "SRT": srt_add, "X": line["X"], "Y": line["Y"], "CNAME": line["CATEGORY_NAME"],
                    "CATEGORY": int(line["CATEGORY_CODE"]), "TAG": "SELECT", "TEST2": 1,
                    "TEST_IS": 0, "TEST_SH": 0, "TEST": 1,
                })
        to_df = pd.DataFrame(to_df_list)
        to_df.to_csv(changext(csv_fn, "-update.csv"), index=False)
        print(to_df)

    def func8():
        csv_fn = _QD_SPLING_FN()
        print(_QD_SAMPLES_DFN.fn(r"3\spl_select.csv"))
        df_n_testing = pd.read_csv(_QD_SAMPLES_DFN.fn(r"3\spl_select.csv"), index_col="NAME")

        data_list = []
        for name in df_n_testing.index:
            to_n = int(df_n_testing["N"][name])
            add_list = get_data(1, name, to_n, csv_fn)
            print("{:>10} {:>6d} {:>6d}".format(name, to_n, len(add_list)))
            data_list.extend(add_list)
        add_list = filter_eq(pd.read_csv(csv_fn).to_dict("records"), "TEST", 0)
        print("{:>10} {:>6d} {:>6d}".format("TEST", -1, len(add_list)))
        data_list.extend(add_list)

        to_df = pd.DataFrame(data_list)
        print(to_df["CNAME"].value_counts())
        to_df.to_csv(_QD_SAMPLES_DFN.fn(r"3\spl_select_qd.csv"), index=False)

    def func9():
        rrc = RasterRandomCoors(_BJ_RASTER_FN)
        csv_fn = _BJ_SAMPLES_DFN.fn("2", "bj2_random300.csv")
        pd.DataFrame(rrc.random(300)).to_csv(csv_fn, index=False)
        GDALSamplingFast(_BJ_RASTER_FN).csvfile(csv_fn, csv_fn)

    def func10():
        GDALSamplingFast(_QD_RASTER_FN).csvfile(
            r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\QingDao\3\spl_select_qd.csv",
            r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\QingDao\3\spl_select_qd.csv",
        )

    def func11():
        df = pd.read_csv(_QD_SPLING_FN())
        df = sampleTypesToDF(df, _QD_SAMPLES_DFN.fn("3", r"spl_select3.txt"))
        df.to_csv(_QD_SAMPLES_DFN.fn("3", r"spl_select3.csv"), index=False)

    def func12():
        sample_update = _SampleUpdate("update1", "qd", "training", _QD_SAMPLES_DFN.fn("3"))
        # sample_update.to_txt()
        # df = sample_update.counts()
        # df.to_csv(_QD_SAMPLES_DFN.fn("3", "qd_data.csv"))

        df = sample_update.update()
        df = sampleTypesToDF(df, _QD_SAMPLES_DFN.fn("3", r"qd_spl_select.txt"))
        df.to_csv(_QD_SAMPLES_DFN.fn("3", r"qd_spl_select.csv"), index=False)

        # sample_update.to_txt_testIS("20240718H145329")
        # sample_update.testIS("20240718H145329")

        # sample_update.to_txt_testSH("20240718H145329")
        # sample_update.testSH("20240718H145329")
        # df = sample_update.cat(
        #     test_is_fn=_QD_SAMPLES_DFN.fn("3", "update2-test_is.csv"),
        #     test_sh_fn=_QD_SAMPLES_DFN.fn("3", "update3-test_sh.csv"),
        #     update_dict={
        #         4496: "IS_SH", 4113: "VEG_SH", 4136: "IS_SH",
        #         4143: "IS_SH", 4216: "VEG_SH", 3981: "WAT_SH",
        #     })
        # df.to_csv(_QD_SAMPLES_DFN.fn("3", "update-test.csv"), index=False)

    def func13():
        sample_update = _SampleUpdate("update3", "bj", "test_sh", _BJ_SAMPLES_DFN.fn("2"))
        sample_update.csv_fn = _sampleCSV(
            _BJ_RASTER_FN,
            r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\BeiJing\2\bj_test_spl.csv", True
        )

        # # Training
        # sample_update.to_txt()
        # df = sample_update.update()
        # df = sampleTypesToDF(df, _BJ_SAMPLES_DFN.fn("2", r"bj_spl_select.txt"))
        # df.to_csv(_BJ_SAMPLES_DFN.fn("2", r"bj_spl_select.csv"), index=False)

        # # TEST IS
        # sample_update.to_txt_testIS("20240719H162158")
        # sample_update.testIS("20240719H162158")

        # # TEST SH
        # sample_update.to_txt_testSH("20240719H162158")
        sample_update.testSH("20240719H162158")

    def func14():
        sample_update = _SampleUpdate("update3", "cd", "test_sh", _CD_SAMPLES_DFN.fn("2"))
        sample_update.csv_fn = _sampleCSV(
            _CD_RASTER_FN,
            r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\ChengDu\2\cd_test_spl.csv", True
        )

        # # Training
        # sample_update.to_txt()
        # df = sample_update.update()
        # df = sampleTypesToDF(df, _CD_SAMPLES_DFN.fn("2", r"cd_spl_select.txt"))
        # df.to_csv(_CD_SAMPLES_DFN.fn("2", r"cd_spl_select.csv"), index=False)

        # TEST IS
        # sample_update.to_txt_testIS("20240719H163237")
        # sample_update.testIS("20240719H163237")

        # # TEST SH
        # sample_update.to_txt_testSH("20240719H163237")
        sample_update.testSH("20240719H163237")

    def func15():
        fns = [
            r"F:\ProjectSet\Shadow\ASDEHSamples\Models\20240719H162158\bj_FREE-Opt_imdc.tif",
            r"F:\ProjectSet\Shadow\ASDEHSamples\Models\20240719H162158\bj_FREE-Opt-AS_imdc.tif",
            r"F:\ProjectSet\Shadow\ASDEHSamples\Models\20240719H162158\bj_FREE-Opt-AS-DE_imdc.tif",
            r"F:\ProjectSet\Shadow\ASDEHSamples\Models\20240719H162158\bj_FREE-Opt-DE_imdc.tif",
            r"F:\ProjectSet\Shadow\ASDEHSamples\Models\20240719H162158\bj_Opt-Opt-AS-DE_imdc.tif",

        ]
        gr = GDALRaster(r"F:\ProjectSet\Shadow\ASDEHSamples\Models\20240719H162158\bj_SAR-Opt-AS-DE_imdc.tif")
        data = gr.readGDALBand(1)
        datas = np.array([GDALRaster(fn).readGDALBand(1) for fn in fns])
        print(datas.shape)
        to_data = (datas == data) * 1
        to_data = np.sum(to_data, axis=0)
        to_data = (to_data == 0) * 1
        print(to_data.shape)
        to_data = (to_data == 1) & (to_data != datas[4])

        gr.save(
            to_data.astype("int8"),
            r"F:\ProjectSet\Shadow\ASDEHSamples\Models\20240719H162158\bj_SAR-Opt-AS-DE_imdc2.tif",
            fmt="GTiff",
            dtype=gdal.GDT_Byte,
        )

    func14()


def accuracyDirectory(csv_fn, mod_dirname):
    df = pd.read_csv(csv_fn)
    df, names = _samplingImdc(df, _MODEL_DFN.fn(mod_dirname), True)
    df["CATEGORY2"] = (df["CATEGORY"] / 10).round()
    acc_dict = {}

    for name in names:
        to_dict = {}
        print("#", name, "-" * 6)

        def cal_cm(name_show, _type):
            if _type == 1:
                _df = df[df["TEST_IS"] == 1]
            else:
                _df = df[df["TEST_SH"] == 1]

            cm = ConfusionMatrix(class_names=_CATEGORY_NAMES)
            cm.addData(_df["CATEGORY2"].tolist(), _df[name].tolist())
            print(name_show)
            print(cm.fmtCM())
            to_dict["IS OA {}".format(name_show)] = cm.accuracyCategory("IS").OA()
            to_dict["IS Kappa {}".format(name_show)] = cm.accuracyCategory("IS").getKappa()

        cal_cm("NOSH", 1)
        cal_cm("SH", 2)
        acc_dict[name] = to_dict
        printDict("", to_dict)

    print(pd.DataFrame(acc_dict).T.sort_values("IS OA NOSH", ascending=False))


def run():
    sw = SRTWriteText(_MODEL_DFN.fn("Models.txt"), "a")
    csv_fn = None

    for model in ["rf", "svm"]:
        def func1():
            # csv_fn = _QD_SAMPLES_DFN.fn("3", r"qd_spl_select.csv")
            # csv_fn = _sampleCSV(_QD_RASTER_FN, csv_fn, True, changext(csv_fn, "_spl.csv"))
            _TrainImdc("qd", csv_fn=csv_fn, model=model, is_save=True, is_imdc=True, feat_funcs_type=None).fit(sw=sw)

        def func2():
            # csv_fn = _BJ_SAMPLES_DFN.fn("2", r"bj_spl_select.csv")
            # csv_fn = _sampleCSV(_BJ_RASTER_FN, csv_fn, True, changext(csv_fn, "_spl.csv"))
            _TrainImdc("bj", csv_fn=csv_fn, model=model, is_save=True, is_imdc=True, feat_funcs_type=None).fit(sw=sw)

        def func3():
            # csv_fn = _CD_SAMPLES_DFN.fn("2", r"cd_spl_select.csv")
            # csv_fn = _sampleCSV(_CD_RASTER_FN, csv_fn, True, changext(csv_fn, "_spl.csv"))
            _TrainImdc("cd", csv_fn=csv_fn, model=model, is_save=True, is_imdc=True, feat_funcs_type=None).fit(sw=sw)

        func1()
        func2()
        func3()


def main():
    accuracyDirectory(
        # _MODEL_DFN.fn(r"20240718H145329\update3-test_sh.csv"),
        r"F:\ProjectSet\Shadow\ASDEHSamples\Samples\QingDao\3\update-test.csv",
        r"20240718H145329",
    )

    return


def draw():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    from PIL import Image

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

    def func1():
        df = pd.read_csv(_CD_SPLING_FN())
        df_training = df[df["TEST"] == 1]
        df_test = df[df["TEST"] == 1]
        df_training = df_training[df["AS_VV"] < -13]
        df_training = df_training[df["DE_VV"] < -13]
        feat_funcs = featFuncs()

        # df = feat_funcs.fits(df)

        def hist(_cname, _x_key, color):
            _data = df[df["CNAME"] == _cname][_x_key].values
            hist_data, bin_edges = np.histogram(_data, bins=256)
            hist_data = hist_data / np.sum(hist_data)
            plt.plot(bin_edges[1:], hist_data, color=color, label="{} {}".format(_cname, _x_key))

        hist("IS", "AS_VV", "red")
        hist("IS", "DE_VV", "green")
        # hist("IS", "AS_VH", "green")
        # hist("VEG", "AS_VV", "yellow")
        # hist("VEG", "AS_VH", "blue")

        plt.legend()
        plt.show()

    def func2():
        df_list = pd.read_csv(_QD_SPLING_FN()).to_dict("records") + \
                  pd.read_csv(_BJ_SPLING_FN()).to_dict("records") + \
                  pd.read_csv(_CD_SPLING_FN()).to_dict("records")

        #     'IS', 'IS_SH', 'IS_AS_SH', 'IS_DE_SH',
        #     'VEG', 'VEG_SH', 'VEG_AS_SH', 'VEG_DE_SH',
        #     'SOIL', 'SOIL_SH', 'SOIL_AS_SH', 'SOIL_DE_SH',
        #     'WAT', 'WAT_SH', 'WAT_AS_SH', 'WAT_DE_SH'

        df_is_as_sh = _DFFilter(_list=df_list).all(("CNAME", "==", "IS_AS_SH")).df()
        df_is_de_sh = _DFFilter(_list=df_list).all(("CNAME", "==", "IS_DE_SH")).df()

        df_veg_as_sh = _DFFilter(_list=df_list).all(("CNAME", "==", "VEG_AS_SH")).df()
        df_veg_de_sh = _DFFilter(_list=df_list).all(("CNAME", "==", "VEG_DE_SH")).df()

        df_soil_as_sh = _DFFilter(_list=df_list).all(("CNAME", "==", "SOIL_AS_SH")).df()
        df_soil_de_sh = _DFFilter(_list=df_list).all(("CNAME", "==", "SOIL_DE_SH")).df()

        df_wat_as_sh = _DFFilter(_list=df_list).all(("CNAME", "==", "WAT_AS_SH")).df()
        df_wat_de_sh = _DFFilter(_list=df_list).all(("CNAME", "==", "WAT_DE_SH")).df()

        show_list = [
            ("IS", df_is_as_sh, df_is_de_sh), ("VEG", df_veg_as_sh, df_veg_de_sh),
            ("SOIL", df_soil_as_sh, df_soil_de_sh), ("WAT", df_wat_as_sh, df_wat_de_sh),
        ]

        name_dict = {"IS": "(a) Impervious Surface", "VEG": "(b) Vegetation", "SOIL": "(c) Soil", "WAT": "(d) Water"}

        def show1():
            plt.rcParams['font.size'] = 12
            plt.rc('text', usetex=True)
            plt.figure(figsize=(12, 4))
            plt.subplots_adjust(top=0.95, bottom=0.05, left=0.09, right=0.99, hspace=0.4, wspace=0.4)

            for i, (name, df_as_sh, df_de_sh) in enumerate(show_list):
                # plt.figure(figsize=(6, 6))
                # plt.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.8, hspace=0.2, wspace=0.2)

                plt.subplot(1, 4, i + 1)
                ax = plt.gca()
                ax.set_aspect("equal", adjustable='box')
                ax.axline((0, 0), (1, 1), color="#262626", linewidth=1)
                x, y = (df_as_sh["AS_VV"] + df_as_sh["AS_VH"]) / 2, (df_as_sh["DE_VV"] + df_as_sh["DE_VH"]) / 2
                plt.scatter(x, y, marker="^", s=56, facecolor='none', edgecolor="#BF1D2D", label="Ascending Shadow")
                x, y = (df_de_sh["AS_VV"] + df_de_sh["AS_VH"]) / 2, (df_de_sh["DE_VV"] + df_de_sh["DE_VH"]) / 2
                plt.scatter(x, y, marker="o", s=56, facecolor='none', edgecolor="#293890", label="Descending Shadow")
                plt.title(name_dict[name])
                plt.legend(frameon=False, prop={"size": 9})
                plt.xlim([-30, 30])
                plt.ylim([-30, 30])
                plt.xlabel("Ascending $(VV+VH)/2$\n")
                plt.ylabel("Descending $(VV+VH)/2$")

            fn = r"F:\ASDEWrite\Images\4.1\ADDE_SH_mean.svg"
            plt.savefig(fn, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.show()

        def show2():
            plt.rcParams['font.size'] = 12
            plt.rc('text', usetex=True)
            plt.figure(figsize=(12, 4))
            plt.subplots_adjust(top=0.95, bottom=0.15, left=0.09, right=0.99, hspace=0.2, wspace=0.2)

            for i, (name, df_as_sh, df_de_sh) in enumerate(show_list):
                # plt.figure(figsize=(6, 6))
                # plt.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.8, hspace=0.2, wspace=0.2)

                plt.subplot(1, 4, i + 1)
                ax = plt.gca()
                ax.set_aspect("equal", adjustable='box')
                ax.axline((0, 0), (1, 1), color="#262626", linewidth=1)
                x, y = (df_as_sh["AS_VV"] + df_as_sh["AS_VH"]) / 2, (df_as_sh["DE_VV"] + df_as_sh["DE_VH"]) / 2
                plt.scatter(x, y, marker="^", s=56, facecolor='none', edgecolor="#BF1D2D", label="Ascending Shadow")
                x, y = (df_de_sh["AS_VV"] + df_de_sh["AS_VH"]) / 2, (df_de_sh["DE_VV"] + df_de_sh["DE_VH"]) / 2
                plt.scatter(x, y, marker="o", s=56, facecolor='none', edgecolor="#293890", label="Descending Shadow")
                plt.title(name_dict[name])
                plt.xlim([-30, 20])
                plt.ylim([-30, 20])
                plt.xlabel("Ascending $(VV+VH)/2$\n")
                if i == 0:
                    plt.ylabel("Descending $(VV+VH)/2$")
                    plt.legend(
                        loc='upper left', bbox_to_anchor=(1.16, -0.25), borderaxespad=0.05,
                        prop={"size": 14}, frameon=False, ncol=2,
                    )

            fn = r"F:\ASDEWrite\Images\4.1\ADDE_SH_mean2.svg"
            plt.savefig(fn, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.show()

        show2()

    def func3():
        gdi = GDALDrawImages(win_size=(31, 31))
        qd_name = gdi.addGeoRange(_QD_RASTER_FN, _QD_RANGE_FN)
        bj_name = gdi.addGeoRange(_BJ_RASTER_FN, _BJ_RANGE_FN)
        cd_name = gdi.addGeoRange(_CD_RASTER_FN, _CD_RANGE_FN)
        gdi.addCategoryColor("color", {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255)})

        gdi.addRasterCenterCollection("RGB", bj_name, cd_name, qd_name, channel_list=["Red", "Green", "Blue"])
        gdi.addRasterCenterCollection("NRG", bj_name, cd_name, qd_name, channel_list=["NIR", "Red", "Green"])
        gdi.addRasterCenterCollection("AS_VV", bj_name, cd_name, qd_name, channel_list=["AS_VV"])
        gdi.addRasterCenterCollection("DE_VV", bj_name, cd_name, qd_name, channel_list=["DE_VV"])

        column_names = ["RGB", "NRG", "AS SAR", "DE SAR"]
        row_names = []

        def add_row(name, x, y):
            n_row = len(row_names)
            gdi.addAxisDataXY(n_row, 0, "RGB", x, y, min_list=[200, 200, 200, ], max_list=[2000, 2000, 2000, ])
            gdi.addAxisDataXY(n_row, 1, "NRG", x, y, min_list=[200, 200, 200, ], max_list=[3000, 2000, 2000, ])
            gdi.addAxisDataXY(n_row, 2, "AS_VV", x, y, min_list=[-14], max_list=[6])
            gdi.addAxisDataXY(n_row, 3, "DE_VV", x, y, min_list=[-14], max_list=[6])
            row_names.append(name)

        add_row(" ", 116.4601889, 39.9578624)

        gdi.draw(n_rows_ex=3.0, n_columns_ex=3.0, row_names=row_names, column_names=column_names)
        fn = r"F:\ASDEWrite\流程图\fig1.jpg"
        plt.savefig(fn, dpi=300)
        remove_white_border(fn, fn)
        plt.show()

    def func4():
        remove_white_border(r"F:\ASDEWrite\流程图\流程图2.jpg")

    def func5():
        gdi = GDALDrawImages(win_size=(31, 31))

        qd_name = gdi.addGeoRange(_QD_RASTER_FN, _QD_RANGE_FN)
        bj_name = gdi.addGeoRange(_BJ_RASTER_FN, _BJ_RANGE_FN)
        cd_name = gdi.addGeoRange(_CD_RASTER_FN, _CD_RANGE_FN)
        gdi.addCategoryColor("color", {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255)})

        gdi.addRCC("RGB", bj_name, cd_name, qd_name, channel_list=["Red", "Green", "Blue"])
        gdi.addRCC("NRG", bj_name, cd_name, qd_name, channel_list=["NIR", "Red", "Green"])
        gdi.addRCC("AS_VV", bj_name, cd_name, qd_name, channel_list=["AS_VV"])
        gdi.addRCC("DE_VV", bj_name, cd_name, qd_name, channel_list=["DE_VV"])

        column_names = ["NRG", "AS SAR", "DE SAR"]
        row_names = []

        def add_row(name, x, y):
            n_row = len(row_names)
            gdi.addAxisDataXY(n_row, 0, "RGB", x, y, min_list=[200, 200, 200, ], max_list=[2000, 2000, 2000, ])
            gdi.addAxisDataXY(n_row, 1, "NRG", x, y, min_list=[200, 200, 200, ], max_list=[3000, 2000, 2000, ])
            gdi.addAxisDataXY(n_row, 2, "AS_VV", x, y, min_list=[-14], max_list=[6])
            gdi.addAxisDataXY(n_row, 3, "DE_VV", x, y, min_list=[-14], max_list=[6])
            row_names.append(name)

        add_row(" ", 116.4601889, 39.9578624)

        gdi.draw(n_rows_ex=3.0, n_columns_ex=3.0, row_names=row_names, column_names=column_names)
        fn = r"F:\ASDEWrite\流程图\fig1.jpg"
        plt.savefig(fn, dpi=300)
        remove_white_border(fn, fn)
        plt.show()

    func5()


if __name__ == "__main__":
    draw()

r"""
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.ShadowASDEHSamples import trainimdcMain; trainimdcMain()"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.ShadowASDEHSamples import samplesFuncs; samplesFuncs()"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.ShadowASDEHSamples import training; training('qd')"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.ShadowASDEHSamples import run; run()"
"""
