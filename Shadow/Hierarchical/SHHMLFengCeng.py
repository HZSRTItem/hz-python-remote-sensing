# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHHMLFengCeng.py
@Time    : 2024/3/8 21:06
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHHMLFengCeng
-----------------------------------------------------------------------------"""
import datetime
import os
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from SRTCodes.GDALRasterClassification import GDALImdcAcc
from SRTCodes.GDALRasterIO import GDALRasterChannel, GDALRaster, tiffAddColorTable
from SRTCodes.GDALUtils import GDALRastersSampling, GDALSamplingFast
from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.NumpyUtils import reHist
from SRTCodes.OGRUtils import sampleSpaceUniform
from SRTCodes.Utils import timeDirName, DirFileName, SRTWriteText, saveJson, readJson, Jdt, filterFileExt, \
    SRTDFColumnCal, getfilenamewithoutext, printList, datasCaiFen, changefiledirname, changext, readLines, \
    changefilename, readLinesList, numberfilename, concatCSV, SRTLog
from Shadow.Hierarchical import SHHConfig
from Shadow.Hierarchical.SHHConfig import SHH_COLOR8, categoryMap
from Shadow.Hierarchical.ShadowHSample import SHH2_SPL
from Shadow.ShadowTraining import trainSVM_RandomizedSearchCV

pd.set_option('display.width', 500)
pd.set_option('max_colwidth', 200)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def feat_norm(d1, d2):
    if isinstance(d1, np.ndarray):
        d1[np.isnan(d1)] = 1
        d1[np.isnan(d2)] = -1.0001
    return (d1 - d2) / (d1 + d2)


def ext_feat(data, *feat_names):
    for feat_name in feat_names:
        feat_name = feat_name.lower()
        if feat_name == "ndvi":
            data[feat_name] = feat_norm(data["B8"], data["B4"])
        elif feat_name == "ndwi":
            data[feat_name] = feat_norm(data["B3"], data["B8"])
        elif feat_name == "mndwi":
            data[feat_name] = feat_norm(data["B3"], data["B12"])
        elif feat_name == "as_vhdvv":
            data["AS_VHDVV"] = data["AS_VH"] - data["AS_VV"]
        elif feat_name == "de_vhdvv":
            data["DE_VHDVV"] = data["DE_VH"] - data["DE_VV"]


def df_get_data(df, x_keys, test_n, fengcheng=-1, cate_name=None):
    # X
    if fengcheng != -1:
        if "TEST" in df:
            df_train = df[df["TEST"] == test_n]
        else:
            df_train = df
        df_train = df_train[df_train["FEN_CENG"] == fengcheng]
    else:
        if "TEST" in df:
            df_train = df[df["TEST"] == test_n]
        else:
            df_train = df
    x = df_train[x_keys].values
    # Y
    if fengcheng != -1:
        if cate_name is None:
            cate_name = "FC"
        y = df_train[cate_name].values
    else:
        if cate_name is None:
            cate_name = "__CODE__"
        y = df_train[cate_name].values
    return x, y


def read_geo_raster(geo_fn, imdc_keys, glcm_fn=None, range_dict=None):
    grc = GDALRasterChannel()
    grc.addGDALDatas(geo_fn, imdc_keys)
    if glcm_fn is not None:
        grc.addGDALDatas(glcm_fn)
    ext_feat(grc, *imdc_keys)
    if range_dict is not None:
        grc = dfTo01(grc, range_dict)
    data = grc.fieldNamesToData(*imdc_keys)
    return data


def feature_importance(mod: RandomForestClassifier, show_keys):
    fi = mod.feature_importances_
    show_data = list(zip(show_keys, fi))
    print(show_data)
    show_data.sort(key=lambda _elem: _elem[1], reverse=False)
    print(show_data)
    show_data = dict(show_data)
    plt.barh(list(show_data.keys()), list(show_data.values()))
    plt.show()


def dfTo01(df, range_dict=None, range_dict_fn=None):
    range_dict = getRangeDict(range_dict, range_dict_fn)
    for k in range_dict:
        if k in df:
            x_min, x_max = tuple(range_dict[k])
            data = df[k]
            data = np.clip(data, a_min=x_min, a_max=x_max)
            df[k] = data
    return df, range_dict


def getRangeDict(range_dict, range_dict_fn):
    if range_dict is None:
        range_dict = {}
    if range_dict_fn is not None:
        lines = readLinesList(range_dict_fn, " ")
        for line in lines:
            name, x_min, x_max = line[0], float(line[1]), float(line[2])
            range_dict[name] = (x_min, x_max)
    return range_dict


class FenCeng:

    def __init__(self, log_wt, init_dfn):
        self.ws_keys = None
        self.wat_sh_clf = None
        self.is_keys = None
        self.is_soil_clf = None
        self.vhl_keys = None
        self.veg_high_low_clf = None
        self.log_wt = log_wt
        self.init_dfn = init_dfn
        self.imdc_keys = []
        self.vhl_index_list = []
        self.is_index_list = []
        self.ws_index_list = []

    def initVHL(self, clf, x_keys):
        self.veg_high_low_clf = clf
        self.vhl_keys = x_keys
        self.logWT("VEG_HIGH_LOW_CLF:", self.veg_high_low_clf)
        self.logWT("VHL_KEYS:", self.vhl_keys)

    def initIS(self, clf, x_keys):
        self.is_soil_clf = clf
        self.is_keys = x_keys
        self.logWT("IS_SOIL_CLF:", self.is_soil_clf)
        self.logWT("IS_KEYS:", self.is_keys)

    def initWS(self, clf, x_keys):
        self.wat_sh_clf = clf
        self.ws_keys = x_keys
        self.logWT("WAT_SH_CLF:", self.wat_sh_clf)
        self.logWT("WS_KEYS:", self.ws_keys)

    def get_keys(self):
        to_keys = []

        def _get_keys(_list):
            for k in _list:
                if k not in to_keys:
                    to_keys.append(k)

        _get_keys(self.vhl_keys)
        _get_keys(self.is_keys)
        _get_keys(self.ws_keys)
        return to_keys

    def init_imdc_keys(self, imdc_keys: list):
        self.imdc_keys = imdc_keys
        self.vhl_index_list = [self.imdc_keys.index(k) for k in self.vhl_keys]
        self.is_index_list = [self.imdc_keys.index(k) for k in self.is_keys]
        self.ws_index_list = [self.imdc_keys.index(k) for k in self.ws_keys]

    def fit(self, df, cate_name=None, test_cate_name=None):
        def get_data(x_keys, test_n, fengcheng=-1):
            return df_get_data(df, x_keys, test_n, fengcheng, cate_name=cate_name)

        def _train(n, _clf, fit_keys, name):
            _x_train, _y_train = get_data(fit_keys, 1, n)
            _x_test, _y_test = get_data(fit_keys, 0, n)
            _clf.fit(_x_train, _y_train)
            print(name, "train acc:", _clf.score(_x_train, _y_train))
            self.log_wt.write(name, "train acc:", _clf.score(_x_train, _y_train))
            print(name, "test acc:", _clf.score(_x_test, _y_test))
            self.log_wt.write(name, "test acc:", _clf.score(_x_test, _y_test))

        _train(1, self.veg_high_low_clf, self.vhl_keys, "veg_high_low")
        _train(2, self.is_soil_clf, self.is_keys, "is_soil")
        _train(3, self.wat_sh_clf, self.ws_keys, "wat_sh")

        if test_cate_name is not None:
            _x_keys = self.get_keys()
            self.init_imdc_keys(_x_keys)
            x_test, y_test = df_get_data(df, _x_keys, 0, cate_name=test_cate_name)
            y_pred = self.predict(x_test)
            cm = ConfusionMatrix(8, SHHConfig.SHH_CNAMES8)
            cm.addData(y_test, y_pred)
            self.logWT(cm.fmtCM())

    def predict(self, x):
        y1_d = self.veg_high_low_clf.predict(x[:, self.vhl_index_list])
        y2_d = self.is_soil_clf.predict(x[:, self.is_index_list])
        y3_d = self.wat_sh_clf.predict(x[:, self.ws_index_list])
        y = []
        for i in range(len(x)):
            y1, y2, y3 = y1_d[i], y2_d[i], y3_d[i]
            if y1 == 1:
                y.append(2)
            elif y1 == 2:
                if y2 == 1:
                    y.append(1)
                elif y2 == 2:
                    y.append(3)
                else:
                    y.append(0)
            elif y1 == 3:
                if y3 == 5:
                    y.append(4)
                else:
                    y.append(y3 + 4)
            else:
                y.append(0)
        return np.array(y)

    def save(self, to_fn=None):
        if to_fn is None:
            to_fn = self.init_dfn.fn("fc")
        json_fn = to_fn + ".mod.json"
        vhl_fn = to_fn + "_vhl.mod"
        joblib.dump(self.veg_high_low_clf, vhl_fn)
        is_fn = to_fn + "_is.mod"
        joblib.dump(self.is_soil_clf, is_fn)
        ws_fn = to_fn + "_ws.mod"
        joblib.dump(self.wat_sh_clf, ws_fn)
        to_dict = {"VHL_FN": vhl_fn, "IS_FN": is_fn, "WS_FN": ws_fn}
        saveJson(to_dict, json_fn)
        return {to_fn: to_dict}

    def load(self, json_fn):
        json_dict = readJson(json_fn)
        self.veg_high_low_clf = joblib.load(json_dict["VHL_FN"])
        self.is_soil_clf = joblib.load(json_dict["IS_FN"])
        self.wat_sh_clf = joblib.load(json_dict["WS_FN"])

    def logWT(self, *text, sep=" ", end="\n", is_print=True):
        self.log_wt.write(*text, sep=sep, end=end)
        if is_print:
            print(*text, sep=sep, end=end)


class NoFenCeng:

    def __init__(self, log_wt, init_dfn):
        self.log_wt = log_wt
        self.init_dfn = init_dfn
        self.nofc_x_keys = []
        self.clf = None

    def initXKeys(self, *x_keys):
        x_keys = datasCaiFen(x_keys)
        self.nofc_x_keys = x_keys
        self.log_wt.write("NOFC_X_KEYS:", x_keys)

    def fit(self, df, cate_name=None, filter_eqs=None):
        if filter_eqs is not None:
            filter_eqs = {}

        def get_data(x_keys, test_n, fengcheng=-1):
            return df_get_data(df, x_keys, test_n, fengcheng, cate_name=cate_name)

        _clf = self.clf
        printList("NOFC_X_KEYS", self.nofc_x_keys)
        x_train, y_train = get_data(self.nofc_x_keys, 1)
        x_test, y_test = get_data(self.nofc_x_keys, 0)
        _clf.fit(x_train, y_train)
        print("train acc:", _clf.score(x_train, y_train))
        self.log_wt.write("train acc:", _clf.score(x_train, y_train))
        print("test acc:", _clf.score(x_test, y_test))
        self.log_wt.write("test acc:", _clf.score(x_test, y_test))
        self.log_wt.write("CLF", _clf)
        print()
        return _clf

    def predict(self, x):
        return self.clf.predict(x)


def imdcFit(clf, imdc_keys, geo_fn, to_geo_fn, code_colors, glcm_fn=None, range_dict=None):
    d = read_geo_raster(geo_fn, imdc_keys, glcm_fn, range_dict=range_dict)
    d[np.isnan(d)] = 0.0
    print(d.shape)
    imdc = np.zeros(d.shape[1:])
    jdt = Jdt(imdc.shape[0], "Imdc")
    jdt.start()
    for i in range(imdc.shape[0]):
        d_tmp = d[:, i, :].T
        y_tmp = clf.predict(d_tmp)
        imdc[i, :] = y_tmp
        jdt.add()
    jdt.end()
    to_fn = to_geo_fn
    gr = GDALRaster(geo_fn)
    gr.save(d=imdc, save_geo_raster_fn=to_fn, fmt="GTiff", dtype=gdal.GDT_Byte, options=["COMPRESS=PACKBITS"])
    tiffAddColorTable(to_fn, code_colors=code_colors)
    return to_fn


class MLFCModel:

    def __init__(self, clf_name=None, clf=None, x_keys=None, range_dict=None, city_name=None):
        self.clf_name = clf_name
        self.clf = clf
        self.x_keys = x_keys
        self.range_dict = range_dict
        self.city_name = city_name

    def getRangeDict(self, range_dict=None, range_dict_fn=None, city_name=None):
        if city_name is not None:
            if city_name == "bj":
                range_dict_fn = r"G:\ImageData\SHH2BeiJingImages\range.txt"
            if city_name == "cd":
                range_dict_fn = r"G:\ImageData\SHH2ChengDuImages\range.txt"
            if city_name == "qd":
                range_dict_fn = r"G:\ImageData\SHH2QingDaoImages\range.txt"
        self.range_dict = getRangeDict(range_dict, range_dict_fn)
        return range_dict

    def xRange(self, x):
        def func_x_range():
            if self.range_dict is None:
                return x
            for k in self.range_dict:
                if k in self.x_keys:
                    i_data = self.x_keys.index(k)
                    x_min, x_max = tuple(self.range_dict[k])
                    data = x[:, i_data]
                    data = np.clip(data, a_min=x_min, a_max=x_max)
                    data = (data - x_min) / (x_max - x_min)
                    x[:, i_data] = data

        if self.clf_name == "svm":
            func_x_range()

        return x

    def fit(self, X, y, sample_weight=None):
        X = self.xRange(X)
        return self.clf.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        X = self.xRange(X)
        return self.clf.predict(X)

    def score(self, X, y, sample_weight=None):
        X = self.xRange(X)
        return self.clf.score(X, y, sample_weight=sample_weight)

    def toDict(self):
        return {
            "clf_name": self.clf_name,
            "clf": self.clf,
            "x_keys": self.x_keys,
            "range_dict": self.range_dict,
            "city_name": self.city_name,
        }

    def log(self, log_wt: SRTWriteText, name, *text):
        log_wt.write("> MLFCModel[{0}]".format(name))
        to_dict = self.toDict()
        if len(text) != 0:
            log_wt.write("# ", *text)
        for k in to_dict:
            log_wt.write("  + {0}: {1}".format(k.upper(), to_dict[k]))


class SHHMLFC:

    def __init__(self, is_fenceng="fc"):
        self.is_fenceng = is_fenceng
        self.init_dirname = timeDirName(r"F:\ProjectSet\Shadow\Hierarchical\MLMods", is_mk=False)
        self.init_dirname += self.is_fenceng
        if not os.path.isdir(self.init_dirname):
            os.mkdir(self.init_dirname)
        self.init_dfn = DirFileName(self.init_dirname)
        self.log_wt = SRTWriteText(self.init_dfn.fn("log.txt"))
        self.logWT("IS_FENCENG:", self.is_fenceng)
        self.logWT("INIT_DIRNAME:", self.init_dirname)
        self.saveCodeFile(__file__)

        self.df = None
        self.range_dict = None
        self.fc = FenCeng(self.log_wt, self.init_dfn)
        self.no_fc = NoFenCeng(self.log_wt, self.init_dfn)

    def loadDF(self, df=None, csv_fn=None, excel_fn=None, sheet_name=None):
        if df is not None:
            self.df = df
        elif excel_fn is not None:
            self.df = pd.read_excel(excel_fn, sheet_name=sheet_name)
            self.log_wt.write("EXCEL_FN:", excel_fn)
            self.log_wt.write("SHEET_NAME:", sheet_name)
        elif csv_fn is not None:
            self.df = pd.read_csv(csv_fn)
            self.log_wt.write("CSV_FN:", excel_fn)
        return df

    def loadDFCity(self, city, df=None, csv_fn=None, excel_fn=None, sheet_name=None):
        self.logWT("CITY:", city)
        self.loadDF(df=df, csv_fn=csv_fn, excel_fn=excel_fn, sheet_name=sheet_name)
        self.df = self.df[self.df["CITY"] == city]
        self.logWT("DF Length:", len(self.df))

    def dfTo01(self, range_dict=None, range_dict_fn=None):
        if (range_dict is None) and (range_dict_fn is None):
            return
        self.df, self.range_dict = dfTo01(self.df, range_dict=range_dict, range_dict_fn=range_dict_fn)
        self.logWT("RANGE_DICT", self.range_dict)

    def saveCodeFile(self, code_filename):
        code_fn = os.path.split(code_filename)[1]
        self.log_wt.write("CODE_FN", code_filename)
        self.log_wt.write("CODE_TO_FN", code_fn)
        with open(self.init_dfn.fn(code_fn), "w", encoding="utf-8") as fw:
            with open(code_filename, "r", encoding="utf-8") as fr:
                fw.write(fr.read())

    def logWT(self, *text, sep=" ", end="\n", is_print=True):
        self.log_wt.write(*text, sep=sep, end=end)
        if is_print:
            print(*text, sep=sep, end=end)

    def dealDF(self):
        ext_feat(self.df, "ndvi", "ndwi", "mndwi", )
        self.logWT("DF.KEYS", self.df.keys())
        self.df.to_csv(self.init_dfn.fn("train_data.csv"), index=False)

    def fitFC(self, cate_name=None, test_cate_name=None):
        self.dealDF()
        self.fc.fit(self.df, cate_name=cate_name, test_cate_name=test_cate_name)
        to_dict = self.fc.save()
        self.log_wt.write("MODEL_SAVE", to_dict)

    def fitNoFC(self, cate_name=None, ):
        self.dealDF()
        self.df = self.df[self.df["NO_FEN_CENG"] == 1]
        clf = self.no_fc.fit(self.df, cate_name=cate_name)
        joblib.dump(clf, self.init_dfn.fn("nofc.mod"))
        self.log_wt.write("MODEL_SAVE", self.init_dfn.fn("nofc.mod"))

    def fit(self):
        if self.is_fenceng == "fc":
            return self.fitFC(cate_name="FC", test_cate_name=None)
        elif self.is_fenceng == "nofc":
            return self.fitNoFC(cate_name="NOFC", )

    def imdcFun(self, geo_fn, to_geo_fn=None, glcm_fn=None):
        if to_geo_fn is None:
            to_geo_fn = changefiledirname(geo_fn, self.init_dirname)
            to_geo_fn = changext(to_geo_fn, "_{0}_imdc.tif".format(self.is_fenceng))
        print("geo_fn   :", geo_fn)
        print("to_geo_fn:", to_geo_fn)
        print("glcm_fn  :", glcm_fn)
        self.log_wt.write("GEO_FN:", geo_fn)
        self.log_wt.write("TO_GEO_FN:", to_geo_fn)
        self.log_wt.write("GLCM_FN:", glcm_fn)

        if self.is_fenceng == "nofc":
            imdc_keys = self.no_fc.nofc_x_keys
            clf = self.no_fc
        elif self.is_fenceng == "fc":
            imdc_keys = self.fc.get_keys()
            self.fc.init_imdc_keys(imdc_keys)
            clf = self.fc
        else:
            print("imdc_fun is_fenceng == \"{}\"".format(self.is_fenceng))
            return

        self.log_wt.write("IMDC_KEYS:", imdc_keys)
        code_colors = SHH_COLOR8
        to_fn = imdcFit(clf, imdc_keys, geo_fn, to_geo_fn, code_colors, range_dict=self.range_dict)
        self.log_wt.write("CODE_COLORS:", code_colors)

        return {"GEO_FN": geo_fn, "TO_FN": to_fn}

    def imdcTiles(self, tiles_dirname, to_fn=None):
        tiles_fns = filterFileExt(tiles_dirname, ".tif")
        if to_fn is None:
            to_fn = os.path.split(tiles_dirname)[-1]
            to_fn = os.path.join(self.init_dirname, to_fn + "_{0}_imdc.tif".format(self.is_fenceng))
        to_tiles_dirname = os.path.splitext(to_fn)[0] + "_imdctiles"
        self.logWT("TILES_DIRNAME", tiles_dirname)
        self.logWT("TO_TILES_DIRNAME", to_tiles_dirname)
        if not os.path.isdir(to_tiles_dirname):
            os.mkdir(to_tiles_dirname)
        to_fn_tmps = []
        for fn in tiles_fns:
            to_fn_tmp = changext(fn, "_{0}_imdc.tif".format(self.is_fenceng))
            to_fn_tmp = changefiledirname(to_fn_tmp, to_tiles_dirname)
            to_fn_tmps.append(to_fn_tmp)
            print("Image:", fn)
            print("Imdc :", to_fn_tmp)
            if os.path.isfile(to_fn_tmp):
                print("Imdc 100%")
                continue
            self.imdcFun(fn, to_fn_tmp)
        print("Merge:", to_fn)

        from osgeo_utils.gdal_merge import main as gdal_merge_main

        gdal_merge_main(["gdal_merge_main",
                         "-of", "GTiff",
                         "-n", "0",
                         "-ot", "Byte",
                         "-co", "COMPRESS=PACKBITS",
                         "-o", to_fn,
                         *to_fn_tmps, ])
        tiffAddColorTable(to_fn, code_colors=SHH_COLOR8)


class MLFCFeatures:

    def __init__(self):
        self.opt_f = [
            "B2", "B3", "B4", "B8", "B11", "B12",
            'ndvi', 'ndwi', 'mndwi',
            "OPT_asm", "OPT_con", "OPT_cor", "OPT_dis", "OPT_ent", "OPT_hom", "OPT_mean", "OPT_var",
        ]
        self.as_f = [
            "AS_VV", "AS_VH",
            "AS_VH_asm", "AS_VH_con", "AS_VH_cor", "AS_VH_dis", "AS_VH_ent", "AS_VH_hom", "AS_VH_mean", "AS_VH_var",
            "AS_VV_asm", "AS_VV_con", "AS_VV_cor", "AS_VV_dis", "AS_VV_ent", "AS_VV_hom", "AS_VV_mean", "AS_VV_var",
        ]
        self.de_f = [
            "DE_VV", "DE_VH",
            "DE_VH_asm", "DE_VH_con", "DE_VH_cor", "DE_VH_dis", "DE_VH_ent", "DE_VH_hom", "DE_VH_mean", "DE_VH_var",
            "DE_VV_asm", "DE_VV_con", "DE_VV_cor", "DE_VV_dis", "DE_VV_ent", "DE_VV_hom", "DE_VV_mean", "DE_VV_var",
        ]

    def to_opt(self):
        return self.opt_f

    def to_opt_as(self):
        return self.opt_f + self.as_f

    def to_opt_de(self):
        return self.opt_f + self.de_f

    def to_opt_as_de(self):
        return self.opt_f + self.as_f + self.de_f


class MLFCModel_trainSVM_RandomizedSearchCV(MLFCModel):

    def __init__(self, clf_name=None, clf=None, x_keys=None, range_dict=None, city_name=None):
        super().__init__(clf_name, clf, x_keys, range_dict, city_name)

    def fit(self, X, y, sample_weight=None):
        X = self.xRange(X)
        self.clf = trainSVM_RandomizedSearchCV(x=X, y=y, n_iter=20, is_return_model=True, )
        return self.clf


def SHHMLFC_main(city_name="qd", is_fenceng="nofc"):
    mlfc = SHHMLFC(is_fenceng)
    mlfc.logWT("CITY_NAME:", city_name)
    mlfc_feats = MLFCFeatures()

    r"""
    "FEN_CENG": model of fenceng 1|2|3
    "FC": CATEGORY of fenceng 
    """
    # mlfc.loadDFCity(city_name, csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\23\sh2_spl23_fc2.csv")
    # mlfc.loadDF(csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\24\sh2_spl24_is7_spl.csv")
    mlfc.loadDF(csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\sh2_spl25_2_spl2_32.csv")

    model = MLFCModel
    model = MLFCModel_trainSVM_RandomizedSearchCV

    # RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=1, min_samples_split=2)
    # log_wt = SRTWriteText(numberfilename(r"F:\ProjectSet\Shadow\Hierarchical\Temp\tmp.txt"))

    mlfc.log_wt.write("``` models")
    if is_fenceng == "nofc":
        nofc_mod = model("svm", city_name=city_name)
        nofc_mod.clf = SVC()
        nofc_mod.x_keys = mlfc_feats.to_opt_as_de()
        nofc_mod.getRangeDict(city_name=city_name)
        nofc_mod.log(mlfc.log_wt, "NOFC")

        mlfc.no_fc.clf = nofc_mod
        mlfc.no_fc.initXKeys(nofc_mod.x_keys)
    else:
        fc_vhl_mod = model("svm", city_name=city_name)
        fc_vhl_mod.clf = SVC()
        fc_vhl_mod.x_keys = mlfc_feats.to_opt()
        fc_vhl_mod.getRangeDict(city_name=city_name)
        fc_vhl_mod.log(mlfc.log_wt, "FC_VHL_MOD")

        fc_is_mod = model("svm", city_name=city_name)
        fc_is_mod.clf = SVC()
        fc_is_mod.x_keys = mlfc_feats.to_opt_as_de()
        fc_is_mod.getRangeDict(city_name=city_name)
        fc_is_mod.log(mlfc.log_wt, "FC_IS_MOD")

        fc_ws_mod = model("svm", city_name=city_name)
        fc_ws_mod.clf = SVC()
        fc_ws_mod.x_keys = mlfc_feats.to_opt_as_de()
        fc_ws_mod.getRangeDict(city_name=city_name)
        fc_is_mod.log(mlfc.log_wt, "FC_IS_MOD")

        mlfc.fc.initVHL(clf=fc_vhl_mod, x_keys=fc_vhl_mod.x_keys)
        mlfc.fc.initIS(clf=fc_is_mod, x_keys=fc_is_mod.x_keys)
        mlfc.fc.initWS(clf=fc_ws_mod, x_keys=fc_ws_mod.x_keys)

    mlfc.log_wt.write("```")

    mlfc.fit()

    if city_name == "qd":
        mlfc.imdcTiles(r"G:\ImageData\SHH2QingDaoImages\qd_sh2_1_opt_sar_glcm")
    elif city_name == "bj":
        mlfc.imdcTiles(r"G:\ImageData\SHH2BeiJingImages\bj_sh2_1_opt_sar_glcm")
    elif city_name == "cd":
        mlfc.imdcTiles(r"G:\ImageData\SHH2ChengDuImages\cd_sh2_1_opt_sar_glcm")

    r"""
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHMLFengCeng import SHHMLFC_main; SHHMLFC_main('qd', 'nofc')"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHMLFengCeng import SHHMLFC_main; SHHMLFC_main('qd', 'fc')"


python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHMLFengCeng import SHHMLFC_main; SHHMLFC_main('bj', 'nofc')"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHMLFengCeng import SHHMLFC_main; SHHMLFC_main('bj', 'fc')"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHMLFengCeng import SHHMLFC_main; SHHMLFC_main('cd', 'nofc')"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHMLFengCeng import SHHMLFC_main; SHHMLFC_main('cd', 'fc')"
    """


def imageRange():
    def x_range(data):
        data = np.expand_dims(data, axis=0)
        print(data.shape)
        x = reHist(data, ratio=0.005)
        x_min, x_max = float(x[0][0]), float(x[0][1])
        return x_min, x_max

    def func1(_geo_fn):
        gr = GDALRaster(_geo_fn)
        data = gr.readAsArray()
        return x_range(data)

    fns = [
        r"G:\ImageData\SHH2BeiJingImages\filelist.txt",
        r"G:\ImageData\SHH2ChengDuImages\filelist.txt",
        r"G:\ImageData\SHH2QingDaoImages\filelist.txt",
    ]

    for fn in fns:
        geo_fns = readLines(fn)
        swt = SRTWriteText(changefilename(fn, "range2.txt"))
        # print(swt.text_fn)
        to_dict = {}
        for geo_fn in geo_fns:
            name = getfilenamewithoutext(geo_fn)
            to_dict[name] = geo_fn
            # x_min, x_max = func1(geo_fn)
            # swt.write(name, x_min, x_max)
            # print(geo_fn)
        print(fn)

        def load_data(name):
            return GDALRaster(to_dict[name]).readAsArray()

        red = load_data("B4")
        green = load_data("B3")
        nir = load_data("B8")
        swir2 = load_data("B12")

        data_dict = {
            "ndvi": feat_norm(nir, red),
            "ndwi": feat_norm(green, nir),
            "mndwi": feat_norm(green, swir2),
        }

        for name in data_dict:
            x_min, x_max = x_range(data_dict[name])
            print(name, x_min, x_max)


def trainMLFC():
    # 分层方法测试，使用机器学习的方法
    # python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHTemp import method_name7; method_name7()"
    def get_data(x_keys, test_n, fengcheng=-1):
        return df_get_data(df, x_keys, test_n, fengcheng)

    """
    'SRT', 'OSRT', 'X', 'Y', 'CNAME', 'CATEGORY', 'TAG', 'TEST', 'CODE',
    'CITY', 'AS_VV', 'AS_VH', 'AS_angle', 'DE_VV', 'DE_VH', 'DE_angle',
    'B2', 'B3', 'B4', 'B8', 'B11', 'B12', '__CODE__', 'FEN_CENG', 'FC',
    'OPT_mean', 'OPT_var', 'OPT_hom', 'OPT_con', 'OPT_dis', 'OPT_ent',
    'OPT_asm', 'OPT_cor', 'ndvi', 'ndwi', 'mndwi'
    """
    is_fenceng = "fc"
    init_dirname = timeDirName(r"F:\ProjectSet\Shadow\Hierarchical\MLMods", is_mk=False)
    init_dirname += is_fenceng
    if not os.path.isdir(init_dirname):
        os.mkdir(init_dirname)
    init_dfn = DirFileName(init_dirname)

    log_wt = SRTWriteText(init_dfn.fn("log.txt"))
    code_fn = os.path.split(__file__)[1]
    log_wt.write("CODE_FN", __file__)
    log_wt.write("CODE_TO_FN", code_fn)
    with open(init_dfn.fn(code_fn), "w", encoding="utf-8") as fw:
        with open(__file__, "r", encoding="utf-8") as fr:
            fw.write(fr.read())

    df_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\5\FenCengSamples_glcm.xlsx"
    df = pd.read_excel(df_fn, sheet_name="GLCM")
    log_wt.write("DF_FN:", df_fn)
    ext_feat(df, "ndvi", "ndwi", "mndwi")
    print(df.keys())
    log_wt.write("IS_FENCENG:", is_fenceng)
    nofc_x_keys = [
        'B2', 'B3', 'B4', 'B8', 'B11', 'B12',
        'ndvi', 'ndwi', 'mndwi',
        'AS_VV', 'AS_VH', 'DE_VV', 'DE_VH',
        'OPT_mean', 'OPT_var', 'OPT_hom', 'OPT_con', 'OPT_dis', 'OPT_ent', 'OPT_asm'
    ]
    log_wt.write("NOFC_X_KEYS:", nofc_x_keys)
    df.to_csv(init_dfn.fn("train_data.csv"))

    class fen_ceng:

        def __init__(self):
            self.veg_high_low_clf = RandomForestClassifier(150)
            self.vhl_keys = [
                'B2', 'B3', 'B4', 'B8', 'B11', 'B12',
                'ndvi', 'ndwi', 'mndwi',
            ]
            log_wt.write("VEG_HIGH_LOW_CLF:", self.veg_high_low_clf)
            log_wt.write("VHL_KEYS:", self.vhl_keys)
            self.is_soil_clf = RandomForestClassifier(150)
            self.is_keys = [
                'B2', 'B3', 'B4', 'B8', 'B11', 'B12',
                'AS_VV', 'AS_VH',
                'DE_VV', 'DE_VH',
                'OPT_mean', 'OPT_var', 'OPT_hom', 'OPT_con', 'OPT_dis', 'OPT_ent', 'OPT_asm',
                'ndvi', 'ndwi', 'mndwi',
            ]
            log_wt.write("IS_SOIL_CLF:", self.is_soil_clf)
            log_wt.write("IS_KEYS:", self.is_keys)
            self.wat_sh_clf = RandomForestClassifier(150)
            self.ws_keys = [
                'B2', 'B3', 'B4', 'B8', 'B11', 'B12',
                'ndvi', 'ndwi', 'mndwi',
                'AS_VV', 'AS_VH',
                'DE_VV', 'DE_VH',
                'OPT_mean', 'OPT_var', 'OPT_hom', 'OPT_con', 'OPT_dis', 'OPT_ent', 'OPT_asm',
            ]
            log_wt.write("WAT_SH_CLF:", self.wat_sh_clf)
            log_wt.write("WS_KEYS:", self.ws_keys)
            self.imdc_keys = []
            self.vhl_index_list = []
            self.is_index_list = []
            self.ws_index_list = []

        def get_keys(self):
            to_keys = []

            def _get_keys(_list):
                for k in _list:
                    if k not in to_keys:
                        to_keys.append(k)

            _get_keys(self.vhl_keys)
            _get_keys(self.is_keys)
            _get_keys(self.ws_keys)
            return to_keys

        def init_imdc_keys(self, imdc_keys: list):
            self.imdc_keys = imdc_keys
            self.vhl_index_list = [self.imdc_keys.index(k) for k in self.vhl_keys]
            self.is_index_list = [self.imdc_keys.index(k) for k in self.is_keys]
            self.ws_index_list = [self.imdc_keys.index(k) for k in self.ws_keys]

        def fit(self):
            def _train(n, _clf, fit_keys, name):
                _x_train, _y_train = get_data(fit_keys, 1, n)
                _x_test, _y_test = get_data(fit_keys, 0, n)
                _clf.fit(_x_train, _y_train)
                print(name, "train acc:", _clf.score(_x_train, _y_train))
                log_wt.write(name, "train acc:", _clf.score(_x_train, _y_train))
                print(name, "test acc:", _clf.score(_x_test, _y_test))
                log_wt.write(name, "test acc:", _clf.score(_x_test, _y_test))

            _train(1, self.veg_high_low_clf, self.vhl_keys, "veg_high_low")
            _train(2, self.is_soil_clf, self.is_keys, "is_soil")
            _train(3, self.wat_sh_clf, self.ws_keys, "wat_sh")

        def predict(self, x):
            y1_d = self.veg_high_low_clf.predict(x[:, self.vhl_index_list])
            y2_d = self.is_soil_clf.predict(x[:, self.is_index_list])
            y3_d = self.wat_sh_clf.predict(x[:, self.ws_index_list])
            y = []
            for i in range(len(x)):
                y1, y2, y3 = y1_d[i], y2_d[i], y3_d[i]
                if y1 == 1:
                    y.append(2)
                elif y1 == 2:
                    if y2 == 1:
                        y.append(1)
                    elif y2 == 2:
                        y.append(3)
                    else:
                        y.append(0)
                elif y1 == 3:
                    if y3 == 5:
                        y.append(4)
                    else:
                        y.append(y3 + 4)
                else:
                    y.append(0)
            return np.array(y)

        def save(self, to_fn=None):
            if to_fn is None:
                to_fn = init_dfn.fn("fc")
            json_fn = to_fn + ".mod.json"
            vhl_fn = to_fn + "_vhl.mod"
            joblib.dump(self.veg_high_low_clf, vhl_fn)
            is_fn = to_fn + "_is.mod"
            joblib.dump(self.is_soil_clf, is_fn)
            ws_fn = to_fn + "_ws.mod"
            joblib.dump(self.wat_sh_clf, ws_fn)
            to_dict = {"VHL_FN": vhl_fn, "IS_FN": is_fn, "WS_FN": ws_fn}
            saveJson(to_dict, json_fn)
            return {to_fn: to_dict}

        def load(self, json_fn):
            json_dict = readJson(json_fn)
            self.veg_high_low_clf = joblib.load(json_dict["VHL_FN"])
            self.is_soil_clf = joblib.load(json_dict["IS_FN"])
            self.wat_sh_clf = joblib.load(json_dict["WS_FN"])

    def fenceng():
        _clf = fen_ceng()
        _clf.fit()
        print()
        return _clf

    def not_fenceng():
        print(nofc_x_keys)
        x_train, y_train = get_data(nofc_x_keys, 1)
        x_test, y_test = get_data(nofc_x_keys, 0)
        _clf = RandomForestClassifier(150)
        _clf.fit(x_train, y_train)
        print("train acc:", _clf.score(x_train, y_train))
        log_wt.write("train acc:", _clf.score(x_train, y_train))
        print("test acc:", _clf.score(x_test, y_test))
        log_wt.write("test acc:", _clf.score(x_test, y_test))
        log_wt.write("CLF", _clf)
        print()
        return _clf

    if is_fenceng == "fc":
        clf = fenceng()
        to_dict = clf.save()
        log_wt.write("MODEL_SAVE", to_dict)
    elif is_fenceng == "nofc":
        clf = not_fenceng()
        joblib.dump(clf, init_dfn.fn("nofc.mod"))
        log_wt.write("MODEL_SAVE", init_dfn.fn("nofc.mod"))
    else:
        clf = None

    # feature_importance(clf, nofc_x_keys)

    def imdc_fun(geo_fn, to_geo_fn, glcm_fn=None):
        print("geo_fn   :", geo_fn)
        print("to_geo_fn:", to_geo_fn)
        print("glcm_fn  :", glcm_fn)
        log_wt.write("GEO_FN:", geo_fn)
        log_wt.write("TO_GEO_FN:", to_geo_fn)
        log_wt.write("GLCM_FN:", glcm_fn)

        if is_fenceng == "nofc":
            imdc_keys = nofc_x_keys
        elif is_fenceng == "fc":
            imdc_keys = clf.get_keys()
            clf.init_imdc_keys(imdc_keys)
        else:
            print("imdc_fun is_fenceng == \"{}\"".format(is_fenceng))
            return

        log_wt.write("IMDC_KEYS:", imdc_keys)
        d = read_geo_raster(geo_fn, imdc_keys, glcm_fn)
        d[np.isnan(d)] = 0.0
        print(d.shape)

        imdc = np.zeros(d.shape[1:])
        jdt = Jdt(imdc.shape[0], "Imdc")
        jdt.start()
        for i in range(imdc.shape[0]):
            d_tmp = d[:, i, :].T
            y_tmp = clf.predict(d_tmp)
            imdc[i, :] = y_tmp
            jdt.add()
        jdt.end()

        to_fn = to_geo_fn
        gr = GDALRaster(geo_fn)
        gr.save(d=imdc, save_geo_raster_fn=to_fn, fmt="GTiff", dtype=gdal.GDT_Byte, options=["COMPRESS=PACKBITS"])
        code_colors = SHH_COLOR8
        tiffAddColorTable(to_fn, code_colors=code_colors)
        log_wt.write("CODE_COLORS:", code_colors)

        print()

        return {"GEO_FN": geo_fn, "TO_FN": to_fn}

    def imdc_fns():
        dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Images")
        qd_to_fn = init_dfn.fn(r"qd_sh1_{0}.tif".format(is_fenceng))
        cd_to_fn = init_dfn.fn(r"cd_sh1_{0}.tif".format(is_fenceng))
        bj_to_fn = init_dfn.fn(r"bj_sh1_{0}.tif".format(is_fenceng))
        print(qd_to_fn, cd_to_fn, bj_to_fn, "", sep="\n")
        log_wt.write(qd_to_fn, cd_to_fn, bj_to_fn, "", sep="\n")
        imdc_dict = [
            imdc_fun(dfn.fn(r"QingDao\qd_sh2_1.tif"), qd_to_fn, dfn.fn(r"QingDao\glcm\qd_sh2_1_gray_envi_mean"), ),
            imdc_fun(dfn.fn(r"ChengDu\cd_sh2_1.tif"), cd_to_fn, dfn.fn(r"ChengDu\glcm\cd_sh2_1_gray_envi_mean"), ),
            imdc_fun(dfn.fn(r"BeiJing\bj_sh2_1.tif"), bj_to_fn, dfn.fn(r"BeiJing\glcm\bj_sh2_1_gray_envi_mean"), ),
        ]
        saveJson(imdc_dict, init_dfn.fn("imdc.json"))
        log_wt.write("IMDC_DICT", imdc_dict)
        with open(init_dfn.fn("imdc_gdaladdo.bat"), "w", encoding="utf-8") as f:
            f.write("gdaladdo \"{0}\"\n".format(qd_to_fn))
            f.write("gdaladdo \"{0}\"\n".format(cd_to_fn))
            f.write("gdaladdo \"{0}\"\n".format(bj_to_fn))

    # imdc_fns()


class t_acc_samples:

    def __init__(self, mod_dfn):
        self.samples = []
        self.sdf = SRTDFColumnCal()
        self.mod_dfn = mod_dfn

    def read_csv(self, csv_fn):
        self.sdf = SRTDFColumnCal()
        self.sdf.read_csv(csv_fn, is_auto_type=True)

    def init_train_data(self):
        csv_fn = self.mod_dfn.fn("train_data.csv")
        self.read_csv(csv_fn)
        return csv_fn

    def filterEQ(self, field_name, *data):
        self.sdf = self.sdf.filterEQ(field_name, *data)

    def sampling(self, field_name, _imdc_fn, map_dict=None):
        o_rasters_sampling = GDALRastersSampling(_imdc_fn)

        def fit_func(line: dict):
            if "TEST" in line:
                if line["TEST"] != 0:
                    return 0
            x, y = line["X"], line["Y"]
            d = o_rasters_sampling.sampling(x, y, 1, 1)
            if d is None:
                return 0
            else:
                d = d.ravel()
            cate = int(d[0])
            if map_dict is not None:
                if cate in map_dict:
                    cate = map_dict[cate]
            return cate

        return self.sdf.fit(field_name, fit_func)

    def getCategory(self, c_name, map_dict=None):
        cate = self.sdf[c_name]
        if map_dict is not None:
            cate = [map_dict[k] if k in map_dict else k for k in cate]
        return cate


def tAccMLFC():
    # python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHMLFengCeng import tAccMLFC; tAccMLFC()"
    mod_dirname = r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240308H120328nofc"
    mod_dirname = r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240308H120401fc"
    # mod_dirname = r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240308H210341nofc"
    # mod_dirname = r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240308H215552fc"
    mod_dirname = r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240521H154413nofc"
    mod_dirname = r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240521H205325fc"
    print(mod_dirname)
    mod_dfn = DirFileName(mod_dirname)

    spls = t_acc_samples(mod_dfn)
    # sample_fn = spls.init_train_data()

    sample_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\6\sh2_spl6_1_211.csv"
    sample_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\24\sh2_spl24_is74.csv"
    spls.read_csv(sample_fn)
    # spls.filterEQ("FT", "qd_random1000_1")

    fn = getfilenamewithoutext(sample_fn)
    to_wt_fn = mod_dfn.fn(fn + "_tacc2.txt")
    print(to_wt_fn)
    to_wt = SRTWriteText(to_wt_fn)
    to_wt.write("MOD_DIRNAME: {0}\n".format(mod_dirname))
    to_wt.write("SAMPLE_FN: {0}\n".format(sample_fn))

    y1_map_dict = {1: 1, 2: 2, 3: 3, 4: 4, 5: 1, 6: 2, 7: 3, 8: 4}
    # y1_map_dict = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}
    to_wt.write("Y1_MAP_DICT: {0}\n".format(y1_map_dict))

    get_cname = "CATEGORY"
    to_wt.write("CNAME: {0}\n".format(get_cname))
    y0_map_dict = {11: 1, 21: 2, 31: 3, 41: 4, 12: 1, 22: 2, 32: 3, 42: 4}
    # y0_map_dict = {11: 1, 21: 2, 31: 3, 41: 4, 12: 5, 22: 6, 32: 7, 42: 8}
    to_wt.write("Y0_MAP_DICT: {0}\n".format(y0_map_dict))
    y0 = spls.getCategory(get_cname, map_dict=y0_map_dict)

    cnames = [
        "IS", "VEG", "SOIL", "WAT",
        # "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH",
    ]
    to_wt.write("CNAMES: {0}\n".format(cnames))

    imdc_fns = filterFileExt(mod_dirname, ".tif")
    to_wt.write("IMDC_FNS: {0}\n".format(fn))
    to_csv_fn = mod_dfn.fn(fn + "_tacc.csv")

    for imdc_fn in imdc_fns:
        fn = getfilenamewithoutext(imdc_fn)
        print(fn)
        to_wt.write("> {0}".format(fn))
        y1 = spls.sampling(fn, imdc_fn, y1_map_dict)
        y0_tmp, y1_tmp = [], []
        for i in range(len(spls.sdf)):
            if y1[i] != 0:
                y0_tmp.append(y0[i])
                y1_tmp.append(y1[i])
        cm = ConfusionMatrix(len(cnames), cnames)
        cm.addData(y0, y1)
        to_wt.write(cm.fmtCM(), "\n")
        print(cm.fmtCM())

    spls.sdf.toCSV(to_csv_fn)
    print("to_csv_fn", to_csv_fn)


def showFeatImp():
    data = [('OPT_var', 0.01576080067114541), ('OPT_dis', 0.016985082447471232), ('OPT_hom', 0.017616540976901025),
            ('OPT_asm', 0.017923906302913552), ('OPT_con', 0.018356980765199923), ('OPT_mean', 0.02191201533008487),
            ('OPT_ent', 0.022383564102921045), ('DE_VH', 0.030301017021598278), ('B3', 0.03451912048116178),
            ('DE_VV', 0.03620504625396766), ('AS_VH', 0.04421460000735884), ('mndwi', 0.04650880058323637),
            ('B2', 0.0466088179151466), ('B4', 0.050477364850457555), ('AS_VV', 0.05259860124221752),
            ('B12', 0.07709950041903517), ('B8', 0.08208353069186401), ('B11', 0.08227222530772832),
            ('ndwi', 0.13597161703821775), ('ndvi', 0.15020086759137327)]
    data = [('OPT_mean', 0.015812921221597682), ('OPT_asm', 0.016189493684199623), ('OPT_hom', 0.016671560648190056),
            ('OPT_var', 0.017104535091576435), ('OPT_dis', 0.017713110302578543), ('B11', 0.017746612066257408),
            ('B3', 0.019814833820484906), ('OPT_con', 0.02069767237997769), ('OPT_ent', 0.021570300884741992),
            ('B12', 0.023372443246070834), ('B4', 0.02478219558047086), ('B8', 0.02713507870334041),
            ('B2', 0.02770782061984352), ('mndwi', 0.04243660277886134), ('DE_VH', 0.06308501142811943),
            ('DE_VV', 0.07650474615203316), ('AS_VH', 0.10794083139911925), ('AS_VV', 0.12874449936121726),
            ('ndwi', 0.1469263044503744), ('ndvi', 0.1680434261809452)]

    # data.reverse()
    print(data)
    data = dict(data)
    plt.figure()
    plt.subplots_adjust(left=0.2)
    # 'Times New Roman'
    plt.xticks(fontproperties="Times New Roman", fontsize=16)
    plt.yticks(fontproperties="Times New Roman", fontsize=16)
    plt.barh(list(data.keys()), list(data.values()))
    plt.show()


def toDictLine(ml_dfn, city_name, clf_name, mod_dirname, model_name=None):
    if model_name is None:
        if "nofc" in mod_dirname:
            model_name = "nofc"
        else:
            model_name = "fc"
    to_mod_dirname = ml_dfn.fn(mod_dirname)
    tif_fn = to_mod_dirname
    for fn in os.listdir(to_mod_dirname):
        if fn.endswith("_imdc.tif"):
            tif_fn = os.path.join(to_mod_dirname, fn)
    return {"CITY": city_name, "CLF": clf_name, "MODEL": model_name, "TIF_FN": tif_fn, }


def get_class_oa_kappa(to_list, df, map_category=None, to_map_category=None,
                       cm_cname="IS", category_field_name=None):
    """ map_category={11: 1, 21: 2, 31: 2, 41: 2, 12: 1, 22: 2, 32: 2, 42: 2} """

    if map_category is None:
        map_category = {11: 1, 21: 2, 31: 2, 41: 2, 12: 1, 22: 2, 32: 2, 42: 2}
    if to_map_category is None:
        to_map_category = {1: 1, 2: 2, 3: 2, 4: 2, 5: 1, 6: 2, 7: 2, 8: 2}

    for i, line in enumerate(to_list):
        categoryOAKappa(df, line, [cm_cname, "NO" + cm_cname], map_category, to_map_category, category_field_name,
                        cm_cname=cm_cname)

        to_list[i] = line

        print(to_list[0].keys())
        print(pd.DataFrame(to_list)[['CITY', 'CLF', 'MODEL', cm_cname + "_OA", cm_cname + "_Kappa", ]])
    return pd.DataFrame(to_list)[['CITY', 'CLF', 'MODEL', cm_cname + "_OA", cm_cname + "_Kappa", "TIF_FN"]]


def categoryOAKappa(df, line, cm_cnames, map_category, to_map_category, category_field_name,
                    is_return_cm=False, cm_cname="",
                    category_func=None, to_category_func=None,
                    ):
    tif_fn = line["TIF_FN"]
    gica = GDALImdcAcc(tif_fn)
    if category_field_name is not None:
        gica.c_column_name = category_field_name
    gica.addDataFrame(df)
    gica.map_category = map_category
    gica.to_map_category = to_map_category
    if category_func is not None:
        gica.category_func = category_func
    if to_category_func is not None:
        gica.to_category_func = to_category_func
    gica.calCM(cm_cnames)
    # print(gica.cm.fmtCM())
    line[cm_cname + "_OA"] = gica.cm.OA()
    line[cm_cname + "_Kappa"] = gica.cm.getKappa()
    if is_return_cm:
        return line, gica.cm
    return line


class TAcc_FC:

    def __init__(self):
        self.ml_dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\MLMods")
        self.df = None
        self.category_field_name = "CATEGORY"
        self.to_list = []
        self.df_list = []
        self.log = None

    def addTDL(self, city_name, clf_name, mod_dirname, model_name=None, df=None):
        if df is None:
            df = self.df
        self.df_list.append(df)
        to_line = toDictLine(self.ml_dfn, city_name, clf_name, mod_dirname, model_name)
        to_line = {"NUMBER": len(self.to_list) + 1, **to_line}
        self.logwl(to_line)
        self.to_list.append(to_line)
        return to_line["NUMBER"]

    def logwl(self, line, end="\n", is_print=None):
        if self.log is not None:
            return self.log.wl(line, end=end, is_print=is_print)
        return line

    def categoryOAKappa(self, cm_cnames, map_category, to_map_category, cm_cname="",
                        category_func=None, to_category_func=None, ):
        for i, line in enumerate(self.to_list):
            df = self.df_list[i]
            self.to_list[i], cm = categoryOAKappa(
                df, line, cm_cnames, map_category, to_map_category,
                category_field_name=self.category_field_name,
                is_return_cm=True, cm_cname=cm_cname,
                category_func=category_func, to_category_func=to_category_func,
            )

            if self.log is not None:
                self.log.kw("MAP_CATEGORY", map_category)
                self.log.kw("TO_MAP_CATEGORY", to_map_category)
                self.log.kw("{0} [{1}]".format(cm_cnames, line["NUMBER"]), cm.fmtCM(), sep=":\n", end="")

    def categoryOAKappaIS(self, cm_cname="IS"):
        self.categoryOAKappa(
            cm_cnames=["IS", "NOIS"],
            map_category={11: 1, 21: 2, 31: 2, 41: 2, 12: 1, 22: 2, 32: 2, 42: 2},
            to_map_category={1: 1, 2: 2, 3: 2, 4: 2, 5: 1, 6: 2, 7: 2, 8: 2},
            cm_cname=cm_cname
        )

    def toDF(self):
        return pd.DataFrame(self.to_list)

    def categoryOAKappa_VHL(self, cm_cname="VHL"):
        self.categoryOAKappa(
            cm_cnames=["VEG", "HIGH", "LOW"],
            map_category={11: 2, 21: 1, 31: 2, 41: 3, 12: 3, 22: 3, 32: 3, 42: 3},
            to_map_category={1: 2, 2: 1, 3: 2, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3},
            cm_cname=cm_cname,
            # to_category_func=to_category_func,
        )

    def categoryOAKappa_IS(self, cm_cname="ISSOIL"):
        self.categoryOAKappa(
            cm_cnames=["SOIL", "IS"],
            map_category={11: 2, 21: 0, 31: 1, 41: 0, 12: 0, 22: 0, 32: 0, 42: 0},
            to_map_category={1: 2, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0},
            cm_cname=cm_cname,
            # to_category_func=to_category_func,
        )

    def categoryOAKappa_SH(self, cm_cname="ISSH"):
        self.categoryOAKappa(
            cm_cnames=["IS_SH", "NOIS_SH"],
            map_category={11: 0, 21: 0, 31: 0, 41: 0, 12: 1, 22: 2, 32: 2, 42: 2},
            to_map_category={1: 1, 2: 2, 3: 2, 4: 2, 5: 1, 6: 2, 7: 2, 8: 2},
            cm_cname=cm_cname,
            # to_category_func=to_category_func,
        )

    def imdcSampling(self, save_csv_fn, number_list=None):
        if number_list is None:
            number_list = []
        x, y = self.df["X"].values, self.df["Y"].values,
        to_dict = {**self.df.to_dict("list")}
        for i in number_list:
            line = self.to_list[i]
            gsf = GDALSamplingFast(line["TIF_FN"])
            tmp_dict = gsf.sampling(x, y)
            to_dict["{0}_{1}_{2}_{3}".format(line["CITY"], line["CLF"], line["MODEL"], line["NUMBER"])] = \
                list(tmp_dict.values())[0]
        pd.DataFrame(to_dict).to_csv(save_csv_fn)


class TAcc_FC_Main:

    def __init__(self, init_dirname, name="TACCFC"):
        self.init_dirname = timeDirName(init_dirname, is_mk=True)
        self.init_dfn = DirFileName(self.init_dirname)
        self.name = name
        self.log = SRTLog(os.path.join(self.init_dirname, "log_{}.txt".format(self.name)), mode="w")
        self.log.log("\n", "-" * 60, "\n", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.tacc_fc = TAcc_FC()
        self.tacc_fc.log = self.log

    def addTDL(self, city_name, clf_name, mod_dirname, model_name=None, df=None):
        self.tacc_fc.addTDL(city_name, clf_name, mod_dirname, model_name, df)

    def qd_df(self):
        log = self.log

        def func1():
            csv_fn = log.kw("CSV_FN", r"F:\ProjectSet\Shadow\Hierarchical\Samples\24\sh2_spl24_is7.csv")
            df = pd.read_csv(csv_fn)
            df = df[df["TEST"] == 0]
            map_dict = log.kw(
                "MAP_DICT",
                {"IS": 11, "VEG": 21, "SOIL": 31, "WAT": 41, "IS_SH": 12, "VEG_SH": 22, "SOIL_SH": 32, "WAT_SH": 42}
            )
            df["CATEGORY"] = categoryMap(df["CNAME"].tolist(), map_dict)
            return df

        return func1()

    def bj_df(self):
        log, tacc_fc = self.log, self.tacc_fc
        csv_fn = log.kw("CSV_FN", r"F:\ProjectSet\Shadow\Hierarchical\MLMods\Analysis\1\sh2_bj_tspl1_spl.csv")
        df = pd.read_csv(csv_fn)
        # df = df[df["TEST"] == 0]
        self.tacc_fc.category_field_name = "CATEGORY_CODE"
        return df

    def cd_df(self):
        log, tacc_fc = self.log, self.tacc_fc
        csv_fn = log.kw("CSV_FN", r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240420H000421nofc\train_data.csv")
        df = pd.read_csv(csv_fn)
        df = df[df["TEST"] == 0]
        return df

    def qd_addTDL(self):
        log, tacc_fc = self.log, self.tacc_fc
        tacc_fc.df = self.qd_df()
        number_list = [
            tacc_fc.addTDL("qd", "SVM", "20240420H145813nofc"),
            tacc_fc.addTDL("qd", "SVM", "20240524H090841fc"),
            tacc_fc.addTDL("qd", "RF", "20240419H234523nofc"),
            tacc_fc.addTDL("qd", "RF", "20240419H222504fc"),
        ]
        tacc_fc.imdcSampling(self.init_dfn.fn("qd_imdcsampling.csv"), number_list=number_list)

    def bj_addTDL(self):
        log, tacc_fc = self.log, self.tacc_fc
        tacc_fc.df = self.bj_df()
        tacc_fc.addTDL("bj", "SVM", "20240420H160934nofc")
        tacc_fc.addTDL("bj", "SVM", "20240420H123300fc")
        tacc_fc.addTDL("bj", "RF", "20240419H235038nofc")
        tacc_fc.addTDL("bj", "RF", "20240419H223557fc")
        tacc_fc.imdcSampling(self.init_dfn.fn("bj_imdcsampling.csv"))

    def cd_addTDL(self):
        log, tacc_fc = self.log, self.tacc_fc
        tacc_fc.df = self.cd_df()
        tacc_fc.addTDL("cd", "SVM", "20240420H184703nofc")
        tacc_fc.addTDL("cd", "SVM", "20240420H140735fc")
        tacc_fc.addTDL("cd", "RF", "20240420H000421nofc")
        tacc_fc.addTDL("cd", "RF", "20240419H230332fc")
        tacc_fc.imdcSampling(self.init_dfn.fn("cd_imdcsampling.csv"))

    def tacc_fc_run(self):
        tacc_fc = self.tacc_fc
        tacc_fc.categoryOAKappaIS()
        tacc_fc.categoryOAKappa_VHL()
        tacc_fc.categoryOAKappa_IS()
        tacc_fc.categoryOAKappa_SH()
        print(tacc_fc.toDF().drop("TIF_FN", axis=1))

    def fit(self):
        self.qd_addTDL()
        self.bj_addTDL()
        self.cd_addTDL()
        self.tacc_fc_run()
        self.tacc_fc.toDF().to_csv(to_csv_fn.format("cd"))

        tacc_fc.imdcSampling(r"F:\ProjectSet\Shadow\Hierarchical\MLMods\Analysis\1\sh2_bj_tspl1.csv")
        self.tacc_fc_run()

        qd_df()


def tAccMLFC2():
    ml_dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\MLMods")

    def split_df(df, field_name, *datas):
        datas = datasCaiFen(datas)
        if len(datas) == 0:
            return pd.DataFrame(), df

        data_unique = pd.unique(df[field_name]).tolist()
        df_list1, df_list2 = [], []
        for k in data_unique:
            if k in datas:
                df_list1.append(df[df[field_name] == k])
            else:
                df_list2.append(df[df[field_name] == k])
        df1, df2 = pd.concat(df_list1), pd.concat(df_list2)

        return df1, df2

    def tdl(city_name, clf_name, mod_dirname, model_name=None):
        return toDictLine(ml_dfn, city_name, clf_name, mod_dirname, model_name)

    def func1():
        df = pd.read_excel(r"F:\ProjectSet\Shadow\Hierarchical\Samples\23\分层有效性实验测试样本.xlsx",
                           sheet_name="青岛")
        print(df)
        df_sh, df_nosh = split_df(df, "CNAME", "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH", "SOIL", "WAT")
        print(df_sh, df_nosh)
        df_no_sh_random, df_nosh = split_df(df_nosh, "TAG", "RANDOM1000")
        coors = [[float(df_nosh["X"][i]), float(df_nosh["Y"][i])] for i in df_nosh.index]
        coors2, out_index_list = sampleSpaceUniform(coors, x_len=400, y_len=400, is_trans_jiaodu=True, ret_index=True)
        df_index = df_nosh.index[out_index_list]
        df_out = df_nosh.loc[df_index]
        print(df_out)
        df_out = pd.concat([df_no_sh_random, df_out])
        print(df_out)
        to_fn = numberfilename(r"F:\ProjectSet\Shadow\Hierarchical\Samples\23\sh2_spl23_fc_qd.csv")
        print(to_fn)
        df_out.to_csv(to_fn, index=False)

        os.system("srt_csv2shp \"{0}\"".format(to_fn))

    def func2():

        def func21(tif_fn, df):
            gica = GDALImdcAcc(tif_fn)
            # df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\23\sh2_spl23_fc_qd.csv")
            gica.addDataFrame(df)
            gica.map_category = SHHConfig.CATE_MAP_SH881
            gica.calCM(SHHConfig.SHH_CNAMES8)
            print(gica.cm.fmtCM())
            for name in gica.cm.CNAMES():
                print(name)
                print(gica.cm.accuracyCategory(name).fmtCM())
            return gica.cm

        def func22():
            gica = GDALImdcAcc(
                r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240419H234523nofc\qd_sh2_1_opt_sar_glcm_imdc.tif")
            df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\23\sh2_spl23_fc_qd.csv")
            gica.addDataFrame(df)
            gica.map_category = {11: 1, 21: 2, 31: 2, 41: 2, 12: 1, 22: 2, 32: 2, 42: 2}
            gica.to_map_category = {1: 1, 2: 2, 3: 2, 4: 2, 5: 1, 6: 2, 7: 2, 8: 2}
            gica.calCM(["IS", "NOIS"])
            print(gica.cm.fmtCM())

        # func22()
        def tif_names():
            dirname_list = [
                r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240419H222504fc",
                r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240419H234523nofc",
                r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240420H115335fc",
                r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240420H145813nofc",
            ]
            for dirname in dirname_list:
                print("r\"{0}\",".format(filterFileExt(dirname, ".tif")[0]))

        init_list = [
            {"CITY": "qd", "CLF": "RF", "MODEL": "fc", },
            {"CITY": "qd", "CLF": "RF", "MODEL": "nofc", },
            {"CITY": "qd", "CLF": "SVM", "MODEL": "fc", },
            {"CITY": "qd", "CLF": "SVM", "MODEL": "nofc", },
        ]

        swt = SRTWriteText(r"F:\ProjectSet\Shadow\Hierarchical\Samples\23\fc_acc_cm.txt")

        def swt_w1(line):
            for k in line:
                swt.write("{0}: {1}".format(k, line[k]))

        def qd():

            to_list = [
                {"CITY": "qd", "CLF": "RF", "MODEL": "fc",
                 "TIF_FN": ml_dfn.fn(r"20240419H222504fc\qd_sh2_1_opt_sar_glcm_imdc.tif"), },
                {"CITY": "qd", "CLF": "RF", "MODEL": "nofc",
                 "TIF_FN": ml_dfn.fn(r"20240419H234523nofc\qd_sh2_1_opt_sar_glcm_imdc.tif"), },
                {"CITY": "qd", "CLF": "SVM", "MODEL": "fc",
                 "TIF_FN": ml_dfn.fn(r"20240420H115335fc\qd_sh2_1_opt_sar_glcm_fc_imdc.tif"), },
                {"CITY": "qd", "CLF": "SVM", "MODEL": "nofc",
                 "TIF_FN": ml_dfn.fn(r"20240420H145813nofc\qd_sh2_1_opt_sar_glcm_nofc_imdc.tif"), },
            ]
            df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\23\sh2_spl23_fc_qd.csv")

            for i, line in enumerate(to_list):
                tif_fn = line["TIF_FN"]
                cm = func21(tif_fn, df)
                swt_w1(line)
                swt.write(cm.fmtCM())
                gica = GDALImdcAcc(tif_fn)
                gica.addDataFrame(df)
                gica.map_category = {11: 1, 21: 2, 31: 2, 41: 2, 12: 1, 22: 2, 32: 2, 42: 2}
                gica.to_map_category = {1: 1, 2: 2, 3: 2, 4: 2, 5: 1, 6: 2, 7: 2, 8: 2}
                gica.calCM(["IS", "NOIS"])
                print(gica.cm.fmtCM())
                to_list[i]["OA"] = gica.cm.OA()
                to_list[i]["Kappa"] = gica.cm.getKappa()

            print(to_list[0].keys())
            print(pd.DataFrame(to_list)[['CITY', 'CLF', 'MODEL', 'OA', 'Kappa', ]])
            return pd.DataFrame(to_list)[['CITY', 'CLF', 'MODEL', 'OA', 'Kappa', "TIF_FN"]]

        def bj():

            to_list = [
                {"CITY": "bj", "CLF": "RF", "MODEL": "fc",
                 "TIF_FN": ml_dfn.fn(r"20240419H223557fc\bj_sh2_1_opt_sar_glcm_imdc.tif"), },
                {"CITY": "bj", "CLF": "RF", "MODEL": "nofc",
                 "TIF_FN": ml_dfn.fn(r"20240419H235038nofc\bj_sh2_1_opt_sar_glcm_imdc.tif"), },
                {"CITY": "bj", "CLF": "SVM", "MODEL": "fc",
                 "TIF_FN": ml_dfn.fn(r"20240420H123300fc\bj_sh2_1_opt_sar_glcm_fc_imdc.tif"), },
                {"CITY": "bj", "CLF": "SVM", "MODEL": "nofc",
                 "TIF_FN": ml_dfn.fn(r"20240420H160934nofc\bj_sh2_1_opt_sar_glcm_nofc_imdc.tif"), },
            ]
            df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240419H223557fc\train_data.csv")
            df = df[df["TEST"] == 0]

            for i, line in enumerate(to_list):
                tif_fn = line["TIF_FN"]
                print(tif_fn)
                cm = func21(tif_fn, df)
                swt_w1(line)
                swt.write(cm.fmtCM())
                gica = GDALImdcAcc(tif_fn)
                gica.addDataFrame(df)
                gica.map_category = {11: 1, 21: 2, 31: 2, 41: 2, 12: 1, 22: 2, 32: 2, 42: 2}
                gica.to_map_category = {1: 1, 2: 2, 3: 2, 4: 2, 5: 1, 6: 2, 7: 2, 8: 2}
                gica.calCM(["IS", "NOIS"])
                print(gica.cm.fmtCM())
                to_list[i]["OA"] = gica.cm.OA()
                to_list[i]["Kappa"] = gica.cm.getKappa()

            print(to_list[0].keys())
            print(pd.DataFrame(to_list)[['CITY', 'CLF', 'MODEL', 'OA', 'Kappa', ]])
            return pd.DataFrame(to_list)[['CITY', 'CLF', 'MODEL', 'OA', 'Kappa', "TIF_FN"]]

        def cd():

            to_list = [
                {"CITY": "cd", "CLF": "RF", "MODEL": "fc",
                 "TIF_FN": ml_dfn.fn(r"20240419H230332fc\cd_sh2_1_opt_sar_glcm_imdc.tif"), },
                {"CITY": "cd", "CLF": "RF", "MODEL": "nofc",
                 "TIF_FN": ml_dfn.fn(r"20240420H000421nofc\cd_sh2_1_opt_sar_glcm_imdc.tif"), },
                {"CITY": "cd", "CLF": "SVM", "MODEL": "fc",
                 "TIF_FN": ml_dfn.fn(r"20240420H140735fc\cd_sh2_1_opt_sar_glcm_fc_imdc.tif"), },
                {"CITY": "cd", "CLF": "SVM", "MODEL": "nofc",
                 "TIF_FN": ml_dfn.fn(r"20240420H184703nofc\cd_sh2_1_opt_sar_glcm_nofc_imdc.tif"), },
            ]
            df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240419H230332fc\train_data.csv")
            df = df[df["TEST"] == 0]

            for i, line in enumerate(to_list):
                tif_fn = line["TIF_FN"]
                print(tif_fn)
                cm = func21(tif_fn, df)
                swt_w1(line)
                swt.write(cm.fmtCM())
                gica = GDALImdcAcc(tif_fn)
                gica.addDataFrame(df)
                gica.map_category = {11: 1, 21: 2, 31: 2, 41: 2, 12: 1, 22: 2, 32: 2, 42: 2}
                gica.to_map_category = {1: 1, 2: 2, 3: 2, 4: 2, 5: 1, 6: 2, 7: 2, 8: 2}
                gica.calCM(["IS", "NOIS"])
                print(gica.cm.fmtCM())
                to_list[i]["OA"] = gica.cm.OA()
                to_list[i]["Kappa"] = gica.cm.getKappa()

            print(to_list[0].keys())
            print(pd.DataFrame(to_list)[['CITY', 'CLF', 'MODEL', 'OA', 'Kappa', ]])
            return pd.DataFrame(to_list)[['CITY', 'CLF', 'MODEL', 'OA', 'Kappa', "TIF_FN"]]

        df_is_oa = pd.concat([qd(), bj(), cd()])
        print(df_is_oa[['CITY', 'CLF', 'MODEL', 'OA', 'Kappa', ]])
        to_csv_fn = numberfilename(r"F:\ProjectSet\Shadow\Hierarchical\Samples\23\fc_acc.csv")
        print(to_csv_fn)
        df_is_oa.to_csv(to_csv_fn, index=False)

    def func3():
        to_list = [tdl("qd", "SVM", "20240522H203233nofc"), tdl("qd", "SVM", "20240524H090841fc")]
        df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\24\sh2_spl24_is7.csv")
        df = df[df["TEST"] == 0]
        map_dict = {"IS": 11, "VEG": 21, "SOIL": 31, "WAT": 41, "IS_SH": 12, "VEG_SH": 22, "SOIL_SH": 32, "WAT_SH": 42}
        df["CATEGORY"] = categoryMap(df["CNAME"].tolist(), map_dict)
        get_class_oa_kappa(to_list, df)

    def func4():

        log = SRTLog(r"F:\ProjectSet\Shadow\Hierarchical\MLMods\Analysis\1\tAccMLFC2.txt", mode="a")
        log.log("\n", "-" * 60, "\n", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        tacc_fc = TAcc_FC()
        tacc_fc.log = log

        def qd_df():
            csv_fn = log.kw("CSV_FN", r"F:\ProjectSet\Shadow\Hierarchical\Samples\24\sh2_spl24_is7.csv")
            df = pd.read_csv(csv_fn)
            df = df[df["TEST"] == 0]
            map_dict = log.kw(
                "MAP_DICT",
                {"IS": 11, "VEG": 21, "SOIL": 31, "WAT": 41, "IS_SH": 12, "VEG_SH": 22, "SOIL_SH": 32, "WAT_SH": 42}
            )
            df["CATEGORY"] = categoryMap(df["CNAME"].tolist(), map_dict)
            tacc_fc.df = df
            tacc_fc.addTDL("qd", "SVM", "20240420H145813nofc")
            tacc_fc.addTDL("qd", "SVM", "20240524H090841fc")
            tacc_fc.addTDL("qd", "RF", "20240419H234523nofc")
            tacc_fc.addTDL("qd", "RF", "20240419H222504fc")
            tacc_fc_run()
            tacc_fc.toDF().to_csv(to_csv_fn.format("qd"))

        def bj_df():
            csv_fn = log.kw("CSV_FN", r"F:\ProjectSet\Shadow\Hierarchical\MLMods\Analysis\1\sh2_bj_tspl1_spl.csv")
            df = pd.read_csv(csv_fn)
            # df = df[df["TEST"] == 0]
            tacc_fc.df = df
            tacc_fc.category_field_name = "CATEGORY_CODE"
            tacc_fc.addTDL("bj", "SVM", "20240420H160934nofc")
            tacc_fc.addTDL("bj", "SVM", "20240420H123300fc")
            tacc_fc.addTDL("bj", "RF", "20240419H235038nofc")
            tacc_fc.addTDL("bj", "RF", "20240419H223557fc")
            tacc_fc.imdcSampling(r"F:\ProjectSet\Shadow\Hierarchical\MLMods\Analysis\1\sh2_bj_tspl1.csv")
            tacc_fc_run()
            tacc_fc.toDF().to_csv(to_csv_fn.format("bj"))

        def cd_df():
            csv_fn = log.kw("CSV_FN", r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240420H000421nofc\train_data.csv")
            df = pd.read_csv(csv_fn)
            df = df[df["TEST"] == 0]
            tacc_fc.df = df
            tacc_fc.addTDL("cd", "SVM", "20240420H184703nofc")
            tacc_fc.addTDL("cd", "SVM", "20240420H140735fc")
            tacc_fc.addTDL("cd", "RF", "20240420H000421nofc")
            tacc_fc.addTDL("cd", "RF", "20240419H230332fc")
            tacc_fc_run()
            tacc_fc.toDF().to_csv(to_csv_fn.format("cd"))

        def tacc_fc_run():
            tacc_fc.categoryOAKappaIS()
            tacc_fc.categoryOAKappa_VHL()
            tacc_fc.categoryOAKappa_IS()
            tacc_fc.categoryOAKappa_SH()
            print(tacc_fc.toDF().drop("TIF_FN", axis=1))

        to_csv_fn = log.kw("TO_CSV_FN", r"F:\ProjectSet\Shadow\Hierarchical\MLMods\Analysis\1\sh2_ml1_{}.csv")

        qd_df()

    func4()


def main():
    def sampling():
        # samplingSHH21OptSarGLCM(
        #     csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\23\sh2_spl23_fc1.csv",
        #     to_csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\23\sh2_spl23_fc2.csv",
        # )
        # samplingSHH21OptSarGLCM(
        #     csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\24\sh2_spl24_is6.csv",
        #     to_csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\24\sh2_spl24_is6_spl.csv",
        # )
        gsf = GDALSamplingFast(r"G:\ImageData\SHH2QingDaoImages\shh2_qd1_2.vrt")
        gsf.csvfile(
            csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\24\sh2_spl24_is7.csv",
            to_csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\24\sh2_spl24_is7_spl.csv",
        )

    def func1():
        city_name = "qd"
        mlfc_feats = MLFCFeatures()

        log_wt = SRTWriteText(r"F:\ProjectSet\Shadow\Hierarchical\Temp\tmp1.txt", "a")
        nofc_mod = MLFCModel("svm", city_name=city_name)
        nofc_mod.clf = SVC()
        nofc_mod.x_keys = mlfc_feats.to_opt_as_de()
        nofc_mod.getRangeDict(city_name=city_name)
        nofc_mod.log(log_wt, "NOFC", "the extraction of UIS")

    def func2():
        gica = GDALImdcAcc(r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240419H222504fc\qd_sh2_1_opt_sar_glcm_imdc.tif")
        df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240419H222504fc\train_data.csv")
        df = df[df["TEST"] == 0]
        df = df[df["CITY"] == "qd"]
        gica.addDataFrame(df)
        gica.map_category = SHHConfig.CATE_MAP_SH881
        gica.calCM(SHHConfig.SHH_CNAMES8)
        print(gica.cm.fmtCM())
        for name in gica.cm.CNAMES():
            print(name)
            print(gica.cm.accuracyCategory(name).fmtCM())

    def func3():
        # "F:\ProjectSet\Shadow\Hierarchical\Samples\Release\qd_VHL_random2000_1"
        # "F:\ProjectSet\Shadow\Hierarchical\Samples\Release\qd_roads_shouhua_tp1"
        # "F:\ProjectSet\Shadow\Hierarchical\Samples\Release\qd_random1000_1"
        # "F:\ProjectSet\Shadow\Hierarchical\Samples\Release\qd_is_random2000"
        # r"F:\ProjectSet\Shadow\Hierarchical\Samples\23\sh2_spl23_fc2.csv"

        map_dict = {1: 2, 2: 1, 3: 3}
        s2spl = SHH2_SPL(map_dict=map_dict, others=0, is_npy=True)
        s2spl.add_qd_VHL_random2000(
            category_field_name="CATEGORY_CODE", field_datas={"FT": "qd_VHL_random2000", "CITY": "qd"})
        map_dict = {11: 11}
        s2spl.add_qd_roads_shouhua_tp1(
            category_field_name="CATEGORY", map_dict=map_dict, field_datas={"FT": "qd_roads_shouhua_tp1", "CITY": "qd"})
        s2spl.add_qd_random1000_1(category_field_name="CATEGORY_CODE",
                                  field_datas={"FT": "qd_random1000_1", "CITY": "qd"})
        s2spl.add_qd_is_random2000(category_field_name="CATEGORY_CODE",
                                   field_datas={"FT": "qd_is_random2000", "CITY": "qd"})
        s2spl.shh2_spl.toCSV(
            r"F:\ProjectSet\Shadow\Hierarchical\Samples\24\sh2_spl24_is4.csv",
            category_field_name="CATEGORY_FC",
            cname_field_name="CNAME_FC",
        )
        concatCSV(
            r"F:\ProjectSet\Shadow\Hierarchical\Samples\24\sh2_spl24_is4.csv",
            r"F:\ProjectSet\Shadow\Hierarchical\Samples\23\sh2_spl23_fc2.csv",
            to_csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\24\sh2_spl24_is5.csv"
        )
        # s2spl.filterEq("IS_TAG", "TRUE")

    def func4():
        df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\24\sh2_spl24_sheet5.csv")
        print(df.keys())
        # df = df[["FCSRT", "X", "Y", "TEST", 'FT', 'CITY',]]
        print(pd.unique(df["FT"]))
        samples = []

        def getsample(x=0.0, y=0.0, fc_cate=0, nofc_cate=0, fen_ceng=0, no_fen_ceng=1,
                      ft="FT", test=1, cname="NOT_KNOW"):
            return {"X": x, "Y": y, "FC": fc_cate, "NOFC": nofc_cate, "FEN_CENG": fen_ceng, "NO_FEN_CENG": no_fen_ceng,
                    "FT": ft, "TEST": test, "CNAME": cname}

        def map_category(map_dict, data, o_data=0):
            if data in map_dict:
                return map_dict[data]
            else:
                return o_data

        print("qd_is_random2000")
        df_qd = pd.DataFrame(df[df["FT"] == "qd_is_random2000"].to_dict("records"))
        map_fc_cate = {11: 1}
        map_nofc_cate = {11: 1, 21: 2, 31: 3, 41: 4, 12: 1, 22: 2, 32: 3, 42: 4}
        map_cname = {11: "IS", 21: "VEG", 31: "SOIL", 41: "WAT", 12: "IS_SH", 22: "VEG_SH", 32: "SOIL_SH", 42: "WAT_SH"}
        for i in range(len(df_qd)):
            samples.append(getsample(
                x=float(df_qd["X"][i]), y=float(df_qd["Y"][i]),
                fc_cate=map_category(map_fc_cate, int(df_qd["CNAME_FC"][i]), 2),
                nofc_cate=map_category(map_nofc_cate, int(df_qd["CNAME_FC"][i])),
                fen_ceng=2,
                no_fen_ceng=1,
                ft="qd_is_random2000",
                test=1,
                cname=map_category(map_cname, df_qd["CNAME_FC"][i])
            ))

        print("qd_roads_shouhua_tp1")
        df_qd = pd.DataFrame(df[df["FT"] == "qd_roads_shouhua_tp1"].to_dict("records"))
        map_fc_cate = {11: 1, }
        map_nofc_cate = {11: 1, 21: 2, 31: 3, 41: 4, 12: 1, 22: 2, 32: 3, 42: 4}
        map_cname = {11: "IS", 21: "VEG", 31: "SOIL", 41: "WAT", 12: "IS_SH", 22: "VEG_SH", 32: "SOIL_SH", 42: "WAT_SH"}
        for i in range(len(df_qd)):
            if random.random() < 0.9:
                test = 1
            else:
                test = 0
            if random.random() < 0.5:
                fen_ceng = 2
            else:
                fen_ceng = 1
            if fen_ceng == 1:
                fc_cate = 2
            else:
                fc_cate = 1

            samples.append(getsample(
                x=float(df_qd["X"][i]), y=float(df_qd["Y"][i]),
                fc_cate=fc_cate,
                nofc_cate=1,
                fen_ceng=fen_ceng,
                no_fen_ceng=1,
                ft="qd_roads_shouhua_tp1",
                test=test,
                cname=map_category(map_cname, df_qd["CNAME_FC"][i])
            ))

        print("qd_random1000_1")
        df_qd = pd.DataFrame(df[df["FT"] == "qd_random1000_1"].to_dict("records"))
        map_fc_cate = {11: 1, 21: 2, 31: 3, 41: 4, 12: 5, 22: 6, 32: 7, 42: 8}
        map_nofc_cate = {11: 1, 21: 2, 31: 3, 41: 4, 12: 1, 22: 2, 32: 3, 42: 4}
        map_cname = {11: "IS", 21: "VEG", 31: "SOIL", 41: "WAT", 12: "IS_SH", 22: "VEG_SH", 32: "SOIL_SH", 42: "WAT_SH"}
        for i in range(len(df_qd)):
            samples.append(getsample(
                x=float(df_qd["X"][i]), y=float(df_qd["Y"][i]),
                fc_cate=map_category(map_fc_cate, int(df_qd["CNAME_FC"][i]), 2),
                nofc_cate=map_category(map_nofc_cate, int(df_qd["CNAME_FC"][i])),
                fen_ceng=0,
                no_fen_ceng=1,
                ft="qd_random1000_1",
                test=0,
                cname=map_category(map_cname, df_qd["CNAME_FC"][i])
            ))

        print("qd_VHL_random2000")
        df_qd = pd.DataFrame(df[df["FT"] == "qd_VHL_random2000"].to_dict("records"))
        map_cname = {2: "VEG", 1: "HIGH", 3: "LOW"}
        map_fc_cate = {1: 2, 2: 1, 3: 3}
        # map_fc_cate = {1: 1, 2: 2, 3: 3}
        map_nofc_cate = {1: 2}

        for i in range(len(df_qd)):
            if int(df_qd["CATEGORY_CODE"][i]) == 2:
                no_fen_ceng = 1
            else:
                no_fen_ceng = 0
            samples.append(getsample(
                x=float(df_qd["X"][i]), y=float(df_qd["Y"][i]),
                fc_cate=map_category(map_fc_cate, int(df_qd["CATEGORY_CODE"][i])),
                nofc_cate=2,
                fen_ceng=1,
                no_fen_ceng=no_fen_ceng,
                ft="qd_VHL_random2000",
                test=1,
                cname=map_category(map_cname, df_qd["CATEGORY_CODE"][i])
            ))

        print("shadow1samples")
        df_qd = pd.DataFrame(df[df["FT"] == "shadow1samples"].to_dict("records"))
        map_nofc_cate = {11: 1, 21: 2, 31: 3, 41: 4, 12: 1, 22: 2, 32: 3, 42: 4}
        map_cname = {11: "IS", 21: "VEG", 31: "SOIL", 41: "WAT", 12: "IS_SH", 22: "VEG_SH", 32: "SOIL_SH",
                     42: "WAT_SH"}
        for i in range(len(df_qd)):
            samples.append(getsample(
                x=float(df_qd["X"][i]), y=float(df_qd["Y"][i]),
                fc_cate=int(df_qd["FC"][i]),
                nofc_cate=map_category(map_nofc_cate, int(df_qd["CATEGORY"][i])),
                fen_ceng=int(df_qd["FEN_CENG"][i]),
                ft="shadow1samples",
                test=int(df_qd["TEST"][i]),
                cname=map_category(map_cname, df_qd["CATEGORY"][i])
            ))

        to_df = pd.DataFrame(samples)
        print(to_df)
        to_df.to_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\24\sh2_spl24_is7.csv", index_label="SRT")

        if True:
            gsf = GDALSamplingFast(r"G:\ImageData\SHH2QingDaoImages\shh2_qd1_2.vrt")
            gsf.csvfile(
                csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\24\sh2_spl24_is7.csv",
                to_csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\24\sh2_spl24_is7_spl.csv",
            )

    def func5():
        data = readJson(r"F:\ProjectSet\Shadow\QingDao\Mods\20240510H224639\SPL_NOSH-SVM-TAG-OPT_args.json")
        plt.plot(data["train"]["gamma"]["args"], data["train"]["gamma"]["accuracy"])
        plt.show()

    SHHMLFC_main('qd', 'fc')
    # func4()


def method_name2():
    clf = joblib.load(r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240308H215552fc\fc_ws.mod")
    feature_importance(clf, ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'ndvi', 'ndwi', 'mndwi', 'AS_VV', 'AS_VH', 'DE_VV',
                             'DE_VH', 'OPT_mean', 'OPT_var', 'OPT_hom', 'OPT_con', 'OPT_dis', 'OPT_ent', 'OPT_asm'])


def method_name1():
    # 计算精度
    df_fn = r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240308H215552fc\20240308H215552fc.xlsx"
    df = pd.read_excel(df_fn, sheet_name="Sheet2")
    df = df[df["FEN_CENG"] == 2]
    k0, k1 = "__CODE__", "FC_CATE"
    y0 = categoryMap(df[k0].values, SHHConfig.CATE_MAP_IS_8)
    y1 = categoryMap(df[k1].values, SHHConfig.CATE_MAP_IS_8)
    # y1 = df[k1].values
    cm = ConfusionMatrix(
        # class_names=SHHConfig.SHH_CNAMES[1:],
        class_names=SHHConfig.IS_CNAMES,
    )
    cm.addData(y0, y1)
    print(cm.fmtCM())
    print("OA Kappa")
    print(cm.OA(), cm.getKappa())


if __name__ == "__main__":
    main()
