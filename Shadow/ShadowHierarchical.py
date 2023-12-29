# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowHierarchical.py
@Time    : 2023/12/9 19:15
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of ShadowHierarchical
-----------------------------------------------------------------------------"""
import json
import os
import random
import time
import warnings

import joblib
import numpy as np
from sklearn.svm import SVC

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.GDALUtils import samplingToCSV
from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.SRTModel import SRTModelInit
from SRTCodes.SRTSample import CSVSamples
from SRTCodes.Utils import printList, DirFileName, Jdt, readJson, writeTexts, saveJson, copyFile
from Shadow.ShadowDraw import cal_10log10
from Shadow.ShadowImdC import ShadowImageClassification
from Shadow.ShadowMain import ShadowMain
from Shadow.ShadowTraining import trainSVM_RandomizedSearchCV


def trainTest(x: np.ndarray, y: np.ndarray):
    clf = SVC(kernel="rbf", cache_size=5000)
    clf.fit(x, y)
    return clf


def yReCategory(y, *categorys):
    out_y = np.zeros(len(y), dtype=int)
    for i, category in enumerate(categorys):
        select_c = np.zeros(len(y), dtype=bool)
        if category is None:
            out_y[out_y == 0] = i + 1
            continue
        for cate in category:
            select_c[y == cate] = True
        out_y[select_c] = i + 1
    return out_y


def reXY(categorys, x_o, y_o):
    select_c = yReCategory(y_o, *categorys)
    x = x_o[select_c != 0]
    y = select_c[select_c != 0]
    return x, y


def getFieldToData(x, select_field_names: list, field_names: list):
    select_codes = [field_names.index(k) for k in select_field_names]
    return x[:, select_codes]


class SHHModel(SRTModelInit):

    def __init__(self, name, c_names, colors=None, features=None):
        super().__init__()
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.model = None
        self.canshu = None

        self.field_names = None
        self.c_names = c_names
        self.cm = ConfusionMatrix(len(self.c_names), self.c_names)
        self.name = name

        self.colors = colors
        self.features = features
        self.re_categorys = ()

    def addColors(self, *colors):
        if len(colors) != len(self.c_names):
            raise Exception("Number of colors can not equal c_names.")
        self.colors = list(colors)

    def addFeatures(self, *features):
        if self.features is None:
            self.features = []
        for feat in features:
            if feat in self.features:
                warnings.warn("Feature {0} have in features {1}".format(feat, self.features))
            self.features.append(feat)

    def sample(self, x_train, y_train, x_test=None, y_test=None):
        if x_train is not None:
            self.x_train = x_train.copy()
        if y_train is not None:
            self.y_train = y_train.copy()
        if x_test is not None:
            self.x_test = x_test.copy()
        if y_test is not None:
            self.y_test = y_test.copy()

    def reCategory(self, *categorys):
        if len(categorys) == 0:
            categorys = self.re_categorys
        if (self.x_train is not None) and (self.y_train is not None):
            self.x_train, self.y_train = reXY(categorys, self.x_train, self.y_train)
        if (self.x_test is not None) and (self.y_test is not None):
            self.x_test, self.y_test = reXY(categorys, self.x_test, self.y_test)

    def addReCategory(self, *categorys):
        self.re_categorys = categorys

    def dfFields(self, field_names=None):
        if field_names is None:
            field_names = self.features
        self.field_names = field_names
        self.x_train = self.x_train[field_names]
        if self.x_test is not None:
            self.x_test = self.x_test[field_names]

    def score(self, x=None, y=None, names=None, *args, **kwargs):
        if (x is None) and (y is None):
            x, y = self.x_test, self.y_test
        if names is not None:
            self.c_names = names
        self.cm = ConfusionMatrix(len(self.c_names), self.c_names)
        if "is_values" not in kwargs:
            y_pred = self.predict(x.values)
        else:
            y_pred = self.predict(x)
        self.cm.addData(y, y_pred)
        return self.cm.OA()

    def calCM(self, y_pred, y=None):
        if y is None:
            y = self.y_test
        self.cm = ConfusionMatrix(len(self.c_names), self.c_names)
        self.cm.addData(y, y_pred)
        return self.cm.OA()

    def save(self, filename, *args, **kwargs):
        joblib.dump(self.model, filename)
        return filename

    def load(self, filename, to_dict=None, *args, **kwargs):
        self.model = joblib.load(filename)
        if to_dict is not None:
            self.name = to_dict["NAME"]
            self.canshu = to_dict["CAN_SHU"]
            self.field_names = to_dict["FIELD_NAMES"]
            self.c_names = to_dict["C_NAMES"]
            self.colors = to_dict["COLORS"]
            self.features = to_dict["FEATURES"]
            self.re_categorys = to_dict["RE_CATEGORYS"]

    def toSaveDict(self):
        to_dict = {
            "NAME": self.name,
            "CAN_SHU": self.canshu,
            "FIELD_NAMES": self.field_names,
            "C_NAMES": self.c_names,
            "CM_LIST": self.cm.toList(),
            "COLORS": [] if self.colors is None else list(self.colors),
            "FEATURES": self.features,
            "RE_CATEGORYS": [] if (len(self.re_categorys) == 0) else list(self.re_categorys),
        }
        return to_dict


class SHHModelSVM(SHHModel):

    def __init__(self, name, c_names, colors=None, features=None):
        super().__init__(name=name, c_names=c_names, colors=colors, features=features)

    def train(self, is_values=True, *args, **kwargs):
        x, y = self.x_train, self.y_train
        if is_values:
            x = x.values
        self.model, self.canshu = trainSVM_RandomizedSearchCV(x, y)
        self.canshu["field_names"] = self.field_names
        return self.model, self.canshu

    def predict(self, x, *args, **kwargs):
        return self.model.predict(x)


def trainModel(train_args, model, x_test, x_train, y_test, y_train):
    model.sample(x_train, y_train, x_test, y_test)
    model.reCategory()
    model.dfFields()
    model.train()
    model.score()
    train_args[model.name] = model.canshu
    print("\nConfusion Matrix {0}:".format(model.name))
    print(model.cm.fmtCM())


class ShadowHierarchicalModel(SHHModel):
    """ Shadow Hierarchical Model """

    CODE_IS = 1
    CODE_SH_IS = 5
    CODE_VEG = 2
    CODE_SH_VEG = 6
    CODE_SOIL = 3
    CODE_SH_SOIL = 7
    CODE_WAT = 4
    CODE_SH_WAT = 8

    def __init__(self, name, c_names, colors=None, features=None):
        super().__init__(name=name, c_names=c_names, colors=colors, features=features)
        self.code_is = self.CODE_IS
        self.code_sh_is = self.CODE_SH_IS
        self.code_veg = self.CODE_VEG
        self.code_sh_veg = self.CODE_SH_VEG
        self.code_soil = self.CODE_SOIL
        self.code_sh_soil = self.CODE_SH_SOIL
        self.code_wat = self.CODE_WAT
        self.code_sh_wat = self.CODE_SH_WAT

        self.optic_field_names = []
        self.as_sar_field_names = []
        self.de_sar_field_names = []

        self.models = {}
        self.veg_mod = SHHModelSVM(name="VEG", c_names=["VEG", "NOT_VEG"])
        self.veg_mod.addColors((0, 255, 0), (0, 125, 0))
        self.is_low_mod = SHHModelSVM(name="LOW_NO", c_names=["HEIGHT", "LOW"])
        self.is_soil_mod = SHHModelSVM(name="IS_SOIL", c_names=["IS", "SOIL"])
        self.ws_mod = SHHModelSVM(name="WS", c_names=["WAT", "IS_SH", "OTHER_SH"])

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.y_veg = None
        self.y_is_low = None
        self.y_is_soil = None
        self.y_ws = None

    def train(self, x_train, y_train, x_test, y_test, *args, **kwargs):
        train_args = super().train()

        self.veg_mod = self.models["VEG"]
        self.is_low_mod = self.models["LOW_NO"]
        self.is_soil_mod = self.models["IS_SOIL"]
        self.ws_mod = self.models["WS"]

        self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test
        self.field_names = list(self.x_train.keys())

        trainModel(train_args, self.veg_mod, x_test, x_train, y_test, y_train)
        trainModel(train_args, self.is_low_mod, x_test, x_train, y_test, y_train)
        trainModel(train_args, self.is_soil_mod, x_test, x_train, y_test, y_train)
        trainModel(train_args, self.ws_mod, x_test, x_train, y_test, y_train)

        self.score(is_test_other=False)
        print("Confusion Matrix:")
        print(self.cm.fmtCM())

        train_args["field_names"] = self.field_names
        return train_args

    def predict(self, x, ret_all_y=False, *args, **kwargs):

        x_veg = getFieldToData(x, self.veg_mod.field_names, self.field_names)
        y_veg = self.veg_mod.predict(x_veg)

        x_is_low = getFieldToData(x, self.is_low_mod.field_names, self.field_names)
        y_is_low = self.is_low_mod.predict(x_is_low)

        x_is_soil = getFieldToData(x, self.is_soil_mod.field_names, self.field_names)
        y_is_soil = self.is_soil_mod.predict(x_is_soil)

        x_ws = getFieldToData(x, self.ws_mod.field_names, self.field_names)
        y_ws = self.ws_mod.predict(x_ws)

        y_out = np.zeros(len(x), dtype=int)
        for i in range(len(x)):
            if y_veg[i] == 1:
                y_out[i] = 2
            elif y_veg[i] == 2:
                if y_is_low[i] == 1:
                    if y_is_soil[i] == 1:
                        y_out[i] = 1
                    elif y_is_soil[i] == 2:
                        y_out[i] = 3
                else:
                    if y_ws[i] == 1:
                        y_out[i] = 4
                    elif y_ws[i] == 2:
                        y_out[i] = 1
                    elif y_ws[i] == 3:
                        y_out[i] = 2

        self.y_veg = y_veg
        self.y_is_low = y_is_low
        self.y_is_soil = y_is_soil
        self.y_ws = y_ws

        if ret_all_y:
            return y_out, y_veg, y_is_low, y_is_soil, y_ws
        else:
            return y_out

    def save(self, dirname, *args, **kwargs):
        to_dict = self.toSaveDict()
        to_dict["MOD_ARGS"] = kwargs["mod_args"]
        to_dict["MODELS_FN"] = {}
        for k in self.models:
            to_dict["MODELS_FN"][k] = self.models[k].save(os.path.join(dirname, f"{k}.mod"))
        saveJson(to_dict, os.path.join(dirname, f"{self.name}.json"))
        # self.veg_mod.save(os.path.join(dirname, "veg_mod.mod"))
        # self.is_low_mod.save(os.path.join(dirname, "is_low_mod.mod"))
        # self.is_soil_mod.save(os.path.join(dirname, "is_soil_mod.mod"))
        # self.ws_mod.save(os.path.join(dirname, "ws_mod.mod"))

    def load(self, dirname, *args, **kwargs):
        json_fn = None
        for fn in os.listdir(dirname):
            if "_args.json" in fn:
                json_fn = os.path.join(dirname, fn)
                break
        if json_fn is None:
            raise Exception("Can not find args json file.")
        json_dict = readJson(json_fn)
        self.field_names = json_dict["field_names"]
        self.veg_mod.load(os.path.join(dirname, "veg_mod.mod"))
        self.veg_mod.field_names = json_dict["veg_mod"]["field_names"]
        self.is_low_mod.load(os.path.join(dirname, "is_low_mod.mod"))
        self.is_low_mod.field_names = json_dict["is_low_mod"]["field_names"]
        self.is_soil_mod.load(os.path.join(dirname, "is_soil_mod.mod"))
        self.is_soil_mod.field_names = json_dict["is_soil_mod"]["field_names"]
        self.ws_mod.load(os.path.join(dirname, "ws_mod.mod"))
        self.ws_mod.field_names = json_dict["ws_mod"]["field_names"]

    def score(self, x=None, y=None, names=None, *args, **kwargs):
        if (x is None) and (y is None):
            x, y = self.x_test, self.y_test
        categorys = ([self.code_is, self.code_sh_is], [self.code_veg, self.code_sh_veg],
                     [self.code_soil, self.code_sh_soil], [self.code_wat, self.code_sh_wat])
        x_test, y_test = reXY(categorys, x, y)

        oa = super(ShadowHierarchicalModel, self).score(x_test, y_test)
        x_train, y_train = None, None

        if "is_test_other" in kwargs:
            if not kwargs["is_test_other"]:
                return oa

        self.veg_mod.sample(x_train, y_train, x_test, y_test)
        self.veg_mod.reCategory([self.code_veg], None)
        self.veg_mod.calCM(self.y_veg)

        self.is_low_mod.sample(x_train, y_train, x_test, y_test)
        self.is_low_mod.reCategory([self.code_is, self.code_soil],
                                   [self.code_wat, self.code_sh_is, self.code_sh_veg, self.code_sh_soil,
                                    self.code_sh_wat])
        self.is_low_mod.calCM(self.y_is_low)

        self.is_soil_mod.sample(x_train, y_train, x_test, y_test)
        self.is_soil_mod.reCategory([self.code_is], [self.code_soil])
        self.is_soil_mod.calCM(self.y_is_soil)

        self.ws_mod.sample(x_train, y_train, x_test, y_test)
        self.ws_mod.reCategory([self.code_wat, self.code_sh_wat], [self.code_sh_is],
                               [self.code_sh_veg, self.code_sh_soil])
        self.ws_mod.calCM(self.y_ws)

        return oa

    def getModels(self):
        models = {"VEG": self.veg_mod,
                  "LOW_OR_NO": self.is_low_mod,
                  "IS_SOIL": self.is_soil_mod,
                  "WS": self.ws_mod, }
        return models

    def addModel(self, name, model):
        self.models[name] = model

    def modelColors(self, name, *colors):
        self.models[name].addColors(*colors)

    def modelFeatures(self, name, *features):
        self.models[name].addFeatures(*features)

    def modelReCategory(self, name, *categorys):
        self.models[name].addReCategory(*categorys)

    def toSaveDict(self):
        to_dict = super(ShadowHierarchicalModel, self).toSaveDict()
        to_dict["MODELS"] = {}
        for k in self.models:
            to_dict["MODELS"][k] = self.models[k].toSaveDict()
        return to_dict


class ShadowHierarchicalImageClassification(ShadowImageClassification):

    def __init__(self, dat_fn, model_dir):
        super().__init__(dat_fn, model_dir)
        self.features = []
        self.category_colors = {}

    def classify(self, mod: ShadowHierarchicalModel, features, mod_name):
        to_f = os.path.join(self.model_dir, mod_name + "_imdc.dat")
        if os.path.isfile(to_f):
            print("Shadow Image RasterClassification: 100%")
            return to_f
        if self.d is None:
            self._readData()
        self._initImdc()
        d = self.getFeaturesData(features)
        print(to_f)
        jdt = Jdt(total=self.n_rows, desc="Shadow Image RasterClassification")
        jdt.start()
        for i in range(0, self.n_rows):
            col_imdc = d[:, i, :].T
            y = self.predict(col_imdc)
            self.imdc[i, :] = y
            jdt.add()
        jdt.end()
        self.saveImdc(to_f)
        return to_f

    def classifySHH(self, mod: ShadowHierarchicalModel, features, mod_name):
        models = mod.getModels()
        to_f = os.path.join(self.model_dir, mod_name + "_imdc.dat")
        if self.d is None:
            self._readData()

        d = self.getFeaturesData(features)
        print(to_f)
        jdt = Jdt(total=self.n_rows, desc="Shadow Image RasterClassification")
        jdt.start()
        for i in range(0, self.n_rows):
            col_imdc = d[:, i, :].T
            y = self.predict(col_imdc)
            self.imdc[i, :] = y
            jdt.add()
        jdt.end()
        self.saveImdc(to_f)
        return to_f

    def addImdcCategoryColor(self, name, cate_name, cate_color):
        if name not in self.category_colors:
            self.category_colors[name] = {"n": 1, "names": ["Unclassified"], "colors": [(0, 0, 0)]}
        self.category_colors[name]["names"] = cate_name
        self.category_colors[name]["colors"] = cate_color
        self.category_colors[name]["n"] += 1

    def getImdcHDR(self, name):
        category_color = self.category_colors[name]
        cate_names = self.category_colors[name]["names"]
        cate_colors = self.category_colors[name]["colors"]
        imdc_hdr = self.save_hdr.copy()
        imdc_hdr["classes"] = str(category_color["n"])

        class_lookup = "{ "
        for c in cate_colors:
            class_lookup += " {0:>3}, {1:>3}, {2:>3},".format(c[0], c[1], c[2])
        class_lookup = class_lookup[:-1]
        class_lookup += " }"
        imdc_hdr["class lookup"] = class_lookup

        class_names = "{ " + cate_names[0]
        for name in self.cate_names[1:]:
            class_names += ", " + name
        class_names += " }"
        imdc_hdr["class names"] = class_names

        imdc_hdr["band names"] = "{" + os.path.split(self.dat_fn)[1] + " Category }"
        return imdc_hdr

    def predict(self, x, *args, **kwargs) -> np.ndarray:
        return self.model.predict(x)


class ShadowHierarchicalTrainImdcOne(ShadowMain):
    GRS = {}

    def __init__(self, model_dir=None):
        super().__init__()
        self.model = None
        self.raster_dfn = ""
        self.sample_dfn = ""
        self.model_dfn = ""
        self.model_dir = ""
        self.raster_fn = ""
        self.sample_fn = ""
        self.sample_csv_fn = ""
        self.sample_csv_spl_fn = ""
        self.mod_name = ""
        self.model_name = None

        dirname = self.initFileNames()

        self.sic: ShadowHierarchicalImageClassification = ShadowHierarchicalImageClassification(
            self.raster_fn, self.model_dir)

        self.test_y = None
        self.test_x = None
        self.test_field_name = "TEST"

        self.csv_spl = CSVSamples()
        self.gr = GDALRaster()
        self.addGDALRaster(self.raster_fn)

        self.feats = []
        self.categorys = []
        self.tags = None

        self.cm_names = []

        self.train_func = None

        self.save_cm_file = os.path.join(self.model_dir, dirname + "_cm.txt")
        self.save_train_spl_file = os.path.join(self.model_dir, dirname + "_train_spl.csv")
        self.imd = None

    def initFileNames(self):
        return ""

    def setSample(self, spl: CSVSamples = None, is_sh_to_no=True):
        if spl is not None:
            self.csv_spl = spl
        self.test_x, self.test_y = self._getTrainTestSample(0)
        if is_sh_to_no:
            self.test_y = self._shCodeToNO(self.test_y)
        self._getNames()
        self.csv_spl.saveToFile(self.save_train_spl_file)

    def _getTrainTestSample(self, select_code, c_names=None, feat_names=None, tags=None):
        x, y = self.csv_spl.get(c_names=c_names, feat_names=feat_names, tags=tags)
        select = x[self.test_field_name].values == select_code
        x = x[select]
        y = y[select]
        return x, y

    def filterSampleTestField(self, select_code, x, y):
        select_list = x[self.test_field_name] == select_code
        return x[select_list], y[select_list]

    def _getNames(self):
        self.feats = self.csv_spl.getFeatureNames()
        self.categorys = self.csv_spl.getCategoryNames()
        self.tags = self.csv_spl.getTagNames()

    def addGDALRaster(self, raster_fn, ):
        raster_fn = os.path.abspath(raster_fn)
        if raster_fn not in self.GRS:
            self.GRS[raster_fn] = GDALRaster(raster_fn)
        self.gr = self.GRS[raster_fn]
        return self.gr

    def fitFeatureNames(self, *feat_names):
        self.feats = list(feat_names)

    def fitCategoryNames(self, *c_names):
        self.categorys = list(c_names)

    def fitTagNames(self, *tag_names):
        self.tags = list(tag_names)

    def fitCMNames(self, *cm_names):
        self.cm_names = list(cm_names)

    def trainFunc(self, train_func):
        self.train_func = train_func

    def getSample(self, spls: list, feats: list, tags: list, is_sh_to_no=True):
        feats.append(self.test_field_name)
        x, y = self.csv_spl.get(c_names=spls, feat_names=feats, tags=tags)
        shuffle_list = [i for i in range(len(y))]
        random.shuffle(shuffle_list)
        x, y = x.loc[shuffle_list], y[shuffle_list]
        x_train, y_train = self.filterSampleTestField(1, x, y)
        x_test, y_test = self.filterSampleTestField(0, x, y)
        return x_train, y_train, x_test, y_test

    def _shCodeToNO(self, y_train):
        y1 = y_train[y_train >= 5]
        y_train[y_train >= 5] = y1 - 4
        return y_train

    def timeModelDir(self):
        dir_name = time.strftime("%Y%m%dH%H%M%S")
        self.model_dir = os.path.join(self.model_dir, dir_name)
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
        return dir_name

    def saveModel(self, model_name, *args, **kwargs):
        if model_name is not None:
            self.model.save(model_name, mod_args=self.mod_args)

    def saveModArgs(self, fn):
        with open(fn, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.mod_args))
            # json.dump(f, self.mod_args)

    def initSIC(self, dat_fn=None):
        if dat_fn is None:
            dat_fn = self.gr.gdal_raster_fn
        self.sic = ShadowImageClassification(dat_fn, self.model_dir)
        self.sic.is_trans = True
        self.sic.is_scale_01 = True

    def sicAddCategory(self, name: str, color: tuple = None):
        self.sic.addCategory(name, color)

    def featureCallBack(self, feat_name, callback_func, is_trans=None):
        self.csv_spl.featureCallBack(feat_name=feat_name, callback_func=callback_func, is_trans=is_trans)
        self.sic.featureCallBack(feat_name=feat_name, callback_func=callback_func, is_trans=is_trans)

    def featureScaleMinMax(self, feat_name, x_min, x_max, is_trans=None, is_01=None):
        self.csv_spl.featureScaleMinMax(feat_name=feat_name, x_min=x_min, x_max=x_max, is_trans=is_trans, is_01=is_01)
        self.sic.featureScaleMinMax(feat_name=feat_name, x_min=x_min, x_max=x_max, is_trans=is_trans, is_01=is_01)

    def addModel(self, mod):
        self.model = mod

    def fit(self, is_sh_to_no=False):
        print("MODEL DIRNAME:", self.model_dir)
        self.csv_spl.saveToFile(self.save_train_spl_file)
        copyFile(__file__, os.path.join(self.model_dir, os.path.split(__file__)[1]))
        mod_name = self.model_name + os.path.split(self.model_dir)[1]
        mod_args_fn = os.path.join(self.model_dir, mod_name + "_args.json")

        # train running ---
        print(">>> Train")
        printList("categorys", self.categorys)
        printList("feats", self.feats)
        printList("tags", self.tags)
        x_train, y_train, x_test, y_test = self.getSample(self.categorys, self.feats, self.tags, is_sh_to_no)

        is_train = True

        if not is_train:
            self.model.load(r"F:\ProjectSet\Shadow\Hierarchical\20231209\20231212H205710")
            self.model.score(x_test, y_test)

        if is_train:
            print("Start Training ...... ", end="")
            mod_args = self.model.train(x_train, y_train, x_test, y_test)
            print("End")
            # Save
            self.mod_args = mod_args
            self.mod_args["model_name"] = mod_name
            self.mod_args["model_filename"] = self.model_dir
            self.mod_args["features"] = self.feats.copy()
            self.mod_args["categorys"] = self.categorys.copy()
            self.mod_args["tags"] = self.tags.copy()
            self.saveModArgs(mod_args_fn)
            self.saveModel(self.model_dir)

        self.printAccuracy()
        with open(self.save_cm_file, "w", encoding="utf-8") as f:
            self.printAcc(f)

    def addCSVFile(self, spl_fn=None, is_spl=False):
        if spl_fn is None:
            spl_fn = self.sample_csv_spl_fn
        if is_spl:
            samplingToCSV(spl_fn, self.gr, spl_fn)
        self.csv_spl = CSVSamples(spl_fn)
        self._getNames()

    def initCSVSamples(self):
        self._getNames()

    def printAcc(self, fs=None):
        print("Confusion Matrix VEG:", file=fs)
        print(self.model.veg_mod.cm.fmtCM(), file=fs)
        print("Confusion Matrix HEIGHT LOW:", file=fs)
        print(self.model.is_low_mod.cm.fmtCM(), file=fs)
        print("Confusion Matrix IS SOIL:", file=fs)
        print(self.model.is_soil_mod.cm.fmtCM(), file=fs)
        print("Confusion Matrix WATER SHADOW:", file=fs)
        print(self.model.ws_mod.cm.fmtCM(), file=fs)
        print("Confusion Matrix:", file=fs)
        print(self.model.cm.fmtCM(), file=fs)
        self.printAccuracy(fs)

    def printAccuracy(self, fs=None):
        print("\n{0:<8} {1:<10} {2:<10} {3:<10}".format("NAME", "OA", "PA", "UA"), file=fs)
        to_dict = self.model.cm.accuracy()
        for k in to_dict:
            print("{0:<8} {1:<10.2f} {2:<10.2f} {3:<10.2f}".format(
                k, to_dict[k][0], to_dict[k][1], to_dict[k][2]), file=fs)

    def classify(self, model_dirname=None):
        for i in range(len(self.model.c_names)):
            self.sic.addCategory(self.model.c_names[i], self.model.colors[i])
        if model_dirname is None:
            model_dirname = self.model_dir
        mod_name = self.model_name + os.path.split(self.model_dir)[1]
        self.sic.model = self.model
        writeTexts(os.path.join(self.model_dir, "sic.txt"), model_dirname)
        self.sic.classify(self.model, self.feats, mod_name)
        return None


class ShadowHierarchicalTrainImdcOneBeiJing(ShadowHierarchicalTrainImdcOne):

    def __init__(self, model_dir=None):
        super().__init__(model_dir)
        self.model_name = "BeiJing"
        print(self.model_name)

    def initFileNames(self):
        self.raster_dfn = DirFileName(r"F:\ProjectSet\Shadow\Release\BeiJingImages")
        self.sample_dfn = DirFileName(r"F:\ProjectSet\Shadow\Release\BeiJingSamples")
        self.model_dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\20231209")
        self.model_dir = self.model_dfn.fn()
        # dirname = os.path.split(self.model_dir)[1]
        dirname = self.timeModelDir()
        self.raster_fn = self.raster_dfn.fn("SH_BJ_envi.dat")
        self.sample_fn = self.sample_dfn.fn("BeiJingSamples.xlsx")
        self.sample_csv_fn = self.model_dfn.fn("sh_bj_sample.csv")
        self.sample_csv_spl_fn = self.model_dfn.fn("sh_bj_sample_spl.csv")
        return dirname

    def featureCallback1(self):
        self.featureScaleMinMax("Blue", 99.76996, 2397.184)
        self.featureScaleMinMax("Green", 45.83414, 2395.735)
        self.featureScaleMinMax("Red", 77.79654, 2726.7026)
        self.featureScaleMinMax("NIR", 87.66086, 3498.4321)
        self.featureScaleMinMax("NDVI", -0.5007727, 0.7354284)
        self.featureScaleMinMax("NDWI", -0.6572631, 0.7623875)
        self.featureScaleMinMax("OPT_asm", 0.02124183, 0.998366)
        self.featureScaleMinMax("OPT_con", 0.0, 169.74791)
        self.featureScaleMinMax("OPT_cor", -0.036879253, 0.99688625)
        self.featureScaleMinMax("OPT_dis", 0.0, 9.799746)
        self.featureScaleMinMax("OPT_ent", 0.0, 3.8249474)
        self.featureScaleMinMax("OPT_hom", 0.12091503, 0.998366)
        self.featureScaleMinMax("OPT_mean", 4.941177, 53.7353)
        self.featureScaleMinMax("OPT_var", 0.0, 236.09961)
        self.featureCallBack("AS_VV", cal_10log10)
        self.featureCallBack("AS_VH", cal_10log10)
        self.featureCallBack("AS_C11", cal_10log10)
        self.featureCallBack("AS_C22", cal_10log10)
        self.featureCallBack("AS_Lambda1", cal_10log10)
        self.featureCallBack("AS_Lambda2", cal_10log10)
        self.featureCallBack("AS_SPAN", cal_10log10)
        self.featureCallBack("AS_Epsilon", cal_10log10)
        self.featureCallBack("DE_VV", cal_10log10)
        self.featureCallBack("DE_VH", cal_10log10)
        self.featureCallBack("DE_C11", cal_10log10)
        self.featureCallBack("DE_C22", cal_10log10)
        self.featureCallBack("DE_Lambda1", cal_10log10)
        self.featureCallBack("DE_Lambda2", cal_10log10)
        self.featureCallBack("DE_SPAN", cal_10log10)
        self.featureCallBack("DE_Epsilon", cal_10log10)
        self.featureScaleMinMax("AS_VV", -24.609674, 5.9092603)
        self.featureScaleMinMax("AS_VH", -31.865038, -5.2615275)
        self.featureScaleMinMax("AS_VHDVV", 0.0, 0.95164585)
        self.featureScaleMinMax("AS_C11", -22.61998, 5.8634768)
        self.featureScaleMinMax("AS_C22", -28.579813, -5.2111626)
        self.featureScaleMinMax("AS_Lambda1", -21.955856, 6.124724)
        self.featureScaleMinMax("AS_Lambda2", -29.869734, -8.284683)
        self.featureScaleMinMax("AS_SPAN", -22.58362, 6.97997)
        self.featureScaleMinMax("AS_Epsilon", 0.0, 35.12922)
        self.featureScaleMinMax("AS_Mu", -0.7263123, 0.7037629)
        self.featureScaleMinMax("AS_RVI", 0.07459847, 2.076324)
        self.featureScaleMinMax("AS_m", 0.26469338, 0.97544414)
        self.featureScaleMinMax("AS_Beta", 0.632338, 0.9869048)
        self.featureScaleMinMax("AS_VH_asm", 0.02124183, 0.050653595)
        self.featureScaleMinMax("AS_VH_con", 6.572378, 59.151405)
        self.featureScaleMinMax("AS_VH_cor", 0.006340516, 0.86876196)
        self.featureScaleMinMax("AS_VH_dis", 1.9767247, 5.8193297)
        self.featureScaleMinMax("AS_VH_ent", 3.0939856, 3.8060431)
        self.featureScaleMinMax("AS_VH_hom", 0.16666667, 0.40849674)
        self.featureScaleMinMax("AS_VH_mean", 7.514706, 54.04412)
        self.featureScaleMinMax("AS_VH_var", 5.9986033, 108.64137)
        self.featureScaleMinMax("AS_VV_asm", 0.022875817, 0.050653595)
        self.featureScaleMinMax("AS_VV_con", 4.5305123, 48.325462)
        self.featureScaleMinMax("AS_VV_cor", 0.21234758, 0.88228023)
        self.featureScaleMinMax("AS_VV_dis", 1.5990733, 5.22229)
        self.featureScaleMinMax("AS_VV_ent", 3.1254923, 3.7871387)
        self.featureScaleMinMax("AS_VV_hom", 0.18464053, 0.45261437)
        self.featureScaleMinMax("AS_VV_mean", 8.544118, 51.573532)
        self.featureScaleMinMax("AS_VV_var", 3.8744159, 96.8604)
        self.featureScaleMinMax("DE_VV", -27.851603, 5.094706)
        self.featureScaleMinMax("DE_VH", -35.427082, -5.4092093)
        self.featureScaleMinMax("DE_VHDVV", 0.0, 1.0289364)
        self.featureScaleMinMax("DE_C11", -26.245598, 4.9907513)
        self.featureScaleMinMax("DE_C22", -32.042320, -5.322515)
        self.featureScaleMinMax("DE_Lambda1", -25.503738, 5.2980003)
        self.featureScaleMinMax("DE_Lambda2", -33.442368, -8.68537)
        self.featureScaleMinMax("DE_SPAN", -24.81076, 4.82663)
        self.featureScaleMinMax("DE_Epsilon", 0.0, 21.882689)
        self.featureScaleMinMax("DE_Mu", -0.6823329, 0.7723537)
        self.featureScaleMinMax("DE_RVI", 0.0940072, 2.1935015)
        self.featureScaleMinMax("DE_m", 0.24836189, 0.9705721)
        self.featureScaleMinMax("DE_Beta", 0.6241778, 0.9852859)
        self.featureScaleMinMax("DE_VH_asm", 0.022875817, 0.05392157)
        self.featureScaleMinMax("DE_VH_con", 5.6798058, 51.11825)
        self.featureScaleMinMax("DE_VH_cor", 0.12444292, 0.87177193)
        self.featureScaleMinMax("DE_VH_dis", 1.8186697, 5.456009)
        self.featureScaleMinMax("DE_VH_ent", 2.9679575, 3.7997417)
        self.featureScaleMinMax("DE_VH_hom", 0.1748366, 0.42810458)
        self.featureScaleMinMax("DE_VH_mean", 7.6176476, 55.176476)
        self.featureScaleMinMax("DE_VH_var", 5.513511, 95.38374)
        self.featureScaleMinMax("DE_VV_asm", 0.02124183, 0.057189543)
        self.featureScaleMinMax("DE_VV_con", 5.0987973, 57.54357)
        self.featureScaleMinMax("DE_VV_cor", 0.19514601, 0.88254523)
        self.featureScaleMinMax("DE_VV_dis", 1.7117102, 5.6928787)
        self.featureScaleMinMax("DE_VV_ent", 2.993163, 3.7997417)
        self.featureScaleMinMax("DE_VV_hom", 0.17320262, 0.44444445)
        self.featureScaleMinMax("DE_VV_mean", 6.4852943, 54.04412)
        self.featureScaleMinMax("DE_VV_var", 4.44714, 111.17851)


class ShadowHierarchicalTrainImdcOneQingDao(ShadowHierarchicalTrainImdcOne):

    def __init__(self, model_dir=None):
        super().__init__(model_dir)
        self.model_name = "QingDao"
        print(self.model_name)

    def initFileNames(self):
        self.raster_dfn = DirFileName(r"F:\ProjectSet\Shadow\Release\QingDaoImages")
        self.sample_dfn = DirFileName(r"F:\ProjectSet\Shadow\Release\QingDaoSamples")
        self.model_dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\20231209")
        self.model_dir = self.model_dfn.fn()
        # dirname = os.path.split(self.model_dir)[1]
        dirname = self.timeModelDir()
        self.raster_fn = self.raster_dfn.fn("SH_QD_envi.dat")
        self.sample_fn = self.sample_dfn.fn("QingDaoSamples.xlsx")
        self.sample_csv_fn = self.model_dfn.fn("sh_qd_sample.csv")
        self.sample_csv_spl_fn = self.model_dfn.fn("sh_qd_sample_spl.csv")
        return dirname

    def featureCallback1(self):
        self.featureScaleMinMax("Blue", 99.76996, 2397.184)
        self.featureScaleMinMax("Green", 45.83414, 2395.735)
        self.featureScaleMinMax("Red", 77.79654, 2726.7026)
        self.featureScaleMinMax("NIR", 87.66086, 3498.4321)
        self.featureScaleMinMax("NDVI", -0.5007727, 0.7354284)
        self.featureScaleMinMax("NDWI", -0.6572631, 0.7623875)
        self.featureScaleMinMax("OPT_asm", 0.02124183, 0.998366)
        self.featureScaleMinMax("OPT_con", 0.0, 169.74791)
        self.featureScaleMinMax("OPT_cor", -0.036879253, 0.99688625)
        self.featureScaleMinMax("OPT_dis", 0.0, 9.799746)
        self.featureScaleMinMax("OPT_ent", 0.0, 3.8249474)
        self.featureScaleMinMax("OPT_hom", 0.12091503, 0.998366)
        self.featureScaleMinMax("OPT_mean", 4.941177, 53.7353)
        self.featureScaleMinMax("OPT_var", 0.0, 236.09961)
        self.featureCallBack("AS_VV", cal_10log10)
        self.featureCallBack("AS_VH", cal_10log10)
        self.featureCallBack("AS_C11", cal_10log10)
        self.featureCallBack("AS_C22", cal_10log10)
        self.featureCallBack("AS_Lambda1", cal_10log10)
        self.featureCallBack("AS_Lambda2", cal_10log10)
        self.featureCallBack("AS_SPAN", cal_10log10)
        self.featureCallBack("AS_Epsilon", cal_10log10)
        self.featureCallBack("DE_VV", cal_10log10)
        self.featureCallBack("DE_VH", cal_10log10)
        self.featureCallBack("DE_C11", cal_10log10)
        self.featureCallBack("DE_C22", cal_10log10)
        self.featureCallBack("DE_Lambda1", cal_10log10)
        self.featureCallBack("DE_Lambda2", cal_10log10)
        self.featureCallBack("DE_SPAN", cal_10log10)
        self.featureCallBack("DE_Epsilon", cal_10log10)
        self.featureScaleMinMax("AS_VV", -24.609674, 5.9092603)
        self.featureScaleMinMax("AS_VH", -31.865038, -5.2615275)
        self.featureScaleMinMax("AS_VHDVV", 0.0, 0.95164585)
        self.featureScaleMinMax("AS_C11", -22.61998, 5.8634768)
        self.featureScaleMinMax("AS_C22", -28.579813, -5.2111626)
        self.featureScaleMinMax("AS_Lambda1", -21.955856, 6.124724)
        self.featureScaleMinMax("AS_Lambda2", -29.869734, -8.284683)
        self.featureScaleMinMax("AS_SPAN", -22.58362, 6.97997)
        self.featureScaleMinMax("AS_Epsilon", 0.0, 35.12922)
        self.featureScaleMinMax("AS_Mu", -0.7263123, 0.7037629)
        self.featureScaleMinMax("AS_RVI", 0.07459847, 2.076324)
        self.featureScaleMinMax("AS_m", 0.26469338, 0.97544414)
        self.featureScaleMinMax("AS_Beta", 0.632338, 0.9869048)
        self.featureScaleMinMax("AS_VH_asm", 0.02124183, 0.050653595)
        self.featureScaleMinMax("AS_VH_con", 6.572378, 59.151405)
        self.featureScaleMinMax("AS_VH_cor", 0.006340516, 0.86876196)
        self.featureScaleMinMax("AS_VH_dis", 1.9767247, 5.8193297)
        self.featureScaleMinMax("AS_VH_ent", 3.0939856, 3.8060431)
        self.featureScaleMinMax("AS_VH_hom", 0.16666667, 0.40849674)
        self.featureScaleMinMax("AS_VH_mean", 7.514706, 54.04412)
        self.featureScaleMinMax("AS_VH_var", 5.9986033, 108.64137)
        self.featureScaleMinMax("AS_VV_asm", 0.022875817, 0.050653595)
        self.featureScaleMinMax("AS_VV_con", 4.5305123, 48.325462)
        self.featureScaleMinMax("AS_VV_cor", 0.21234758, 0.88228023)
        self.featureScaleMinMax("AS_VV_dis", 1.5990733, 5.22229)
        self.featureScaleMinMax("AS_VV_ent", 3.1254923, 3.7871387)
        self.featureScaleMinMax("AS_VV_hom", 0.18464053, 0.45261437)
        self.featureScaleMinMax("AS_VV_mean", 8.544118, 51.573532)
        self.featureScaleMinMax("AS_VV_var", 3.8744159, 96.8604)
        self.featureScaleMinMax("DE_VV", -27.851603, 5.094706)
        self.featureScaleMinMax("DE_VH", -35.427082, -5.4092093)
        self.featureScaleMinMax("DE_VHDVV", 0.0, 1.0289364)
        self.featureScaleMinMax("DE_C11", -26.245598, 4.9907513)
        self.featureScaleMinMax("DE_C22", -32.042320, -5.322515)
        self.featureScaleMinMax("DE_Lambda1", -25.503738, 5.2980003)
        self.featureScaleMinMax("DE_Lambda2", -33.442368, -8.68537)
        self.featureScaleMinMax("DE_SPAN", -24.81076, 4.82663)
        self.featureScaleMinMax("DE_Epsilon", 0.0, 21.882689)
        self.featureScaleMinMax("DE_Mu", -0.6823329, 0.7723537)
        self.featureScaleMinMax("DE_RVI", 0.0940072, 2.1935015)
        self.featureScaleMinMax("DE_m", 0.24836189, 0.9705721)
        self.featureScaleMinMax("DE_Beta", 0.6241778, 0.9852859)
        self.featureScaleMinMax("DE_VH_asm", 0.022875817, 0.05392157)
        self.featureScaleMinMax("DE_VH_con", 5.6798058, 51.11825)
        self.featureScaleMinMax("DE_VH_cor", 0.12444292, 0.87177193)
        self.featureScaleMinMax("DE_VH_dis", 1.8186697, 5.456009)
        self.featureScaleMinMax("DE_VH_ent", 2.9679575, 3.7997417)
        self.featureScaleMinMax("DE_VH_hom", 0.1748366, 0.42810458)
        self.featureScaleMinMax("DE_VH_mean", 7.6176476, 55.176476)
        self.featureScaleMinMax("DE_VH_var", 5.513511, 95.38374)
        self.featureScaleMinMax("DE_VV_asm", 0.02124183, 0.057189543)
        self.featureScaleMinMax("DE_VV_con", 5.0987973, 57.54357)
        self.featureScaleMinMax("DE_VV_cor", 0.19514601, 0.88254523)
        self.featureScaleMinMax("DE_VV_dis", 1.7117102, 5.6928787)
        self.featureScaleMinMax("DE_VV_ent", 2.993163, 3.7997417)
        self.featureScaleMinMax("DE_VV_hom", 0.17320262, 0.44444445)
        self.featureScaleMinMax("DE_VV_mean", 6.4852943, 54.04412)
        self.featureScaleMinMax("DE_VV_var", 4.44714, 111.17851)


class ShadowHierarchicalTrainImdcOneChengDu(ShadowHierarchicalTrainImdcOne):

    def __init__(self, model_dir=None):
        super().__init__(model_dir)
        self.model_name = "ChengDu"
        print(self.model_name)

    def initFileNames(self):
        self.raster_dfn = DirFileName(r"F:\ProjectSet\Shadow\Release\ChengDuImages")
        self.sample_dfn = DirFileName(r"F:\ProjectSet\Shadow\Release\ChengDuSamples")
        self.model_dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\20231209")
        self.model_dir = self.model_dfn.fn()
        # dirname = os.path.split(self.model_dir)[1]
        dirname = self.timeModelDir()
        self.raster_fn = self.raster_dfn.fn("SH_CD_envi.dat")
        self.sample_fn = self.sample_dfn.fn("ChengDuSamples.xlsx")
        self.sample_csv_fn = self.model_dfn.fn("sh_cd_sample.csv")
        self.sample_csv_spl_fn = self.model_dfn.fn("sh_cd_sample_spl.csv")
        return dirname

    def featureCallback1(self):
        self.featureScaleMinMax("Blue", 99.76996, 2397.184)
        self.featureScaleMinMax("Green", 45.83414, 2395.735)
        self.featureScaleMinMax("Red", 77.79654, 2726.7026)
        self.featureScaleMinMax("NIR", 87.66086, 3498.4321)
        self.featureScaleMinMax("NDVI", -0.5007727, 0.7354284)
        self.featureScaleMinMax("NDWI", -0.6572631, 0.7623875)
        self.featureScaleMinMax("OPT_asm", 0.02124183, 0.998366)
        self.featureScaleMinMax("OPT_con", 0.0, 169.74791)
        self.featureScaleMinMax("OPT_cor", -0.036879253, 0.99688625)
        self.featureScaleMinMax("OPT_dis", 0.0, 9.799746)
        self.featureScaleMinMax("OPT_ent", 0.0, 3.8249474)
        self.featureScaleMinMax("OPT_hom", 0.12091503, 0.998366)
        self.featureScaleMinMax("OPT_mean", 4.941177, 53.7353)
        self.featureScaleMinMax("OPT_var", 0.0, 236.09961)
        self.featureCallBack("AS_VV", cal_10log10)
        self.featureCallBack("AS_VH", cal_10log10)
        self.featureCallBack("AS_C11", cal_10log10)
        self.featureCallBack("AS_C22", cal_10log10)
        self.featureCallBack("AS_Lambda1", cal_10log10)
        self.featureCallBack("AS_Lambda2", cal_10log10)
        self.featureCallBack("AS_SPAN", cal_10log10)
        self.featureCallBack("AS_Epsilon", cal_10log10)
        self.featureCallBack("DE_VV", cal_10log10)
        self.featureCallBack("DE_VH", cal_10log10)
        self.featureCallBack("DE_C11", cal_10log10)
        self.featureCallBack("DE_C22", cal_10log10)
        self.featureCallBack("DE_Lambda1", cal_10log10)
        self.featureCallBack("DE_Lambda2", cal_10log10)
        self.featureCallBack("DE_SPAN", cal_10log10)
        self.featureCallBack("DE_Epsilon", cal_10log10)
        self.featureScaleMinMax("AS_VV", -24.609674, 5.9092603)
        self.featureScaleMinMax("AS_VH", -31.865038, -5.2615275)
        self.featureScaleMinMax("AS_VHDVV", 0.0, 0.95164585)
        self.featureScaleMinMax("AS_C11", -22.61998, 5.8634768)
        self.featureScaleMinMax("AS_C22", -28.579813, -5.2111626)
        self.featureScaleMinMax("AS_Lambda1", -21.955856, 6.124724)
        self.featureScaleMinMax("AS_Lambda2", -29.869734, -8.284683)
        self.featureScaleMinMax("AS_SPAN", -22.58362, 6.97997)
        self.featureScaleMinMax("AS_Epsilon", 0.0, 35.12922)
        self.featureScaleMinMax("AS_Mu", -0.7263123, 0.7037629)
        self.featureScaleMinMax("AS_RVI", 0.07459847, 2.076324)
        self.featureScaleMinMax("AS_m", 0.26469338, 0.97544414)
        self.featureScaleMinMax("AS_Beta", 0.632338, 0.9869048)
        self.featureScaleMinMax("AS_VH_asm", 0.02124183, 0.050653595)
        self.featureScaleMinMax("AS_VH_con", 6.572378, 59.151405)
        self.featureScaleMinMax("AS_VH_cor", 0.006340516, 0.86876196)
        self.featureScaleMinMax("AS_VH_dis", 1.9767247, 5.8193297)
        self.featureScaleMinMax("AS_VH_ent", 3.0939856, 3.8060431)
        self.featureScaleMinMax("AS_VH_hom", 0.16666667, 0.40849674)
        self.featureScaleMinMax("AS_VH_mean", 7.514706, 54.04412)
        self.featureScaleMinMax("AS_VH_var", 5.9986033, 108.64137)
        self.featureScaleMinMax("AS_VV_asm", 0.022875817, 0.050653595)
        self.featureScaleMinMax("AS_VV_con", 4.5305123, 48.325462)
        self.featureScaleMinMax("AS_VV_cor", 0.21234758, 0.88228023)
        self.featureScaleMinMax("AS_VV_dis", 1.5990733, 5.22229)
        self.featureScaleMinMax("AS_VV_ent", 3.1254923, 3.7871387)
        self.featureScaleMinMax("AS_VV_hom", 0.18464053, 0.45261437)
        self.featureScaleMinMax("AS_VV_mean", 8.544118, 51.573532)
        self.featureScaleMinMax("AS_VV_var", 3.8744159, 96.8604)
        self.featureScaleMinMax("DE_VV", -27.851603, 5.094706)
        self.featureScaleMinMax("DE_VH", -35.427082, -5.4092093)
        self.featureScaleMinMax("DE_VHDVV", 0.0, 1.0289364)
        self.featureScaleMinMax("DE_C11", -26.245598, 4.9907513)
        self.featureScaleMinMax("DE_C22", -32.042320, -5.322515)
        self.featureScaleMinMax("DE_Lambda1", -25.503738, 5.2980003)
        self.featureScaleMinMax("DE_Lambda2", -33.442368, -8.68537)
        self.featureScaleMinMax("DE_SPAN", -24.81076, 4.82663)
        self.featureScaleMinMax("DE_Epsilon", 0.0, 21.882689)
        self.featureScaleMinMax("DE_Mu", -0.6823329, 0.7723537)
        self.featureScaleMinMax("DE_RVI", 0.0940072, 2.1935015)
        self.featureScaleMinMax("DE_m", 0.24836189, 0.9705721)
        self.featureScaleMinMax("DE_Beta", 0.6241778, 0.9852859)
        self.featureScaleMinMax("DE_VH_asm", 0.022875817, 0.05392157)
        self.featureScaleMinMax("DE_VH_con", 5.6798058, 51.11825)
        self.featureScaleMinMax("DE_VH_cor", 0.12444292, 0.87177193)
        self.featureScaleMinMax("DE_VH_dis", 1.8186697, 5.456009)
        self.featureScaleMinMax("DE_VH_ent", 2.9679575, 3.7997417)
        self.featureScaleMinMax("DE_VH_hom", 0.1748366, 0.42810458)
        self.featureScaleMinMax("DE_VH_mean", 7.6176476, 55.176476)
        self.featureScaleMinMax("DE_VH_var", 5.513511, 95.38374)
        self.featureScaleMinMax("DE_VV_asm", 0.02124183, 0.057189543)
        self.featureScaleMinMax("DE_VV_con", 5.0987973, 57.54357)
        self.featureScaleMinMax("DE_VV_cor", 0.19514601, 0.88254523)
        self.featureScaleMinMax("DE_VV_dis", 1.7117102, 5.6928787)
        self.featureScaleMinMax("DE_VV_ent", 2.993163, 3.7997417)
        self.featureScaleMinMax("DE_VV_hom", 0.17320262, 0.44444445)
        self.featureScaleMinMax("DE_VV_mean", 6.4852943, 54.04412)
        self.featureScaleMinMax("DE_VV_var", 4.44714, 111.17851)


def main():
    shti_class = ShadowHierarchicalTrainImdcOneQingDao
    run(shti_class)


def run(shti_class):
    opt_feats = (
        "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
        "OPT_asm", "OPT_con", "OPT_cor", "OPT_dis", "OPT_ent", "OPT_hom", "OPT_mean", "OPT_var")
    as_feats = (
        "AS_VV", "AS_VH", "AS_VHDVV", "AS_C11", "AS_C12_imag", "AS_C12_real", "AS_C22", "AS_Lambda1", "AS_Lambda2",
        "AS_SPAN", "AS_Epsilon", "AS_Mu", "AS_RVI", "AS_m", "AS_Beta",
        "AS_VV_asm", "AS_VV_con", "AS_VV_cor", "AS_VV_dis", "AS_VV_ent", "AS_VV_hom", "AS_VV_mean", "AS_VV_var",
        "AS_VH_asm", "AS_VH_con", "AS_VH_cor", "AS_VH_dis", "AS_VH_ent", "AS_VH_hom", "AS_VH_mean", "AS_VH_var")
    de_feats = (
        "DE_VV", "DE_VH", "DE_VHDVV", "DE_C11", "DE_C12_imag", "DE_C12_real", "DE_C22", "DE_Lambda1", "DE_Lambda2",
        "DE_SPAN", "DE_Epsilon", "DE_Mu", "DE_RVI", "DE_Beta", "DE_m",
        "DE_VH_asm", "DE_VH_con", "DE_VH_cor", "DE_VH_dis", "DE_VH_ent", "DE_VH_hom", "DE_VH_mean", "DE_VH_var",
        "DE_VV_asm", "DE_VV_con", "DE_VV_cor", "DE_VV_dis", "DE_VV_ent", "DE_VV_hom", "DE_VV_mean", "DE_VV_var")
    as_de_feats = as_feats + de_feats
    fit_feats = opt_feats + as_feats + de_feats
    c_is = ShadowHierarchicalModel.CODE_IS
    c_sh_is = ShadowHierarchicalModel.CODE_SH_IS
    c_veg = ShadowHierarchicalModel.CODE_VEG
    c_sh_veg = ShadowHierarchicalModel.CODE_SH_VEG
    c_soil = ShadowHierarchicalModel.CODE_SOIL
    c_sh_soil = ShadowHierarchicalModel.CODE_SH_SOIL
    c_wat = ShadowHierarchicalModel.CODE_WAT
    c_sh_wat = ShadowHierarchicalModel.CODE_SH_WAT
    shtio_bj = shti_class()
    # shtio_bj.sampleToCsv()
    # shtio_bj.sampling()
    shtio_bj.addCSVFile()
    shtio_bj.csv_spl.fieldNameCategory("CNAME")  # CNAME
    shtio_bj.csv_spl.fieldNameTag("TAG")
    shtio_bj.csv_spl.addCategoryNames(["NOT_KNOW", "IS", "VEG", "SOIL", "WAT", "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH"])
    shtio_bj.csv_spl.readData()
    shtio_bj.featureCallback1()
    shtio_bj.initCSVSamples()
    model = ShadowHierarchicalModel("ShadowHierarchical", ["IS", "VEG", "SOIL", "WAT"])
    model.addColors((255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255))
    model.addFeatures(*fit_feats)
    model.addModel("VEG", SHHModelSVM(name="VEG", c_names=["VEG", "NOT_VEG"]))
    model.modelColors("VEG", (0, 255, 0), (0, 125, 0))
    model.modelFeatures("VEG", *opt_feats)
    model.modelReCategory("VEG", [c_veg], None)
    model.addModel("LOW_NO", SHHModelSVM(name="LOW_NO", c_names=["NO_LOW", "LOW"]))
    model.modelColors("LOW_NO", (255, 255, 0), (125, 125, 0))
    model.modelFeatures("LOW_NO", *opt_feats)
    model.modelReCategory("LOW_NO", [c_is, c_soil], [c_wat, c_sh_is, c_sh_veg, c_sh_soil, c_sh_wat])
    model.addModel("IS_SOIL", SHHModelSVM(name="IS_SOIL", c_names=["IS", "SOIL"]))
    model.modelColors("IS_SOIL", (255, 0, 0), (125, 125, 0))
    model.modelFeatures("IS_SOIL", *fit_feats)
    model.modelReCategory("IS_SOIL", [c_is], [c_soil])
    model.addModel("WS", SHHModelSVM(name="WS", c_names=["WAT", "IS_SH", "VEG_SH"]))
    model.modelColors("WS", (0, 0, 255), (125, 0, 0), (0, 125, 0))
    model.modelFeatures("WS", *fit_feats)
    model.modelReCategory("WS", [c_wat, c_sh_wat], [c_sh_is], [c_sh_veg, c_sh_soil])
    shtio_bj.addModel(model)
    shtio_bj.fitCategoryNames("IS", "VEG", "SOIL", "WAT", "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH")
    shtio_bj.fitFeatureNames(*fit_feats)
    shtio_bj.fit()
    shtio_bj.classify()


if __name__ == "__main__":
    """

python -c "import sys; sys.path.append('F:\\PyCodes'); from Shadow.ShadowHierarchical import run, ShadowHierarchicalTrainImdcOneBeiJing; run(ShadowHierarchicalTrainImdcOneBeiJing)"
python -c "import sys; sys.path.append('F:\\PyCodes'); from Shadow.ShadowHierarchical import run, ShadowHierarchicalTrainImdcOneQingDao; run(ShadowHierarchicalTrainImdcOneQingDao)"
python -c "import sys; sys.path.append('F:\\PyCodes'); from Shadow.ShadowHierarchical import run, ShadowHierarchicalTrainImdcOneChengDu; run(ShadowHierarchicalTrainImdcOneChengDu)"

    """
    main()
