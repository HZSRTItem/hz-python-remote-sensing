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

import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.GDALUtils import samplingToCSV
from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.NumpyUtils import saveCM, fmtCM
from SRTCodes.SRTModel import SRTModelInit
from SRTCodes.SRTSample import CSVSamples
from SRTCodes.Utils import printList, DirFileName, Jdt, readJson, writeTexts
from Shadow.ShadowDraw import cal_10log10
from Shadow.ShadowImdC import ShadowImageClassification
from Shadow.ShadowMain import ShadowMain


def trainSVM_RandomizedSearchCV(x: np.ndarray, y: np.ndarray, find_grid: dict = None, n_iter=None, **kwargs, ):
    if find_grid is None:
        find_grid = {}
    if len(kwargs) != 0:
        for k in kwargs:
            find_grid[k] = kwargs[k]
    if len(find_grid) == 0:
        find_grid = {"gamma": np.logspace(-1, 1, 20), "C": np.linspace(0.01, 10, 20)}
    if n_iter is None:
        n = 1
        for k in find_grid:
            n *= len(find_grid[k])
        n_iter = int(0.2 * n)
    svm_rs_cv = RandomizedSearchCV(
        estimator=SVC(kernel="rbf", cache_size=5000),
        param_distributions=find_grid,
        n_iter=n_iter
    )
    svm_rs_cv.fit(x, y)
    svm_canchu = svm_rs_cv.best_params_
    svm_mod = svm_rs_cv.best_estimator_
    return svm_mod, svm_canchu


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

    def __init__(self, c_names):
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
        if (self.x_train is not None) and (self.y_train is not None):
            self.x_train, self.y_train = reXY(categorys, self.x_train, self.y_train)
        if (self.x_test is not None) and (self.y_test is not None):
            self.x_test, self.y_test = reXY(categorys, self.x_test, self.y_test)

    def dfFields(self, field_names):
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

    def load(self, filename, *args, **kwargs):
        self.model = joblib.load(filename)


class SHHModelSVM(SHHModel):

    def __init__(self, c_names):
        super().__init__(c_names)

    def train(self, is_values=True, *args, **kwargs):
        x, y = self.x_train, self.y_train
        if is_values:
            x = x.values
        self.model, self.canshu = trainSVM_RandomizedSearchCV(x, y)
        self.canshu["field_names"] = self.field_names
        return self.model, self.canshu

    def predict(self, x, *args, **kwargs):
        return self.model.predict(x)


class ShadowHierarchicalModel(SHHModel):

    def __init__(self):
        super().__init__(["IS", "VEG", "SOIL", "WAT"])
        self.code_is = 1
        self.code_sh_is = 5
        self.code_veg = 2
        self.code_sh_veg = 6
        self.code_soil = 3
        self.code_sh_soil = 7
        self.code_wat = 4
        self.code_sh_wat = 8
        self.optic_field_names = []
        self.as_sar_field_names = []
        self.de_sar_field_names = []

        self.veg_mod = SHHModelSVM(c_names=["VEG", "NOT_VEG"])
        self.is_low_mod = SHHModelSVM(c_names=["HEIGHT", "LOW"])
        self.is_soil_mod = SHHModelSVM(c_names=["IS", "SOIL"])
        self.ws_mod = SHHModelSVM(c_names=["WAT", "IS_SH", "OTHER_SH"])

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

        self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test
        self.field_names = list(self.x_train.keys())

        self.veg_mod.sample(x_train, y_train, x_test, y_test)
        self.veg_mod.reCategory([self.code_veg], None)
        self.veg_mod.dfFields(["Blue", "Green", "Red", "NIR", "NDVI", "NDWI"])
        self.veg_mod.train()
        self.veg_mod.score()
        train_args["veg_mod"] = self.veg_mod.canshu
        print("\nConfusion Matrix VEG:")
        print(self.veg_mod.cm.fmtCM())

        # IS Low
        self.is_low_mod.sample(x_train, y_train, x_test, y_test)
        self.is_low_mod.reCategory([self.code_is, self.code_soil],
                                   [self.code_wat, self.code_sh_is, self.code_sh_veg, self.code_sh_soil,
                                    self.code_sh_wat])
        self.is_low_mod.dfFields(["Blue", "Green", "Red", "NIR", "NDVI", "NDWI"])
        self.is_low_mod.train()
        self.is_low_mod.score()
        train_args["is_low_mod"] = self.is_low_mod.canshu
        print("Confusion Matrix HEIGHT LOW:")
        print(self.is_low_mod.cm.fmtCM())

        # IS Soil
        self.is_soil_mod.sample(x_train, y_train, x_test, y_test)
        self.is_soil_mod.reCategory([self.code_is], [self.code_soil])
        self.is_soil_mod.dfFields([
            "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
            "AS_VV", "AS_VH", "AS_VHDVV", "AS_C11", "AS_C22", "AS_Lambda1", "AS_Lambda2",
            "DE_VV", "DE_VH", "DE_VHDVV", "DE_C11", "DE_C22", "DE_Lambda1", "DE_Lambda2",
        ])
        self.is_soil_mod.train()
        self.is_soil_mod.score()
        train_args["is_soil_mod"] = self.is_soil_mod.canshu
        print("Confusion Matrix IS SOIL:")
        print(self.is_soil_mod.cm.fmtCM())

        # Water Shadow
        self.ws_mod.sample(x_train, y_train, x_test, y_test)
        self.ws_mod.reCategory([self.code_wat, self.code_sh_wat], [self.code_sh_is],
                               [self.code_sh_veg, self.code_sh_soil])
        self.ws_mod.dfFields([
            "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
            "AS_VV", "AS_VH", "AS_VHDVV", "AS_C11", "AS_C22", "AS_Lambda1", "AS_Lambda2",
            "DE_VV", "DE_VH", "DE_VHDVV", "DE_C11", "DE_C22", "DE_Lambda1", "DE_Lambda2",
        ])
        self.ws_mod.train()
        self.ws_mod.score()
        train_args["ws_mod"] = self.ws_mod.canshu
        print("Confusion Matrix WATER SHADOW:")
        print(self.ws_mod.cm.fmtCM())

        # self.load(r"F:\ProjectSet\Shadow\Hierarchical\20231209\20231212H205710")

        self.score(is_test_other=False)
        print("Confusion Matrix:")
        print(self.cm.fmtCM())

        train_args["field_names"] = self.field_names
        return train_args

    def predict(self, x, *args, **kwargs):

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

        return y_out

    def save(self, dirname, *args, **kwargs):
        self.veg_mod.save(os.path.join(dirname, "veg_mod.mod"))
        self.is_low_mod.save(os.path.join(dirname, "is_low_mod.mod"))
        self.is_soil_mod.save(os.path.join(dirname, "is_soil_mod.mod"))
        self.ws_mod.save(os.path.join(dirname, "ws_mod.mod"))

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


class ShadowHierarchicalImageClassification(ShadowImageClassification):

    def __init__(self, dat_fn, model_dir):
        super().__init__(dat_fn, model_dir)
        self.features = []

    def classify(self, mod, features, mod_name):
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

    def predict(self, x, *args, **kwargs) -> np.ndarray:
        return self.model.predict(x)


class ShadowHierarchicalTrainImdcOne(ShadowMain):
    GRS = {}

    def __init__(self, model_dir=None):
        super().__init__()
        self.raster_dfn = ""
        self.sample_dfn = ""
        self.model_dfn = ""
        self.model_dir = ""
        self.raster_fn = ""
        self.sample_fn = ""
        self.sample_csv_fn = ""
        self.sample_csv_spl_fn = ""
        self.mod_name = ""

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
            self.model.save(model_name)

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

    def fit(self, is_sh_to_no=True):
        mod_name = self.mod_name
        mod_fn = os.path.join(self.model_dir, mod_name + "_mod.model")
        mod_args_fn = os.path.join(self.model_dir, mod_name + "_args.json")

        # train running ---
        print(">>> Train")
        printList("categorys", self.categorys)
        printList("feats", self.feats)
        printList("tags", self.tags)
        x_train, y_train, x_test, y_test = self.getSample(self.categorys, self.feats, self.tags, is_sh_to_no)
        mod, mod_args = self.train_func(x_train, y_train)

        # Save
        self.model = mod
        self.saveModel(mod_fn)
        self.mod_args = mod_args
        self.mod_args["model_name"] = mod_name
        self.mod_args["model_filename"] = mod_fn
        self.mod_args["features"] = self.feats.copy()
        self.mod_args["categorys"] = self.categorys.copy()
        self.mod_args["tags"] = self.tags.copy()
        self.saveModArgs(mod_args_fn)

        # Confusion Matrix
        print(">>> Confusion Matrix")
        cm = ConfusionMatrix(len(self.cm_names), self.cm_names)
        # Train
        y_train_2 = self.model.predict(x_train)
        cm.addData(y_train, y_train_2)
        cm_arr = cm.calCM()
        n = saveCM(cm_arr, self.save_cm_file, cate_names=self.cm_names, infos=["TRAIN"])
        print("* Train")
        print(fmtCM(cm_arr, cate_names=self.cm_names))
        cm.clear()
        # Test
        y_test_2 = self.model.predict(x_test)
        cm.addData(y_test, y_test_2)
        cm_arr = cm.calCM()
        n = saveCM(cm_arr, self.save_cm_file, cate_names=self.cm_names, infos=["TEST"])
        print("* Test")
        print(fmtCM(cm_arr, cate_names=self.cm_names))

        # Class
        self.sic.classify(mod_fn, self.feats, mod_name)

    def addCSVFile(self, spl_fn=None, is_spl=False):
        if spl_fn is None:
            spl_fn = self.sample_csv_spl_fn
        if is_spl:
            samplingToCSV(spl_fn, self.gr, spl_fn)
        self.csv_spl = CSVSamples(spl_fn)
        self._getNames()

    def initCSVSamples(self, is_save_csv=True):
        self._getNames()
        if is_save_csv:
            self.csv_spl.saveToFile(self.save_train_spl_file)

    def printAcc(self, fs=None):
        print("\nConfusion Matrix VEG:", file=fs)
        print(self.model.veg_mod.cm.fmtCM(), file=fs)
        print("\nConfusion Matrix HEIGHT LOW:", file=fs)
        print(self.model.is_low_mod.cm.fmtCM(), file=fs)
        print("\nConfusion Matrix IS SOIL:", file=fs)
        print(self.model.is_soil_mod.cm.fmtCM(), file=fs)
        print("\nConfusion Matrix WATER SHADOW:", file=fs)
        print(self.model.ws_mod.cm.fmtCM(), file=fs)
        print("\nConfusion Matrix:", file=fs)
        print(self.model.cm.fmtCM(), file=fs)
        self.printAccuracy(fs)

        # print("{0:<16} {1:<10.2f} {2:<10.2f} {3:<10.2f}\n".format("VEG", "OA", "PA", "UA"))
        # print("{0:<16} {1:<10.2f} {2:<10.2f} {3:<10.2f}\n".format("HEIGHT_LOW", "OA", "PA", "UA"))
        # print("{0:<16} {1:<10.2f} {2:<10.2f} {3:<10.2f}\n".format("IS_SOIL", "OA", "PA", "UA"))
        # print("{0:<16} {1:<10.2f} {2:<10.2f} {3:<10.2f}\n".format("WATER_SHADOW", "OA", "PA", "UA"))
        # print("{0:<16} {1:<10.2f} {2:<10.2f} {3:<10.2f}\n".format("", self.model.cm.OA(), "PA", "UA"))

    def printAccuracy(self, fs=None):
        print("\n{0:<8} {1:<10} {2:<10} {3:<10}".format("NAME", "OA", "PA", "UA"), file=fs)
        to_dict = self.model.cm.accuracy()
        for k in to_dict:
            print("{0:<8} {1:<10.2f} {2:<10.2f} {3:<10.2f}".format(
                k, to_dict[k][0], to_dict[k][1], to_dict[k][2]), file=fs)


class ShadowHierarchicalTrainImdcOneBeiJing(ShadowHierarchicalTrainImdcOne):

    def __init__(self, model_dir=None):
        super().__init__(model_dir)

        self.model_name = "BeiJing"

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

    def fit(self, is_sh_to_no=False):
        print("MODEL DIRNAME:", self.model_dir)
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

        self.printAccuracy()
        with open(self.save_cm_file, "w", encoding="utf-8") as f:
            self.printAcc(f)

        if is_train:
            # Save
            self.saveModel(self.model_dir)
            self.mod_args = mod_args
            self.mod_args["model_name"] = mod_name
            self.mod_args["model_filename"] = self.model_dir
            self.mod_args["features"] = self.feats.copy()
            self.mod_args["categorys"] = self.categorys.copy()
            self.mod_args["tags"] = self.tags.copy()
            self.saveModArgs(mod_args_fn)

    def classify(self, model_dirname=None):
        self.sicAddCategory("IS", (255, 0, 0))
        self.sicAddCategory("VEG", (0, 255, 0))
        self.sicAddCategory("SOIL", (255, 255, 0))
        self.sicAddCategory("WAT", (0, 0, 255))
        if model_dirname is None:
            model_dirname = self.model_dir
        self.model.load(model_dirname)
        mod_name = self.model_name + os.path.split(self.model_dir)[1]
        self.sic.model = self.model
        writeTexts(os.path.join(self.model_dir, "sic.txt"), model_dirname)
        self.sic.classify(self.model, self.feats, mod_name)


def main():
    shtio_bj = ShadowHierarchicalTrainImdcOneBeiJing()
    # shtio_bj.sampleToCsv()
    # shtio_bj.sampling()

    shtio_bj.addCSVFile()
    shtio_bj.csv_spl.fieldNameCategory("CNAME")  # CNAME
    shtio_bj.csv_spl.fieldNameTag("TAG")
    shtio_bj.csv_spl.addCategoryNames(["NOT_KNOW", "IS", "VEG", "SOIL", "WAT", "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH"])
    shtio_bj.csv_spl.readData()

    featureCallback1(shtio_bj)

    shtio_bj.initCSVSamples(is_save_csv=False)

    shtio_bj.fitCategoryNames("IS", "VEG", "SOIL", "WAT", "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH")
    shtio_bj.fitFeatureNames(
        "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
        "OPT_asm", "OPT_con", "OPT_cor", "OPT_dis", "OPT_ent", "OPT_hom", "OPT_mean", "OPT_var",
        "AS_VV", "AS_VH", "AS_VHDVV", "AS_C11", "AS_C12_imag", "AS_C12_real", "AS_C22", "AS_Lambda1", "AS_Lambda2",
        "AS_SPAN", "AS_Epsilon", "AS_Mu", "AS_RVI", "AS_m", "AS_Beta",
        "AS_VV_asm", "AS_VV_con", "AS_VV_cor", "AS_VV_dis", "AS_VV_ent", "AS_VV_hom", "AS_VV_mean", "AS_VV_var",
        "AS_VH_asm", "AS_VH_con", "AS_VH_cor", "AS_VH_dis", "AS_VH_ent", "AS_VH_hom", "AS_VH_mean", "AS_VH_var",
        "DE_VV", "DE_VH", "DE_VHDVV", "DE_C11", "DE_C12_imag", "DE_C12_real", "DE_C22", "DE_Lambda1", "DE_Lambda2",
        "DE_SPAN", "DE_Epsilon", "DE_Mu", "DE_RVI", "DE_Beta", "DE_m",
        "DE_VH_asm", "DE_VH_con", "DE_VH_cor", "DE_VH_dis", "DE_VH_ent", "DE_VH_hom", "DE_VH_mean", "DE_VH_var",
        "DE_VV_asm", "DE_VV_con", "DE_VV_cor", "DE_VV_dis", "DE_VV_ent", "DE_VV_hom", "DE_VV_mean", "DE_VV_var"
    )

    shtio_bj.addModel(ShadowHierarchicalModel())

    shtio_bj.fit()

    shtio_bj.classify()


def featureCallback1(shtio_bj):
    shtio_bj.featureScaleMinMax("Blue", 99.76996, 2397.184)
    shtio_bj.featureScaleMinMax("Green", 45.83414, 2395.735)
    shtio_bj.featureScaleMinMax("Red", 77.79654, 2726.7026)
    shtio_bj.featureScaleMinMax("NIR", 87.66086, 3498.4321)
    shtio_bj.featureScaleMinMax("NDVI", -0.5007727, 0.7354284)
    shtio_bj.featureScaleMinMax("NDWI", -0.6572631, 0.7623875)
    shtio_bj.featureScaleMinMax("OPT_asm", 0.02124183, 0.998366)
    shtio_bj.featureScaleMinMax("OPT_con", 0.0, 169.74791)
    shtio_bj.featureScaleMinMax("OPT_cor", -0.036879253, 0.99688625)
    shtio_bj.featureScaleMinMax("OPT_dis", 0.0, 9.799746)
    shtio_bj.featureScaleMinMax("OPT_ent", 0.0, 3.8249474)
    shtio_bj.featureScaleMinMax("OPT_hom", 0.12091503, 0.998366)
    shtio_bj.featureScaleMinMax("OPT_mean", 4.941177, 53.7353)
    shtio_bj.featureScaleMinMax("OPT_var", 0.0, 236.09961)
    shtio_bj.featureCallBack("AS_VV", cal_10log10)
    shtio_bj.featureCallBack("AS_VH", cal_10log10)
    shtio_bj.featureCallBack("AS_C11", cal_10log10)
    shtio_bj.featureCallBack("AS_C22", cal_10log10)
    shtio_bj.featureCallBack("AS_Lambda1", cal_10log10)
    shtio_bj.featureCallBack("AS_Lambda2", cal_10log10)
    shtio_bj.featureCallBack("AS_SPAN", cal_10log10)
    shtio_bj.featureCallBack("AS_Epsilon", cal_10log10)
    shtio_bj.featureCallBack("DE_VV", cal_10log10)
    shtio_bj.featureCallBack("DE_VH", cal_10log10)
    shtio_bj.featureCallBack("DE_C11", cal_10log10)
    shtio_bj.featureCallBack("DE_C22", cal_10log10)
    shtio_bj.featureCallBack("DE_Lambda1", cal_10log10)
    shtio_bj.featureCallBack("DE_Lambda2", cal_10log10)
    shtio_bj.featureCallBack("DE_SPAN", cal_10log10)
    shtio_bj.featureCallBack("DE_Epsilon", cal_10log10)
    shtio_bj.featureScaleMinMax("AS_VV", -24.609674, 5.9092603)
    shtio_bj.featureScaleMinMax("AS_VH", -31.865038, -5.2615275)
    shtio_bj.featureScaleMinMax("AS_VHDVV", 0.0, 0.95164585)
    shtio_bj.featureScaleMinMax("AS_C11", -22.61998, 5.8634768)
    shtio_bj.featureScaleMinMax("AS_C22", -28.579813, -5.2111626)
    shtio_bj.featureScaleMinMax("AS_Lambda1", -21.955856, 6.124724)
    shtio_bj.featureScaleMinMax("AS_Lambda2", -29.869734, -8.284683)
    shtio_bj.featureScaleMinMax("AS_SPAN", -22.58362, 6.97997)
    shtio_bj.featureScaleMinMax("AS_Epsilon", 0.0, 35.12922)
    shtio_bj.featureScaleMinMax("AS_Mu", -0.7263123, 0.7037629)
    shtio_bj.featureScaleMinMax("AS_RVI", 0.07459847, 2.076324)
    shtio_bj.featureScaleMinMax("AS_m", 0.26469338, 0.97544414)
    shtio_bj.featureScaleMinMax("AS_Beta", 0.632338, 0.9869048)
    shtio_bj.featureScaleMinMax("AS_VH_asm", 0.02124183, 0.050653595)
    shtio_bj.featureScaleMinMax("AS_VH_con", 6.572378, 59.151405)
    shtio_bj.featureScaleMinMax("AS_VH_cor", 0.006340516, 0.86876196)
    shtio_bj.featureScaleMinMax("AS_VH_dis", 1.9767247, 5.8193297)
    shtio_bj.featureScaleMinMax("AS_VH_ent", 3.0939856, 3.8060431)
    shtio_bj.featureScaleMinMax("AS_VH_hom", 0.16666667, 0.40849674)
    shtio_bj.featureScaleMinMax("AS_VH_mean", 7.514706, 54.04412)
    shtio_bj.featureScaleMinMax("AS_VH_var", 5.9986033, 108.64137)
    shtio_bj.featureScaleMinMax("AS_VV_asm", 0.022875817, 0.050653595)
    shtio_bj.featureScaleMinMax("AS_VV_con", 4.5305123, 48.325462)
    shtio_bj.featureScaleMinMax("AS_VV_cor", 0.21234758, 0.88228023)
    shtio_bj.featureScaleMinMax("AS_VV_dis", 1.5990733, 5.22229)
    shtio_bj.featureScaleMinMax("AS_VV_ent", 3.1254923, 3.7871387)
    shtio_bj.featureScaleMinMax("AS_VV_hom", 0.18464053, 0.45261437)
    shtio_bj.featureScaleMinMax("AS_VV_mean", 8.544118, 51.573532)
    shtio_bj.featureScaleMinMax("AS_VV_var", 3.8744159, 96.8604)
    shtio_bj.featureScaleMinMax("DE_VV", -27.851603, 5.094706)
    shtio_bj.featureScaleMinMax("DE_VH", -35.427082, -5.4092093)
    shtio_bj.featureScaleMinMax("DE_VHDVV", 0.0, 1.0289364)
    shtio_bj.featureScaleMinMax("DE_C11", -26.245598, 4.9907513)
    shtio_bj.featureScaleMinMax("DE_C22", -32.042320, -5.322515)
    shtio_bj.featureScaleMinMax("DE_Lambda1", -25.503738, 5.2980003)
    shtio_bj.featureScaleMinMax("DE_Lambda2", -33.442368, -8.68537)
    shtio_bj.featureScaleMinMax("DE_SPAN", -24.81076, 4.82663)
    shtio_bj.featureScaleMinMax("DE_Epsilon", 0.0, 21.882689)
    shtio_bj.featureScaleMinMax("DE_Mu", -0.6823329, 0.7723537)
    shtio_bj.featureScaleMinMax("DE_RVI", 0.0940072, 2.1935015)
    shtio_bj.featureScaleMinMax("DE_m", 0.24836189, 0.9705721)
    shtio_bj.featureScaleMinMax("DE_Beta", 0.6241778, 0.9852859)
    shtio_bj.featureScaleMinMax("DE_VH_asm", 0.022875817, 0.05392157)
    shtio_bj.featureScaleMinMax("DE_VH_con", 5.6798058, 51.11825)
    shtio_bj.featureScaleMinMax("DE_VH_cor", 0.12444292, 0.87177193)
    shtio_bj.featureScaleMinMax("DE_VH_dis", 1.8186697, 5.456009)
    shtio_bj.featureScaleMinMax("DE_VH_ent", 2.9679575, 3.7997417)
    shtio_bj.featureScaleMinMax("DE_VH_hom", 0.1748366, 0.42810458)
    shtio_bj.featureScaleMinMax("DE_VH_mean", 7.6176476, 55.176476)
    shtio_bj.featureScaleMinMax("DE_VH_var", 5.513511, 95.38374)
    shtio_bj.featureScaleMinMax("DE_VV_asm", 0.02124183, 0.057189543)
    shtio_bj.featureScaleMinMax("DE_VV_con", 5.0987973, 57.54357)
    shtio_bj.featureScaleMinMax("DE_VV_cor", 0.19514601, 0.88254523)
    shtio_bj.featureScaleMinMax("DE_VV_dis", 1.7117102, 5.6928787)
    shtio_bj.featureScaleMinMax("DE_VV_ent", 2.993163, 3.7997417)
    shtio_bj.featureScaleMinMax("DE_VV_hom", 0.17320262, 0.44444445)
    shtio_bj.featureScaleMinMax("DE_VV_mean", 6.4852943, 54.04412)
    shtio_bj.featureScaleMinMax("DE_VV_var", 4.44714, 111.17851)


if __name__ == "__main__":
    main()
