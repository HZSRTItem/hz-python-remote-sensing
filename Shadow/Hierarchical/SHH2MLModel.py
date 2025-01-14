# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2MLModel.py
@Time    : 2024/7/4 14:19
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2MLModel
-----------------------------------------------------------------------------"""

import os.path

import joblib
import pandas as pd
from osgeo import gdal
from sklearn.ensemble import RandomForestClassifier

from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.SRTFeature import SRTFeaturesMemory, SRTFeaturesCalculation
from SRTCodes.SRTModel import mapDict
from SRTCodes.SRTModelImage import GDALImdc
from Shadow.Hierarchical.SHH2Config import samplesDescription


class SHH2MLTraining:

    def __init__(self):
        self.category_names = None
        self.df = None
        self.models = {}
        self.categorys = {}
        self.acc_dict = {}
        self.clf = None
        self.map_dict = None
        self.test_filters = {0: [], 1: []}
        self.sfc = None

    def train(self, name, x_keys=None, c_fn="CATEGORY", map_dict=None, clf=None, *args, **kwargs):
        if x_keys is None:
            x_keys = []
        if map_dict is None:
            map_dict = self.map_dict

        x_train, y_train, category_names, df_train = self.train_test(1, x_keys, c_fn, map_dict, )
        x_test, y_test, category_names, df_test = self.train_test(0, x_keys, c_fn, map_dict, )

        self.category_names = category_names

        if clf is None:
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=1, min_samples_split=2)
            # clf = SVC(kernel="rbf", C=4.742, gamma=0.42813)

        clf.fit(x_train, y_train)
        train_acc, test_acc = clf.score(x_train, y_train) * 100, clf.score(x_test, y_test) * 100
        self.clf = clf

        to_dict = self.addAccuracy(df_test, x_test, y_test)

        self.categorys[name] = to_dict
        self.acc_dict[name] = {}

        self.models[name] = clf
        return train_acc, test_acc

    def addAccuracy(self, df_test, x_test, y_test):
        y2 = self.clf.predict(x_test)
        to_dict = {"y1": y_test, "y2": y2.tolist(), }
        if "X" in df_test:
            to_dict["X"] = df_test["X"].tolist()
        if "Y" in df_test:
            to_dict["Y"] = df_test["Y"].tolist()
        if "SRT" in df_test:
            to_dict["SRT"] = df_test["SRT"].tolist()
        if "CNAME" in df_test:
            to_dict["CNAME"] = df_test["CNAME"].tolist()
        return to_dict

    def readCSV(self, csv_fn):
        self.df = pd.read_csv(csv_fn)

    def toCSV(self, csv_fn, **kwargs):
        self.df.to_csv(csv_fn, index=False, **kwargs)

    def train_test(self, n, x_keys, c_fn=None, map_dict=None):
        _df = self.df[self.df["TEST"] == n]
        self.sfc.initData("df", _df)
        self.sfc.fit()

        data_filter = self.test_filters[n]
        if len(data_filter) != 0:
            df_list = []
            for k, data in data_filter:
                df_list.extend(_df[_df[k] == data].to_dict("records"))
            _df = pd.DataFrame(df_list)

        y, data_select = mapDict(_df[c_fn].tolist(), map_dict, return_select=True)
        _df = _df[data_select]
        x = _df[x_keys].values
        return x, y, _df["CNAME"].tolist(), _df


class TIC(SHH2MLTraining):
    r""" train image classification

    :param name: to name
    :param df: samples
    :param map_dict: cname -> category. if cname not in map_dict, will remove this sample
    :param raster_fn: raster filename
    :param x_keys: fit field names
    :param cm_names: confusion matrix names
    :param clf: RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=1, min_samples_split=2)
    :param category_field_name: in df
    :param color_table: imdc. default:{1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), }
    :param sfm: SRTFeaturesMemory(names=names).initCallBacks(); sfm.callbacks(name).add(func)
    :param is_save_model: save model
    :param is_save_imdc: save_imdc
    :param td: TimeDirectory(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods").initLog()
    """

    def __init__(
            self,
            name="TIC",
            df=pd.DataFrame(),
            map_dict=None,
            raster_fn=None,
            x_keys=None,
            cm_names=None,
            clf=None,
            sfc=None,

            category_field_name="CNAME",
            color_table=None,
            sfm: SRTFeaturesMemory = None,
            is_save_model=True,
            is_save_imdc=True,
            td=None,
            func_save_model=None,
    ):

        super(TIC, self).__init__()

        if cm_names is None:
            cm_names = []
        if color_table is None:
            color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), }
        if x_keys is None:
            x_keys = []
        if map_dict is None:
            map_dict = {}
        if clf is None:
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=1, min_samples_split=2)
        if sfc is None:
            sfc = SRTFeaturesCalculation(*x_keys)

        self.name = name
        self.df = df
        self.map_dict = map_dict
        self.raster_fn = raster_fn
        self.x_keys = x_keys
        self.cm_names = cm_names
        self.clf = clf
        self.sfc = sfc

        self.category_field_name = category_field_name
        self.color_table = color_table
        self.sfm = sfm
        self.is_save_model = is_save_model
        self.is_save_imdc = is_save_imdc
        self.td = td

        print(self.td.time_dfn.dirname)

        self.log("#", "-" * 20, "SHH2ML Image classification", "-" * 20, "#")
        self.kw("NAME", self.name)
        self.kw("data", self.df, sep=":\n")
        self.kw("CLF", self.clf)
        self.kw("MAP_DICT", self.map_dict)
        self.kw("CATEGORY_FIELD_NAME", self.map_dict)
        self.kw("X_KEYS", self.x_keys)
        self.kw("COLOR_TABLE", self.color_table)

        if self.td is not None:
            self.td.saveDF("{}_data.csv".format(self.name), self.df, index=False)

        self.accuracy_dict = {}
        self.cm = ConfusionMatrix()

        self.func_save_model = lambda to_mod_fn: joblib.dump(self.clf, to_mod_fn)
        if func_save_model is not None:
            self.func_save_model = func_save_model

    def train(self, to_mod_fn=None, **kwargs):
        self.td.kw("NAME", self.name)
        self.kw("DF_DES", samplesDescription(self.df), sep=":\n", end="\n")

        x_train, y_train, category_names, df_train = self.train_test(
            1, self.x_keys, self.category_field_name, self.map_dict)
        x_test, y_test, category_names, df_test = self.train_test(
            0, self.x_keys, self.category_field_name, self.map_dict)
        self.kw("DF_DES DF_TRAIN", samplesDescription(df_train), sep=":\n", end="\n")
        self.kw("DF_DES DF_TEST", samplesDescription(df_test), sep=":\n", end="\n")

        self.clf.fit(x_train, y_train)
        to_mod_fn = self.saveModel(to_mod_fn)

        train_acc, test_acc = self.clf.score(x_train, y_train) * 100, self.clf.score(x_test, y_test) * 100
        self.accuracy_dict = self.addAccuracy(df_test, x_test, y_test)
        self.calCM()

        self.td.kw("TO_MOD_FN", to_mod_fn)
        self.td.kw("TRAIN_ACC", train_acc)
        self.td.kw("TEST_ACC", test_acc)
        self.td.kw("CM", self.cm.fmtCM(), sep=":\n", end="\n")
        self.td.kw("OA", self.cm.OA())
        self.td.kw("KAPPA", self.cm.getKappa())

        if self.td is not None:
            self.td.saveJson("{}_accuracy_data.json".format(self.name), self.accuracy_dict)
            self.td.saveDF("{}_accuracy_data.csv".format(self.name), pd.DataFrame(self.accuracy_dict), index=False)

        return self

    def saveModel(self, to_mod_fn):
        if to_mod_fn is None:
            if self.td is not None:
                to_mod_fn = self.td.time_dfn.fn("{}_mod.mod".format(self.name))
        if to_mod_fn is not None:
            if self.is_save_model:
                self.func_save_model(to_mod_fn)
        return to_mod_fn

    def calCM(self):
        cm = ConfusionMatrix(class_names=self.cm_names)
        cm.addData(self.accuracy_dict["y1"], self.accuracy_dict["y2"])
        self.cm = cm
        return cm

    def kw(self, key, value, sep=": ", end="\n", is_print=None):
        if self.td is None:
            print(key, value, sep=sep, end=end, )
        else:
            self.td.kw(key, value, sep=sep, end=end, is_print=is_print)

    def log(self, *text, sep=" ", end="\n", is_print=None):
        if self.td is None:
            print(*text, sep=sep, end=end, )
        else:
            self.td.log(*text, sep=sep, end=end, is_print=is_print)

    def imdc(self, to_imdc_fn=None, **kwargs):
        if self.sfm is not None:
            gimdc = GDALImdc(self.raster_fn, is_sfm=True, sfc=self.sfc)
            gimdc.sfm = self.sfm
        else:
            gimdc = GDALImdc(self.raster_fn, is_sfm=False, sfc=self.sfc)

        to_imdc_fn = self.getImdcFn(to_imdc_fn)

        gimdc.imdc1(self.clf, to_imdc_fn, self.x_keys, color_table=self.color_table)

        if to_imdc_fn == r"F:\ProjectSet\Shadow\Hierarchical\Temp\tmp_imdc.tif":
            return gdal.Open(to_imdc_fn).ReadAsArray()

        if not self.is_save_imdc:
            os.remove(to_imdc_fn)

        return None

    def getImdcFn(self, to_imdc_fn):
        if to_imdc_fn is None:
            if self.td is not None:
                to_imdc_fn = self.td.time_dfn.fn("{}_imdc.tif".format(self.name))
        if to_imdc_fn is None:
            to_imdc_fn = r"F:\ProjectSet\Shadow\Hierarchical\Temp\tmp_imdc.tif"
        self.td.kw("TO_IMDC_FN", to_imdc_fn)
        return to_imdc_fn


def main():
    pass


if __name__ == "__main__":
    main()
