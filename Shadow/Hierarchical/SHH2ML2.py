# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2ML2.py
@Time    : 2024/6/17 10:20
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2ML2
-----------------------------------------------------------------------------"""
import os.path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.SRTFeature import SRTFeaturesMemory
from SRTCodes.SRTModelImage import GDALImdc
from SRTCodes.SRTTimeDirectory import TimeDirectory
from SRTCodes.Utils import FRW, DirFileName
from Shadow.Hierarchical.SHH2Accuracy import accuracyY12


def mapDict(data, map_dict):
    if map_dict is None:
        return data
    to_list = []
    for d in data:
        if d in map_dict:
            to_list.append(map_dict[d])
    return to_list


class SHH2MLTrainSamplesTiao:

    def __init__(self):
        self.map_dict = None
        self.clf = None
        self.df = None
        self.categorys = {}
        self.acc_dict = {}
        self.category_names = []

    def train(self, name, x_keys=None, c_fn="CATEGORY", map_dict=None, clf=None):
        if x_keys is None:
            x_keys = []

        def train_test(n):
            _df = self.df[self.df["TEST"] == n]
            x = _df[x_keys].values
            y = mapDict(_df[c_fn].tolist(), map_dict)
            return x, y, _df["CNAME"].tolist()

        x_train, y_train, category_names = train_test(1)
        x_test, y_test, category_names = train_test(0)
        self.category_names=category_names

        if clf is None:
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=2, min_samples_split=4)
            # clf = SVC(kernel="rbf", C=4.742, gamma=0.42813)

        clf.fit(x_train, y_train)

        train_acc, test_acc = clf.score(x_train, y_train) * 100, clf.score(x_test, y_test) * 100
        # print("  Train Accuracy: {:.2f}".format(train_acc))
        # print("  Test  Accuracy: {:.2f}".format(test_acc))

        y2 = clf.predict(x_test)
        self.categorys[name] = {"y1": y_test, "y2": y2}
        self.acc_dict[name] = {}

        self.clf = clf
        return train_acc, test_acc

    def accuracyOAKappa(self, cm_name, cnames, y1_map_dict=None, y2_map_dict=None, fc_category=None):
        cm_str = ""
        for name, line in self.categorys.items():
            # y1 = mapDict(line["y1"], y1_map_dict)
            # y2 = mapDict(line["y2"], y2_map_dict)
            # cm = ConfusionMatrix(class_names=cnames)
            # cm.addData(y1, y2)

            cm = accuracyY12(
                self.category_names, line["y2"],
                self.map_dict, y2_map_dict,
                cnames=["IS", "VEG", "SOIL", "WAT"],
                fc_category=fc_category,
            )

            cm_str += "> {} IS\n".format(name)
            cm_str += "{}\n\n".format(cm.fmtCM())
            self.acc_dict[name]["{}_OA".format(cm_name)] = cm.OA()
            self.acc_dict[name]["{}_Kappa".format(cm_name)] = cm.getKappa()

            cm2 = ConfusionMatrix(class_names=["IS", "VEG", "SOIL", "WAT"])
            cm2.addData(line["y1"], line["y2"])
            cm_str += "> {} {}\n".format(name, " ".join(["IS", "VEG", "SOIL", "WAT"]))
            cm_str += "{}\n\n".format(cm2.fmtCM())

            # cm3 = cm2.accuracyCategory("IS")
            # self.acc_dict[name]["{}_OA2".format(cm_name)] = cm3.OA()
            # self.acc_dict[name]["{}_Kappa2".format(cm_name)] = cm3.getKappa()

        return cm_str


class SHH2ML_TST_main:

    def __init__(self, city_name, is_save_model=False):
        self.category_cnames = None
        self.fc_category = None
        self.dfn = None
        self.raster_fn = None
        self.city_name = city_name
        self.train_test_df = None
        self.y2_map_dict = None
        self.y1_map_dict = None
        self.color_table = None
        self.csv_fn = None
        self.json_fn = None
        self.json_dict = None
        self.td = None
        self.map_dict = {}
        self.sfm = None
        self.acc_dict_list = []
        self.mod_list = []
        self.is_save_model = is_save_model

    def mapCategory(self, data_list):
        map_dict = self.map_dict
        to_data_list = []
        for d in data_list:
            to_data_list.append(map_dict[d])
        return to_data_list

    def getDf(self):
        _df = pd.read_csv(self.csv_fn)
        self.category_cnames = _df["CNAME"].tolist()
        _df["CATEGORY"] = self.mapCategory(self.category_cnames)

        for k in self.sfm.names:
            if k in _df:
                _df[k] = self.sfm.callbacks(k).fit(_df[k])
        return _df

    def accRun(self, n_train):
        self.td.log("\n> ", n_train)
        sh2_tst = SHH2MLTrainSamplesTiao()
        sh2_tst.map_dict = self.map_dict
        sh2_tst.df = self.train_test_df

        model_dict = {}
        for n, k in enumerate(self.json_dict):
            self.td.log("[{}] {}".format(n, k), end=" ")
            train_acc, test_acc = sh2_tst.train(k, self.json_dict[k])
            self.td.log("Accuracy: {:>3.2f}% {:>3.2f}%".format(train_acc, test_acc), end=" ")
            self.td.log("Model: {}".format(str(sh2_tst.clf)))
            model_dict[k] = sh2_tst.clf

        cm_str = sh2_tst.accuracyOAKappa(
            "IS", ["IS", "NOT_KNOW"],
            y1_map_dict=self.y1_map_dict,
            y2_map_dict=self.y2_map_dict,
            fc_category=self.fc_category
        )
        self.td.buildWriteText("cm{}.txt".format(n_train)).write(cm_str)

        _df_acc = pd.DataFrame(sh2_tst.acc_dict).T
        self.td.saveDF("acc{}.csv".format(n_train), _df_acc, index=True)
        self.td.log(_df_acc.sort_values("IS_OA", ascending=False))

        self.acc_dict_list.append(sh2_tst.acc_dict)
        self.mod_list.append(model_dict)

    def train(self):
        self.td.log("#", "-" * 20, "SHH2ML Five Mean", "-" * 20, )
        self.td.log("# Test category is IS")

        for i in range(5):
            self.accRun(i + 1)
        self.td.log()

        df_acc = None
        for df in self.acc_dict_list:
            if df_acc is None:
                df_acc = pd.DataFrame(df)
            else:
                df_acc += pd.DataFrame(df)

        df_acc = df_acc.T
        df_acc = df_acc / len(self.acc_dict_list)
        df_acc.to_csv(self.td.fn("accuracy.csv"))
        df_acc = df_acc.sort_values("IS_OA", ascending=False)
        self.td.log(df_acc)

    def imdc(self):
        self.td.log("#", "-" * 20, "SHH2ML Image classification", "-" * 20, )
        self.td.log("# Test category is IS")

        gimdc = GDALImdc(self.raster_fn)
        gimdc.sfm = self.sfm
        sh2_tst = SHH2MLTrainSamplesTiao()
        sh2_tst.map_dict = self.map_dict
        sh2_tst.df = self.train_test_df

        for n, k in enumerate(self.json_dict):
            self.td.log("[{}] {}".format(n, k), end=" ")
            train_acc, test_acc = sh2_tst.train(k, self.json_dict[k])
            self.td.log("Accuracy: {:>3.2f}% {:>3.2f}%".format(train_acc, test_acc))
            filename = self.td.time_dfn.fn("{}_mod.mod".format(k))
            joblib.dump(sh2_tst.clf, filename)
            to_fn = self.td.time_dfn.fn("{}_imdc.tif".format(k))
            self.td.kw("TO_IMDC_FN", to_fn)
            gimdc.imdc1(sh2_tst.clf, to_fn, self.json_dict[k], color_table=self.color_table)

    def saveModel(self):
        if not self.is_save_model:
            return
        to_mod_list = []
        for i, model_dict in enumerate(self.mod_list):
            to_mod_dirname = self.td.dn("model{}".format(i + 1), True)
            dfn = DirFileName(to_mod_dirname)
            to_model_dict = {}
            for k in model_dict:
                to_mod_fn = dfn.fn("{}.mod".format(k))
                model = model_dict[k]
                to_model_dict[k] = {"model": str(model), "filename": to_mod_fn}
                joblib.dump(model, to_mod_fn)
            to_mod_list.append(to_model_dict)
        FRW(self.td.fn("model.json")).saveJson(to_mod_list)

    def getSfm(self):
        if self.city_name == "cd":
            range_json_fn = self.td.kw(
                "RANGE_JSON_FN", r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\SHH2_CD2_range.json")
        elif self.city_name == "qd":
            range_json_fn = self.td.kw(
                "RANGE_JSON_FN", r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\SHH2_QD2_range.json")
        elif self.city_name == "bj":
            range_json_fn = self.td.kw(
                "RANGE_JSON_FN", r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\SHH2_BJ2_range.json")
        else:
            range_json_fn = None

        range_json_dict = None
        if range_json_fn is not None:
            if os.path.isfile(range_json_fn):
                range_json_dict = FRW(range_json_fn).readJson()

        if range_json_dict is not None:
            _sfm = SRTFeaturesMemory(names=list(range_json_dict.keys()))
            _sfm.initCallBacks()
            for name in range_json_dict:
                _sfm.callbacks(name).addScaleMinMax(
                    range_json_dict[name]["min"], range_json_dict[name]["max"], is_trans=True, is_to_01=True)
        else:
            _sfm = SRTFeaturesMemory()

        return _sfm

    def getRasterFn(self):
        self.td.buildWriteText("{}.txt".format(self.city_name)).write(self.city_name)
        if self.city_name == "qd":
            raster_fn = r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\SHH2_QD2_envi.dat"
        elif self.city_name == "cd":
            raster_fn = r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\SHH2_CD2_envi.dat"
        elif self.city_name == "bj":
            raster_fn = r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\SHH2_BJ2_envi.dat"
        else:
            raise Exception("City name \"{}\"".format(self.city_name))
        return raster_fn

    def dfDes(self):
        _df = self.train_test_df
        self.td.log("Training samples categorys numbers:")
        df_des_data = _df[_df["TEST"] == 1].groupby("CNAME").count()["SRT"]
        self.td.log(df_des_data)
        self.td.log(df_des_data.sum())
        self.td.log("Test samples categorys numbers:")
        df_des_data = _df[_df["TEST"] == 0].groupby("CNAME").count()["SRT"]
        self.td.log(df_des_data)
        self.td.log(df_des_data.sum())

    def getCSVFn(self):
        if self.city_name == "qd":
            csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\2\sh2_spl252_1.csv"
            csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\2\sh2_spl252_4_spl.csv"
        elif self.city_name == "cd":
            csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\2\sh2_spl26_4_spl.csv"
        elif self.city_name == "bj":
            csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\2\sh2_spl271_32.csv"
            csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\2\sh2_spl271_4_spl.csv"
        else:
            raise Exception("City name \"{}\"".format(self.city_name))
        return csv_fn

    def main(self):
        self.dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods")
        self.td = TimeDirectory(self.dfn.fn())
        self.td.initLog()
        self.td.log(self.td.time_dfn.dirname)
        self.td.buildWriteText("city_name_{}.txt".format(self.city_name)).write(self.city_name)

        self.city_name = self.td.kw("CITY_NAME", self.city_name)
        self.csv_fn = self.td.kw("CSV_FN", self.getCSVFn())
        self.raster_fn = self.td.kw("RASTER_FN", self.getRasterFn())
        self.json_fn = self.td.kw("JSON_FN", self.dfn.fn(r"Temp\{}.json".format("feats4")))

        self.map_dict = self.td.kw("MAP_DICT", {
            "IS": 1, "VEG": 2, "SOIL": 3, "WAT": 4,
            "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 3, "WAT_SH": 4
        })
        self.sfm = self.getSfm()
        self.acc_dict_list = []
        self.mod_list = []

        self.td.copyfile(__file__)

        self.color_table = self.td.kw(
            "COLOR_TABLE", {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), })
        self.y1_map_dict = self.td.kw("Y1_MAP_DICT", {1: 1, 2: 2, 3: 3, 4: 4})
        self.y2_map_dict = self.td.kw("Y2_MAP_DICT", {1: 1, 2: 2, 3: 3, 4: 4})
        self.fc_category = self.td.kw("FC_CATEGORY", [["IS"], ["SOIL"]])

        self.json_dict = FRW(self.json_fn).readJson()
        self.train_test_df = self.getDf()
        self.dfDes()


def imdc(city_name="qd"):
    tst_main = SHH2ML_TST_main(city_name, is_save_model=True)
    tst_main.main()
    tst_main.imdc()


def main():
    tst_main = SHH2ML_TST_main("bj", is_save_model=True)
    tst_main.main()
    tst_main.train()
    # tst_main.saveModel()
    return


if __name__ == "__main__":
    r"""
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2ML2 import imdc; imdc('qd')"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2ML2 import imdc; imdc('bj')"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2ML2 import imdc; imdc('cd')"
    
    """
    main()
