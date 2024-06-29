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
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from SRTCodes.GDALUtils import GDALSamplingFast, GDALSampling
from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.SRTFeature import SRTFeaturesMemory
from SRTCodes.SRTModelImage import GDALImdc
from SRTCodes.SRTTimeDirectory import TimeDirectory
from SRTCodes.Utils import FRW, DirFileName
from Shadow.Hierarchical import SHH2Config
from Shadow.Hierarchical.SHH2Accuracy import accuracyY12, SHH2SamplesManage


def mapDict(data, map_dict):
    if map_dict is None:
        return data
    to_list = []
    for d in data:
        if d in map_dict:
            to_list.append(map_dict[d])
    return to_list


class SHH2MLTraining:

    def __init__(self):
        self.category_names = None
        self.df = None
        self.models = {}
        self.categorys = {}
        self.acc_dict = {}
        self.clf = None

    def train(self, name, x_keys=None, c_fn="CATEGORY", map_dict=None, clf=None):
        if x_keys is None:
            x_keys = []

        def train_test(n):
            _df = self.df[self.df["TEST"] == n]
            x = _df[x_keys].values
            y = mapDict(_df[c_fn].tolist(), map_dict)
            return x, y, _df["CNAME"].tolist(), _df

        x_train, y_train, category_names, df_train = train_test(1)
        x_test, y_test, category_names, df_test = train_test(0)
        self.category_names = category_names

        if clf is None:
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=1, min_samples_split=2)
            # clf = SVC(kernel="rbf", C=4.742, gamma=0.42813)

        clf.fit(x_train, y_train)

        train_acc, test_acc = clf.score(x_train, y_train) * 100, clf.score(x_test, y_test) * 100
        # print("  Train Accuracy: {:.2f}".format(train_acc))
        # print("  Test  Accuracy: {:.2f}".format(test_acc))

        y2 = clf.predict(x_test)
        to_dict = {"y1": y_test, "y2": y2.tolist(), }
        if "X" in df_test:
            to_dict["X"] = df_test["X"].tolist()
        if "Y" in df_test:
            to_dict["Y"] = df_test["Y"].tolist()
        if "SRT" in df_test:
            to_dict["SRT"] = df_test["SRT"].tolist()
        if "CNAME" in df_test:
            to_dict["CNAME"] = df_test["CNAME"].tolist()

        self.categorys[name] = to_dict
        self.acc_dict[name] = {}

        self.clf = clf
        self.models[name] = clf
        return train_acc, test_acc


class SHH2MLTrainSamplesTiao(SHH2MLTraining):

    def __init__(self):
        super(SHH2MLTrainSamplesTiao, self).__init__()
        self.map_dict = None
        self.clf = None
        self.category_names = []

    def accuracyOAKappa(self, cm_name, cnames, y1_map_dict=None, y2_map_dict=None, fc_category=None, is_eq_number=True):
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
                is_eq_number=is_eq_number,
            )
            cm = cm.accuracyCategory("IS")
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


class SHH2ML_TST_Init:

    def __init__(self, city_name=None, is_save_model=False):
        self.is_imdc = False
        self.city_name = city_name
        self.y1_map_dict = None
        self.y2_map_dict = None
        self.fc_category = None
        self.json_dict = None
        self.train_test_df = None
        self.sfm = SRTFeaturesMemory()
        self.csv_fn = None
        self.td = None
        self.map_dict = {}
        self.category_cnames = None
        self.acc_dict_list = []
        self.mod_list = []
        self.color_table = None
        self.raster_fn = None
        self.n_run = 5
        self.dfn = None
        self.json_fn = None
        self.is_save_model = is_save_model
        self.is_eq_number = True
        self.sh2_tst_list = []

    def mapCategory(self, data_list):
        map_dict = self.map_dict
        to_data_list = []
        for d in data_list:
            to_data_list.append(map_dict[d])
        return to_data_list

    def getDf(self):
        _df = pd.read_csv(self.csv_fn)
        _df.to_csv(self.td.fn("training_data.csv"), index=False)
        self.category_cnames = _df["CNAME"].tolist()
        _df["CATEGORY"] = self.mapCategory(self.category_cnames)

        for k in self.sfm.names:
            if k in _df:
                _df[k] = self.sfm.callbacks(k).fit(_df[k])
        return _df

    def dfDes(self):
        _df = self.train_test_df
        self.td.log("Training samples categorys numbers:")
        df_des_data = _df[_df["TEST"] == 1].groupby("CNAME").count()["TEST"]
        self.td.log(df_des_data)
        self.td.log(df_des_data.sum())
        self.td.log("Test samples categorys numbers:")
        df_des_data = _df[_df["TEST"] == 0].groupby("CNAME").count()["TEST"]
        self.td.log(df_des_data)
        self.td.log(df_des_data.sum())

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

        self.sh2_tst_list.append(sh2_tst)

        cm_str = sh2_tst.accuracyOAKappa(
            "IS", ["IS", "NOT_KNOW"],
            y1_map_dict=self.y1_map_dict,
            y2_map_dict=self.y2_map_dict,
            fc_category=self.fc_category,
            is_eq_number=self.is_eq_number,
        )
        self.td.buildWriteText("cm{}.txt".format(n_train)).write(cm_str)

        _df_acc = pd.DataFrame(sh2_tst.acc_dict).T
        self.td.saveDF("acc{}.csv".format(n_train), _df_acc, index=True)
        self.td.log(_df_acc.sort_values("IS_OA", ascending=False))
        self.td.saveJson("categorys{}.json".format(n_train), sh2_tst.categorys)

        self.acc_dict_list.append(sh2_tst.acc_dict)
        self.mod_list.append(model_dict)

    def accRuns(self, n):
        for i in range(n):
            self.accRun(i + 1)

        self.td.log()
        df_acc = None
        for df in self.acc_dict_list:
            if df_acc is None:
                df_acc = pd.DataFrame(df)
            else:
                df_acc += pd.DataFrame(df)

        return df_acc

    def train(self, is_imdc=False):
        self.is_imdc = is_imdc
        self.td.log("#", "-" * 20, "SHH2ML Five Mean", "-" * 20, )
        self.td.log("# Test category is IS")
        df_acc = self.accRuns(self.n_run)
        df_acc = df_acc.T
        df_acc = df_acc / len(self.acc_dict_list)
        df_acc.to_csv(self.td.fn("accuracy.csv"))
        df_acc = df_acc.sort_values("IS_OA", ascending=False)
        self.td.log(df_acc)

        if self.is_imdc:

            self.td.log("#", "-" * 20, "SHH2ML Image classification", "-" * 20, )

            for i in range(self.n_run):
                acc = self.acc_dict_list[i]
                opt_as_de_oa = acc["OPT+AS+DE"]["IS_OA"]
                opt_as_oa = acc["OPT+AS"]["IS_OA"]
                opt_de_oa = acc["OPT+DE"]["IS_OA"]
                opt_oa = acc["OPT"]["IS_OA"]
                if ((opt_as_de_oa > opt_as_oa) and (opt_as_de_oa > opt_de_oa) and (opt_as_de_oa > opt_oa)) \
                        and (opt_as_oa > opt_oa) and (opt_de_oa > opt_oa):
                    df_acc = pd.DataFrame(acc)
                    df_acc = df_acc.T
                    df_acc = df_acc.sort_values("IS_OA", ascending=False)
                    print(df_acc)

                    gimdc = GDALImdc(self.raster_fn, is_sfm= False)
                    gimdc.sfm = self.sfm
                    model_dict = self.mod_list[i]
                    for k, clf in model_dict.items():
                        self.td.log("> {:<12} {:6.2f}% {:6.4f}".format(k, acc[k]["IS_OA"], acc[k]["IS_Kappa"]), end=" ")
                        filename = self.td.time_dfn.fn("{}_mod.mod".format(k))
                        joblib.dump(clf, filename)
                        to_fn = self.td.time_dfn.fn("{}_imdc.tif".format(k))
                        self.td.kw("TO_IMDC_FN", to_fn)
                        gimdc.imdc1(clf, to_fn, self.json_dict[k], color_table=self.color_table)

                    break

    def imdc(self):
        self.td.log("#", "-" * 20, "SHH2ML Image classification", "-" * 20, )
        self.td.log("# Test category is IS")

        gimdc = GDALImdc(self.raster_fn, is_sfm= False)
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

    def saveModel(self, *args, **kwargs):
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

    def getSfm(self, *args, **kwargs):
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

    def getRasterFn(self, *args, **kwargs):
        self.td.buildWriteText("{}.txt".format(self.city_name)).write(self.city_name)
        if self.city_name == "qd":
            raster_fn = SHH2Config.QD_ENVI_FN
        elif self.city_name == "cd":
            raster_fn = SHH2Config.CD_ENVI_FN
        elif self.city_name == "bj":
            raster_fn = SHH2Config.BJ_ENVI_FN
        else:
            raise Exception("City name \"{}\"".format(self.city_name))
        return raster_fn

    def getCSVFn(self, *args, **kwargs):
        return ""


class SHH2ML_TST_main(SHH2ML_TST_Init):

    def __init__(self, city_name, is_save_model=False):
        super().__init__(city_name, is_save_model=is_save_model)

    def getCSVFn(self):
        if self.city_name == "qd":
            csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\2\sh2_spl252_1.csv"
            csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\2\sh2_spl252_4_spl.csv"
            csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\vhl\sh2_spl25_vhl_2_spl.csv"
            csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\vhl\sh2_spl25_vhl_2_spl2.csv"
        elif self.city_name == "cd":
            csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\2\sh2_spl26_4_spl.csv"
            csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\2\sh2_spl26_4_spl2.csv"
        elif self.city_name == "bj":
            csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\2\sh2_spl271_32.csv"
            csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\2\sh2_spl271_4_spl.csv"
            csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\3\sh2_spl273_5_spl.csv"
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
        self.n_run = self.td.kw("N_RUN", 10)

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

        self.fc_category = self.td.kw(
            "FC_CATEGORY",
            # [["IS", "IS_SH", ], ["VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH"]]
            None
        )

        self.json_dict = FRW(self.json_fn).readJson()
        self.train_test_df = self.getDf()
        self.dfDes()


def imdc(city_name="qd"):
    tst_main = SHH2ML_TST_main(city_name, is_save_model=True)
    tst_main.main()
    tst_main.imdc()


def train(city_name="qd"):
    tst_main = SHH2ML_TST_main(city_name, is_save_model=True)
    tst_main.main()
    tst_main.train(True)


def main():
    def func1():
        tst_main = SHH2ML_TST_main("cd", is_save_model=True)
        tst_main.main()
        tst_main.train()
        # tst_main.saveModel()

    def func2():
        json_dict = FRW(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240619H204022\categorys1.json").readJson()
        to_dict = {}
        for k in json_dict:
            to_dict["X"] = json_dict[k]["X"]
            to_dict["Y"] = json_dict[k]["Y"]
            to_dict["y1"] = json_dict[k]["y1"]
            to_dict["SRT"] = json_dict[k]["SRT"]
            to_dict[k] = json_dict[k]["y2"]
        pd.DataFrame(to_dict).to_csv(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240619H204022\categorys1_data.csv",
                                     index=False)

    def func3():
        df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240619H204022\categorys1_data.csv")
        dirname = r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240618H114833"
        fns = ['OPT', 'OPT+AS', 'OPT+DE', 'OPT+AS+DE', 'OPT+GLCM', 'OPT+BS', 'OPT+C2', 'OPT+HA', 'OPT+SARGLCM']
        for fn in fns:
            fn2 = os.path.join(dirname, "{}_imdc.tif".format(fn))
            gs = GDALSampling(fn2)
            df[fn] = gs.sampling(df["X"].tolist(), df["Y"].tolist())["FEATURE_1"]
        sum_data = None
        fns = ['OPT', 'OPT+AS', 'OPT+DE', 'OPT+AS+DE', ]
        fns2 = ["SUM"]
        for fn in fns:
            k = "{}_T".format(fn)
            fns2.append(k)
            df[k] = (df[fn] == df["y1"]) * 1
            if sum_data is None:
                sum_data = df[k].copy()
            else:
                sum_data += df[k]
        df["SUM"] = sum_data
        df = df.sort_values(['OPT+AS+DE_T', 'OPT_T', 'OPT+AS_T', 'OPT+DE_T', ], ascending=[True, False, False, False])
        df["CATEGORY"] = df["y1"] * 10 + 1
        df.to_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\3\sh2_spl273_2.csv", index=False)

    def func4():
        df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\3\sh2_spl273_21.csv")
        y1 = (df["CATEGORY_CODE"] / 10).values
        fns = ['OPT', 'OPT+AS', 'OPT+DE', 'OPT+AS+DE', 'OPT+GLCM', 'OPT+BS', 'OPT+C2', 'OPT+HA', 'OPT+SARGLCM']
        to_dict = {}
        for fn in fns:
            print("-" * 10, fn, "-" * 10)
            y2 = df[fn].values
            cm = ConfusionMatrix(class_names=["IS", "VEG", "SOIL", "WAT"])
            cm.addData(y1, y2)
            print(cm.fmtCM())
            cm2 = cm.accuracyCategory("IS")
            to_dict[fn] = {"IS_OA": cm2.OA(), "IS_Kappa": cm2.getKappa()}
            print(cm2.fmtCM())
        print(pd.DataFrame(to_dict).T.sort_values("IS_OA", ascending=False))

    func1()
    return


class SHH2ML_TST_SamplesTiao(SHH2ML_TST_Init):

    def __init__(self, city_name, is_save_model=False):
        super().__init__(city_name, is_save_model=is_save_model)
        self.csv_fn = ""

    def getCSVFn(self, *args, **kwargs):
        # raster_fn = self.getRasterFn()
        # spl_type = "iter"
        # city_name = self.city_name
        # to_fn = self.td.fn("splt_{}_data.csv".format(city_name))
        # get_csv_func(raster_fn, spl_type, city_name, to_fn)
        return self.csv_fn

    def main(self):
        self.dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods")
        self.td = TimeDirectory(self.dfn.fn())
        self.td.initLog()
        self.td.log(self.td.time_dfn.dirname)
        self.td.buildWriteText("city_name_{}.txt".format(self.city_name)).write(self.city_name)

        self.city_name = self.td.kw("CITY_NAME", self.city_name)
        self.csv_fn = self.td.kw("CSV_FN", self.getCSVFn())
        self.raster_fn = self.td.kw("RASTER_FN", self.getRasterFn())
        self.json_fn = self.td.kw("JSON_FN", self.dfn.fn(r"Temp\{}.json".format("feats5")))
        self.n_run = self.td.kw("N_RUN", 5)

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
        self.is_eq_number = self.td.kw("IS_EQ_NUMBER", False)
        self.fc_category = self.td.kw(
            # "FC_CATEGORY", [["IS", "IS_SH", ], ["VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH"]],
            "FC_CATEGORY", [["IS"], ["SOIL"]],
        )

        self.json_dict = FRW(self.json_fn).readJson()
        self.train_test_df = self.getDf()
        self.dfDes()


def sampling_tiao():
    def df_des(_front_str, _df):
        print(_front_str)
        df_des_data = _df.groupby("CNAME").count()["TEST"]
        print(df_des_data)
        print(df_des_data.sum())

    def get_csv_func(raster_fn, spl_type, city_name, to_fn):

        def _sampling(_df):
            if spl_type == "fast":
                gs = GDALSamplingFast(raster_fn)
            elif spl_type == "iter":
                gs = GDALSampling(raster_fn)
            elif spl_type == "npy":
                gs = GDALSampling()
                gs.initNPYRaster(raster_fn)
            else:
                raise Exception("Can not format sampling type of \"{}\"".format(spl_type))
            _df = gs.samplingDF(_df)
            return _df

        if city_name == "qd":
            dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\3")
            df = pd.read_csv(dfn.fn(r"sh2_spl252_1.csv"))
            sh2_sm = SHH2SamplesManage()

            df_data = sh2_sm.addDF(df[df["TEST"] == 1], field_datas={"SPLT": "training sh2_spl252_1"})
            df_des("", df_data)

            df_data = sh2_sm.addQJY(
                dfn.fn(r"sh2_spl253_training.txt"), _sampling,
                field_datas={"SPLT": "training add", "TEST": 1}
            )
            df_des("", df_data)

            sh2_sm_test = SHH2SamplesManage()

            sh2_sm_test.addDF(df[df["TEST"] == 0], field_datas={"SPLT": "testing sh2_spl252_1"})
            df_des("", df_data)

            to_df = sh2_sm_test.toDF()
            print(to_df)
            to_df.to_csv(to_fn, index=False)

        elif city_name == "bj":
            dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\3")
            csv_name = "sh2_spl271_4_spl2"
            df = pd.read_csv(dfn.fn(r"{}.csv".format(csv_name)))
            sh2_sm = SHH2SamplesManage()

            df_data = sh2_sm.addDF(df[df["TEST"] == 1], field_datas={"SPLT": "training {}".format(csv_name)})
            df_des("> training sh2_spl271_4_spl", df_data)

            df_data = sh2_sm.addQJY(
                dfn.fn(r"sh2_spl273_training.txt"), _sampling,
                field_datas={"SPLT": "training add", "TEST": 1}
            )
            df_des("> training add", df_data)

            df_data = sh2_sm.addDF(df[df["TEST"] == 0], field_datas={"SPLT": "testing {}".format(csv_name)})
            df_des("> testing sh2_spl271_4_spl", df_data)

            to_df = sh2_sm.toDF()
            print(to_df)
            to_df.to_csv(to_fn, index=False)

    # get_csv_func(
    #     raster_fn=SHH2Config.BJ_ENVI_FN,
    #     spl_type="iter",
    #     city_name="bj",
    #     to_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\3\test.csv"
    # )

    sh2_tst_st = SHH2ML_TST_SamplesTiao("bj")
    sh2_tst_st.csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\3\test.csv"
    # get_csv_func(
    #     raster_fn=SHH2Config.BJ_ENVI_FN, spl_type="iter", city_name=sh2_tst_st.city_name, to_fn=sh2_tst_st.csv_fn)
    sh2_tst_st.main()
    sh2_tst_st.train()

    return


class SHH2ML_TrainImdc(SHH2ML_TST_Init):

    def __init__(self, city_name, is_save_model=False):
        super().__init__(city_name, is_save_model=is_save_model)
        self.csv_fn = ""

    def getCSVFn(self, *args, **kwargs):
        return self.csv_fn

    def main(self):
        self.dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods")
        self.td = TimeDirectory(self.dfn.fn())
        self.td.initLog()
        self.td.log(self.td.time_dfn.dirname)
        self.td.buildWriteText("city_name_{}.txt".format(self.city_name)).write(self.city_name)
        records_dict = {}

        self.city_name = self.td.kw("CITY_NAME", self.city_name)
        self.csv_fn = self.td.kw("CSV_FN", self.getCSVFn())
        self.raster_fn = self.td.kw("RASTER_FN", self.getRasterFn())
        self.json_fn = self.td.kw("JSON_FN", self.dfn.fn(r"Temp\{}.json".format("feats5")))
        self.n_run = self.td.kw("N_RUN", 5)

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
        self.is_eq_number = self.td.kw("IS_EQ_NUMBER", False)
        self.fc_category = self.td.kw(
            "FC_CATEGORY", [["IS", "IS_SH", ], ["VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH"]],
            # "FC_CATEGORY", [["IS"], ["SOIL"]],
        )

        self.json_dict = FRW(self.json_fn).readJson()
        self.train_test_df = self.getDf()
        self.dfDes()

    def toToJson(self):
        self.td.saveToJson("td.json")


class Citys:

    def __init__(self):
        self.qd = None
        self.bj = None
        self.cd = None

        self._next_qd = False
        self._next_bj = False
        self._next_cd = False

    def __len__(self):
        return 3

    def __iter__(self):
        return self

    def __next__(self):
        if not self._next_qd:
            self._next_qd = True
            return self.qd
        elif not self._next_bj:
            self._next_bj = True
            return self.bj
        elif not self._next_cd:
            self._next_cd = True
            return self.cd
        else:
            self._next_qd = False
            self._next_bj = False
            self._next_cd = False
            raise StopIteration()

    def __contains__(self, item):
        return item in [self.qd, self.bj, self.cd]


class SHH2ML_TST_Three_Main:

    def __init__(self):
        self.csv_fn = ""
        self.td = None
        self.dfn = None
        self.sh2_tsts = Citys()
        self.dfns = Citys()
        self.oa_kappa_cms = Citys()
        self.oa_kappa_IO_cms = Citys()
        self.oa_kappa_IS_cms = Citys()
        self.oa_kappa_SH_cms = Citys()
        self.accuracy_list = []

    def sh2Tst(self, city_name):
        if city_name == "qd":
            self.sh2_tsts.qd = SHH2ML_TrainImdc("qd")
            self.sh2_tsts.qd.csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\2\sh2_spl252_4_spl2.csv"
            self.sh2_tsts.qd.main()
            self.sh2_tsts.qd.train()
            self.sh2_tsts.qd.toToJson()
            fn = self.sh2_tsts.qd.td.time_dfn.fn()
            return os.path.split(fn)[1]
        elif city_name == "bj":
            self.sh2_tsts.bj = SHH2ML_TrainImdc("bj")
            self.sh2_tsts.bj.csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\3\sh2_spl273_5_spl.csv"
            self.sh2_tsts.bj.main()
            self.sh2_tsts.bj.train()
            self.sh2_tsts.bj.toToJson()
            fn = self.sh2_tsts.bj.td.time_dfn.fn()
            return os.path.split(fn)[1]
        elif city_name == "cd":
            self.sh2_tsts.cd = SHH2ML_TrainImdc("cd")
            self.sh2_tsts.cd.csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\2\sh2_spl26_4_spl2.csv"
            self.sh2_tsts.cd.main()
            self.sh2_tsts.cd.train()
            self.sh2_tsts.cd.toToJson()
            fn = self.sh2_tsts.cd.td.time_dfn.fn()
            return os.path.split(fn)[1]

    def accuracyCM(self, accuracy_name, city_name, citys, map_dict1, map_dict2, cnames):
        self.td.kw("{} {}".format(accuracy_name, "MAP_DICT1"), map_dict1)
        self.td.kw("{} {}".format(accuracy_name, "MAP_DICT2"), map_dict2)
        self.td.kw("{} {}".format(accuracy_name, "CNAMES"), cnames)

        def _oa_kappa_cm(_json_fn):
            json_dict = FRW(_json_fn).readJson()
            to_dict = {}
            for k in json_dict:
                data = json_dict[k]
                y1 = mapDict(data["CNAME"], map_dict1)
                y2 = mapDict(data["y2"], map_dict2)
                cm = ConfusionMatrix(class_names=cnames)
                cm.addData(y1, y2)
                to_dict[k] = cm
            return to_dict

        def _oa_kappa_files(dirname):
            to_list = []
            n = 1
            while True:
                fn = os.path.join(dirname, "categorys{}.json".format(n))
                if os.path.isfile(fn):
                    to_list.append(_oa_kappa_cm(fn))
                else:
                    return to_list
                n += 1

        if city_name == "qd":
            citys.qd = _oa_kappa_files(self.dfns.qd.fn())

        elif city_name == "bj":
            citys.bj = _oa_kappa_files(self.dfns.bj.fn())

        elif city_name == "cd":
            citys.cd = _oa_kappa_files(self.dfns.cd.fn())

    def accuracyOAKappa(self, city_name):
        map_dict1 = {
            "IS": 1, "VEG": 2, "SOIL": 3, "WAT": 4,
            "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 3, "WAT_SH": 4
        }
        map_dict2 = {1: 1, 2: 2, 3: 3, 4: 4}
        cnames = ["IS", "VEG", "SOIL", "WAT"]
        citys = self.oa_kappa_cms
        accuracy_name = "ACCURACY OA KAPPA"
        self.accuracyCM(accuracy_name, city_name, citys, map_dict1, map_dict2, cnames)

    def accuracyOAKappaISO(self, city_name):
        map_dict1 = {
            "IS": 1, "VEG": 0, "SOIL": 2, "WAT": 0,
            "IS_SH": 0, "VEG_SH": 0, "SOIL_SH": 0, "WAT_SH": 0
        }
        map_dict2 = {1: 1, 2: 0, 3: 2, 4: 0}
        cnames = ["IS", "SOIL"]
        citys = self.oa_kappa_IO_cms
        accuracy_name = "ACCURACY OA KAPPA IS SOIL"
        self.accuracyCM(accuracy_name, city_name, citys, map_dict1, map_dict2, cnames)

    def accuracyOAKappaSH(self, city_name):
        map_dict1 = {
            "IS": 0, "VEG": 0, "SOIL": 0, "WAT": 0,
            "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 3, "WAT_SH": 4
        }
        map_dict2 = {1: 1, 2: 2, 3: 3, 4: 4}
        cnames = ["IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH"]
        accuracy_name = "ACCURACY OA KAPPA SHADOW"
        citys = self.oa_kappa_SH_cms
        self.accuracyCM(accuracy_name, city_name, citys, map_dict1, map_dict2, cnames)

    def accuracyOAKappaIS(self, city_name):
        map_dict1 = {
            "IS": 1, "VEG": 2, "SOIL": 2, "WAT": 2,
            "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 2, "WAT_SH": 2
        }
        map_dict2 = {1: 1, 2: 0, 3: 2, 4: 0}
        cnames = ["IS", "NOIS"]
        accuracy_name = "ACCURACY OA KAPPA SHADOW"
        citys = self.oa_kappa_IS_cms
        self.accuracyCM(accuracy_name, city_name, citys, map_dict1, map_dict2, cnames)

    def accuracy(self):

        def calculate_city_cm(name, citys_cms, ):

            def calculate(cm_list):
                df_list = []
                for cm_dict in cm_list:
                    df_dict = {}
                    for k in cm_dict:
                        cm = cm_dict[k]
                        df_dict[k] = {"{}OA".format(name): cm.OA(), "{}Kappa".format(name): cm.getKappa(), }
                    df_list.append(df_dict)
                df = pd.DataFrame(df_list[0])
                for df_dict in df_list[1:]:
                    df += pd.DataFrame(df_dict)
                df /= len(df_list)
                return df

            def add_df(df, city_name):
                _to_dict = {"City": [city_name, city_name], "Accuracy": list(df.index), **(df.to_dict("list"))}
                to_list.append({k: _to_dict[k][0] for k in _to_dict})
                to_list.append({k: _to_dict[k][1] for k in _to_dict})

            qd_df = calculate(citys_cms.qd)
            add_df(qd_df, "QingDao")

            bj_df = calculate(citys_cms.bj)
            add_df(bj_df, "BeiJing")

            cd_df = calculate(citys_cms.cd)
            add_df(cd_df, "ChengDu")

        to_list = []
        calculate_city_cm("", self.oa_kappa_cms)
        calculate_city_cm("IS_", self.oa_kappa_IS_cms)
        calculate_city_cm("IO_", self.oa_kappa_IO_cms)
        calculate_city_cm("SH_", self.oa_kappa_SH_cms)
        return pd.DataFrame(to_list)

    def saveCM(self):
        sw = self.td.buildWriteText("cm.md")

        def calculate_city_cm(name, citys_cms, ):
            sw.write("# {}\n".format(name))

            def calculate(cm_list, city_name):
                sw.write("## {}\n".format(city_name))
                df_list = []
                for i, cm_dict in enumerate(cm_list):
                    df_dict = {}
                    sw.write("### {}\n".format(i + 1))
                    for k in cm_dict:
                        sw.write("#### {}\n".format(k))
                        cm = cm_dict[k]
                        df_dict[k] = cm.fmtCM()
                        sw.write(df_dict[k])
                    df_list.append(df_dict)
                return {city_name: df_list}

            return {name: {
                **calculate(citys_cms.qd, "QingDao"),
                **calculate(citys_cms.qd, "BeiJing"),
                **calculate(citys_cms.qd, "ChengDu"),
            }}

        to_dict = {
            **calculate_city_cm("oa_kappa_cms", self.oa_kappa_cms),
            **calculate_city_cm("oa_kappa_IS_cms", self.oa_kappa_IS_cms),
            **calculate_city_cm("oa_kappa_IO_cms", self.oa_kappa_IO_cms),
            **calculate_city_cm("oa_kappa_SH_cms", self.oa_kappa_SH_cms),
        }

        self.td.saveJson("cm.json", to_dict)

    def main(self):
        self.dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods")
        self.td = TimeDirectory(self.dfn.fn())
        self.td.initLog()
        self.td.log("#", "-" * 34, "SHH2ML Three Accuracy", "-" * 34, "#")
        self.td.log(self.td.time_dfn.dirname)
        time.sleep(2)

        # QingDao
        self.td.log("#", "-" * 30, "SHH2ML Three Accuracy QingDao", "-" * 30, "#")
        # dirname = self.td.kw("QingDao Dirname", self.dfn.fn(self.sh2Tst("qd")))
        dirname = self.td.kw("QingDao Dirname", self.dfn.fn("20240620H100549"))
        self.dfns.qd = DirFileName(dirname)
        self.accuracyOAKappa("qd")
        self.accuracyOAKappaIS("qd")
        self.accuracyOAKappaSH("qd")
        self.accuracyOAKappaISO("qd")

        # BeiJing
        self.td.log("#", "-" * 30, "SHH2ML Three Accuracy BeiJing", "-" * 30, "#")
        # dirname = self.td.kw("BeiJing Dirname", self.dfn.fn(self.sh2Tst("bj")))
        dirname = self.td.kw("BeiJing Dirname", self.dfn.fn("20240620H100627"))
        self.dfns.bj = DirFileName(dirname)
        self.accuracyOAKappa("bj")
        self.accuracyOAKappaIS("bj")
        self.accuracyOAKappaSH("bj")
        self.accuracyOAKappaISO("bj")

        # ChengDu
        self.td.log("#", "-" * 30, "SHH2ML Three Accuracy ChengDu", "-" * 30, "#")
        # dirname = self.td.kw("ChengDu Dirname", self.dfn.fn(self.sh2Tst("cd")))
        dirname = self.td.kw("ChengDu Dirname", self.dfn.fn("20240620H100711"))
        self.dfns.cd = DirFileName(dirname)
        self.accuracyOAKappa("cd")
        self.accuracyOAKappaIS("cd")
        self.accuracyOAKappaSH("cd")
        self.accuracyOAKappaISO("cd")

        df = self.accuracy()
        print(self.td.saveDF("accuracy.csv", df))
        print(df)
        self.saveCM()
        return


def shh2ml_tst_three_main():
    def func1():
        sh2_tst_three = SHH2ML_TST_Three_Main()
        sh2_tst_three.main()

    def func2():
        df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240625H095752\accuracy.csv")

        class df_filter:

            def __init__(self, _df):
                self.df = _df

            def eq(self, field_name, data):
                if not isinstance(data, list):
                    data = [data]
                self.df = self.df[self.df[field_name].isin(data)]
                return self

            def datain(self, field_name, data):
                self.df = self.df[self.df[field_name].apply(lambda x: data in x)]
                return self

        city_name = "ChengDu"
        df = df_filter(df) \
            .datain("Accuracy", "OA") \
            .df.reset_index(drop=True)
        #             .eq("City", city_name) \

        print(df, "\n")
        fns = ['OPT', 'OPT+AS', 'OPT+DE', 'OPT+AS+DE', 'OPT+GLCM', 'OPT+BS', 'OPT+C2', 'OPT+HA', 'OPT+SARGLCM']
        df2 = df[fns]
        df2 = df2.reset_index(drop=True)
        df2.index = ["{} {}".format(df["City"][i], df["Accuracy"][i], ) for i in range(len(df))]
        print(df2, "\n")
        print(df2.T.sort_values("{} IS_OA".format(city_name), ascending=False), "\n")

    func2()


if __name__ == "__main__":
    r"""
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2ML2 import imdc; imdc('qd')"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2ML2 import imdc; imdc('bj')"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2ML2 import imdc; imdc('cd')"

python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2ML2 import train; train('qd')"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2ML2 import train; train('bj')"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2ML2 import train; train('cd')"

    
    """
    train()
