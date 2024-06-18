# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2ML.py
@Time    : 2024/6/5 16:53
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2ML
-----------------------------------------------------------------------------"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from osgeo import gdal
from osgeo_utils.gdal_merge import main as gdal_merge_main
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from SRTCodes.GDALRasterIO import GDALRaster, tiffAddColorTable
from SRTCodes.GDALUtils import GDALSampling, RasterToVRTS, vrtAddDescriptions
from SRTCodes.ModelTraining import TrainLog, ConfusionMatrix
from SRTCodes.NumpyUtils import reHist
from SRTCodes.SRTFeature import SRTFeaturesMemory
from SRTCodes.SRTModelImage import SRTModImSklearn, GDALImdc
from SRTCodes.SRTTimeDirectory import TimeDirectory
from SRTCodes.Utils import SRTLog, timeDirName, changext, changefiledirname, FRW, Jdt, DirFileName, printList
from Shadow.Hierarchical import SHH2Config
from Shadow.ShadowData import ShadowData
from Shadow.ShadowImageDraw import _10log10


class SHH2MLTrainImageClassification:

    def __init__(self, csv_fn=None, raster_fn=None, raster_tile_fns=None):
        self.fit_keys = None
        self.name = "SHH2ML"
        self.smis = SRTModImSklearn()
        self.slog = SRTLog()
        self.model_dirname = r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods"
        self.model_dirname = timeDirName(self.model_dirname, is_mk=True)

        self.slog.__init__(os.path.join(self.model_dirname, "{0}_log.txt".format(self.name)), mode="a", )
        self.slog.kw("NAME", self.name)
        self.slog.kw("MODEL_DIRNAME", self.model_dirname)

        self.cnames = ["NOT_KNOW", "IS", "VEG", "SOIL", "WAT"]

        self.saveCodeFile()

        self.df = None
        self.datas = None
        self.datas_tile_grs = {}
        self.gr = None
        self.initDF(csv_fn=csv_fn)
        self.initRaster(raster_fn=raster_fn)
        self.raster_tile_fns = raster_tile_fns
        self.initRasterTiles(raster_tile_fns=raster_tile_fns)

        self.train_log = TrainLog(
            save_csv_file=os.path.join(self.model_dirname, "train_save{}.csv".format(self.name)),
            log_filename=os.path.join(self.model_dirname, "train_log{}.csv".format(self.name)),
        )
        self.addTrainLogFields()
        self.train_log.saveHeader()
        self.n = 1

        self.color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), }

    def initRaster(self, raster_fn=None):
        if raster_fn is None:
            return
        self.gr = GDALRaster(raster_fn)
        self.datas = {k: None for k in self.gr.names}

    def initRasterTiles(self, raster_tile_fns=None):
        if raster_tile_fns is None:
            return
        for fn in raster_tile_fns:
            gr = GDALRaster(fn)
            self.datas_tile_grs[fn] = gr

    def initDF(self, csv_fn=None):
        if csv_fn is not None:
            self.df = pd.read_csv(csv_fn)
            to_csv_fn = self.slog.kw(
                "to_csv_fn", os.path.join(self.model_dirname, "{}_train_data.csv".format(self.name)))
            print(self.df)
            self.df.to_csv(to_csv_fn, index=False)

    def addTrainLogFields(self):
        self.train_log.addField("NUMBER", "int")
        self.train_log.addField("NAME", "string")
        self.train_log.addField("OA_TEST", "float")
        self.train_log.addField("KAPPA_TEST", "float")
        self.train_log.addField("UA_TEST", "float")
        self.train_log.addField("PA_TEST", "float")
        self.train_log.addField("OA_TRAIN", "float")
        self.train_log.addField("KAPPA_TRAIN", "float")
        self.train_log.addField("UA_TRAIN", "float")
        self.train_log.addField("PA_TRAIN", "float")

    def train(self, name=None, fit_keys=None):
        fit_keys, name = self._initFitKeysName(fit_keys, name)

        self.train_log.updateField("NAME", name)
        self.slog.wl("-" * 80)
        self.slog.wl("** Training --".format(name))
        self.name = name

        self.smis = SRTModImSklearn()
        self.smis.initColorTable(self.color_table)
        self.smis.category_names = self.cnames
        self.slog.kw("CATEGORY_NAMES", self.smis.category_names)
        self.slog.kw("COLOR_TABLE", self.smis.color_table)
        self.smis.initPandas(self.df)
        self.slog.kw("shh_mis.df.keys()", list(self.smis.df.keys()))
        self.slog.kw("Category Field Name:", self.smis.initCategoryField())
        self.fit_keys = fit_keys
        self.slog.kw("fit_keys", fit_keys)
        self.smis.initXKeys(fit_keys)
        self.smis.testFieldSplitTrainTest()
        self.slog.kw("LEN X", len(self.smis.x))
        self.slog.kw("LEN Train", len(self.smis.x_train))
        self.slog.kw("LEN Test", len(self.smis.y_test))
        self.smis.initCLF(RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=2, min_samples_split=4))
        self.smis.train()
        self.smis.scoreTrainCM()
        self.slog.kw("Train CM", self.smis.train_cm.fmtCM(), sep="\n")
        self.train_log.updateField("OA_TRAIN", self.smis.train_cm.OA())
        self.train_log.updateField("KAPPA_TRAIN", self.smis.train_cm.getKappa())
        self.train_log.updateField("UA_TRAIN", self.smis.train_cm.UA(2))
        self.train_log.updateField("PA_TRAIN", self.smis.train_cm.PA(2))
        self.smis.scoreTestCM()
        self.slog.kw("Test CM", self.smis.test_cm.fmtCM(), sep="\n")
        self.train_log.updateField("OA_TEST", self.smis.test_cm.OA())
        self.train_log.updateField("KAPPA_TEST", self.smis.test_cm.getKappa())
        self.train_log.updateField("UA_TEST", self.smis.test_cm.UA(2))
        self.train_log.updateField("PA_TEST", self.smis.test_cm.PA(2))
        mod_fn = self.slog.kw("Model FileName", os.path.join(self.model_dirname, "{0}.model".format(name)))
        self.smis.saveModel(mod_fn)
        self.train_log.updateField("NUMBER", self.n)
        self.train_log.saveLine()
        self.train_log.newLine()
        self.n += 1
        return self.smis.clf

    def imdc(self, fit_keys=None, name=None):
        self.slog.wl("** Image Classification --", name)
        fit_keys, name = self._initFitKeysName(fit_keys, name)
        to_imdc_fn = self.slog.kw("to_imdc_fn", os.path.join(self.model_dirname, "{}_imdc.tif".format(name)))
        data = np.zeros((len(fit_keys), self.gr.n_rows, self.gr.n_columns))
        for i, k in enumerate(fit_keys):
            data[i] = self.gr.readGDALBand(k)
        self.gr.d = data
        self.smis.imdcGR(to_imdc_fn, self.gr)

    def imdcTiles(self, fit_keys=None, name=None):
        self.slog.wl("** Image Classification --", name)
        fit_keys, name = self._initFitKeysName(fit_keys, name)
        to_imdc_fn = self.slog.kw("to_imdc_fn", os.path.join(self.model_dirname, "{}_imdc.tif".format(name)))
        to_imdc_dirname = changext(to_imdc_fn, "_tiles")
        if not os.path.isdir(to_imdc_dirname):
            os.mkdir(to_imdc_dirname)
        to_fn_tmps = []

        for fn in self.datas_tile_grs:
            self.slog.wl("{}".format(fn))
            gr = self.datas_tile_grs[fn]
            data = np.zeros((len(fit_keys), gr.n_rows, gr.n_columns))
            for i, k in enumerate(fit_keys):
                data[i] = gr.readGDALBand(k)
            gr.d = data
            to_imdc_fn_tmp = os.path.join(to_imdc_dirname, changext(os.path.split(fn)[1], "_imdc.tif"))
            to_fn_tmps.append(to_imdc_fn_tmp)
            self.slog.wl("TO_IMDC_FN_TMP: {}".format(to_imdc_fn_tmp))
            self.smis.imdcGR(to_imdc_fn_tmp, gr)
            del gr.d
            del data
            gr.d = None

        gdal_merge_main(["gdal_merge_main",
                         "-of", "GTiff",
                         "-n", "0",
                         "-ot", "Byte",
                         "-co", "COMPRESS=PACKBITS",
                         "-o", to_imdc_fn,
                         *to_fn_tmps, ])
        tiffAddColorTable(to_imdc_fn, code_colors=self.smis.color_table)

    def _initFitKeysName(self, fit_keys, name):
        if fit_keys is None:
            fit_keys = self.fit_keys
        if name is None:
            name = self.name
        return fit_keys, name

    def saveCodeFile(self):
        to_code_fn = self.slog.kw("to_code_fn", changefiledirname(__file__, self.model_dirname))
        self.smis.saveCodeFile(code_fn=__file__, to_code_fn=to_code_fn)


def mapDict(data, map_dict):
    if map_dict is None:
        return data
    to_list = []
    for d in data:
        if d in map_dict:
            to_list.append(map_dict[d])
    return to_list


def SHH2ML_TIC_mian(city_name="qd", json_fn="feats3", is_single=False):
    def qd():
        return SHH2MLTrainImageClassification(
            csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\sh2_spl25_3_spl.csv",
            raster_fn=r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\SHH2_QD2_envi.dat",
            raster_tile_fns=[
                r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\Tiles\SHH2_QD2_envi_1_1.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\Tiles\SHH2_QD2_envi_1_2.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\Tiles\SHH2_QD2_envi_1_3.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\Tiles\SHH2_QD2_envi_1_4.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\Tiles\SHH2_QD2_envi_2_1.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\Tiles\SHH2_QD2_envi_2_2.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\Tiles\SHH2_QD2_envi_2_3.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\Tiles\SHH2_QD2_envi_2_4.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\Tiles\SHH2_QD2_envi_3_1.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\Tiles\SHH2_QD2_envi_3_2.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\Tiles\SHH2_QD2_envi_3_3.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\Tiles\SHH2_QD2_envi_3_4.tif",
            ])

    def cd():
        return SHH2MLTrainImageClassification(
            csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\shh2_spl26_3_spl.csv",
            raster_fn=r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\SHH2_CD2_envi.dat",
            raster_tile_fns=[
                r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_1_1.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_1_2.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_1_3.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_2_1.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_2_2.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_2_3.tif",
            ])

    def bj():
        return SHH2MLTrainImageClassification(
            csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\sh2_spl27_1_spl.csv",
            raster_fn=r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\SHH2_BJ2_envi.dat",
            raster_tile_fns=[
                r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\Tiles\SHH2_BJ2_envi_1_1.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\Tiles\SHH2_BJ2_envi_1_2.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\Tiles\SHH2_BJ2_envi_1_3.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\Tiles\SHH2_BJ2_envi_1_4.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\Tiles\SHH2_BJ2_envi_2_1.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\Tiles\SHH2_BJ2_envi_2_2.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\Tiles\SHH2_BJ2_envi_2_3.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\Tiles\SHH2_BJ2_envi_2_4.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\Tiles\SHH2_BJ2_envi_3_1.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\Tiles\SHH2_BJ2_envi_3_2.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\Tiles\SHH2_BJ2_envi_3_3.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\Tiles\SHH2_BJ2_envi_3_4.tif",
            ])

    if city_name == "qd":
        tic = qd()
    elif city_name == "bj":
        tic = bj()
    elif city_name == "cd":
        tic = cd()
    else:
        print("city name no eq {}.".format(city_name))
        return

    json_dict = FRW(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\Temp\{}.json".format(json_fn)).readJson()
    if is_single:
        k = sys.argv[1]
        print(k, json_dict[k])
        tic.train(k, json_dict[k])
        tic.imdcTiles()
    else:
        for k in json_dict:
            printList(k, json_dict[k])
            tic.train(k, json_dict[k])
            # tic.imdcTiles()


class SHH2MLTrainSamplesTiao:

    def __init__(self):
        self.clf = None
        self.df = None
        self.categorys = {}
        self.acc_dict = {}

    def train(self, name, x_keys=None, c_fn="CATEGORY", map_dict=None, clf=None):
        if x_keys is None:
            x_keys = []

        def train_test(n):
            _df = self.df[self.df["TEST"] == n]
            x = _df[x_keys].values
            y = mapDict(_df[c_fn].tolist(), map_dict)
            return x, y

        x_train, y_train = train_test(1)
        x_test, y_test = train_test(0)

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

    def accuracyOAKappa(self, cm_name, cnames, y1_map_dict=None, y2_map_dict=None, ):
        cm_str = ""
        for name, line in self.categorys.items():
            y1 = mapDict(line["y1"], y1_map_dict)
            y2 = mapDict(line["y2"], y2_map_dict)
            cm = ConfusionMatrix(class_names=cnames)
            cm.addData(y1, y2)
            self.acc_dict[name]["{}_OA".format(cm_name)] = cm.OA()
            self.acc_dict[name]["{}_Kappa".format(cm_name)] = cm.getKappa()
            cm_str += "> {} IS\n".format(name)
            cm_str += "{}\n\n".format(cm.fmtCM())

            cm2 = ConfusionMatrix(class_names=["IS", "VEG", "SOIL", "WAT"])
            cm2.addData(line["y1"], line["y2"])
            cm_str += "> {} {}\n".format(name, " ".join(["IS", "VEG", "SOIL", "WAT"]))
            cm_str += "{}\n\n".format(cm2.fmtCM())

            # cm3 = cm2.accuracyCategory("IS")
            # self.acc_dict[name]["{}_OA2".format(cm_name)] = cm3.OA()
            # self.acc_dict[name]["{}_Kappa2".format(cm_name)] = cm3.getKappa()
        return cm_str


def SHH2ML_TST_main():
    def map_category(data_list):
        map_dict = {
            "IS": 1, "VEG": 2, "SOIL": 3, "WAT": 4,
            "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 3, "WAT_SH": 4
        }
        to_data_list = []
        for d in data_list:
            to_data_list.append(map_dict[d])
        return to_data_list

    def get_sfm():
        if city_name == "cd":
            range_json_fn = td.kw(
                "RANGE_JSON_FN", r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\SHH2_CD2_range.json")
            range_json_dict = FRW(range_json_fn).readJson()
        else:
            range_json_dict = None
        if range_json_dict is not None:
            _sfm = SRTFeaturesMemory(names=list(range_json_dict.keys()))
            _sfm.initCallBacks()
            for name in range_json_dict:
                _sfm.callbacks(name).addScaleMinMax(
                    range_json_dict[name]["min"], range_json_dict[name]["max"], is_trans=True, is_to_01=True)
        else:
            _sfm = SRTFeaturesMemory()
        return _sfm

    def get_df():
        _df = pd.read_csv(csv_fn)
        _df["CATEGORY"] = map_category(_df["CNAME"].tolist())
        for k in sfm.names:
            if k in _df:
                _df[k] = sfm.callbacks(k).fit(_df[k])
        return _df

    def df_des():
        td.log("Training samples categorys numbers:")
        df_des_data = train_test_df[train_test_df["TEST"] == 1].groupby("CNAME").count()["SRT"]
        td.log(df_des_data)
        td.log(df_des_data.sum())
        td.log("Test samples categorys numbers:")
        df_des_data = train_test_df[train_test_df["TEST"] == 0].groupby("CNAME").count()["SRT"]
        td.log(df_des_data)
        td.log(df_des_data.sum())

    acc_dict_list = []
    mod_list = []

    def acc_run(n_train):
        td.log("\n> ", n_train)
        sh2_tst = SHH2MLTrainSamplesTiao()
        sh2_tst.df = train_test_df

        for n, k in enumerate(json_dict):
            td.log("[{}] {}".format(n, k), end=" ")
            train_acc, test_acc = sh2_tst.train(k, json_dict[k])
            td.log("Accuracy: {:>3.2f}% {:>3.2f}%".format(train_acc, test_acc), end=" ")
            td.log("Model: {}".format(str(sh2_tst.clf)))

        cm_str = sh2_tst.accuracyOAKappa("IS", ["IS", "NOT_KNOW"], y1_map_dict=y1_map_dict, y2_map_dict=y2_map_dict)
        td.buildWriteText("cm{}.txt".format(n_train)).write(cm_str)

        _df_acc = pd.DataFrame(sh2_tst.acc_dict).T
        td.saveDF("acc{}.csv".format(n_train), _df_acc, index=True)
        td.log(_df_acc.sort_values("IS_OA", ascending=False))

        acc_dict_list.append(sh2_tst.acc_dict)

    def train():

        for i in range(5):
            acc_run(i + 1)
        td.log()

        df_acc = None
        for df in acc_dict_list:
            if df_acc is None:
                df_acc = pd.DataFrame(df)
            else:
                df_acc += pd.DataFrame(df)
        df_acc = df_acc.T
        df_acc = df_acc / len(acc_dict_list)
        td.log(df_acc.sort_values("IS_OA", ascending=False))

    def imdc():
        gimdc = GDALImdc(SHH2Config.BJ_ENVI_FN)
        gimdc.sfm = sfm
        sh2_tst = SHH2MLTrainSamplesTiao()
        sh2_tst.df = train_test_df

        for n, k in enumerate(json_dict):
            td.log("[{}] {}".format(n, k), end=" ")
            train_acc, test_acc = sh2_tst.train(k, json_dict[k])
            td.log("Accuracy: {:>3.2f}% {:>3.2f}%".format(train_acc, test_acc))
            filename = td.time_dfn.fn("{}_mod.mod".format(k))
            joblib.dump(sh2_tst.clf, filename)
            to_fn = td.time_dfn.fn("{}_imdc.tif".format(k))
            td.kw("TO_IMDC_FN", to_fn)
            gimdc.imdc1(sh2_tst.clf, to_fn, json_dict[k], color_table=color_table)

    td = TimeDirectory(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods")
    td.initLog()
    td.log(td.time_dfn.dirname)
    td.log("#", "-" * 20, "SHH2ML Five Mean", "-" * 20, )
    td.copyfile(__file__)

    city_name = td.kw("CITY_NAME", "bj")
    td.buildWriteText("{}.txt".format(city_name)).write(city_name)
    # "F:\ProjectSet\Shadow\Hierarchical\Samples\27\2\sh2_spl271_32.csv"
    # "F:\ProjectSet\Shadow\Hierarchical\Samples\27\2\sh2_spl271_4_spl.csv"
    # "F:\ProjectSet\Shadow\Hierarchical\Samples\25\2\sh2_spl252_1.csv"
    # "F:\ProjectSet\Shadow\Hierarchical\Samples\26\2\sh2_spl26_4_spl.csv"
    csv_fn = td.kw("CSV_FN", r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\2\sh2_spl271_4_spl.csv")
    color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), }
    y1_map_dict = td.kw("Y1_MAP_DICT", {1: 1, 2: 2, 3: 2, 4: 2})
    y2_map_dict = td.kw("Y2_MAP_DICT", {1: 1, 2: 2, 3: 2, 4: 2})
    json_fn = td.kw("JSON_FN", r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\Temp\{}.json".format("feats4"))
    json_dict = FRW(json_fn).readJson()
    sfm = get_sfm()
    train_test_df = get_df()
    df_des()
    td.log("# Test category is IS")

    imdc()


def featExtHA():
    # init_dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\2\Temp")
    init_dfn = DirFileName(r"G:\S1\BJ\2\SH2_BJ1")

    # raster_dfn = DirFileName(r"F:\ProjectSet\Shadow\Release\BeiJingImages")
    # raster_fn = raster_dfn.fn("SH_BJ_envi.dat")

    def func3(city_name, raster_fn):

        gr = GDALRaster(raster_fn)

        def func1(name, dfn):
            print(dfn.fn("{}_H.dat".format(name)))

            c11_key = "{}_C11".format(name)
            c22_key = "{}_C22".format(name)
            c12_real_key = "{}_C12_real".format(name)
            c12_imag_key = "{}_C12_imag".format(name)

            d_c11 = gr.readGDALBand(c11_key)
            d_c22 = gr.readGDALBand(c22_key)
            d_c12_real = gr.readGDALBand(c12_real_key)
            d_c12_imag = gr.readGDALBand(c12_imag_key)

            lamd1, lamd2 = np.zeros((gr.n_rows, gr.n_columns)), np.zeros((gr.n_rows, gr.n_columns))
            alp1, alp2 = np.zeros((gr.n_rows, gr.n_columns)), np.zeros((gr.n_rows, gr.n_columns))

            jdt = Jdt(gr.n_rows, "{0} {1} featExtHA".format(city_name, name)).start()
            for i in range(gr.n_rows):
                for j in range(gr.n_columns):
                    c2 = np.array([
                        [d_c11[i, j], d_c12_real[i, j] + (d_c12_imag[i, j] * 1j)],
                        [d_c12_real[i, j] - (d_c12_imag[i, j] * 1j), d_c22[i, j]],
                    ])
                    eigenvalue, featurevector = np.linalg.eig(c2)
                    lamd1[i, j] = np.abs(eigenvalue[0])
                    lamd2[i, j] = np.abs(eigenvalue[1])
                    alp1[i, j] = np.arccos(abs(featurevector[0, 0]))
                    alp2[i, j] = np.arccos(abs(featurevector[0, 1]))
                jdt.add()
            jdt.end()

            # dfn = DirFileName(r"F:\ProjectSet\Shadow\Analysis\14")
            # gr.save(lamd1, dfn.fn("lamd1.dat"))
            # gr.save(lamd2, dfn.fn("lamd2.dat"))
            # gr.save(alp1, dfn.fn("alp1.dat"))
            # gr.save(alp2, dfn.fn("alp2.dat"))

            p1 = lamd1 / (lamd1 + lamd2)
            p2 = lamd2 / (lamd1 + lamd2)
            d_h = p1 * (np.log(p1) / np.log(3)) + p2 * (np.log(p2) / np.log(3))
            a = p1 - p2
            alp = p1 * alp1 + p2 * alp2

            gr.save(d_h, dfn.fn("{}_H.dat".format(name)), descriptions=["{}_H".format(name)])
            gr.save(a, dfn.fn("{}_A.dat".format(name)), descriptions=["{}_A".format(name)])
            gr.save(alp, dfn.fn("{}_Alpha.dat".format(name)), descriptions=["{}_Alpha".format(name)])

        func1("AS", DirFileName(init_dfn.fn(city_name)))
        func1("DE", DirFileName(init_dfn.fn(city_name)))

    # func3("BJ", r"F:\ProjectSet\Shadow\Release\BeiJingImages\SH_BJ_envi.dat")
    # func3("CD", r"F:\ProjectSet\Shadow\Release\ChengDuImages\SH_CD_envi.dat")
    # func3("QD", r"F:\ProjectSet\Shadow\Release\QingDaoImages\SH_QD_envi.dat")
    func3("BJ", r"G:\S1\BJ\2\bj_sh2_1_c2_2_envi.dat")

    def func4(city_name, raster_fn):
        print(city_name, raster_fn)
        gr = GDALRaster(raster_fn)
        names = gr.names.copy()
        data_list = [gr.readAsArray()]

        def func1(name, dfn):
            names.extend(["{}_H".format(name), "{}_A".format(name), "{}_Alpha".format(name)])
            data_list.extend([
                [GDALRaster(dfn.fn("{}_H.dat".format(name))).readAsArray()],
                [GDALRaster(dfn.fn("{}_A.dat".format(name))).readAsArray()],
                [GDALRaster(dfn.fn("{}_Alpha.dat".format(name))).readAsArray()]
            ])

        func1("AS", DirFileName(init_dfn.fn(city_name)))
        func1("DE", DirFileName(init_dfn.fn(city_name)))
        data = np.concatenate(data_list)
        to_fn = DirFileName(init_dfn.fn(city_name)).fn(os.path.split(raster_fn)[1])
        print(to_fn)
        gr.save(data, to_fn, descriptions=names)

    # func4("BJ", r"F:\ProjectSet\Shadow\Release\BeiJingImages\SH_BJ_envi.dat")
    # func4("CD", r"F:\ProjectSet\Shadow\Release\ChengDuImages\SH_CD_envi.dat")
    # func4("QD", r"F:\ProjectSet\Shadow\Release\QingDaoImages\SH_QD_envi.dat")

    def func5(city_name, filename):
        gr = GDALRaster(init_dfn.fn(city_name, filename))
        print(city_name)
        names = ["AS_H", "AS_A", "AS_Alpha", "DE_H", "DE_A", "DE_Alpha", ]
        for name in names:
            data = gr.readGDALBand(name)
            print("obj_feat.featureScaleMinMax(\"{0}\", {1}, {2})".format(name, data.min(), data.max()))

            # y, x = np.histogram(data, bins=256)
            # plt.plot(x[:-1], y)
            # plt.title(name)
            # plt.show()

    # func5("BJ", "SH_BJ_envi.dat")
    # func5("QD", "SH_QD_envi.dat")
    # func5("CD", "SH_CD_envi.dat")

    def func2():
        gr = GDALRaster()
        c11_key = "AS_C11"
        c22_key = "AS_C22"
        c12_real_key = "AS_C12_real"
        c12_imag_key = "AS_C12_imag"

        d_c11 = gr.readGDALBand(c11_key)
        d_c22 = gr.readGDALBand(c22_key)
        d_c12_real = gr.readGDALBand(c12_real_key)
        d_c12_imag = gr.readGDALBand(c12_imag_key)
        lamd1_tmp = gr.readGDALBand("AS_Lambda1")
        lamd2_tmp = gr.readGDALBand("AS_Lambda2")
        dfn = DirFileName(r"F:\ProjectSet\Shadow\Analysis\14")
        lamd1 = GDALRaster(dfn.fn("lamd1.dat")).readGDALBand(1)
        lamd2 = GDALRaster(dfn.fn("lamd2.dat")).readGDALBand(1)
        alp1 = GDALRaster(dfn.fn("alp1.dat")).readGDALBand(1)
        alp2 = GDALRaster(dfn.fn("alp2.dat")).readGDALBand(1)

        p1 = lamd1 / (lamd1 + lamd2)
        p2 = lamd2 / (lamd1 + lamd2)
        d_h = p1 * (np.log(p1) / np.log(3)) + p2 * (np.log(p2) / np.log(3))
        alp = p1 * alp1 + p2 * alp2

        gr.save(d_h, dfn.fn("h.dat"))
        gr.save(alp, dfn.fn("alp.dat"))


def featExt():
    sd = ShadowData()
    raster_fn = r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\cat\filelist2_vrt.vrt"  # 青岛
    raster_fn = r"H:\ChengDu\Temp\cd_sh2_1_c2_envi.dat"  # 成都
    raster_fn = r"G:\S1\BJ\2\bj_sh2_1_c2_2_envi.dat"  # 北京
    # optics
    sd.addGDALData(raster_fn, "Blue")
    sd.addGDALData(raster_fn, "Green")
    sd.addGDALData(raster_fn, "Red")
    sd.addGDALData(raster_fn, "NIR")
    sd.extractNDVI("Red", "NIR", "NDVI")
    sd.extractNDWI("Green", "NIR", "NDWI")
    print("optics")
    # AS SAR
    sar_t = "AS"
    vv_n = sd.addGDALData(raster_fn, sar_t + "_VV")
    sd.update10EDivide10(vv_n)
    vh_n = sd.addGDALData(raster_fn, sar_t + "_VH")
    sd.update10EDivide10(vh_n)
    sd.extractSigmaRatio(sar_t + "_VV", sar_t + "_VH", sar_t + "_VHDVV")
    c11_n = sd.addGDALData(raster_fn, sar_t + "_C11")
    c12_imag_n = sd.addGDALData(raster_fn, sar_t + "_C12_imag")
    c12_real_n = sd.addGDALData(raster_fn, sar_t + "_C12_real")
    c22_n = sd.addGDALData(raster_fn, sar_t + "_C22")
    lamd1_n, lamd2_n = sd.extractC2EIG(c11_n, c22_n, c12_real_n, c12_imag_n, sar_t + "_Lambda1", sar_t + "_Lambda2")
    sd.extractSPAN(c11_n, c22_n, sar_t + "_SPAN")
    sd.extractEpsilon(c11_n, c22_n, sar_t + "_Epsilon")
    sd.extractMu(c12_real_n, c11_n, c22_n, sar_t + "_Mu")
    sd.extractRVI(c11_n, c22_n, sar_t + "_RVI")
    sd.extractDegreePolarization(c11_n, c22_n, c12_real_n, c12_imag_n, sar_t + "_m")
    sd.extractBeta(lamd1_n, lamd2_n, sar_t + "_Beta")
    print("AS SAR")
    # DE SAR
    sar_t = "DE"
    vv_n = sd.addGDALData(raster_fn, sar_t + "_VV")
    sd.update10EDivide10(vv_n)
    vh_n = sd.addGDALData(raster_fn, sar_t + "_VH")
    sd.update10EDivide10(vh_n)
    sd.extractSigmaRatio(sar_t + "_VV", sar_t + "_VH", sar_t + "_VHDVV")
    c11_n = sd.addGDALData(raster_fn, sar_t + "_C11")
    c12_imag_n = sd.addGDALData(raster_fn, sar_t + "_C12_imag")
    c12_real_n = sd.addGDALData(raster_fn, sar_t + "_C12_real")
    c22_n = sd.addGDALData(raster_fn, sar_t + "_C22")
    lamd1_n, lamd2_n = sd.extractC2EIG(c11_n, c22_n, c12_real_n, c12_imag_n, sar_t + "_Lambda1", sar_t + "_Lambda2")
    sd.extractSPAN(c11_n, c22_n, sar_t + "_SPAN")
    sd.extractEpsilon(c11_n, c22_n, sar_t + "_Epsilon")
    sd.extractMu(c12_real_n, c11_n, c22_n, sar_t + "_Mu")
    sd.extractRVI(c11_n, c22_n, sar_t + "_RVI")
    sd.extractDegreePolarization(c11_n, c22_n, c12_real_n, c12_imag_n, sar_t + "_m")
    sd.extractBeta(lamd1_n, lamd2_n, sar_t + "_Beta")
    print("DE SAR")
    sd.print()
    sd.saveToSingleImageFile(r"G:\S1\BJ\2\SH2_BJ1", "", raster_fn, vrt_fn="SH2_BJ1.vrt")


def funcs():
    def func1():
        grf = GDALSampling(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\2\SHH2_QD2_envi.dat")
        print(grf)
        grf.csvfile(
            csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\sh2_spl25_2.csv",
            to_csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\sh2_spl25_2_spl2.csv",
        )

    def func2():
        gr = GDALRaster(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\2\SHH2_QD2.vrt")
        FRW(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\Temp\tmp2.json").saveJson(gr.names)

    def func3():
        rtvrts = RasterToVRTS(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\2\SHH2_QD1_envi.dat")
        rtvrts.save(to_dirname=r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\2\Temp")
        "H:\ChengDu\Temp\tmp4.tif"

    def func4():
        json_dict = FRW(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\Temp\feats.json").readJson()
        to_dict = {
            "OPT": json_dict["OPT"],
            "OPT+AS": json_dict["OPT"] + json_dict["OPTGLCM"] + json_dict["AS"],
            "OPT+DE": json_dict["OPT"] + json_dict["OPTGLCM"] + json_dict["DE"],
            "OPT+AS+DE": json_dict["OPT"] + json_dict["OPTGLCM"] + json_dict["AS"] + json_dict["DE"],
            "OPT+GLCM": json_dict["OPT"] + json_dict["OPTGLCM"],
            "OPT+BS": json_dict["OPT"] + json_dict["BS"],
            "OPT+C2": json_dict["OPT"] + json_dict["C2"],
            "OPT+HA": json_dict["OPT"] + json_dict["HA"],
            "OPT+SARGLCM": json_dict["OPT"] + json_dict["SARGLCM"],
        }
        json_dict = FRW(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\Temp\feats4.json").saveJson(to_dict)
        print(json_dict.keys())
        # for k in to_dict:
        #     print("python -c \"import sys; sys.path.append(r'F:\\PyCodes');"
        #           " from Shadow.Hierarchical.SHH2ML import SHH2ML_TIC_mian; SHH2ML_TIC_mian()\" {}".format(k))
        cmd_line = "python -c \"import sys; sys.path.append(r'F:\\PyCodes');" \
                   " from Shadow.Hierarchical.SHH2ML import SHH2ML_TIC_mian; SHH2ML_TIC_mian(\'{}\', \'{}\')\" "
        print(cmd_line.format("qd", "feats4"))
        print(cmd_line.format("bj", "feats4"))
        print(cmd_line.format("cd", "feats4"))
        print()

    def func5():
        df = pd.read_excel(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\2\SHH青岛C2影像.xlsx",
                           sheet_name="Sheet3")
        print(df)
        FRW(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\2\filelist2.txt").writeLines(df["FILE"].tolist())
        vrtAddDescriptions(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\2\SHH2_QD2.vrt",
                           r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\2\SHH2_QD2.vrt",
                           df["NAME"].tolist()
                           )

    def func6():
        gr = GDALRaster(r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\SHH2_CD2_envi.dat", gdal.GA_Update)
        print(gr.names)

        def func61():
            names = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'AS_VV', 'AS_VH', 'DE_VV', 'DE_VH', ]
            data = np.zeros((len(names), gr.n_rows, gr.n_columns))
            for i, name in enumerate(names):
                print(name)
                data[i] = gr.readGDALBand(name)
            gr.save(data, r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\SHH2_CD2_look.tif", fmt="GTiff",
                    dtype=gdal.GDT_Float32, descriptions=names)

        def func62():
            is_w = False

            def writedata(name, _data):
                if is_w:
                    band = gr.getGDALBand(name)
                    band.WriteArray(data.astype("float32"))

            names = ['AS_C11', 'AS_C22', 'AS_Lambda1', 'AS_Lambda2', 'AS_SPAN', 'AS_Epsilon',
                     'DE_C11', 'DE_C22', 'DE_Lambda1', 'DE_Lambda2', 'DE_SPAN', 'DE_Epsilon', ]

            for i, name in enumerate(names):
                print(name)
                data = gr.readGDALBand(name)
                # data[data == 0] = data[data == 0] + 1.1381005e-15
                # data = _10log10(data)
                print(data.min(), data.max(), data.mean())
                writedata(name, data)

            names = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDWI', ]

            for i, name in enumerate(names):
                print(name)
                data = gr.readGDALBand(name)
                data[np.isnan(data)] = 0
                print(data.min(), data.max(), data.mean())
                writedata(name, data)

            print("AS_VHDVV")
            data_vv = gr.readGDALBand("AS_VV")
            data_vh = gr.readGDALBand("AS_VH")
            data = data_vh - data_vv
            writedata("AS_VHDVV", data)
            print(data.min(), data.max(), data.mean())

            print("DE_VHDVV")
            data_vv = gr.readGDALBand("DE_VV")
            data_vh = gr.readGDALBand("DE_VH")
            writedata("DE_VHDVV", data)
            data = data_vh - data_vv
            print(data.min(), data.max(), data.mean())

        func61()

    def func7():
        def x_range(data):
            data = np.expand_dims(data, axis=0)
            print(data.shape)
            x = reHist(data, ratio=0.005)
            return float(x[0][0]), float(x[0][1])

        gr = GDALRaster(SHH2Config.BJ_ENVI_FN)
        to_dict = {}

        for fn in gr.names:
            x_min, x_max = x_range(gr.readGDALBand(fn))
            to_dict[fn] = {"min": x_min, "max": x_max}
            print(fn, x_min, x_max)

        FRW(r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\SHH2_BJ2_range2.json").saveJson(to_dict)

    def func8():
        json_dict = FRW(r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\SHH2_CD2_range.json").readJson()
        print("{:<12} | {:<20} | {:<20}".format("NAME", "MIN", "MAX"))
        print("-" * 60)
        for k in json_dict:
            print("{:<12} | {:>20.6f} | {:>20.6f}".format(k, json_dict[k]["min"], json_dict[k]["max"], ))

    def func9():
        gr = GDALRaster(r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\SHH2_BJ2_envi.dat", gdal.GA_Update)
        print(gr.names)

        # names = ["AS_VV", "AS_VH", "DE_VV", "DE_VH"]
        # for name in names:
        #     data = gr.readGDALBand(name)
        #     data = _10log10(data)
        #     # band = gr.getGDALBand(name)
        #     # band.WriteArray(data.astype("float32"))
        #     print(name, data.min(), data.max())

        print("AS_VHDVV")
        data_vv = gr.readGDALBand("AS_VV")
        data_vh = gr.readGDALBand("AS_VH")
        data = data_vh - data_vv
        band = gr.getGDALBand("AS_VHDVV")
        band.WriteArray(data.astype("float32"))
        print(data.min(), data.max(), data.mean())

        print("DE_VHDVV")
        data_vv = gr.readGDALBand("DE_VV")
        data_vh = gr.readGDALBand("DE_VH")
        data = data_vh - data_vv
        band = gr.getGDALBand("DE_VHDVV")
        band.WriteArray(data.astype("float32"))
        print(data.min(), data.max(), data.mean())

    def func10():
        df_list = []

        def df_des(csv_fn):
            df = pd.read_csv(csv_fn)
            df_des_data = df[df["TEST"] == 1].groupby("CNAME").count()["SRT"]
            print(df_des_data)
            print(df_des_data.sum())
            df_list.append(df_des_data)
            print("Test samples categorys numbers:")
            df_des_data = df[df["TEST"] == 0].groupby("CNAME").count()["SRT"]
            df_list.append(df_des_data)
            print(df_des_data)
            print(df_des_data.sum())

        df_des(r"F:\Week\20240623\Data\sh2_spl252_1.csv")
        df_des(r"F:\Week\20240623\Data\sh2_spl26_4_spl.csv")
        df_des(r"F:\Week\20240623\Data\sh2_spl271_4_spl.csv")
        pd.concat(df_list, axis=1).T.to_csv(r"F:\Week\20240623\Data\spl_n.csv")
        print(pd.concat(df_list, axis=1))

    func10()
    return


def main():
    funcs()
    pass


if __name__ == "__main__":
    main()

r"""
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2ML import main; main()"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2ML import SHH2ML_TIC_mian; SHH2ML_TIC_mian()"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2ML import SHH2ML_TST_main; SHH2ML_TST_main()"
"""
