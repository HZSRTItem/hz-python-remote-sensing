# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2GD2.py
@Time    : 2024/7/20 20:33
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2GD2
-----------------------------------------------------------------------------"""
import os
import time

from plotly.express import pd
from tabulate import tabulate

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.GDALUtils import sampleCSV
from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.NumpyUtils import update10EDivide10, update10Log10
from SRTCodes.SRTModel import FeatFuncs, SVM_RGS, RF_RGS, DataScale, MLModel, SamplesData
from SRTCodes.SRTSample import samplesDescription
from SRTCodes.SRTTimeDirectory import TimeDirectory
from SRTCodes.Utils import DirFileName, SRTWriteText, printDict, timeStringNow
from Shadow.Hierarchical import SHH2Config
from Shadow.Hierarchical.SHH2Config import FEAT_NAMES_CLS

_DIRNAME = r"F:\ProjectSet\Shadow\Hierarchical\GD2"
_DFN = DirFileName(_DIRNAME)

_QD_SPL_FN = _DFN.fn("GD2_QD.csv")
_BJ_SPL_FN = _DFN.fn("GD2_BJ.csv")
_CD_SPL_FN = _DFN.fn("GD2_CD.csv")

_QD_RASTER_FN = SHH2Config.QD_ENVI_FN
_BJ_RASTER_FN = SHH2Config.BJ_ENVI_FN
_CD_RASTER_FN = SHH2Config.CD_ENVI_FN

_QD_RANGE_FN = SHH2Config.QD_RANGE_FN
_BJ_RANGE_FN = SHH2Config.BJ_RANGE_FN
_CD_RANGE_FN = SHH2Config.CD_RANGE_FN

_QD_SPLING_FN = lambda is_spl=False: sampleCSV(_QD_RASTER_FN, _QD_SPL_FN, is_spl)
_BJ_SPLING_FN = lambda is_spl=False: sampleCSV(_BJ_RASTER_FN, _BJ_SPL_FN, is_spl)
_CD_SPLING_FN = lambda is_spl=False: sampleCSV(_CD_RASTER_FN, _CD_SPL_FN, is_spl)

_CATEGORY_NAMES = ["IS", "VEG", "SOIL", "WAT"]

_MODEL_DIRNAME = _DFN.fn("Models")
_MODEL_DIR = DirFileName(_MODEL_DIRNAME)

_FEAT_NAMES = FEAT_NAMES_CLS()


def _CITY_NAME_GET(city_name, qd, bj, cd):
    if city_name == "qd":
        return qd
    if city_name == "bj":
        return bj
    if city_name == "cd":
        return cd


_RANGE_FN = lambda city_name: _CITY_NAME_GET(city_name, _QD_RANGE_FN, _BJ_RANGE_FN, _CD_RANGE_FN)
_RASTER_FN = lambda city_name: _CITY_NAME_GET(city_name, _QD_RASTER_FN, _BJ_RASTER_FN, _CD_RASTER_FN)


def _SPLING_FN(city_name):
    if city_name == "qd":
        return _QD_SPLING_FN()
    if city_name == "bj":
        return _BJ_SPLING_FN()
    if city_name == "cd":
        return _CD_SPLING_FN()
    return None


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
        func1(update10EDivide10)
    elif _type == "10Log10":
        func1(update10Log10)

    return ff


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

        if self.feat_funcs is None:
            self.feat_funcs = FeatFuncs()
        if self.map_dict is None:
            self.map_dict = {
                "IS": 1, "VEG": 2, "SOIL": 3, "WAT": 4,
                "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 3, "WAT_SH": 4
            }

    def fit(self):

        if self.model == "svm":
            model = SVM_RGS()
        elif self.model == "rf":
            model = RF_RGS()
        else:
            print("Can not find model of \"{}\"".format(self.model))
            return

        data_scale = DataScale()
        if self.range_json is not None:
            data_scale = DataScale().readJson(self.range_json)

        ml_mod = MLModel()
        ml_mod.name = self.name
        ml_mod.filename = os.path.join(self.dirname, "{}.hm".format(self.name))
        ml_mod.x_keys = self.x_keys
        ml_mod.feat_funcs = self.feat_funcs
        ml_mod.map_dict = self.map_dict
        ml_mod.data_scale = data_scale
        ml_mod.color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255)}
        ml_mod.clf = model
        ml_mod.train_filters = [("TEST", "==", 1)]
        ml_mod.test_filter = [("TEST", "==", 0)]
        # Import samples
        ml_mod.sampleData(self.sd)
        ml_mod.samples.dataDescription(print_func=self.print_func, end="\n")
        # Training
        ml_mod.train()
        # Save
        if self.is_save:
            ml_mod.save()
        if self.raster_fn is not None:
            self.print_func("model file name:", ml_mod.filename)
            ml_mod.imdc(self.raster_fn)
        # Load
        # ml_mod = MLModel().load("*.shh2mod")

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


class _TrainImdc:

    def __init__(self, city_name, csv_fn=None, model="rf", is_save=False,
                 is_imdc=False, feat_funcs_type=None, is_td=False):
        self.city_name = city_name
        self.csv_fn = _SPLING_FN(self.city_name) if csv_fn is None else csv_fn
        self.model = model
        self.range_json = _RANGE_FN(city_name) if model == "svm" else None
        self.is_save = is_save
        self.is_imdc = is_imdc
        self.feat_funcs = featFuncs(feat_funcs_type)
        self.raster_fn = _RASTER_FN(city_name) if is_imdc else None
        self.is_td = is_td

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

        to_dict = {}
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

        func12("{}-Opt-AS-DE".format(df_name), _FEAT_NAMES.f_opt_as_de())
        func12("{}-Opt-AS".format(df_name), _FEAT_NAMES.f_opt_as())
        func12("{}-Opt-DE".format(df_name), _FEAT_NAMES.f_opt_de())
        func12("{}-Opt".format(df_name), _FEAT_NAMES.f_opt())

        df_acc = pd.DataFrame(to_dict).T
        td.kw("accuracy", tabulate(df_acc, headers="keys", tablefmt="simple"), sep=":\n")
        td.saveDF("accuracy.csv", df_acc)

        _sw("{} -> {} {:<3} {}\n".format(timeStringNow(), self.city_name, self.model, td.time_dirname()))

        return self


def main():
    input(">")
    t1 = time.time()

    # ds = ENVIRaster(SHH2Config.QD_ENVI_FN)
    ds = GDALRaster(SHH2Config.QD_ENVI_FN)
    data = ds.readAsArray()

    t2 = time.time()
    print(t2 - t1)
    input(">")


if __name__ == "__main__":
    main()
    r"""
    python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2GD2 import main; main()"
    """
