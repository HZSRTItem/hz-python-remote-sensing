# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2MLFCMain.py
@Time    : 2024/8/8 21:05
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2MLFCMain
-----------------------------------------------------------------------------"""
import os.path
from shutil import copyfile

import pandas as pd
from tabulate import tabulate

from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.SRTLinux import W2LF
from SRTCodes.SRTModel import RF_RGS, SVM_RGS, MLModel
from SRTCodes.SRTTimeDirectory import TimeDirectory
from SRTCodes.Utils import readJson, DirFileName, changext, getfilenamewithoutext, RumTime
from Shadow.Hierarchical import SHH2Config
from Shadow.Hierarchical.SHH2Model import MLFCModel, SamplesData
from Shadow.Hierarchical.SHH2Sample import SAMPLING_CITY_NAME

_MAP_DICT = {"IS": 1, "VEG": 2, "SOIL": 3, "WAT": 4, "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 3, "WAT_SH": 4}
_MAP_DICT_ISO = {"IS": 1, "VEG": 0, "SOIL": 2, "WAT": 0, "IS_SH": 0, "VEG_SH": 0, "SOIL_SH": 0, "WAT_SH": 0}
_MAP_DICT_WS = {"IS": 0, "VEG": 0, "SOIL": 0, "WAT": 3, "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 2, "WAT_SH": 3}

_BACK_SPL_DIRNAME = W2LF(r"F:\ProjectSet\Shadow\Hierarchical\Samples\Update")

_NOFC_X_KEYS = [
    # Opt
    "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "NDVI", "NDWI",
    # Opt glcm
    "OPT_asm", "OPT_con", "OPT_cor", "OPT_dis", "OPT_ent", "OPT_hom", "OPT_mean", "OPT_var",
    # AS
    "AS_VV", "AS_VH", "AS_VHDVV",
    "AS_C11", "AS_C22", "AS_SPAN",
    "AS_H", "AS_A", "AS_Alpha",
    # AS glcm
    "AS_VH_asm", "AS_VH_con", "AS_VH_cor", "AS_VH_dis", "AS_VH_ent", "AS_VH_hom", "AS_VH_mean", "AS_VH_var",
    "AS_VV_asm", "AS_VV_con", "AS_VV_cor", "AS_VV_dis", "AS_VV_ent", "AS_VV_hom", "AS_VV_mean", "AS_VV_var",
    # DE
    "DE_VV", "DE_VH", "DE_angle", "DE_VHDVV",
    "DE_C11", "DE_C22", "DE_SPAN",
    "DE_H", "DE_A", "DE_Alpha",
    # DE glcm
    "DE_VH_asm", "DE_VH_con", "DE_VH_cor", "DE_VH_dis", "DE_VH_ent", "DE_VH_hom", "DE_VH_mean", "DE_VH_var",
    "DE_VV_asm", "DE_VV_con", "DE_VV_cor", "DE_VV_dis", "DE_VV_ent", "DE_VV_hom", "DE_VV_mean", "DE_VV_var",
]

_COLOR_TABLE_4 = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), }

_F_O = SHH2Config.FEAT_NAMES.OPT + SHH2Config.FEAT_NAMES.OPT_GLCM
_F_A = SHH2Config.FEAT_NAMES.AS
_F_D = SHH2Config.FEAT_NAMES.DE
_F = SHH2Config.FEAT_NAMES


def _GET_MODEL(name):
    name = name.upper()
    if name == "RF":
        return RF_RGS()
    if name == "SVM":
        return SVM_RGS()
    return None


class _MLFCModel(MLFCModel):

    def __init__(self):
        super().__init__(is_init=False)

    def buildModel(
            self,
            mod, clf_name, cm_names, color_table, map_dict, name,
            test_filters, train_filters, x_keys
    ):
        mod.name = name
        mod.filename = None
        mod.x_keys = x_keys
        mod.map_dict = map_dict
        mod.data_scale = self.data_scale
        mod.color_table = color_table
        mod.clf = _GET_MODEL(clf_name)
        mod.train_filters = train_filters
        mod.test_filters = test_filters
        mod.cm_names = cm_names

    def initWSModel(self, clf_name="RF", x_keys=None, map_dict=None, color_table=None, cm_names=None, *args, **kwargs):
        if cm_names is None:
            cm_names = ["IS_SH", "NOIS_SH", "WAT"]
        if color_table is None:
            color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), }
        if map_dict is None:
            map_dict = {"WAT": 3, "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 2, "WAT_SH": 2}
        if x_keys is None:
            x_keys = [*SHH2Config.FEAT_NAMES.ALL]
        train_filters = [("FCNAME", "==", "WS")]
        test_filters = []
        name = "WSMod"

        self.buildModel(
            self.ws_ml_mod, clf_name, cm_names, color_table,
            map_dict, name, test_filters, train_filters, x_keys
        )
        return self

    def initISModel(self, clf_name="RF", x_keys=None, map_dict=None, color_table=None, cm_names=None, *args, **kwargs):
        if cm_names is None:
            cm_names = ["IS", "SOIL"]
        if color_table is None:
            color_table = {1: (255, 0, 0), 2: (255, 255, 0)}
        if map_dict is None:
            map_dict = {"IS": 1, "SOIL": 2, }
        if x_keys is None:
            x_keys = [*SHH2Config.FEAT_NAMES.ALL]
        train_filters = [("FCNAME", "==", "ISO")]
        test_filters = []
        name = "ISMod"

        self.buildModel(
            self.is_ml_mod, clf_name, cm_names, color_table,
            map_dict, name, test_filters, train_filters, x_keys
        )
        return self

    def intiVHLModel(self, clf_name="RF", x_keys=None, map_dict=None, color_table=None, cm_names=None, *args, **kwargs):
        if cm_names is None:
            cm_names = ["HIGH", "VEG", "LOW"]
        if color_table is None:
            color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 0), }
        if map_dict is None:
            map_dict = {
                "IS": 1, "VEG": 2, "SOIL": 1, "WAT": 3,
                "IS_SH": 3, "VEG_SH": 3, "SOIL_SH": 3, "WAT_SH": 3
            }
        if x_keys is None:
            x_keys = [*SHH2Config.FEAT_NAMES.OPT]
        train_filters = [("FCNAME", "==", "VHL")]
        test_filters = []
        name = "VHLMod"

        self.buildModel(
            self.vhl_ml_mod, clf_name, cm_names, color_table,
            map_dict, name, test_filters, train_filters, x_keys
        )
        return self


class SHH2MLFC_TIC:

    def __init__(self, city_name, csv_fn, td: TimeDirectory = None):
        self._dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\Temp")

        self.city_name = city_name
        self.range_fn = SHH2Config.GET_RANGE_FN(self.city_name)
        self.raster_fn = SHH2Config.GET_RASTER_FN(self.city_name)

        self.csv_fn = csv_fn
        self.sd = None
        self.initSD(csv_fn)

        self.td = td
        self.initTD(td)

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
        self.td.copyfile(self.csv_fn)

    def initSD(self, csv_fn):
        csv_fn = os.path.abspath(csv_fn)
        self.csv_fn = csv_fn
        dfn = DirFileName(_BACK_SPL_DIRNAME)
        json_fn = dfn.fn("SHH2SamplesUpdate.json")
        if os.path.isfile(json_fn):
            copyfile(json_fn, json_fn + "-back")
            to_dict = readJson(json_fn)
        else:
            to_dict = {self.city_name: []}
        if self.csv_fn not in to_dict[self.city_name]:
            to_dict[self.city_name].append(self.csv_fn)
        to_csv_fn = dfn.fn("SHH2_{}.csv".format(self.city_name.upper()))
        if os.path.isfile(to_csv_fn):
            os.remove(to_csv_fn)
        copyfile(csv_fn, to_csv_fn)

        to_spl_csv_fn = changext(csv_fn, "_spl.csv")
        if not os.path.isfile(to_spl_csv_fn):
            SAMPLING_CITY_NAME(self.city_name, csv_fn, to_spl_csv_fn)
        self.sd = SamplesData()
        self.sd.addCSV(to_spl_csv_fn)

    def modelNOFC(self, name, clf_name="rf", x_keys=None, map_dict=None, color_table=None,
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
        ml_mod.filename = self._dfn.fn("{}-{}-NOFC-{}.shh2mod".format(self.city_name, name, clf_name))
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
        self.models_records[name] = {"TYPE": "NOFC", "CLF": clf_name, }
        return ml_mod

    def modelFC(
            self, name, clf_name="RF",
            vhl_clf_name=None, vhl_x_keys=None, vhl_map_dict=None, vhl_color_table=None, vhl_cm_names=None,
            iso_clf_name=None, iso_x_keys=None, iso_map_dict=None, iso_color_table=None, iso_cm_names=None,
            ws_clf_name=None, ws_x_keys=None, ws_map_dict=None, ws_color_table=None, ws_cm_names=None,
    ):
        self.log("#", "-" * 30, self.city_name.upper(), "FC", name.upper(), clf_name.upper(), "-" * 30, "#\n")
        mlfc_mod = _MLFCModel()
        mlfc_mod.filename = self._dfn.fn("{}-{}-FC-{}.shh2mod".format(self.city_name, name, clf_name))
        mlfc_mod.data_scale.readJson(SHH2Config.GET_RANGE_FN(self.city_name))
        self.kw("NAME", getfilenamewithoutext(mlfc_mod.filename))

        if vhl_clf_name is None:
            vhl_clf_name = clf_name
        if iso_clf_name is None:
            iso_clf_name = clf_name
        if ws_clf_name is None:
            ws_clf_name = clf_name

        mlfc_mod.intiVHLModel(
            vhl_clf_name, x_keys=vhl_x_keys, map_dict=vhl_map_dict, color_table=vhl_color_table,
            cm_names=vhl_cm_names, )
        self.kw("VHL.MLMOD.X_KEYS", mlfc_mod.vhl_ml_mod.x_keys)
        self.kw("VHL.MLMOD.MAP_DICT", mlfc_mod.vhl_ml_mod.map_dict)
        self.kw("VHL.MLMOD.COLOR_TABLE", mlfc_mod.vhl_ml_mod.color_table)
        self.kw("VHL.MLMOD.CLF", mlfc_mod.vhl_ml_mod.clf)
        self.kw("VHL.MLMOD.TRAIN_FILTERS", mlfc_mod.vhl_ml_mod.train_filters)
        self.kw("VHL.MLMOD.TEST_FILTERS", mlfc_mod.vhl_ml_mod.test_filters)
        self.kw("VHL.MLMOD.CM_NAMES", mlfc_mod.vhl_ml_mod.cm_names)

        mlfc_mod.initISModel(
            iso_clf_name, x_keys=iso_x_keys, map_dict=iso_map_dict, color_table=iso_color_table,
            cm_names=iso_cm_names, )
        self.kw("ISO.MLMOD.X_KEYS", mlfc_mod.is_ml_mod.x_keys)
        self.kw("ISO.MLMOD.MAP_DICT", mlfc_mod.is_ml_mod.map_dict)
        self.kw("ISO.MLMOD.COLOR_TABLE", mlfc_mod.is_ml_mod.color_table)
        self.kw("ISO.MLMOD.CLF", mlfc_mod.is_ml_mod.clf)
        self.kw("ISO.MLMOD.TRAIN_FILTERS", mlfc_mod.is_ml_mod.train_filters)
        self.kw("ISO.MLMOD.TEST_FILTERS", mlfc_mod.is_ml_mod.test_filters)
        self.kw("ISO.MLMOD.CM_NAMES", mlfc_mod.is_ml_mod.cm_names)

        mlfc_mod.initWSModel(
            ws_clf_name, x_keys=ws_x_keys, map_dict=ws_map_dict, color_table=ws_color_table,
            cm_names=ws_cm_names, )
        self.kw("WS.MLMOD.X_KEYS", mlfc_mod.ws_ml_mod.x_keys)
        self.kw("WS.MLMOD.MAP_DICT", mlfc_mod.ws_ml_mod.map_dict)
        self.kw("WS.MLMOD.COLOR_TABLE", mlfc_mod.ws_ml_mod.color_table)
        self.kw("WS.MLMOD.CLF", mlfc_mod.ws_ml_mod.clf)
        self.kw("WS.MLMOD.TRAIN_FILTERS", mlfc_mod.ws_ml_mod.train_filters)
        self.kw("WS.MLMOD.TEST_FILTERS", mlfc_mod.ws_ml_mod.test_filters)
        self.kw("WS.MLMOD.CM_NAMES", mlfc_mod.ws_ml_mod.cm_names)

        mlfc_mod.sampleData(self.sd)

        self.models[name] = mlfc_mod
        self.model = mlfc_mod
        self.model_name = name
        self.models_records[name] = {"TYPE": "FC", "CLF": clf_name, }
        return mlfc_mod

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
    td = TimeDirectory(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods").initLog()
    tic = SHH2MLFC_TIC("qd", r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\qd\sh2_spl30_qd6.csv", td=td)
    is_imdc = True

    run_list = []

    def nofc(*args, **kwargs):
        run_list.append(("NOFC", args, kwargs))

    def fc(*args, **kwargs):
        run_list.append(("FC", args, kwargs))

    for clf_name in ["RF", "SVM"]:
        nofc("{}_O".format(clf_name), clf_name, x_keys=_F_O)
        nofc("{}_OA".format(clf_name), clf_name, x_keys=_F_O + _F.AS_BS)
        nofc("{}_OD".format(clf_name), clf_name, x_keys=_F_O + _F.DE_BS)
        nofc("{}_OAD_BS".format(clf_name), clf_name, x_keys=_F_O + _F.AS_BS + _F.DE_BS)
        nofc("{}_OAD_HA".format(clf_name), clf_name, x_keys=_F_O + _F.AS_HA + _F.DE_HA)
        nofc("{}_OAD_HABS".format(clf_name), clf_name, x_keys=_F_O + _F.AS_HA + _F.DE_HA + _F.AS_BS + _F.DE_BS)
        fc("{}_FC_BS".format(clf_name), clf_name,
           vhl_x_keys=_F_O,
           iso_x_keys=_F_O + _F.AS_BS + _F.DE_BS,
           ws_x_keys=_F_O + _F.AS_BS + _F.DE_BS, )
        fc("{}_FC_HA".format(clf_name), clf_name,
           vhl_x_keys=_F_O,
           iso_x_keys=_F_O + _F.AS_HA + _F.DE_HA,
           ws_x_keys=_F_O + _F.AS_HA + _F.DE_HA, )
        fc("{}_FC_HABS".format(clf_name), clf_name,
           vhl_x_keys=_F_O,
           iso_x_keys=_F_O + _F.AS_HA + _F.DE_HA + _F.AS_BS + _F.DE_BS,
           ws_x_keys=_F_O + _F.AS_HA + _F.DE_HA + _F.AS_BS + _F.DE_BS, )

    run_time = RumTime(len(run_list), tic.log).strat()
    for name, _args, _kwargs in run_list:
        if name.upper() == "NOFC":
            tic.modelNOFC(*_args, **_kwargs)
        elif name.upper() == "FC":
            tic.modelFC(*_args, **_kwargs)
        tic.train(is_imdc=is_imdc)
        run_time.add().printInfo()
    run_time.end()

    tic.showAccuracy()
    pass


if __name__ == "__main__":
    main()
    r"""
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2MLFCMain import main; main()"
    """
