# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2MLThree.py
@Time    : 2024/6/19 22:28
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2MLThree
-----------------------------------------------------------------------------"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.Utils import DirFileName, SRTWriteText, timeStringNow
from Shadow.Hierarchical import SHH2Config
from Shadow.Hierarchical.SHH2Model import SamplesData, MLModel, MLFCModel, RF_RGS, SVM_RGS


def main():
    dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\ZuHui20240708")
    nofc_rf_sw = SRTWriteText(dfn.fn("nofc_rf.txt"), mode="a", is_show=True)
    nofc_svm_sw = SRTWriteText(dfn.fn("nofc_svm.txt"), mode="a", is_show=True)
    fc_rf_sw = SRTWriteText(dfn.fn("fc_rf.txt"), mode="a", is_show=True)
    fc_svm_sw = SRTWriteText(dfn.fn("fc_svm.txt"), mode="a", is_show=True)
    accuracy_sw = SRTWriteText(dfn.fn("accuracy.txt"), mode="a", is_show=True)

    n_run = 5
    map_dict_4 = {"IS": 1, "VEG": 2, "SOIL": 3, "WAT": 4, "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 3, "WAT_SH": 4}
    map_dict_iso = {"IS": 1, "VEG": 0, "SOIL": 2, "WAT": 0, "IS_SH": 0, "VEG_SH": 0, "SOIL_SH": 0, "WAT_SH": 0}
    map_dict_iso_code = {1: 1, 2: 0, 3: 2, 4: 0}
    map_dict_ws = {"IS": 0, "VEG": 0, "SOIL": 0, "WAT": 3, "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 2, "WAT_SH": 3}
    map_dict_ws_code = {1: 1, 2: 2, 3: 2, 4: 3}
    color_table_4 = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), }
    cm_names = ["IS", "VEG", "SOIL", "WAT"]

    sd = SamplesData()
    sd.addCSV(r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\cd\sh2_spl30_cd6_spl.csv")

    accuracy_collection = []

    def getmodel(name):
        if name == "RF":
            # return RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=1, min_samples_split=2)
            return RF_RGS()
        if name == "SVM":
            # return SVC(kernel="rbf", C=8.42, gamma=0.127)
            return SVM_RGS()
        return None

    def _accuracy(spls_test, is_oa_list, oa_list, ws_oa_list, y2_list_init, _sw, n):
        # OA
        y1_list, y2_list = [], []
        for i, spl in enumerate(spls_test):
            if spl["TAG"] == "shh2_spl26_4_random800_spl2":
                y1_list.append(spl.code(map_dict_4))
                y2_list.append(y2_list_init[i])
        cm_oa = ConfusionMatrix(class_names=cm_names)
        cm_oa.addData(y1_list, y2_list)
        oa_list.append(cm_oa.accuracyCategory("IS").OA())
        # IS SOIL OA
        y1_list, y2_list = [], []
        for i, spl in enumerate(spls_test):
            if spl["IS"] == 1:
                y1_list.append(spl.code(map_dict_iso))
                y2_list.append(map_dict_iso_code[y2_list_init[i]])
        cm_iso = ConfusionMatrix(class_names=["IS", "SOIL"])
        cm_iso.addData(y1_list, y2_list)
        is_oa_list.append(cm_iso.OA())
        # WAT SHADOW
        y1_list, y2_list = [], []
        for i, spl in enumerate(spls_test):
            if spl["WS"] == 1:
                y1_list.append(spl.code(map_dict_ws))
                y2_list.append(map_dict_ws_code[y2_list_init[i]])
        cm_ws = ConfusionMatrix(class_names=["IS_SH", "VEG_SH", "WAT"])
        cm_ws.addData(y1_list, y2_list)
        ws_oa_list.append(cm_ws.accuracyCategory("IS_SH").OA())
        _sw.write("IS CM")
        _sw.write(cm_oa.fmtCM())
        _sw.write("IS SOIL CM")
        _sw.write(cm_iso.fmtCM())
        _sw.write("WATER SHADOW CM")
        _sw.write(cm_ws.fmtCM())
        _sw.write("{:>2d}. IS:{:<10.3f} ISO:{:<10.3f} WS:{:<10.3f} ".format(
            n + 1, oa_list[-1], is_oa_list[-1], ws_oa_list[-1]
        ))
        return cm_iso, cm_oa, cm_ws

    def nofc_ml_train(_clf, _sw):
        _save_dirname = dfn.fn("nofc_{}".format(_clf.lower()))
        _dfn = DirFileName(_save_dirname)
        _dfn.mkdir()

        _sw.write("\n\n#", "*" * 30, timeStringNow(), "*" * 30, "#\n", )

        models = []
        oa_list, is_oa_list, ws_oa_list = [], [], []

        for n in range(n_run):
            ml_mod = MLModel()
            ml_mod.filename = _dfn.fn("NOFC-{}-{}.shh2mod".format(_clf, n + 1))
            ml_mod.x_keys = SHH2Config.FEAT_NAMES.ALL
            ml_mod.map_dict = map_dict_4
            ml_mod.color_table = color_table_4
            ml_mod.data_scale.readJson(SHH2Config.CD_RANGE_FN)
            ml_mod.clf = getmodel(_clf)
            ml_mod.test_filters = []  # ("TAG", "==", "shh2_spl26_4_random800_spl2")
            ml_mod.sampleData(sd)
            ml_mod.train()

            if is_save:
                if n == 0:
                    ml_mod.save()
                    ml_mod.imdc(SHH2Config.CD_ENVI_FN)

            _sw.write("#", "-" * 30, "NOFC", _clf, n + 1, "-" * 30, "#", )
            _sw.write("Sample Counts")
            df = ml_mod.samples.showCounts(is_show=False)
            _sw.write(df)

            cm_iso, cm_oa, cm_ws = _accuracy(
                ml_mod.samples.spls_test, is_oa_list, oa_list, ws_oa_list,
                ml_mod.accuracy_dict["y2"], _sw, n
            )

            # ml_mod.save()
            models.append(ml_mod)

        accuracy_collection.append({
            "NAME": "NOFC", "MODEL": _clf,
            "IS OA": np.mean(oa_list),
            "ISO OA": np.mean(is_oa_list),
            "WS OA": np.mean(ws_oa_list),
        })
        _sw.write(_clf, "OA  MEAN:", np.mean(oa_list))
        _sw.write(_clf, "ISO MEAN:", np.mean(is_oa_list))
        _sw.write(_clf, "WS  MEAN:", np.mean(ws_oa_list))

    def fc_ml_train(_clf, _sw):
        _save_dirname = dfn.fn("fc_{}".format(_clf.lower()))
        _dfn = DirFileName(_save_dirname)
        _dfn.mkdir()

        models = []
        oa_list, is_oa_list, ws_oa_list = [], [], []

        for n in range(n_run):
            mlfc_mod = MLFCModel()
            mlfc_mod.filename = _dfn.fn("FC-{}-{}.shh2mod".format(_clf, n + 1))
            mlfc_mod.data_scale.readJson(SHH2Config.CD_RANGE_FN)
            mlfc_mod.vhl_ml_mod.clf = getmodel(_clf)
            mlfc_mod.vhl_ml_mod.x_keys = SHH2Config.FEAT_NAMES.OPT + SHH2Config.FEAT_NAMES.OPT_GLCM
            mlfc_mod.is_ml_mod.clf = getmodel(_clf)
            mlfc_mod.is_ml_mod.x_keys = SHH2Config.FEAT_NAMES.ALL
            mlfc_mod.ws_ml_mod.clf = getmodel(_clf)
            mlfc_mod.ws_ml_mod.x_keys = SHH2Config.FEAT_NAMES.ALL
            mlfc_mod.sampleData(sd)
            mlfc_mod.samples.showCounts()
            mlfc_mod.train()

            if is_save:
                if n == 0:
                    mlfc_mod.save()
                    mlfc_mod.imdc(SHH2Config.CD_ENVI_FN)

            _sw.write("#", "-" * 30, "NOFC", _clf, n + 1, "-" * 30, "#", )
            _sw.write("Sample Counts")
            df = mlfc_mod.samples.showCounts(is_show=False)
            _sw.write(df)

            cm_iso, cm_oa, cm_ws = _accuracy(
                mlfc_mod.samples.spls_test, is_oa_list, oa_list, ws_oa_list,
                mlfc_mod.accuracy_dict["y2"], _sw, n
            )

            # ml_mod.save()
            models.append(mlfc_mod)

        accuracy_collection.append({
            "NAME": "FC", "MODEL": _clf,
            "IS OA": np.mean(oa_list),
            "ISO OA": np.mean(is_oa_list),
            "WS OA": np.mean(ws_oa_list),
        })
        _sw.write(_clf, "OA  MEAN:", np.mean(oa_list))
        _sw.write(_clf, "ISO MEAN:", np.mean(is_oa_list))
        _sw.write(_clf, "WS  MEAN:", np.mean(ws_oa_list))

    is_save = False

    # nofc_ml_train("SVM", nofc_svm_sw)
    # nofc_ml_train("RF", nofc_rf_sw)
    fc_ml_train("SVM", fc_svm_sw)
    fc_ml_train("RF", fc_rf_sw)

    df_acc = pd.DataFrame(accuracy_collection)
    accuracy_sw.write("\n#", "-" * 30, timeStringNow(), "-" * 30, "#\n", )
    accuracy_sw.write(df_acc)

    # ml_mod.imdc(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\TestRegions\QDTR0.tif")
    # ml_mod = MLModel().load(r"F:\ProjectSet\Shadow\Hierarchical\Temp\tmp9.shh2mod")
    # ml_mod.imdc(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\TestRegions\QDTR1.tif")

    pass


if __name__ == "__main__":
    main()
r"""
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2MLThree import main; main()"
"""