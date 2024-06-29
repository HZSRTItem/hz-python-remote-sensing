# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2VHL.py
@Time    : 2024/6/25 10:10
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2VHL
-----------------------------------------------------------------------------"""
from time import sleep

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from SRTCodes.GDALRasterIO import GDALRaster, saveGTIFFImdc
from SRTCodes.GDALUtils import RasterRandomCoors
from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.SRTModelImage import GDALImdc
from SRTCodes.SRTTimeDirectory import TimeDirectory
from SRTCodes.Utils import DirFileName, FRW
from Shadow.Hierarchical import SHH2Config
from Shadow.Hierarchical.SHH2Accuracy import accuracyY12
from Shadow.Hierarchical.SHH2ML2 import SHH2MLTraining
from Shadow.Hierarchical.SHH2Sample import samplingImdc

KEY_NAMES = SHH2Config.NAMES


def csvFN(city_name):
    if city_name == "qd":
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\2\sh2_spl252_4_spl2.csv"
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\28\sh2_spl28_is1_spl.csv"
    elif city_name == "bj":
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\3\sh2_spl273_5_spl.csv"
    elif city_name == "cd":
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\2\sh2_spl26_4_spl2.csv"
    else:
        csv_fn = None
    return csv_fn


def accuracy1(self, cm_name, cm_str, fc_category, is_eq_number, line, name, y2_map_dict):
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
    return cm_str


def trainimdc(city_name, dfn, td, train_dict, map_dict, y2_map_dict, cnames, clf=None, x_keys=None,
              is_save_model=False, n_run=5, color_table=None, is_imdc=False):
    # dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\VHLModels")
    if clf is None:
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=1, min_samples_split=2)
    if x_keys is None:
        x_keys = KEY_NAMES

    td.initLog()
    td.log("#", "-" * 34, "SHH2 IS", "-" * 34, "#")
    td.log(td.time_dfn.dirname)
    sleep(2)

    city_name = td.kw("CITY_NAME", city_name)
    x_keys = td.kw("X_KEYS", x_keys)
    map_dict = td.kw("MAP_DICT", map_dict)
    y2_map_dict = td.kw("Y2_MAP_DICT", y2_map_dict)
    clf = td.kw("MODEL", clf)
    csv_fn = td.kw("CSV_FN", csvFN(city_name))
    cnames = td.kw("CNAMES", cnames)
    raster_fn = td.kw("RASTER_FN", SHH2Config.GET_RASTER_FN(city_name))
    color_table = td.kw("COLOR_TABLE", color_table)

    td.copyfile(csv_fn)
    td.copyfile(__file__)

    df = pd.read_csv(csv_fn)

    def imdc(ml_train, name):
        gimdc = GDALImdc(raster_fn, is_sfm=False)
        to_fn = td.fn("vhl_{}_{}_imdc.tif".format(city_name, name))
        gimdc.imdc1(ml_train.clf, to_fn, fit_names=x_keys, color_table=color_table)

    def train(_n_run):
        acc_list = []
        for name in train_dict:
            td.log("#", "-" * 10, "Training", name, "-" * 10, "#")
            ml_train = SHH2MLTraining()
            ml_train.map_dict = map_dict
            ml_train.df = df
            ml_train.train(name, x_keys, c_fn="CNAME", map_dict=map_dict, clf=clf)

            cm = accuracyY12(
                ml_train.category_names, ml_train.categorys[name]["y2"],
                map_dict, y2_map_dict,
                cnames=cnames,
                fc_category=None,
                is_eq_number=None,
            )

            td.log(cm.fmtCM())
            td.log("OA:", cm.OA())
            td.log("Kappa:", cm.getKappa())
            td.saveJson("categorys.json", ml_train.categorys)
            acc_list.append({"NAME": name, "OA": cm.OA(), "Kappa": cm.getKappa()})
            if is_save_model:
                joblib.dump(ml_train.clf, td.fn("{}.mod".format(name)), )
            if is_imdc:
                joblib.dump(ml_train.clf, td.fn("{}.mod".format(name)), )
                imdc(ml_train, name)

        df_acc = pd.DataFrame(acc_list)
        df_acc.to_csv(td.fn("accuracy{}.csv".format(_n_run)), index=False)
        print(df_acc.sort_values("OA", ascending=False))

        return df_acc

    df_accs = None
    for i in range(n_run):
        df_acc_tmp = train(i + 1)
        if df_accs is None:
            df_accs = df_acc_tmp
        else:
            df_accs[["OA", "Kappa"]] = df_accs[["OA", "Kappa"]] + df_acc_tmp[["OA", "Kappa"]]

    df_accs[["OA", "Kappa"]] = df_accs[["OA", "Kappa"]] / n_run
    print(df_accs.sort_values("OA", ascending=False))

    pass


def trainimdc_main():
    dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\ISModels")
    td = TimeDirectory(dfn.fn())
    map_dict = {
        "IS": 2, "VEG": 1, "SOIL": 1, "WAT": 1,
        "IS_SH": 1, "VEG_SH": 1, "SOIL_SH": 1, "WAT_SH": 1
    }
    y2_map_dict = {1: 1, 2: 2}
    cnames = ["SOIL", "IS"]
    json_dict = FRW(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\Temp\feats5.json").readJson()

    trainimdc(
        city_name="qd",
        dfn=dfn,
        td=td,
        train_dict=json_dict,
        map_dict=map_dict,
        y2_map_dict=y2_map_dict,
        cnames=cnames,
        clf=None,
        x_keys=None,
        is_save_model=False,
        n_run=5,
        color_table={1: (255, 255, 0), 2: (255, 0, 0)},
        is_imdc=False
    )


def main():
    def func1():
        rrc = RasterRandomCoors(r"F:\ProjectSet\Shadow\Hierarchical\VHLModels\20240625H144500\vhl_qd_imdc.tif")
        rrc.fit(2, 1000)
        df = pd.DataFrame(rrc.coors)
        fns = samplingImdc(df, r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240618H112230")
        print(df.keys())
        data = df[fns].values
        category = []
        for i in range(len(df)):
            if len(np.unique(data[i])) == 1:
                category.append(data[i, 0] * 10 + 1)
            else:
                category.append(0)
        df["CATEGORY2"] = df["CATEGORY"]
        df["CATEGORY"] = np.array(category)
        df = df.sort_values(["CATEGORY", "OPT"])
        df.to_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\28\sh2_spl28_2.csv", index=False)

    def func2():
        df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\2\sh2_spl252_4_spl2.csv")

        print("Train samples categorys numbers:")
        df_des_data = df[df["TEST"] == 1].groupby("CNAME").count()["SRT"]
        print(df_des_data)
        print(df_des_data.sum())
        print("Test samples categorys numbers:")
        df_des_data = df[df["TEST"] == 0].groupby("CNAME").count()["SRT"]
        print(df_des_data)
        print(df_des_data.sum())

    def func3():
        gr_vhl = GDALRaster(r"F:\ProjectSet\Shadow\Hierarchical\VHLModels\20240625H151538\vhl_qd_imdc.tif")
        gr_is = GDALRaster(r"F:\ProjectSet\Shadow\Hierarchical\ISModels\20240625H164518\vhl_qd_imdc.tif")

        vhl_imdc = gr_vhl.readAsArray()
        is_imdc = gr_is.readAsArray()

        to_imdc = np.zeros_like(vhl_imdc)
        to_imdc[vhl_imdc == 1] = 2
        to_imdc[vhl_imdc == 3] = 5
        to_imdc[vhl_imdc == 4] = 4

        is_imdc_tmp = np.zeros_like(is_imdc)
        is_imdc_tmp[is_imdc == 2] = 1
        is_imdc_tmp[is_imdc == 1] = 3
        to_imdc[vhl_imdc == 2] = is_imdc_tmp[vhl_imdc == 2]

        to_fn = r"F:\ProjectSet\Shadow\Hierarchical\ISModels\20240625H164518\vhl_is_qd_imdc.tif"
        saveGTIFFImdc(gr_vhl, to_imdc.astype("int8"), to_fn,
                      color_table={1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), 5: (0, 0, 0)})

    func3()


if __name__ == "__main__":
    trainimdc_main()

    r"""
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2VHL import trainimdc; trainimdc('qd')"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2VHL import trainimdc; trainimdc('bj')"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2VHL import trainimdc; trainimdc('cd')"


    """
