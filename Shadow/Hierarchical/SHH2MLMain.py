# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2MLMain.py
@Time    : 2024/7/3 12:21
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2MLMain
-----------------------------------------------------------------------------"""
import os.path

import joblib
import pandas as pd

from SRTCodes.GDALUtils import uniqueSamples
from SRTCodes.SRTFeature import SRTFeaturesCalculation
from SRTCodes.SRTModel import RF_RGS
from SRTCodes.SRTTimeDirectory import TimeDirectory
from SRTCodes.Utils import changext, samplesFilterOR, printList
from Shadow.Hierarchical import SHH2Config
from Shadow.Hierarchical.SHH2MLModel import TIC
from Shadow.Hierarchical.SHH2Sample import SAMPLING_CITY_NAME

_CITY_NAME = "qd"

_CSV_FNS = [
    ("qd", r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\qd\sh2_spl30_qd1.csv"),
    ("bj", r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\bj\sh2_spl30_bj1.csv"),
    ("cd", r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\cd\sh2_spl30_cd1.csv"),
    ("cd", r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\cd\sh2_spl30_cd2_spl.csv"),
    ("qd", r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\qd\sh2_spl30_qd2.csv"),
    ("bj", r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\bj\sh2_spl30_bj2.csv"),
    ("cd", r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\cd\sh2_spl30_cd7.csv"),
    ("qd", r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\qd\sh2_spl30_qd4.csv"),
    ("qd", r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\qd\sh2_spl30_qd5.csv"),
    ("qd", r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\qd\sh2_spl30_qd5_nolook.csv"),
    ("qd", r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\qd\sh2_spl30_qd6.csv"),
]

_NAME = "VHL-{}-{}".format(_CITY_NAME.upper(), "O")
_MAP_DICT = {
    "IS": 1, "VEG": 2, "SOIL": 1, "WAT": 3,
    "IS_SH": 3, "VEG_SH": 3, "SOIL_SH": 3, "WAT_SH": 3,
}
_COLOR_TABLE = {
    # 1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255),
    1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 0), 4: (0, 0, 255),
}
_CM_NAMES = [
    # "IS", "VEG", "SOIL", "WAT",
    "IS", "VEG", "WS"
]

_X_KEYS = [
    # Opt
    "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "NDVI", "NDWI",
    # # Opt glcm
    # "OPT_asm", "OPT_con", "OPT_cor", "OPT_dis", "OPT_ent", "OPT_hom", "OPT_mean", "OPT_var",
    # # AS
    # "AS_VV", "AS_VH", "AS_VHDVV",
    # "AS_C11", "AS_C22", "AS_SPAN",
    # "AS_H", "AS_A", "AS_Alpha",
    # # AS glcm
    # "AS_VH_asm", "AS_VH_con", "AS_VH_cor", "AS_VH_dis", "AS_VH_ent", "AS_VH_hom", "AS_VH_mean", "AS_VH_var",
    # "AS_VV_asm", "AS_VV_con", "AS_VV_cor", "AS_VV_dis", "AS_VV_ent", "AS_VV_hom", "AS_VV_mean", "AS_VV_var",
    # # DE
    # "DE_VV", "DE_VH", "DE_angle", "DE_VHDVV",
    # "DE_C11", "DE_C22", "DE_SPAN",
    # "DE_H", "DE_A", "DE_Alpha",
    # # DE glcm
    # "DE_VH_asm", "DE_VH_con", "DE_VH_cor", "DE_VH_dis", "DE_VH_ent", "DE_VH_hom", "DE_VH_mean", "DE_VH_var",
    # "DE_VV_asm", "DE_VV_con", "DE_VV_cor", "DE_VV_dis", "DE_VV_ent", "DE_VV_hom", "DE_VV_mean", "DE_VV_var",
]

_FUNC_DN = lambda x, y: (x - y) / (x + y + 0.0000001)
_SFC = SRTFeaturesCalculation()
_SFC.init_names = _X_KEYS
_SFC.add("MNDWI", ["Green", "SWIR2"], lambda data: _FUNC_DN(data["Green"], data["SWIR2"]))
printList("_X_KEYS:", _X_KEYS)

# _CLF = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=1, min_samples_split=2)
_CLF = RF_RGS()

_RASTER_FN = SHH2Config.GET_RASTER_FN(_CITY_NAME)


def _GET_CSV_FN(city_name):
    for i in range(len(_CSV_FNS)):
        city_name_tmp, csv_fn = _CSV_FNS[-(i + 1)]
        print(city_name_tmp, csv_fn)
        if city_name == city_name_tmp:
            csv_spl_fn = changext(csv_fn, "_spl.csv")
            if not os.path.isfile(csv_spl_fn):
                SAMPLING_CITY_NAME(city_name, csv_fn, to_csv_fn=csv_spl_fn)
                samples = pd.read_csv(csv_spl_fn).to_dict("records")
                samples = uniqueSamples(_RASTER_FN, samples)
                pd.DataFrame(samples).to_csv(csv_spl_fn, index=False)
            return csv_spl_fn


def _GET_DF(city_name, train_filter=None, test_filter=None):
    csv_fn = _GET_CSV_FN(city_name)
    df = pd.read_csv(csv_fn)
    if (train_filter is None) and (test_filter is None):
        return df
    df_train_list, df_test_list = df[df["TEST"] == 1].to_dict("records"), df[df["TEST"] == 0].to_dict("records")
    if train_filter is not None:
        df_train_list = samplesFilterOR(df_train_list, *train_filter)
    if test_filter is not None:
        df_test_list = samplesFilterOR(df_test_list, *test_filter)
    return pd.DataFrame(df_train_list + df_test_list)


_DF = _GET_DF(
    _CITY_NAME,
    train_filter=[("FCNAME", "==", "VHL")]
)


# sys.exit()


def trainimdc():
    td = TimeDirectory(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods").initLog("{}_log.txt".format(_CITY_NAME))
    td.kw("CITY_NAME", _CITY_NAME)
    tic = TIC(
        name=_NAME, df=_DF, map_dict=_MAP_DICT, raster_fn=_RASTER_FN, x_keys=_X_KEYS,
        cm_names=_CM_NAMES, clf=_CLF, category_field_name="CNAME", color_table=_COLOR_TABLE, sfm=None,
        is_save_model=True, is_save_imdc=True, td=td,
        func_save_model=lambda to_mod_fn: joblib.dump(_CLF.clf, to_mod_fn),
        sfc=_SFC,
    ).train().imdc()


def main():
    pass


if __name__ == "__main__":
    trainimdc()
    r"""
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2MLMain import trainimdc; trainimdc()" qd
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2MLMain import trainimdc; trainimdc()" bj
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2MLMain import trainimdc; trainimdc()" cd


    """
