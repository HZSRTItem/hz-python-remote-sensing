# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2MLMain.py
@Time    : 2024/7/3 12:21
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2MLMain
-----------------------------------------------------------------------------"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from SRTCodes.SRTTimeDirectory import TimeDirectory
from Shadow.Hierarchical import SHH2Config
from Shadow.Hierarchical.SHH2ML2 import TIC


def trainimdc(city_name="qd"):

    if city_name == "qd":
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\qd\sh2_spl30_qd1.csv"
    elif city_name == "bj":
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\bj\sh2_spl30_bj1.csv"
    elif city_name == "cd":
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\cd\sh2_spl30_cd1.csv"
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\cd\sh2_spl30_cd2_spl.csv"
    else:
        raise Exception("City name \"{}\"".format(city_name))

    if city_name == "qd":
        raster_fn = SHH2Config.QD_ENVI_FN
    elif city_name == "cd":
        raster_fn = SHH2Config.CD_ENVI_FN
    elif city_name == "bj":
        raster_fn = SHH2Config.BJ_ENVI_FN
    else:
        raise Exception("City name \"{}\"".format(city_name))

    td = TimeDirectory(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods").initLog("{}_log.txt".format(city_name))
    td.kw("CITY_NAME", city_name)
    tic = TIC(
        name="VHL3-Opt",
        df=pd.read_csv(csv_fn),
        map_dict={
            "IS": 1,
            # "VEG": 2,
            "SOIL": 2,
            # "WAT": 3,
            # "IS_SH": 3, "VEG_SH": 3, "SOIL_SH": 3, "WAT_SH": 3
        },
        raster_fn=raster_fn,
        x_keys=[
            # Opt
            "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "NDVI", "NDWI",
            # Opt glcm
            "OPT_asm", "OPT_con", "OPT_cor", "OPT_dis", "OPT_ent", "OPT_hom", "OPT_mean", "OPT_var",
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
        ],
        cm_names=["IS", "SOIL"],
        clf=RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=1, min_samples_split=2),

        category_field_name="CNAME",
        color_table = {1: (255, 0, 0), 2: (255, 255, 0), 3: (0, 0, 0), 4: (0, 0, 0), },
        sfm=None,
        is_save_model=True,
        is_save_imdc=True,
        td = td,
    ).initTD().train().imdc()


def main():
    pass


if __name__ == "__main__":
    # trainimdc("qd")
    # trainimdc("bj")
    trainimdc("cd")
    r"""
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2MLMain import trainimdc; trainimdc('qd')"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2MLMain import trainimdc; trainimdc('bj')"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2MLMain import trainimdc; trainimdc('cd')"


    """
