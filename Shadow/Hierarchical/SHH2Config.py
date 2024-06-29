# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2Config.py
@Time    : 2024/6/8 16:57
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2Config
-----------------------------------------------------------------------------"""
import numpy as np
import pandas as pd

from SRTCodes.GDALRasterIO import GDALRaster, saveGTIFFImdc

QD_ENVI_FN = r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\SHH2_QD2_envi.dat"
BJ_ENVI_FN = r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\SHH2_BJ2_envi.dat"
CD_ENVI_FN = r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\SHH2_CD2_envi.dat"

QD_NPY_FN = r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\SHH2_QD2_data.npy"
BJ_NPY_FN = r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\SHH2_BJ2_data.npy"
CD_NPY_FN = r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\SHH2_CD2_data.npy"

QD_LOOK_FN = r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\SHH2_QD2_look.tif"
BJ_LOOK_FN = r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\SHH2_BJ2_look.tif"
CD_LOOK_FN = r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\SHH2_CD2_look.tif"

QD_RANGE_FN = r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\SHH2_QD2_range2.json"
BJ_RANGE_FN = r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\SHH2_BJ2_range2.json"
CD_RANGE_FN = r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\SHH2_CD2_range2.json"

NAMES = [
    "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "NDVI", "NDWI",
    "OPT_asm", "OPT_con", "OPT_cor", "OPT_dis", "OPT_ent", "OPT_hom", "OPT_mean", "OPT_var",
    "AS_VV", "AS_VH", "AS_angle", "AS_VHDVV",
    "AS_C11", "AS_C12_imag", "AS_C12_real", "AS_C22", "AS_Lambda1", "AS_Lambda2", "AS_SPAN",
    "AS_Epsilon", "AS_Mu", "AS_RVI", "AS_m", "AS_Beta",
    "AS_H", "AS_A", "AS_Alpha",
    "AS_VH_asm", "AS_VH_con", "AS_VH_cor", "AS_VH_dis", "AS_VH_ent", "AS_VH_hom", "AS_VH_mean", "AS_VH_var",
    "AS_VV_asm", "AS_VV_con", "AS_VV_cor", "AS_VV_dis", "AS_VV_ent", "AS_VV_hom", "AS_VV_mean", "AS_VV_var",
    "DE_VV", "DE_VH", "DE_angle", "DE_VHDVV",
    "DE_C11", "DE_C12_imag", "DE_C12_real", "DE_C22", "DE_SPAN", "DE_Lambda1", "DE_Lambda2",
    "DE_Epsilon", "DE_Mu", "DE_RVI", "DE_m", "DE_Beta",
    "DE_H", "DE_A", "DE_Alpha",
    "DE_VH_asm", "DE_VH_con", "DE_VH_cor", "DE_VH_dis", "DE_VH_ent", "DE_VH_hom", "DE_VH_mean", "DE_VH_var",
    "DE_VV_asm", "DE_VV_con", "DE_VV_cor", "DE_VV_dis", "DE_VV_ent", "DE_VV_hom", "DE_VV_mean", "DE_VV_var",
]


def GET_RASTER_FN(city_name):
    if city_name == "qd":
        return QD_ENVI_FN
    elif city_name == "bj":
        return BJ_ENVI_FN
    elif city_name == "cd":
        return CD_ENVI_FN
    else:
        return None


def QD_GR():
    return GDALRaster(QD_ENVI_FN)


def BJ_GR():
    return GDALRaster(BJ_ENVI_FN)


def CD_GR():
    return GDALRaster(CD_ENVI_FN)


def _grReadData(gr, names, data):
    if isinstance(names, str) or isinstance(names, int):
        return gr.readGDALBand(names)
    if isinstance(names, list) or isinstance(names, tuple):
        if data is None:
            data = np.zeros((len(names), gr.n_rows, gr.n_columns))
        for i in range(len(names)):
            data[i] = gr.readGDALBand(names[i])
        return data
    return None


class SHH2ImageReadWrite:

    def __init__(self):
        self.qd_gr = None
        self.bj_gr = None
        self.cd_gr = None
        self.color_table = {
            1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255),
            5: (0, 255, 0), 6: (255, 0, 0), 7: (0, 0, 255), 8: (0, 0, 0),
        }

    def readQD(self, names=None, data=None):
        if self.qd_gr is None:
            self.qd_gr = QD_GR()
        return _grReadData(self.qd_gr, names, data)

    def readBJ(self, names=None, data=None):
        if self.bj_gr is None:
            self.bj_gr = BJ_GR()
        return _grReadData(self.bj_gr, names, data)

    def readCD(self, names=None, data=None):
        if self.cd_gr is None:
            self.cd_gr = CD_GR()
        return _grReadData(self.cd_gr, names, data)

    def read(self, city_name, names=None, data=None):
        if city_name == "qd":
            return self.readQD(names=names, data=data)
        elif city_name == "bj":
            return self.readBJ(names=names, data=data)
        elif city_name == "cd":
            return self.readCD(names=names, data=data)

    def writeQD(self, data: np.array = None, save_geo_raster_fn=None, fmt="ENVI", dtype=None, geo_transform=None,
                probing=None, start_xy=None, interleave='band', options=None, descriptions=None):
        if self.qd_gr is None:
            self.qd_gr = QD_GR()
        gr = self.qd_gr
        return gr.save(
            data, save_geo_raster_fn=save_geo_raster_fn, fmt=fmt, dtype=dtype, geo_transform=geo_transform,
            probing=probing, start_xy=start_xy, interleave=interleave, options=options, descriptions=descriptions)

    def writeBJ(self, data: np.array = None, save_geo_raster_fn=None, fmt="ENVI", dtype=None, geo_transform=None,
                probing=None, start_xy=None, interleave='band', options=None, descriptions=None):
        if self.bj_gr is None:
            self.bj_gr = BJ_GR()
        gr = self.bj_gr
        return gr.save(
            data, save_geo_raster_fn=save_geo_raster_fn, fmt=fmt, dtype=dtype, geo_transform=geo_transform,
            probing=probing, start_xy=start_xy, interleave=interleave, options=options, descriptions=descriptions)

    def writeCD(self, data: np.array = None, save_geo_raster_fn=None, fmt="ENVI", dtype=None, geo_transform=None,
                probing=None, start_xy=None, interleave='band', options=None, descriptions=None):
        if self.cd_gr is None:
            self.cd_gr = CD_GR()
        gr = self.cd_gr
        return gr.save(
            data, save_geo_raster_fn=save_geo_raster_fn, fmt=fmt, dtype=dtype, geo_transform=geo_transform,
            probing=probing, start_xy=start_xy, interleave=interleave, options=options, descriptions=descriptions)

    def write(self, city_name, data: np.array = None, save_geo_raster_fn=None, fmt="ENVI", dtype=None,
              geo_transform=None, probing=None, start_xy=None, interleave='band', options=None, descriptions=None):
        if city_name == "qd":
            return self.writeQD(
                data, save_geo_raster_fn=save_geo_raster_fn, fmt=fmt, dtype=dtype, geo_transform=geo_transform,
                probing=probing, start_xy=start_xy, interleave=interleave, options=options, descriptions=descriptions)
        elif city_name == "bj":
            return self.writeCD(
                data, save_geo_raster_fn=save_geo_raster_fn, fmt=fmt, dtype=dtype, geo_transform=geo_transform,
                probing=probing, start_xy=start_xy, interleave=interleave, options=options, descriptions=descriptions)
        elif city_name == "cd":
            return self.writeBJ(
                data, save_geo_raster_fn=save_geo_raster_fn, fmt=fmt, dtype=dtype, geo_transform=geo_transform,
                probing=probing, start_xy=start_xy, interleave=interleave, options=options, descriptions=descriptions)

    def writeImdcQD(self, data, to_fn, color_table=None):
        if color_table is None:
            color_table = self.color_table
        saveGTIFFImdc(self.qd_gr, data, to_fn=to_fn, color_table=color_table)

    def writeImdcBJ(self, data, to_fn, color_table=None):
        if color_table is None:
            color_table = self.color_table
        saveGTIFFImdc(self.bj_gr, data, to_fn=to_fn, color_table=color_table)

    def writeImdcCD(self, data, to_fn, color_table=None):
        if color_table is None:
            color_table = self.color_table
        saveGTIFFImdc(self.cd_gr, data, to_fn=to_fn, color_table=color_table)

    def writeImdc(self, city_name, data, to_fn, color_table=None):
        if city_name == "qd":
            return self.writeImdcQD(data=data, to_fn=to_fn, color_table=color_table)
        elif city_name == "bj":
            return self.writeImdcBJ(data=data, to_fn=to_fn, color_table=color_table)
        elif city_name == "CD":
            return self.writeImdcCD(data=data, to_fn=to_fn, color_table=color_table)


IMD_RW = SHH2ImageReadWrite()


class SHH2ReleaseSamples:

    def __init__(self):
        self.init_dirname = r"F:\ProjectSet\Shadow\Hierarchical\Samples\SaveSamples"
        self.current_fn = ""
        self.filelist


def samplesDescription(df):
    df_des = pd.DataFrame({
        "Training": df[df["TEST"] == 1].groupby("CNAME").count()["TEST"].to_dict(),
        "Testing": df[df["TEST"] == 0].groupby("CNAME").count()["TEST"].to_dict()
    })
    df_des[pd.isna(df_des)] = 0
    df_des["SUM"] = df_des.apply(lambda x:x.sum(),axis =1)
    df_des.loc["SUM"] = df_des.apply(lambda x:x.sum())
    return df_des


def main():
    pass


if __name__ == "__main__":
    main()
