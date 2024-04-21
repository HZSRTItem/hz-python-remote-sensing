# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHHDraw.py
@Time    : 2024/3/9 10:51
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHHDraw
-----------------------------------------------------------------------------"""
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

from SRTCodes import Utils
from SRTCodes.GDALDraw import GDALDrawImages
from SRTCodes.GDALRasterIO import GDALRasterRange, GDALRaster
from SRTCodes.NumpyUtils import reHist
from SRTCodes.Utils import printList, DirFileName
from Shadow.Hierarchical import SHHConfig
from Shadow.Hierarchical.SHHConfig import SHHFNImages


class SHHGDALDrawImages(GDALDrawImages):

    def __init__(self, win_size=None, is_min_max=True, is_01=True):
        super().__init__(win_size, is_min_max, is_01)
        self.addCategoryColor("color4", SHHConfig.SHH_COLOR4)
        self.addCategoryColor("color8", SHHConfig.SHH_COLOR8)
        self.addCategoryColor("color_vnl_8", SHHConfig.SHH_COLOR_VNL_8)
        self.addCategoryColor("color_is_8", SHHConfig.SHH_COLOR_IS_8)
        self.addCategoryColor("color_ws_8", SHHConfig.SHH_COLOR_WS_8)
        self.addCategoryColor("color_vhl_3", SHHConfig.SHH_COLOR_VNL)

    def addRCC_Im1(self):
        imfn = SHHFNImages.images1()
        qd_name = self.addGeoRange(imfn.qd_fn.fn(), imfn.qd_fn.changext(".range"))
        bj_name = self.addGeoRange(imfn.bj_fn.fn(), imfn.bj_fn.changext(".range"))
        cd_name = self.addGeoRange(imfn.cd_fn.fn(), imfn.cd_fn.changext(".range"))

        self.addRasterCenterCollection("RGB", bj_name, cd_name, qd_name, channel_list=["B4", "B3", "B2"])
        self.addRasterCenterCollection("NRG", bj_name, cd_name, qd_name, channel_list=["B8", "B4", "B3"])
        self.addRasterCenterCollection("AS_VV", bj_name, cd_name, qd_name, channel_list=["AS_VV"])
        self.addRasterCenterCollection("AS_VH", bj_name, cd_name, qd_name, channel_list=["AS_VH"])
        self.addRasterCenterCollection("DE_VV", bj_name, cd_name, qd_name, channel_list=["DE_VV"])
        self.addRasterCenterCollection("DE_VH", bj_name, cd_name, qd_name, channel_list=["DE_VH"])
        self.addRasterCenterCollection(
            "NDVI",
            imfn.qd_fn.changext("_ndvi.tif"), imfn.bj_fn.changext("_ndvi.tif"), imfn.cd_fn.changext("_ndvi.tif"),
            channel_list=["NDVI"], min_list=[-0.53], max_list=[0.76]
        )
        self.addRasterCenterCollection(
            "NDWI",
            imfn.qd_fn.changext("_ndwi.tif"), imfn.bj_fn.changext("_ndwi.tif"), imfn.cd_fn.changext("_ndwi.tif"),
            channel_list=["NDWI"], min_list=[-0.66], max_list=[0.76]
        )

        return self

    def addMLImdc(self, name, dirname):
        fns = Utils.filterFileExt(dirname, ".tif")
        self.addRasterCenterCollection(name, *fns, channel_list=[0], is_min_max=False, )


class _ColumnKey:

    def __init__(self, name, grcc_name, win_size=None, color_name=None, *args, **kwargs):
        self.name = name
        self.grcc_name = grcc_name
        self.color_name = color_name
        self.args = args
        self.kwargs = kwargs
        self.win_size = win_size


class _RowKey:

    def __init__(self, name, x, y, win_size=None, *args, **kwargs):
        self.name = name
        self.x = x
        self.y = y
        self.args = args
        self.kwargs = kwargs
        self.win_size = win_size


def catColumnRowArgs(*args, **kwargs):
    return args, kwargs


class SHHGDALDrawImagesColumn(SHHGDALDrawImages):
    """ SHHGDALDrawImagesColumn """

    def __init__(self, win_size=None, is_min_max=True, is_01=True):
        super().__init__(win_size, is_min_max, is_01)
        self.columns = []
        self.rows = []

    def addColumn(self, name, grcc_name, win_size=None, color_name=None, *args, **kwargs):
        self.columns.append(_ColumnKey(
            name, grcc_name, win_size=win_size, color_name=color_name, *args, **kwargs))
        return self.columns[-1]

    def addRow(self, name, x, y, win_size=None, *args, **kwargs):
        self.rows.append(_RowKey(name, x, y, win_size=win_size, *args, **kwargs))
        return self.rows[-1]

    def fitColumn(self, n_columns_ex=1.0, n_rows_ex=1.0, fontdict=None):
        row_names, column_names = [], []
        for column, column_key in enumerate(self.columns):
            column_key: _ColumnKey
            column_names.append(column_key.name)
            for row, row_key in enumerate(self.rows):
                if column == 0:
                    row_names.append(row_key.name)
                row_key: _RowKey
                win_size = column_key.win_size
                if row_key.win_size is not None:
                    win_size = row_key.win_size
                _args, _kwargs = catColumnRowArgs(*column_key.args, *row_key.args, **column_key.kwargs,
                                                  **row_key.kwargs)
                self.addAxisDataXY(n_row=row, n_column=column, grcc_name=column_key.grcc_name, x=row_key.x, y=row_key.y,
                                   color_name=column_key.color_name, win_size=win_size, *_args, **_kwargs)
        self.draw(n_columns_ex=n_columns_ex, n_rows_ex=n_rows_ex, row_names=row_names, column_names=column_names,
                  fontdict=fontdict)

    def addColumnRGB(self, name="RGB", win_size=None, *args, **kwargs):
        return self.addColumn(name, "RGB", win_size=win_size, *args, **kwargs)

    def addColumnNRG(self, name="NRG", win_size=None, *args, **kwargs):
        return self.addColumn(name, "NRG", win_size=win_size, *args, **kwargs)

    def addColumnASVV(self, name="AS_VV", win_size=None, *args, **kwargs):
        return self.addColumn(name, "AS_VV", win_size=win_size, *args, **kwargs)

    def addColumnASVH(self, name="AS_VH", win_size=None, *args, **kwargs):
        return self.addColumn(name, "AS_VH", win_size=win_size, *args, **kwargs)

    def addColumnDEVV(self, name="AS_VV", win_size=None, *args, **kwargs):
        return self.addColumn(name, "AS_VV", win_size=win_size, *args, **kwargs)

    def addColumnDEVH(self, name="DE_VH", win_size=None, *args, **kwargs):
        return self.addColumn(name, "DE_VH", win_size=win_size, *args, **kwargs)


def main():
    sgdic = SHHGDALDrawImagesColumn((60, 60))
    sgdic.fontdict["size"] = 16
    sgdic.addRCC_Im1()
    printList("Image Keys: ", sgdic.keys())
    ml_dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\MLMods")
    print(ml_dfn.fn(r"20240419H222504fc\qd_sh2_1_opt_sar_glcm_imdc.tif"),
          ml_dfn.fn(r"20240419H234523nofc\qd_sh2_1_opt_sar_glcm_imdc.tif"), )
    sgdic.addRasterCenterCollection(
        "FC",
        ml_dfn.fn(r"20240419H222504fc\qd_sh2_1_opt_sar_glcm_imdc.tif"),
        ml_dfn.fn(r"20240419H223557fc\bj_sh2_1_opt_sar_glcm_imdc.tif"),
        ml_dfn.fn(r"20240419H230332fc\cd_sh2_1_opt_sar_glcm_imdc.tif"),
        channel_list=[0], is_min_max=False, )
    sgdic.addRasterCenterCollection(
        "NOFC",
        ml_dfn.fn(r"20240419H234523nofc\qd_sh2_1_opt_sar_glcm_imdc.tif"),
        ml_dfn.fn(r"20240419H235038nofc\bj_sh2_1_opt_sar_glcm_imdc.tif"),
        ml_dfn.fn(r"20240420H000421nofc\cd_sh2_1_opt_sar_glcm_imdc.tif"),
        channel_list=[0], is_min_max=False, )
    column_names = ["RGB", "NRG", "AS SAR", "DE SAR", "FC", "NOFC"]
    row_names = []

    def add1(name, x, y):
        n_row = len(row_names)
        row_names.append(name)
        sgdic.addAxisDataXY(n_row, 0, "RGB", x, y, min_list=[100, 100, 100], max_list=[2000, 2000, 2000])
        sgdic.addAxisDataXY(n_row, 1, "NRG", x, y, min_list=[100, 100, 100], max_list=[3000, 2000, 2000])
        sgdic.addAxisDataXY(n_row, 2, "AS_VV", x, y)
        sgdic.addAxisDataXY(n_row, 3, "DE_VV", x, y)
        sgdic.addAxisDataXY(n_row, 4, "FC", x, y, color_name="color8")
        sgdic.addAxisDataXY(n_row, 5, "NOFC", x, y, color_name="color8")

    add1("(1)    ", 120.297934,36.175996)
    add1("(2)    ", 120.222028,36.266922)
    add1("(3)    ", 120.304915,36.249621)
    add1("(4)    ", 120.369167,36.244726)
    add1("(5)    ", 120.028982,36.271066)
    sgdic.draw(n_columns_ex=1.6, n_rows_ex=1.6, row_names=row_names, column_names=column_names)
    plt.show()

    pass


def method_name13():
    # 流程图
    sgdic = SHHGDALDrawImagesColumn((20, 20))
    sgdic.fontdict["size"] = 16
    sgdic.addRCC_Im1()
    printList("Image Keys: ", sgdic.keys())

    column_names = ["RGB", "NRG", "AS SAR", "DE SAR"]
    row_names = []

    def add1(name, x, y):
        n_row = len(row_names)
        row_names.append(name)
        sgdic.addAxisDataXY(n_row, 0, "RGB", x, y, min_list=[100, 100, 100], max_list=[2000, 2000, 2000])
        sgdic.addAxisDataXY(n_row, 1, "NRG", x, y, min_list=[100, 100, 100], max_list=[3000, 2000, 2000])
        sgdic.addAxisDataXY(n_row, 2, "AS_VV", x, y)
        sgdic.addAxisDataXY(n_row, 3, "DE_VV", x, y)

    add1("(1)    ", 120.3848296, 36.1100934)
    add1("(2)    ", 120.4051095, 36.1039619)
    sgdic.draw(n_columns_ex=2, n_rows_ex=2, row_names=row_names, column_names=column_names)
    plt.show()


def method_name12():
    # STEP1 结果 2024年4月7日19:30:01 组会
    sgdic = SHHGDALDrawImagesColumn((40, 40))
    sgdic.fontdict["size"] = 16
    sgdic.addRCC_Im1()
    city_name = "qd"
    if city_name == "bj":
        sgdic.addRasterCenterCollection(
            "Transformer",
            r"F:\ProjectSet\Shadow\Hierarchical\VHLModels\20240407H133608\VHLModel_VIT1_epoch98_imdc1.tif",
            channel_list=[0], is_min_max=False, )
        sgdic.addRasterCenterCollection(
            "RF", r"F:\ProjectSet\Shadow\Hierarchical\VHLModels\20240406H195636\BJ_VHLML_imdc.tif",
            channel_list=[0], is_min_max=False, )
    elif city_name == "qd":
        sgdic.addRasterCenterCollection(
            "Transformer",
            r"F:\ProjectSet\Shadow\Hierarchical\VHLModels\20240406H192733\VHLModel_VIT1_epoch90_imdc1.tif",
            channel_list=[0], is_min_max=False, )
        sgdic.addRasterCenterCollection(
            "RF", r"F:\ProjectSet\Shadow\Hierarchical\VHLModels\20240405H205015\QD_VHLML_imdc.tif",
            channel_list=[0], is_min_max=False, )
    printList("Image Keys: ", sgdic.keys())
    column_names = ["RGB", "NRG", "AS SAR", "DE SAR", "Transformer", "RF"]
    row_names = []

    def add1(name, x, y):
        n_row = len(row_names)
        row_names.append(name)
        sgdic.addAxisDataXY(n_row, 0, "RGB", x, y, min_list=[100, 100, 100], max_list=[2000, 2000, 2000])
        sgdic.addAxisDataXY(n_row, 1, "NRG", x, y, min_list=[100, 100, 100], max_list=[3000, 2000, 2000])
        sgdic.addAxisDataXY(n_row, 2, "AS_VV", x, y)
        sgdic.addAxisDataXY(n_row, 3, "DE_VV", x, y)
        sgdic.addAxisDataXY(n_row, 4, "Transformer", x, y, color_name="color_vhl_3")
        sgdic.addAxisDataXY(n_row, 5, "RF", x, y, color_name="color_vhl_3")

    if city_name == "bj":
        add1("(1)    ", 116.427091, 40.010610)
        # sgdic.addEllipse((8, 8), 6, 6, n_row=0, angle=0, linewidth=1.5, fill=False, zorder=2, edgecolor="lime", )
        add1("(2)    ", 116.458256, 40.009214)
        # sgdic.addEllipse((12, 9), 6, 6, n_row=1, angle=0, linewidth=1.5, fill=False, zorder=2, edgecolor="red", )
        add1("(3)    ", 116.824781, 39.800360)
        # sgdic.addEllipse((10, 12), 6, 6, n_row=2, angle=0, linewidth=1.5, fill=False, zorder=2, edgecolor="red", )
        add1("(4)    ", 116.7622717, 39.8056862)
        # sgdic.addEllipse((9, 9), 5, 8, n_row=3, angle=0, linewidth=1.5, fill=False, zorder=2, edgecolor="red", )
        # add1("(1)    ", 120.372189,36.067929)

    elif city_name == "qd":
        add1("(1)    ", 120.203040, 36.220417)
        # sgdic.addEllipse((8, 8), 6, 6, n_row=0, angle=0, linewidth=1.5, fill=False, zorder=2, edgecolor="lime", )
        add1("(2)    ", 120.3201389, 36.3350312)
        # sgdic.addEllipse((12, 9), 6, 6, n_row=1, angle=0, linewidth=1.5, fill=False, zorder=2, edgecolor="red", )
        add1("(3)    ", 120.298611, 36.170982)
        # sgdic.addEllipse((10, 12), 6, 6, n_row=2, angle=0, linewidth=1.5, fill=False, zorder=2, edgecolor="red", )
        add1("(4)    ", 120.4097894, 36.0690048)
        # sgdic.addEllipse((9, 9), 5, 8, n_row=3, angle=0, linewidth=1.5, fill=False, zorder=2, edgecolor="red", )
        # add1("(1)    ", 120.372189,36.067929)
    sgdic.draw(n_columns_ex=2, n_rows_ex=2, row_names=row_names, column_names=column_names)
    plt.show()


def method_name11():
    # WSModel 结果
    sgdic = SHHGDALDrawImagesColumn((100, 100))
    sgdic.fontdict["size"] = 16
    sgdic.addRCC_Im1()
    sgdic.addRasterCenterCollection(
        "DL", r"F:\ProjectSet\Shadow\Hierarchical\WSModels\20240330H114435\VIT_WS_T1_epoch90_imdc1.tif"
        , channel_list=[0], is_min_max=False, )
    printList("Image Keys: ", sgdic.keys())
    column_names = ["RGB", "NRG", "AS SAR", "DE SAR", "Imdc"]
    row_names = []

    def add1(name, x, y):
        n_row = len(row_names)
        row_names.append(name)
        sgdic.addAxisDataXY(n_row, 0, "RGB", x, y, min_list=[100, 100, 100], max_list=[2000, 2000, 2000])
        sgdic.addAxisDataXY(n_row, 1, "NRG", x, y, min_list=[100, 100, 100], max_list=[3000, 2000, 2000])
        sgdic.addAxisDataXY(n_row, 2, "AS_VV", x, y)
        sgdic.addAxisDataXY(n_row, 3, "DE_VV", x, y)
        sgdic.addAxisDataXY(n_row, 4, "DL", x, y, color_name="color_ws_8")

    add1("(1)    ", 120.379059, 36.067248)
    # sgdic.addEllipse((8, 8), 6, 6, n_row=0, angle=0, linewidth=1.5, fill=False, zorder=2, edgecolor="lime", )
    add1("(2)    ", 120.246732, 36.263659)
    # sgdic.addEllipse((12, 9), 6, 6, n_row=1, angle=0, linewidth=1.5, fill=False, zorder=2, edgecolor="red", )
    add1("(3)    ", 120.328479, 36.283658)
    # sgdic.addEllipse((10, 12), 6, 6, n_row=2, angle=0, linewidth=1.5, fill=False, zorder=2, edgecolor="red", )
    add1("(4)    ", 120.461028, 36.210270)
    # sgdic.addEllipse((9, 9), 5, 8, n_row=3, angle=0, linewidth=1.5, fill=False, zorder=2, edgecolor="red", )
    # add1("(1)    ", 120.372189,36.067929)
    sgdic.draw(n_columns_ex=2, n_rows_ex=2, row_names=row_names, column_names=column_names)
    plt.show()


def method_name10():
    # 阴影下数据范围改变之后看看阴影下的光学特征
    sgdic = SHHGDALDrawImagesColumn((3, 3))
    sgdic.fontdict["size"] = 16
    sgdic.addRCC_Im1()
    sgdic.addMLImdc("FC", r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240308H210341nofc")
    printList("Image Keys: ", sgdic.keys())
    column_names = ["RGB[100,3000]", "RGB[100,500]", "NRG[100,500]", "NRG[100,500]", "AS SAR", "DE SAR"]
    row_names = []

    def add1(name, x, y):
        n_row = len(row_names)
        row_names.append(name)
        sgdic.addAxisDataXY(n_row, 0, "RGB", x, y)
        sgdic.addAxisDataXY(n_row, 1, "RGB", x, y, min_list=[100, 100, 100], max_list=[500, 500, 500])
        sgdic.addAxisDataXY(n_row, 2, "NRG", x, y)
        sgdic.addAxisDataXY(n_row, 3, "NRG", x, y, min_list=[100, 100, 100], max_list=[700, 500, 500])
        sgdic.addAxisDataXY(n_row, 4, "AS_VV", x, y, )
        sgdic.addAxisDataXY(n_row, 5, "DE_VV", x, y, )

    # add1("(1)    ", 120.3959076,36.1139105)
    # sgdic.addEllipse((8, 8), 6, 6, n_row=0, angle=0, linewidth=1.5, fill=False, zorder=2, edgecolor="lime", )
    #
    # add1("(2)    ", 104.1408601,30.6022039)
    # sgdic.addEllipse((12, 9), 6, 6, n_row=1, angle=0, linewidth=1.5, fill=False, zorder=2, edgecolor="red", )
    #
    # add1("(3)    ", 120.3358679,36.1221974)
    # sgdic.addEllipse((10, 12), 6, 6, n_row=2, angle=0, linewidth=1.5, fill=False, zorder=2, edgecolor="red", )
    #
    # add1("(4)    ", 104.0790883,30.6216332)
    # sgdic.addEllipse((9, 9), 5, 8, n_row=3, angle=0, linewidth=1.5, fill=False, zorder=2, edgecolor="red", )
    add1("(1)    ", 104.0534023, 30.6078706)
    sgdic.draw(n_columns_ex=2, n_rows_ex=2, row_names=row_names, column_names=column_names)
    plt.show()


def method_name9():
    sgdic = SHHGDALDrawImagesColumn((31, 31))
    sgdic.fontdict["size"] = 12
    sgdic.addRCC_Im1()
    sgdic.addRasterCenterCollection(
        "DL", r"F:\ProjectSet\Shadow\Hierarchical\Mods\20240315H221134\model_epoch_50_qd_imdc.tif"
        , channel_list=[0], is_min_max=False, )
    printList("Image Keys: ", sgdic.keys())
    column_names = ["RGB", "NRG", "AS SAR", "DE SAR", "DL"]
    row_names = []

    def add1(name, x, y):
        n_row = len(row_names)
        row_names.append(name)
        sgdic.addAxisDataXY(n_row, 0, "RGB", x, y, min_list=[100, 100, 100], max_list=[2000, 2000, 2000])
        sgdic.addAxisDataXY(n_row, 1, "NRG", x, y, min_list=[100, 100, 100], max_list=[3000, 2000, 2000])
        sgdic.addAxisDataXY(n_row, 2, "AS_VV", x, y)
        sgdic.addAxisDataXY(n_row, 3, "DE_VV", x, y)
        sgdic.addAxisDataXY(n_row, 4, "DL", x, y, color_name="color_is_8")

    # SOIL
    # add1("City(1)        ", 120.430321, 36.133210)
    # add1("City(2)        ", 120.411755, 36.121639)
    # add1("Village(1)            ", 120.147212, 36.295521)
    # add1("Village(2)            ", 120.263935, 36.307509)
    # IS
    add1("City(1)        ", 120.411226, 36.126965)
    add1("City(2)        ", 120.356780, 36.080849)
    add1("Village(1)            ", 120.183232, 36.286523)
    add1("Village(2)            ", 120.361619, 36.349536)
    sgdic.draw(n_columns_ex=2, n_rows_ex=2, row_names=row_names, column_names=column_names)
    plt.show()


def method_name8():
    # 【Soil & Impervious Surface】分类图
    sgdic = SHHGDALDrawImagesColumn((31, 31))
    sgdic.fontdict["size"] = 16
    sgdic.addRCC_Im1()
    sgdic.addMLImdc("FC", r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240308H215552fc")
    printList("Image Keys: ", sgdic.keys())
    column_names = ["RGB", "NRG", "AS SAR", "DE SAR", "FC"]
    row_names = []

    def add1(name, x, y):
        n_row = len(row_names)
        row_names.append(name)
        sgdic.addAxisDataXY(n_row, 0, "RGB", x, y, min_list=[100, 100, 100], max_list=[2000, 2000, 2000])
        sgdic.addAxisDataXY(n_row, 1, "NRG", x, y, min_list=[100, 100, 100], max_list=[3000, 2000, 2000])
        sgdic.addAxisDataXY(n_row, 2, "AS_VV", x, y)
        sgdic.addAxisDataXY(n_row, 3, "DE_VV", x, y)
        sgdic.addAxisDataXY(n_row, 4, "FC", x, y, color_name="color_is_8")

    add1("IS      ", 120.387223, 36.099332)
    add1("IS      ", 120.450069, 36.128606)
    add1("Soil      ", 120.286270, 36.046728)
    add1("IS_SH        ", 120.3761108, 36.0946242)
    sgdic.addEllipse((16, 15), 6, 6, n_row=3, angle=0, linewidth=1.5, fill=False, zorder=2, edgecolor="green", )
    sgdic.draw(n_columns_ex=2, n_rows_ex=2, row_names=row_names, column_names=column_names)
    plt.show()


def method_name7():
    # 【Soil & Impervious Surface】图像分析
    sgdic = SHHGDALDrawImagesColumn((31, 31))
    sgdic.fontdict["size"] = 16
    sgdic.addRCC_Im1()
    # sgdic.addMLImdc("FC", r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240308H215552fc")
    printList("Image Keys: ", sgdic.keys())
    column_names = ["RGB", "NRG", "AS SAR", "DE SAR"]
    row_names = []

    def add1(name, x, y):
        n_row = len(row_names)
        row_names.append(name)
        sgdic.addAxisDataXY(n_row, 0, "RGB", x, y, min_list=[100, 100, 100], max_list=[2000, 2000, 2000])
        sgdic.addAxisDataXY(n_row, 1, "NRG", x, y, min_list=[100, 100, 100], max_list=[3000, 2000, 2000])
        sgdic.addAxisDataXY(n_row, 2, "AS_VV", x, y)
        sgdic.addAxisDataXY(n_row, 3, "DE_VV", x, y)

    add1("IS      ", 120.387223, 36.099332)
    add1("IS      ", 120.450069, 36.128606)
    add1("Soil      ", 120.286270, 36.046728)
    add1("Soil      ", 120.058122, 36.376711)
    # sgdic.addEllipse((9, 22), 6, 6, n_row=0, angle=0, linewidth=1.5, fill=False, zorder=2, edgecolor="red", )
    sgdic.draw(n_columns_ex=2, n_rows_ex=2, row_names=row_names, column_names=column_names)
    plt.show()


def method_name6():
    # step1
    sgdic = SHHGDALDrawImagesColumn((31, 31))
    sgdic.fontdict["size"] = 16
    sgdic.addRCC_Im1()
    sgdic.addMLImdc("FC", r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240308H215552fc")
    printList("Image Keys: ", sgdic.keys())
    column_names = ["RGB", "NRG", "FC"]
    row_names = []

    def add1(name, x, y):
        n_row = len(row_names)
        row_names.append(name)
        sgdic.addAxisDataXY(n_row, 0, "RGB", x, y, min_list=[100, 100, 100], max_list=[2000, 2000, 2000])
        sgdic.addAxisDataXY(n_row, 1, "NRG", x, y, min_list=[100, 100, 100], max_list=[3000, 2000, 2000])
        sgdic.addAxisDataXY(n_row, 2, "FC", x, y, color_name="color_vnl_8")

    # add1("(1)    ", 120.352316,36.093032)
    # sgdic.addEllipse((9, 22), 6, 6, n_row=0, angle=0, linewidth=1.5, fill=False, zorder=2, edgecolor="red", )
    # add1("(2)    ", 120.079303, 36.300055)
    add1("(3)    ", 120.323208, 36.112238)
    add1("(4)    ", 120.081612, 36.140330)
    sgdic.draw(n_columns_ex=2, n_rows_ex=2, row_names=row_names, column_names=column_names)
    plt.show()


def method_name5():
    # 植被光学和SAR特征的比较
    sgdic = SHHGDALDrawImagesColumn((15, 15))
    sgdic.fontdict["size"] = 16
    sgdic.addRCC_Im1()
    printList("Image Keys: ", sgdic.keys())
    column_names = ["RGB", "NRG", "AS SAR", "DE SAR"]
    row_names = []

    def add1(name, x, y):
        n_row = len(row_names)
        row_names.append(name)
        sgdic.addAxisDataXY(n_row, 0, "RGB", x, y, min_list=[100, 100, 100], max_list=[2000, 2000, 2000])
        sgdic.addAxisDataXY(n_row, 1, "NRG", x, y, min_list=[100, 100, 100], max_list=[3000, 2000, 2000])
        sgdic.addAxisDataXY(n_row, 2, "AS_VV", x, y)
        sgdic.addAxisDataXY(n_row, 3, "DE_VV", x, y)

    add1("(1)    ", 120.398953, 36.087511)
    add1("(2)    ", 120.404530, 36.073103)
    sgdic.addEllipse((7, 7), 6, 6, n_row=1, angle=0, linewidth=1.5, fill=False, zorder=2, edgecolor="lime", )
    sgdic.draw(n_columns_ex=2, n_rows_ex=2, row_names=row_names, column_names=column_names)
    plt.show()


def method_name4():
    # 阴影下数据范围改变之后看看阴影下的光学特征
    sgdic = SHHGDALDrawImagesColumn((21, 21))
    sgdic.fontdict["size"] = 16
    sgdic.addRCC_Im1()
    sgdic.addMLImdc("FC", r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240308H210341nofc")
    printList("Image Keys: ", sgdic.keys())
    column_names = ["RGB[100,3000]", "RGB[100,500]", "NRG[100,500]", "NRG[100,500]", "FC"]
    row_names = []

    def add1(name, x, y):
        n_row = len(row_names)
        row_names.append(name)
        sgdic.addAxisDataXY(n_row, 0, "RGB", x, y)
        sgdic.addAxisDataXY(n_row, 1, "RGB", x, y, min_list=[100, 100, 100], max_list=[500, 500, 500])
        sgdic.addAxisDataXY(n_row, 2, "NRG", x, y)
        sgdic.addAxisDataXY(n_row, 3, "NRG", x, y, min_list=[100, 100, 100], max_list=[700, 500, 500])
        sgdic.addAxisDataXY(n_row, 4, "FC", x, y, color_name="color8")

    # add1("(1)    ", 120.3959076,36.1139105)
    # sgdic.addEllipse((8, 8), 6, 6, n_row=0, angle=0, linewidth=1.5, fill=False, zorder=2, edgecolor="lime", )
    #
    # add1("(2)    ", 104.1408601,30.6022039)
    # sgdic.addEllipse((12, 9), 6, 6, n_row=1, angle=0, linewidth=1.5, fill=False, zorder=2, edgecolor="red", )
    #
    # add1("(3)    ", 120.3358679,36.1221974)
    # sgdic.addEllipse((10, 12), 6, 6, n_row=2, angle=0, linewidth=1.5, fill=False, zorder=2, edgecolor="red", )
    #
    # add1("(4)    ", 104.0790883,30.6216332)
    # sgdic.addEllipse((9, 9), 5, 8, n_row=3, angle=0, linewidth=1.5, fill=False, zorder=2, edgecolor="red", )
    add1("(1)    ", 104.0534023, 30.6078706)
    sgdic.draw(n_columns_ex=2, n_rows_ex=2, row_names=row_names, column_names=column_names)
    plt.show()


def method_name3():
    sgdic = SHHGDALDrawImagesColumn((21, 21))
    sgdic.addRCC_Im1()
    sgdic.addMLImdc("FC", r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240308H210341nofc")
    printList("Image Keys: ", sgdic.keys())
    sgdic.addColumnRGB()
    sgdic.addColumnNRG()
    # sgdic.addColumnASVV()
    # sgdic.addColumnASVH()
    sgdic.addColumn("FC", "FC", color_name="color8")
    sgdic.addRow("(1)", 120.3704180, 36.0695225)
    # sgdic.addRow("(2)", 120.3709133,36.0705602)
    # sgdic.addRow("(3)", 120.364846,36.058329)
    sgdic.fitColumn(n_columns_ex=5, n_rows_ex=5)
    plt.show()


def method_name2():
    sgdi = SHHGDALDrawImages((100, 100))
    sgdi.addRCC_Im1()
    sgdi.addMLImdc("FC", r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240308H210341nofc")
    printList("Image Keys: ", sgdi.keys())
    sgdi.addAxisDataXY(0, 0, "RGB", 120.330806, 36.135239)
    sgdi.addAxisDataXY(0, 1, "NRG", 120.330806, 36.135239)
    sgdi.addAxisDataXY(1, 0, "AS_VV", 120.330806, 36.135239)
    sgdi.addAxisDataXY(0, 2, "FC", 120.330806, 36.135239, color_name="color4")
    sgdi.addAxisDataXY(1, 1, "FC", 104.07385, 30.65005, color_name="color8")
    sgdi.addAxisDataXY(2, 0, "NDVI", 120.330806, 36.135239)
    sgdi.draw(n_rows_ex=2, n_columns_ex=2)
    plt.show()


def method_name1():
    imfn = SHHFNImages.images1()
    for fn in imfn.iterFN():
        print(fn.fn())

        def func1():
            gr = GDALRaster(fn.fn())
            d = gr.readAsArray()
            out_d = reHist(d, 0.02)
            np.save(fn.changext("_range.npy"), out_d)

        def func2():
            grr = GDALRasterRange(fn.fn())
            grr.loadNPY(fn.changext("_range.npy"))
            grr.save()

        def func3():
            gr = GDALRaster(fn.fn())
            d = gr.readAsArray()
            ndvi = (d[9] - d[8]) / (d[9] + d[8] + 0.00001)
            ndwi = (d[7] - d[9]) / (d[7] + d[9] + 0.00001)
            gr.save(ndvi.astype("float32"), fn.changext("_ndvi.tif"), fmt="GTiff", descriptions=["NDVI"],
                    dtype=gdal.GDT_Float32)
            gr.save(ndwi.astype("float32"), fn.changext("_ndwi.tif"), fmt="GTiff", descriptions=["NDWI"],
                    dtype=gdal.GDT_Float32)

        func3()


if __name__ == "__main__":
    main()
