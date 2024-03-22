# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowGeoDraw.py
@Time    : 2023/7/15 16:10
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : GEOCodes of ShadowGeoDraw
-----------------------------------------------------------------------------"""

import os.path
from inspect import isfunction

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.colors import ListedColormap
from osgeo import gdal

from SRTCodes.GDALRasterIO import GDALRasterRange, GDALMBTiles, GDALRaster
from SRTCodes.GDALUtils import GDALRasterCenter, readGDALRasterCenter
from SRTCodes.GeoRasterRW import GeoRasterWrite
from SRTCodes.SRTDraw import MplPatchesEllipse, MplPatchesEllipseColl
from SRTCodes.SRTFeature import SRTFeatureCallBackScaleMinMax as SFCBSM
from SRTCodes.Utils import saveJson, readcsv, changext, filterFileContain, sdf_read_csv, Jdt, printList


def isCloseInt(d, eps=0.000001):
    d_int = int(d)
    return abs(d - d_int) < eps


def toMinMax(x):
    d_min = np.expand_dims(np.min(x, axis=(0, 1)), axis=(0, 1))
    d_max = np.expand_dims(np.max(x, axis=(0, 1)), axis=(0, 1))
    x = (x - d_min) / (d_max - d_min)
    return x


class GDALResterCenterDraw(GDALRasterCenter):
    """ GDAL Rester Center Draw """

    def __init__(self, name, raster_fn, channel_list=None):
        self.raster_fn = raster_fn
        channel_list, raster_fn = self.initGeoRaster(channel_list, name, raster_fn)
        super().__init__(name, self.GDALDatasetCollection[raster_fn], channel_list)

        self.d_min = None
        self.d_max = None

    def toImage(self, fn, width=0.0, height=0.0, is_expand=False):
        fn = changext(fn, ".jpg")
        self.d: np.ndarray
        if self.c_d is None:
            rgb_arr = np.ones([self.d.shape[0], self.d.shape[1], 3])
            if self.d.shape[2] == 1:
                for i in range(3):
                    rgb_arr[:, :, i] = self.d[:, :, 0]
            else:
                for i in range(3):
                    rgb_arr[:, :, i] = self.d[:, :, i]
            rgb_arr = toMinMax(rgb_arr)
            rgb_arr = rgb_arr * 255
            rgb_arr = rgb_arr.astype("uint8")
        else:
            rgb_arr = self.d
        im = Image.fromarray(rgb_arr, mode="RGB")
        if is_expand:
            if isCloseInt(width) and isCloseInt(height):
                width = int(width)
                height = int(height)
                rgb_arr_emp = np.zeros([self.d.shape[0] * int(height), self.d.shape[1] * int(width), 3])
                for k in range(3):
                    for i in range(rgb_arr.shape[0]):
                        for j in range(rgb_arr.shape[1]):
                            rgb_arr_emp[height * i:height * (i + 1), width * j:width * (j + 1), k] = rgb_arr[i, j, k]
                rgb_arr_emp = rgb_arr_emp.astype("uint8")
                im = Image.fromarray(rgb_arr_emp, mode="RGB")
            else:
                width = int(im.width * width)
                height = int(im.height * height)
        else:
            if 0 < width < 1 and 0 < height < 1:
                width = int(im.width * width)
                height = int(im.height * height)
            else:
                width, height = int(width), int(height)
        if width != 0 and height != 0:
            if is_expand:
                if isCloseInt(width) and isCloseInt(height):
                    out_im = im
                else:
                    out_im = im.resize((width, height), Image.BILINEAR)
            else:
                out_im = im.resize((width, height), Image.BILINEAR)
            out_im.save(fn, "JPEG", dpi=(300, 300))
        else:
            im.save(fn, "JPEG", dpi=(300, 300))

    def toImageJpg1(self, fn, height, is_expand, width):
        fn = changext(fn, ".jpg")
        self.d: np.ndarray
        if self.c_d is None:
            rgb_arr = np.ones([self.d.shape[0], self.d.shape[1], 3])
            if self.d.shape[2] == 1:
                for i in range(3):
                    rgb_arr[:, :, i] = self.d[:, :, 0]
            else:
                for i in range(3):
                    rgb_arr[:, :, i] = self.d[:, :, i]
            rgb_arr = toMinMax(rgb_arr)
            rgb_arr = rgb_arr * 255
            rgb_arr = rgb_arr.astype("uint8")
        else:
            rgb_arr = self.d
        im = Image.fromarray(rgb_arr, mode="RGB")
        if is_expand:
            width = int(im.width * width)
            height = int(im.height * height)
        else:
            if 0 < width < 1 and 0 < height < 1:
                width = int(im.width * width)
                height = int(im.height * height)
            else:
                width, height = int(width), int(height)
        if width != 0 and height != 0:
            out_im = im.resize((width, height), Image.BILINEAR)
            out_im.save(fn, "JPEG", dpi=(300, 300))
        else:
            im.save(fn, "JPEG", dpi=(300, 300))

    def readWhole(self):
        self.d = self.ds.ReadAsArray()


class ShadowGeoDraw:

    def __init__(self, name, x_row, y_column, win_row_size=0, win_column_size=0, is_geo=True, no_data=0):
        self.name = name
        self.x_row = x_row
        self.y_column = y_column
        self.win_row_size = win_row_size
        self.win_column_size = win_column_size
        self.no_data = no_data
        self.is_geo = is_geo

        self.raster_centers = {}
        self.ds_dict = {}

    def addRasterCenter(self, name, raster_fn, channel_list=None, save_dict=None):
        if name in raster_fn:
            raise Exception("Name \"{0}\" have in raster centers.".format(name))
        if channel_list is None:
            channel_list = [name]
        if save_dict is None:
            save_dict = {}
        raster_fn = os.path.abspath(raster_fn)
        if raster_fn not in self.ds_dict:
            self.ds_dict[raster_fn] = gdal.Open(raster_fn)
        self.raster_centers[name] = GDALRasterCenter(name=name, ds=self.ds_dict[raster_fn], channel_list=channel_list,
                                                     save_dict=save_dict)

    def fit(self, save_dir):
        save_dict = {
            "Name": self.name,
            "X Row": self.x_row,
            "Y Column": self.y_column,
            "Row Number": self.win_row_size,
            "Column Number": self.win_column_size,
            "Is Geo": True,
            "No Data": 0,
            "Npy File Name": os.path.join(save_dir, self.name) + ".npy",
            "Draw": []
        }

        i_data = 0
        d_list = []
        for name in self.raster_centers:
            r_c = self.raster_centers[name]
            d = r_c.read(x_row=self.x_row,
                         y_column=self.y_column,
                         win_row_size=self.win_row_size,
                         win_column_size=self.win_column_size,
                         is_geo=self.is_geo,
                         no_data=self.no_data)
            d_list.append(d)
            save_d = {"Name": r_c.name, "Raster File Name": r_c.raster_fn, "Data": []}
            for i in range(d.shape[2]):
                save_d["Data"].append(i_data)
                i_data += 1
            save_dict["Draw"].append(save_d)

        d_arr = np.concatenate(d_list, axis=2)
        np.save(save_dict["Npy File Name"], d_arr)
        saveJson(save_dict, os.path.join(save_dir, self.name) + ".json")
        print(d_arr.shape)
        print(save_dict)


def _10log10(x, *args, **kwargs):
    return 10 * np.log10(x)


def _power10(x, *args, **kwargs):
    return np.power(10.0, x / 10.0)


def drawShadowImage_Optical(name, x, y, rows, columns, raster_fn, to_fn, d_min=0, d_max=3500, channel_list=None,
                            width=0.0, height=0.0, is_expand=False):
    grcd = GDALResterCenterDraw(name=name, raster_fn=raster_fn, channel_list=channel_list)
    grcd.read(x, y, rows, columns)

    grcd.scaleMinMax(d_min, d_max)
    to_fn = os.path.join(to_fn, name + ".png")
    grcd.toImage(to_fn, width=width, height=height, is_expand=is_expand)


def drawShadowImage_SAR(name, x, y, rows, columns, raster_fn, to_fn, d_min=-60.0, d_max=60.0, channel_list=None,
                        width=0.0, height=0.0, is_expand=False):
    grcd = GDALResterCenterDraw(name=name, raster_fn=raster_fn, channel_list=channel_list)
    grcd.read(x, y, rows, columns)
    grcd.callBackFunc(_10log10)
    grcd.scaleMinMax(d_min, d_max)
    to_fn = os.path.join(to_fn, name + ".png")
    grcd.toImage(to_fn, width=width, height=height, is_expand=is_expand)


def drawShadowImage_Imdc(name, x, y, rows, columns, raster_fn, to_fn, categorys,
                         width=0.0, height=0.0, is_expand=False):
    grcd = GDALResterCenterDraw(name=name, raster_fn=raster_fn, channel_list=[0])
    grcd.read(x, y, rows, columns)
    grcd.toCategory()
    for i in range(0, len(categorys) - 1, 2):
        grcd.categoryColor(categorys[i], categorys[i + 1])
    to_fn = os.path.join(to_fn, name + ".png")
    grcd.toImage(to_fn, width=width, height=height, is_expand=is_expand)


def drawShadowImage_Index(name, x, y, rows, columns, raster_fn, to_fn, d_min=-60.0, d_max=60.0,
                          width=0.0, height=0.0, is_expand=False):
    grcd = GDALResterCenterDraw(name=name, raster_fn=raster_fn, channel_list=[name])
    grcd.read(x, y, rows, columns)
    grcd.scaleMinMax(d_min, d_max)
    to_fn = os.path.join(to_fn, name + ".png")
    grcd.toImage(to_fn, width=width, height=height, is_expand=is_expand)


def drawShadowImageSingle(name, raster_fn, to_fn=None, to_dir=None, x=0.0, y=0.0, rows=0, columns=0, channel=None,
                          callbacks=None, width=0.0, height=0.0, is_geo=True):
    if callbacks is None:
        callbacks = []
    if channel is not None:
        if isinstance(channel, int) or isinstance(channel, str):
            channel = [channel]
    grcd = GDALResterCenterDraw(name=name, raster_fn=raster_fn, channel_list=channel)
    grcd.read(x, y, rows, columns, is_geo=is_geo)
    for callback in callbacks:
        if isfunction(callback):
            grcd.callBackFunc(callback)
        else:
            grcd.callBack(callback)
    if to_dir is not None:
        to_fn = os.path.join(to_dir, name + ".jpg")
    else:
        to_fn = changext(to_fn, ".jpg")
    grcd.toImage(fn=to_fn, width=width, height=height)
    print(to_fn)


def method_name2():
    # is_show = True
    # if is_show:
    #     showcsv(r"F:\ProjectSet\Shadow\QingDao\mktu\featuredingxing\qd_feat_1.csv")
    #     return
    """
        120.2987326, 36.06256204
        120.3700395, 36.06311117
        120.376254, 36.08912129
        120.4064965, 36.11689935
        120.3309573, 36.08681761
        """
    name = "QingDao_2"
    x, y, rows, columns = 120.366316, 36.057372, 50, 50
    categorys = [1, (255, 0, 0), 2, (0, 255, 0), 3, (255, 255, 0), 4, (0, 0, 255)]
    to_fn = os.path.join(r"F:\ProjectSet\Shadow\QingDao\mktu\featuredingxing", name)
    if not os.path.isdir(to_fn):
        os.mkdir(to_fn)
    qd_stack_fn = r"F:\ProjectSet\Shadow\QingDao\Image\stack2.vrt"
    imd_fn1 = r"F:\ProjectSet\Shadow\QingDao\Mods\20230707H200910\SPL_NOSH" \
              r"-RF-TAG-OPTICS-AS_SIGMA-AS_C2-AS_LAMD-DE_SIGMA-DE_C2_imdc.dat"


def method_name1():
    # drawShadowImage_1(columns, qd_stack_fn, rows, to_fn, x, y)
    #
    # d = {
    #     "Name": name, "x": x, "y": y, "rows": rows, "columns": columns,
    #     "categorys": categorys
    # }
    # saveJson(d, os.path.join(to_fn, name + ".json"))

    #
    # drawShadowImage_Imdc("IMDC_1", x, y, rows, columns, imd_fn1,
    #                      to_fn=to_fn,
    #                      categorys=categorys)

    raster_fn = r"G:\ImageData\QingDao\20211023\qd20211023\Temp\qd_im1.vrt"
    to_dir = r"F:\ProjectSet\Shadow\QingDao\Dissertation\4_Analyze\mktu\1"
    width = 0.5
    height = 0.5
    drawShadowImageSingle("NDVI", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[SFCBSM(-0.6, 0.9, True, True)])
    drawShadowImageSingle("NDWI", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[SFCBSM(-0.7, 0.8, True, True)])
    drawShadowImageSingle("AS_VV", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[_10log10, SFCBSM(-24.609674, 5.9092603, True, True)])
    drawShadowImageSingle("AS_VH", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[_10log10, SFCBSM(-31.865038, -5.2615275, True, True)])
    drawShadowImageSingle("AS_C11", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[_10log10, SFCBSM(-22.61998, 5.8634768, True, True)])
    drawShadowImageSingle("AS_C22", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[_10log10, SFCBSM(-28.579813, -5.2111626, True, True)])
    drawShadowImageSingle("AS_Lambda1", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[_10log10, SFCBSM(-21.955856, 6.124724, True, True)])
    drawShadowImageSingle("AS_Lambda2", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[_10log10, SFCBSM(-29.869734, -8.284683, True, True)])
    drawShadowImageSingle("DE_VV", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[_10log10, SFCBSM(-27.851603, 5.094706, True, True)])
    drawShadowImageSingle("DE_VH", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[_10log10, SFCBSM(-35.427082, -5.4092093, True, True)])
    drawShadowImageSingle("DE_C11", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[_10log10, SFCBSM(-26.245598, 4.9907513, True, True)])
    drawShadowImageSingle("DE_C22", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[_10log10, SFCBSM(-32.04232, -5.322515, True, True)])
    drawShadowImageSingle("DE_Lambda1", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[_10log10, SFCBSM(-25.503738, 5.2980003, True, True)])
    drawShadowImageSingle("DE_Lambda2", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[_10log10, SFCBSM(-33.442368, -8.68537, True, True)])


class DrawShadowImage_0:
    """
    TODO: 这里加一个导入数据范围的代码
    """

    def __init__(self, rows, columns, x, y, raster_fn, to_dirname, width=0.0, height=0.0, is_expand=False):
        self.rows = rows
        self.columns = columns
        self.x = x
        self.y = y
        self.raster_fn = raster_fn
        self.to_dirname = to_dirname
        self.width = width
        self.height = height
        self.is_expand = is_expand
        self.d_json = {}
        self.json_fn = os.path.join(to_dirname, "dsi.json")
        self.grr = GDALRasterRange(raster_fn)

    def drawOptical(self, name, d_min=0, d_max=3500, channel_list=None):
        drawShadowImage_Optical(name, self.x, self.y, self.rows, self.columns, self.raster_fn, to_fn=self.to_dirname,
                                d_min=d_min, d_max=d_max, channel_list=channel_list,
                                width=self.width, height=self.height, is_expand=self.is_expand)
        self.saveToJson(name, d_min=d_min, d_max=d_max, channel_list=channel_list,
                        width=self.width, height=self.height, is_expand=self.is_expand)

    def drawIndex(self, name, d_min=-60.0, d_max=60.0):
        drawShadowImage_Index(name, self.x, self.y, self.rows, self.columns, self.raster_fn, to_fn=self.to_dirname,
                              d_min=d_min, d_max=d_max,
                              width=self.width, height=self.height, is_expand=self.is_expand)
        self.saveToJson(name, d_min=d_min, d_max=d_max,
                        width=self.width, height=self.height, is_expand=self.is_expand)

    def drawSAR(self, name, d_min=-60.0, d_max=60.0, channel_list=None):
        drawShadowImage_SAR(name, self.x, self.y, self.rows, self.columns, self.raster_fn, to_fn=self.to_dirname,
                            d_min=d_min, d_max=d_max,
                            width=self.width, height=self.height, is_expand=self.is_expand)
        self.saveToJson(name, d_min=d_min, d_max=d_max,
                        width=self.width, height=self.height, is_expand=self.is_expand)

    def drawImdc(self, name, raster_fn, categorys=None):
        drawShadowImage_Imdc(name, self.x, self.y, self.rows, self.columns, raster_fn, to_fn=self.to_dirname,
                             categorys=categorys, width=self.width, height=self.height, is_expand=self.is_expand)
        self.saveToJson(name, categorys=categorys, width=self.width, height=self.height, is_expand=self.is_expand)

    def saveToJson(self, name, *args, **kwargs):
        d_json = dict(
            rows=self.rows,
            columns=self.columns,
            x=self.x,
            y=self.y,
            raster_fn=self.raster_fn,
            to_dirname=self.to_dirname,
            width=self.width,
            height=self.height,
            is_expand=self.is_expand,
        )
        d_json["__LIST__"] = list(args)
        for k in kwargs:
            d_json[k] = kwargs[k]
        self.d_json[name] = d_json
        saveJson(self.d_json, self.json_fn)


class DrawShadowImage_1(DrawShadowImage_0):

    def __init__(self, rows, columns, x, y, raster_fn, to_dirname):
        super(DrawShadowImage_1, self).__init__(rows, columns, x, y, raster_fn, to_dirname)


def method_name():
    qd_stack_fn = r"F:\ProjectSet\Shadow\QingDao\Image\stack2.vrt"
    imd_fn1 = r"F:\ProjectSet\Shadow\QingDao\Mods\20230707H200910\SPL_NOSH" \
              r"-RF-TAG-OPTICS-AS_SIGMA-AS_C2-AS_LAMD-DE_SIGMA-DE_C2_imdc.dat"
    save = r"F:\ProjectSet\Shadow\QingDao\mktu"
    sgimd = ShadowGeoDraw("Draw1", 120.350277, 36.083884)
    sgimd.win_row_size = 160
    sgimd.win_column_size = 160
    sgimd.addRasterCenter("RGB", qd_stack_fn, channel_list=["Red", "Green", "Blue"])
    sgimd.addRasterCenter("AS_VV", qd_stack_fn)
    sgimd.addRasterCenter("AS_VH", qd_stack_fn)
    sgimd.addRasterCenter("IMDC_1", imd_fn1, channel_list=[0])
    sgimd.fit(save)


def showcsv(fn):
    csv_dict = readcsv(fn)
    for i in range(len(csv_dict["X"])):
        print("{0}, {1}".format(csv_dict["X"][i], csv_dict["Y"][i]))


def imdcCount():
    output_vrt = r"F:\ProjectSet\Shadow\QingDao\Mods\20230707H200910\output.vrt"
    ds = gdal.Open(output_vrt)
    d = ds.ReadAsArray()
    out_d = np.zeros([5, d.shape[1], d.shape[2]])
    for i in range(d.shape[1]):
        for j in range(d.shape[2]):
            out_d[:, i, j] = np.bincount(d[:, i, j], minlength=5)
        print(i)
    grw = GeoRasterWrite(output_vrt)
    grw.save(out_d, r"F:\ProjectSet\Shadow\QingDao\Mods\20230707H200910\imdc2.dat", dtype=gdal.GDT_Int16)


def method_name3():
    raster_fn = r"G:\ImageData\QingDao\20211023\qd20211023\Temp\qd_im1.vrt"
    to_dir = r"F:\ProjectSet\Shadow\QingDao\Dissertation\4_Analyze\mktu\2"
    # width = 0.5
    # height = 0.5
    drawShadowImageSingle("RGB", raster_fn,
                          to_fn=r"F:\ProjectSet\Shadow\QingDao\Dissertation\4_Analyze\mktu\temp\test1.jpg",
                          x=120.377854, y=36.088972, rows=300, columns=300, channel=[2, 1, 0],
                          callbacks=[SFCBSM(300, 2500, True, True)])
    drawShadowImageSingle("NRG", raster_fn,
                          to_fn=r"F:\ProjectSet\Shadow\QingDao\Dissertation\4_Analyze\mktu\temp\test2.jpg",
                          x=120.377854, y=36.088972, rows=300, columns=300, channel=[3, 2, 1],
                          callbacks=[SFCBSM(300, 3500, True, True)])


def qdDrawGeoImage():
    raster_fn = r"F:\ProjectSet\Shadow\Release\QingDaoImages\SH_QD_envi.dat"
    to_dirname = r"F:\ProjectSet\Shadow\MkTu\4.1Details\1"
    dsi = DrawShadowImage_0(60, 60, 120.384609, 36.106485, raster_fn=raster_fn, to_dirname=to_dirname,
                            width=6, height=6, is_expand=True)
    dsi.drawOptical("RGB", channel_list=[2, 1, 0])
    dsi.drawOptical("NRB", channel_list=[3, 2, 1])
    dsi.drawIndex("NDVI", d_min=-0.6, d_max=0.9)
    dsi.drawIndex("NDWI", d_min=-0.7, d_max=0.8)
    dsi.drawSAR("AS_VV", d_min=-24.609674, d_max=5.9092603)
    dsi.drawSAR("AS_VH", d_min=-31.865038, d_max=-5.2615275)
    dsi.drawSAR("AS_C11", d_min=-22.61998, d_max=5.8634768)
    dsi.drawSAR("AS_C22", d_min=-28.579813, d_max=-5.2111626)
    dsi.drawSAR("AS_Lambda1", d_min=-21.955856, d_max=6.124724)
    dsi.drawSAR("AS_Lambda2", d_min=-29.869734, d_max=-8.284683)
    dsi.drawSAR("DE_VV", d_min=-27.851603, d_max=5.094706)
    dsi.drawSAR("DE_VH", d_min=-35.427082, d_max=-5.4092093)
    dsi.drawSAR("DE_C11", d_min=-26.245598, d_max=4.9907513)
    dsi.drawSAR("DE_C22", d_min=-32.04232, d_max=-5.322515)
    dsi.drawSAR("DE_Lambda1", d_min=-25.503738, d_max=5.2980003)
    dsi.drawSAR("DE_Lambda2", d_min=-33.442368, d_max=-8.68537)


def bjDrawGeoImage():
    raster_fn = r"F:\ProjectSet\Shadow\Release\BeiJingImages\SH_BJ_envi.dat"
    to_dirname = r"F:\ProjectSet\Shadow\MkTu\4.1Details\2"
    width = 6
    height = 6
    is_expand = True

    def draw(rows, columns, x, y):
        if not os.path.isdir(to_dirname):
            os.mkdir(to_dirname)

        dsi = DrawShadowImage_0(rows, columns, x, y, raster_fn=raster_fn, to_dirname=to_dirname,
                                width=width, height=height, is_expand=is_expand)
        dsi.drawOptical("RGB", channel_list=[2, 1, 0])
        dsi.drawOptical("NRB", channel_list=[3, 2, 1])
        dsi.drawIndex("NDVI", d_min=-0.6, d_max=0.9)
        dsi.drawIndex("NDWI", d_min=-0.7, d_max=0.8)
        dsi.drawSAR("AS_VV", d_min=-24.609674, d_max=5.9092603)
        dsi.drawSAR("AS_VH", d_min=-31.865038, d_max=-5.2615275)
        dsi.drawSAR("AS_C11", d_min=-22.61998, d_max=5.8634768)
        dsi.drawSAR("AS_C22", d_min=-28.579813, d_max=-5.2111626)
        dsi.drawSAR("AS_Lambda1", d_min=-21.955856, d_max=6.124724)
        dsi.drawSAR("AS_Lambda2", d_min=-29.869734, d_max=-8.284683)
        dsi.drawSAR("DE_VV", d_min=-27.851603, d_max=5.094706)
        dsi.drawSAR("DE_VH", d_min=-35.427082, d_max=-5.4092093)
        dsi.drawSAR("DE_C11", d_min=-26.245598, d_max=4.9907513)
        dsi.drawSAR("DE_C22", d_min=-32.04232, d_max=-5.322515)
        dsi.drawSAR("DE_Lambda1", d_min=-25.503738, d_max=5.2980003)
        dsi.drawSAR("DE_Lambda2", d_min=-33.442368, d_max=-8.68537)

    draw(60, 60, 116.461316, 39.896712)


class ShadowGeoDrawChannel:

    def __init__(self, name, win_row_size, win_column_size, channel_list=None, min_list=None, max_list=None,
                 callback_funcs=None, is_geo=True, no_data=0, ):
        self.file_list = [
            r"F:\ProjectSet\Shadow\Release\BeiJingImages\SH_BJ_look_tif.tif",
            r"F:\ProjectSet\Shadow\Release\ChengDuImages\SH_CD_look_tif.tif",
            r"F:\ProjectSet\Shadow\Release\QingDaoImages\SH_QD_look_tif.tif",
        ]
        self.geo_ranges = [
            GDALRasterRange(range_fn=r"F:\ProjectSet\Shadow\MkTu\Draw\SH_BJ_envi.dat.npy.json"),
            GDALRasterRange(range_fn=r"F:\ProjectSet\Shadow\MkTu\Draw\SH_CD_envi.dat.npy.json"),
            GDALRasterRange(range_fn=r"F:\ProjectSet\Shadow\MkTu\Draw\SH_QD_envi.dat.npy.json"),
        ]
        self.name = name
        self.win_row_size = win_row_size
        self.win_column_size = win_column_size
        self.channel_list = channel_list
        self.min_list = min_list
        self.max_list = max_list
        self.callback_funcs = callback_funcs
        self.is_geo = is_geo
        self.no_data = no_data

    def read(self, x, y):
        return readGDALRasterCenter(
            x=x, y=y, win_row_size=self.win_row_size, win_column_size=self.win_column_size,
            channel_list=self.channel_list, min_list=self.min_list, max_list=self.max_list,
            callback_funcs=self.callback_funcs, is_geo=self.is_geo, no_data=self.no_data,
            geo_ranges=self.geo_ranges, file_list=self.file_list
        )


class ShadowGeoDrawMultiChannel(ShadowGeoDrawChannel):

    def __init__(self, name, win_row_size, win_column_size, channel_list=None, min_list=None, max_list=None,
                 callback_funcs=None, is_geo=True, no_data=0, ):
        if channel_list is None:
            channel_list = [0, 1, 2]
        if min_list is None:
            min_list = [None, None, None]
        if max_list is None:
            max_list = [None, None, None]
        super().__init__(name=name, win_row_size=win_row_size, win_column_size=win_column_size,
                         channel_list=channel_list, min_list=min_list, max_list=max_list,
                         callback_funcs=callback_funcs, is_geo=is_geo, no_data=no_data, )


class ShadowGeoDrawSingleChannel(ShadowGeoDrawChannel):

    def __init__(self, name, win_row_size, win_column_size, channel_list=None, min_list=None, max_list=None,
                 callback_funcs=None, is_geo=True, no_data=0, ):
        if channel_list is None:
            channel_list = [0]
        if min_list is None:
            min_list = [None]
        if max_list is None:
            max_list = [None]
        super().__init__(name=name, win_row_size=win_row_size, win_column_size=win_column_size,
                         channel_list=channel_list, min_list=min_list, max_list=max_list,
                         callback_funcs=callback_funcs, is_geo=is_geo, no_data=no_data, )

    def read(self, x, y):
        d = super(ShadowGeoDrawSingleChannel, self).read(x, y)
        d_rgb = np.zeros((d.shape[0], d.shape[1], 3))
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                d_rgb[i, j, :] = d[i, j]
        return d_rgb


class ShadowGeoDrawMBTiles(ShadowGeoDrawChannel):

    def __init__(self, name, win_row_size, win_column_size, mb_tiles_fns=None,
                 callback_funcs=None, is_geo=True, no_data=0, ):
        if mb_tiles_fns is None:
            mb_tiles_fns = []
        if mb_tiles_fns is not None:
            self.file_list = mb_tiles_fns
        self.mb_tiles = [GDALMBTiles(fn) for fn in self.file_list]

        super().__init__(name=name, win_row_size=win_row_size, win_column_size=win_column_size,
                         channel_list=None, min_list=None, max_list=None,
                         callback_funcs=callback_funcs, is_geo=is_geo, no_data=no_data, )

    def read(self, x, y):
        d = None
        for mb_tile in self.mb_tiles:
            d = mb_tile.getCenterImage(x, y, (self.win_row_size, self.win_column_size))
            if d is not None:
                break
        return d


class ShadowGeoDrawCategoryChannel(GDALRasterCenter):

    def __init__(self, name, imdc_fns, win_row_size, win_column_size, is_geo=True, no_data=0, color_dict=None):
        super().__init__(name=name, channel_list=[0], raster_fn=imdc_fns[0])
        if color_dict is None:
            color_dict = {}
        self.name = name
        self.win_row_size = win_row_size
        self.win_column_size = win_column_size
        self.is_geo = is_geo
        self.no_data = no_data
        self.imdc_fns = imdc_fns
        self.color_dict = color_dict

    def read(self, x=0.0, y=0.0, win_row_size=0, win_column_size=0, is_geo=True, no_data=0, *args, **kwargs):
        win_row_size = self.win_row_size
        win_column_size = self.win_column_size
        is_geo = self.is_geo
        no_data = self.no_data
        d = None
        for i in range(len(self.imdc_fns)):
            self.initRaster(None, raster_fn=self.imdc_fns[i])
            d = super(ShadowGeoDrawCategoryChannel, self).read(
                x_row=x, y_column=y, win_row_size=win_row_size,
                win_column_size=win_column_size, is_geo=is_geo, no_data=no_data)
            if d is not None:
                break
        self.d = d
        if d is not None:
            self.toCategory()
            for k in self.color_dict:
                self.categoryColor(k, self.color_dict[k])
        return self.d / 255


class _mpl_patches_Ellipse(MplPatchesEllipse):

    def __init__(self, xy, width, height, angle=0, is_ratio=False, select_columns=None, **kwargs):
        super().__init__(xy, width, height, angle, is_ratio, select_columns, **kwargs)


class _mpl_patches_Ellipse_Coll(MplPatchesEllipseColl):

    def __init__(self):
        super().__init__()


class EveryAxDeal:

    def __init__(self, fit_func):
        self.fit_func = fit_func

    def fit(self, *args, _sgdgi=None, ax=None, i_row=None, j_column=None, d=None, **kwargs):
        self.fit_func(self, *args, _sgdgi=_sgdgi, ax=ax, i_row=i_row, j_column=j_column, d=d, **kwargs)


class EveryAxDealColl:

    def __init__(self):
        self.eads = []

    def add(self, fit_func):
        self.eads.append(EveryAxDeal(fit_func))

    def fit(self, *args, _sgdgi=None, ax=None, i_row=None, j_column=None, d=None, **kwargs):
        for ead in self.eads:
            ead.fit_func(ead, *args, _sgdgi=_sgdgi, ax=ax, i_row=i_row, j_column=j_column, d=d, **kwargs)


class ShadowGeoDrawGradImage:

    def __init__(self, win_row_size, win_column_size):
        super().__init__()

        self.is_row_name_show = True
        self.is_column_name_show = True
        self.win_row_size = win_row_size
        self.win_column_size = win_column_size
        self.columns = []
        self.row_names = []
        self.xys = []
        self.axes = None

        self.mb_tile_file_list = [
            r"F:\ProjectSet\Shadow\MkTu\4.1Details\BingImages\cd_googleimages.mbtiles",
            r"F:\ProjectSet\Shadow\MkTu\4.1Details\BingImages\bj_googleimages.mbtiles",
            r"F:\ProjectSet\Shadow\MkTu\4.1Details\BingImages\qd_googleimages.mbtiles",
        ]

        self.imdc_file_dict = {}
        self.color_dict = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255)}

        self.ell_coll = _mpl_patches_Ellipse_Coll()
        self.ead_coll = EveryAxDealColl()

        self.column_name_map = {}

    def initSize(self, win_column_size, win_row_size):
        if win_row_size is None:
            win_row_size = self.win_row_size
        if win_column_size is None:
            win_column_size = self.win_column_size
        return win_column_size, win_row_size

    def checkColumnKwargs(self, kws, **kwargs):
        return

    def addColumnRGB(self, name, min_list=None, max_list=None, win_row_size=None, win_column_size=None, **kwargs):
        win_column_size, win_row_size = self.initSize(win_column_size, win_row_size)
        sgdmc = ShadowGeoDrawMultiChannel(
            name=name, win_row_size=win_row_size, win_column_size=win_column_size,
            channel_list=["Red", "Green", "Blue"], min_list=min_list, max_list=max_list,
            callback_funcs=[], is_geo=True, no_data=0,
        )
        self.columns.append(sgdmc)
        self.checkColumnKwargs(kwargs, name=name, min_list=min_list, max_list=max_list,
                               win_row_size=win_row_size, win_column_size=win_column_size, )
        return sgdmc

    def addColumnGoogle(self, name, win_row_size=None, win_column_size=None, **kwargs):
        win_column_size, win_row_size = self.initSize(win_column_size, win_row_size)
        dgdmbt = ShadowGeoDrawMBTiles(name=name, win_row_size=win_row_size, win_column_size=win_column_size,
                                      mb_tiles_fns=self.mb_tile_file_list, callback_funcs=[], is_geo=True, no_data=0, )
        self.columns.append(dgdmbt)
        self.checkColumnKwargs(kwargs, name=name, win_row_size=win_row_size, win_column_size=win_column_size, )
        return dgdmbt

    def addColumnNRG(self, name, min_list=None, max_list=None, win_row_size=None, win_column_size=None, **kwargs):
        win_column_size, win_row_size = self.initSize(win_column_size, win_row_size)
        sgdmc = ShadowGeoDrawMultiChannel(
            name=name, win_row_size=win_row_size, win_column_size=win_column_size,
            channel_list=["NIR", "Red", "Green"], min_list=min_list, max_list=max_list,
            callback_funcs=[], is_geo=True, no_data=0,
        )
        self.columns.append(sgdmc)
        self.checkColumnKwargs(kwargs, name=name, min_list=min_list, max_list=max_list,
                               win_row_size=win_row_size, win_column_size=win_column_size, )
        return sgdmc

    def addColumn(self, name, channel_name, min_list=None, max_list=None, win_row_size=None, win_column_size=None,
                  **kwargs):
        win_column_size, win_row_size = self.initSize(win_column_size, win_row_size)
        sgdsc = ShadowGeoDrawSingleChannel(
            name=name, win_row_size=win_row_size, win_column_size=win_column_size,
            channel_list=[channel_name], min_list=min_list, max_list=max_list,
            # callback_funcs=[SRTFeatureCallBack(_power10, is_trans=True)],
            callback_funcs=[],
            is_geo=True, no_data=0,
        )
        self.columns.append(sgdsc)
        self.checkColumnKwargs(kwargs, name=name, channel_name=channel_name, min_list=min_list, max_list=max_list,
                               win_row_size=win_row_size, win_column_size=win_column_size, )
        return sgdsc

    def addColumnNDVI(self, name, min_list=None, max_list=None, win_row_size=None, win_column_size=None, **kwargs):
        return self.addColumn(name=name, channel_name="NDVI", min_list=min_list, max_list=max_list,
                              win_row_size=win_row_size, win_column_size=win_column_size, **kwargs)

    def addColumnNDWI(self, name, min_list=None, max_list=None, win_row_size=None, win_column_size=None, **kwargs):
        return self.addColumn(name=name, channel_name="NDWI", min_list=min_list, max_list=max_list,
                              win_row_size=win_row_size, win_column_size=win_column_size, **kwargs)

    def addColumnSAR(self, name, channel_name, min_list=None, max_list=None, win_row_size=None, win_column_size=None,
                     **kwargs):
        win_column_size, win_row_size = self.initSize(win_column_size, win_row_size)
        sgdsc = ShadowGeoDrawSingleChannel(
            name=name, win_row_size=win_row_size, win_column_size=win_column_size,
            channel_list=[channel_name], min_list=min_list, max_list=max_list,
            # callback_funcs=[SRTFeatureCallBack(_power10, is_trans=True)],
            callback_funcs=[],
            is_geo=True, no_data=0,
        )
        self.columns.append(sgdsc)
        self.checkColumnKwargs(kwargs, name=name, channel_name=channel_name, min_list=min_list, max_list=max_list,
                               win_row_size=win_row_size, win_column_size=win_column_size, )
        return sgdsc

    def addColumnSAR_ASVV(self, name, min_list=None, max_list=None, win_row_size=None, win_column_size=None, **kwargs):
        return self.addColumnSAR(name=name, channel_name="AS_VV", min_list=min_list, max_list=max_list,
                                 win_row_size=win_row_size, win_column_size=win_column_size, **kwargs)

    def addColumnSAR_ASVH(self, name, min_list=None, max_list=None, win_row_size=None, win_column_size=None, **kwargs):
        return self.addColumnSAR(name=name, channel_name="AS_VH", min_list=min_list, max_list=max_list,
                                 win_row_size=win_row_size, win_column_size=win_column_size, **kwargs)

    def addColumnSAR_DEVV(self, name, min_list=None, max_list=None, win_row_size=None, win_column_size=None, **kwargs):
        return self.addColumnSAR(name=name, channel_name="DE_VV", min_list=min_list, max_list=max_list,
                                 win_row_size=win_row_size, win_column_size=win_column_size, **kwargs)

    def addColumnSAR_DEVH(self, name, min_list=None, max_list=None, win_row_size=None, win_column_size=None, **kwargs):
        return self.addColumnSAR(name=name, channel_name="DE_VH", min_list=min_list, max_list=max_list,
                                 win_row_size=win_row_size, win_column_size=win_column_size, **kwargs)

    def addColumnSAR_ASC11(self, name, min_list=None, max_list=None, win_row_size=None, win_column_size=None, **kwargs):
        return self.addColumnSAR(name=name, channel_name="AS_C11", min_list=min_list, max_list=max_list,
                                 win_row_size=win_row_size, win_column_size=win_column_size, **kwargs)

    def addColumnSAR_ASC22(self, name, min_list=None, max_list=None, win_row_size=None, win_column_size=None, **kwargs):
        return self.addColumnSAR(name=name, channel_name="AS_C22", min_list=min_list, max_list=max_list,
                                 win_row_size=win_row_size, win_column_size=win_column_size, **kwargs)

    def addColumnSAR_DEC11(self, name, min_list=None, max_list=None, win_row_size=None, win_column_size=None, **kwargs):
        return self.addColumnSAR(name=name, channel_name="DE_C11", min_list=min_list, max_list=max_list,
                                 win_row_size=win_row_size, win_column_size=win_column_size, **kwargs)

    def addColumnSAR_DEC22(self, name, min_list=None, max_list=None, win_row_size=None, win_column_size=None, **kwargs):
        return self.addColumnSAR(name=name, channel_name="DE_C22", min_list=min_list, max_list=max_list,
                                 win_row_size=win_row_size, win_column_size=win_column_size, **kwargs)

    def addColumnImdc(self, name, imdc_fns, color_dict=None, win_row_size=None, win_column_size=None, **kwargs):
        win_column_size, win_row_size = self.initSize(win_column_size, win_row_size)
        if color_dict is None:
            color_dict = {}
        sgdcc = ShadowGeoDrawCategoryChannel(
            name, imdc_fns=imdc_fns, win_row_size=win_row_size, win_column_size=win_column_size, is_geo=True, no_data=0,
            color_dict=color_dict
        )
        self.columns.append(sgdcc)
        self.checkColumnKwargs(kwargs, name=name, imdc_fns=imdc_fns, color_dict=imdc_fns,
                               win_row_size=win_row_size, win_column_size=win_column_size, **kwargs)
        return sgdcc

    def addColumnImdcKey(self, name, imdc_key, color_dict=None, win_row_size=None, win_column_size=None, **kwargs):
        if color_dict is None:
            color_dict = self.color_dict
        self.addColumnImdc(name, self.imdc_file_dict[imdc_key], color_dict=color_dict,
                           win_row_size=win_row_size, win_column_size=win_column_size)

    def addRow(self, name, x, y):
        self.row_names.append(name)
        self.xys.append((x, y))

    def addImdcs(self, dirname_list=None):
        if dirname_list is None:
            dirname_list = [
                r"F:\ProjectSet\Shadow\ChengDu\Mods\20231226H203759",
                r"F:\ProjectSet\Shadow\BeiJing\Mods\20231227H151054",
                r"F:\ProjectSet\Shadow\QingDao\Mods\20231226H093225",
            ]
            dirname_list = [
                r"F:\ProjectSet\Shadow\ChengDu\Mods\20240222H170152",
                r"F:\ProjectSet\Shadow\QingDao\Mods\20231226H093225",
                r"F:\ProjectSet\Shadow\BeiJing\Mods\20231225H110303",
            ]
        files_dict = {}
        for dirname in dirname_list:
            fns = filterFileContain(dirname, "_imdc.dat")
            for fn in fns:
                if not fn.endswith("_imdc.dat"):
                    continue
                fn1 = os.path.split(fn)[1]
                fn2 = fn1[:fn1.index("_imdc.dat")]
                if fn2 not in files_dict:
                    files_dict[fn2] = []
                    print("\"{0}\"".format(fn2))
                files_dict[fn2].append(fn)
        self.imdc_file_dict = files_dict

    def addRowEllipse(self, n_row, xy, width, height, angle=0, is_ratio=False, select_columns=None, **kwargs):
        ell = self.ell_coll.add(n_row=n_row, xy=xy, width=width, height=height, angle=angle, is_ratio=is_ratio,
                                select_columns=select_columns, **kwargs)
        return ell

    def addEllipse(self, xy, width, height, n_row=None, n_column=None, angle=0, is_ratio=False, select_columns=None,
                   not_rows=None, not_columns=None, **kwargs):
        ell = self.ell_coll.add2(xy, width, height, n_row=n_row, n_column=n_column, angle=angle, is_ratio=is_ratio,
                                 select_columns=select_columns, not_rows=not_rows, not_columns=not_columns, **kwargs)
        return ell

    def imshow(self, n_rows_ex=1.0, n_columns_ex=1.0):
        n_rows, n_columns = len(self.row_names), len(self.columns)
        fig = plt.figure(
            figsize=(n_columns * n_columns_ex, n_rows * n_rows_ex),
            # dpi=300
        )
        axes = fig.subplots(n_rows, n_columns)
        # fig.tight_layout()
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, hspace=0.04, wspace=0.03)

        for i in range(n_rows):
            x, y = self.xys[i]

            for j in range(n_columns):
                if n_rows != 1:
                    ax = axes[i, j]
                else:
                    ax = axes[j]
                if j == 0:
                    if self.is_row_name_show:
                        ax.set_ylabel(self.row_names[i], rotation=0, fontdict={'family': 'Times New Roman', 'size': 16})
                if i == 0:
                    if self.is_column_name_show:
                        column_name = self.columns[j].name
                        if column_name in self.column_name_map:
                            column_name = self.column_name_map[column_name]
                        ax.set_title(column_name, fontdict={'family': 'Times New Roman', 'size': 16})
                d = self.columns[j].read(x, y)
                ax.imshow(d)

                self.ell_coll.fit(ax, i, j)
                self.ead_coll.fit(_sgdgi=self, ax=ax, i_row=i, j_column=j, d=d, axes=axes)

                ax.set_xticks([])
                ax.set_yticks([])
                pass

    def isColumnName(self, is_show: bool = True):
        self.is_column_name_show = is_show

    def isRowName(self, is_show: bool = True):
        self.is_row_name_show = is_show


def main():
    # df = sdf_read_csv(r"F:\ProjectSet\Shadow\Analysis\12\mktu423_coors2.csv")
    # df.asColumnType("X", float)
    # df.asColumnType("Y", float)

    dirname_list = [
        r"F:\ProjectSet\Shadow\ChengDu\Mods\20240222H170152",
        r"F:\ProjectSet\Shadow\QingDao\Mods\20231226H093225",
        r"F:\ProjectSet\Shadow\BeiJing\Mods\20231225H110303",
    ]

    def func1():
        mods_dirname = dirname_list[0]
        fns = filterFileContain(mods_dirname, "_imdc.dat")
        files_dict = {}
        des_list = []
        datas = []
        for fn in fns:
            if not fn.endswith("_imdc.dat"):
                continue
            fn1 = os.path.split(fn)[1]
            fn2 = fn1[:fn1.index("_imdc.dat")]
            files_dict[fn2] = fn
            des_list.append(fn2)
            datas.append([GDALRaster(fn).readAsArray()])
            print(fn, datas[-1][0].shape)
        data = np.concatenate(datas)
        print(data.shape)
        gr = GDALRaster(fns[0])
        gr.save(data, r"F:\ProjectSet\Shadow\Analysis\13\cd_imdc2.tif", fmt="GTiff", dtype=gdal.GDT_Byte,
                descriptions=des_list, options=["COMPRESS=PACKBITS"])
        return

    def func2():
        imdc_fn = r"F:\ProjectSet\Shadow\Analysis\13\cd_imdc2.tif"
        to_fn = r"F:\ProjectSet\Shadow\Analysis\13\cd_imdc2_count1.tif"
        # grc = GDALRasterChannel()
        # grc.addGDALDatas(imdc_fn)
        np.all()
        # output_vrt = r"F:\ProjectSet\Shadow\QingDao\Mods\20230707H200910\output.vrt"
        ds = gdal.Open(imdc_fn)
        d = ds.ReadAsArray()
        out_d = np.zeros([5, d.shape[1], d.shape[2]])
        jdt = Jdt("Imdc Count")
        for i in range(d.shape[1]):
            for j in range(d.shape[2]):
                out_d[:, i, j] = np.bincount(d[:, i, j], minlength=5)
            print(i)
        gr = GDALRaster(imdc_fn)
        gr.save(out_d, to_fn, dtype=gdal.GDT_Int16, fmt="GTiff", options=["COMPRESS=PACKBITS"])

        return

    def func3():
        mods_dirname = dirname_list[1]
        fns = filterFileContain(mods_dirname, "_imdc.dat")
        files_dict = {}
        datas = {}

        for fn in fns:
            if not fn.endswith("_imdc.dat"):
                continue
            fn1 = os.path.split(fn)[1]
            fn2 = fn1[:fn1.index("_imdc.dat")]

            files_dict[fn2] = fn
            datas[fn2] = GDALRaster(fn).readAsArray()

            print(fn, datas[fn2].shape)

        printList("", list(datas.keys()))
        """
          "SPL_NOSH-RF-TAG-AS-DE", "SPL_NOSH-RF-TAG-AS", 
          "SPL_NOSH-RF-TAG-DE", "SPL_NOSH-RF-TAG-OPTICS-AS-DE", 
          "SPL_NOSH-RF-TAG-OPTICS-AS", "SPL_NOSH-RF-TAG-OPTICS-DE", "SPL_NOSH-RF-TAG-OPTICS", 
          "SPL_NOSH-SVM-TAG-AS-DE", "SPL_NOSH-SVM-TAG-AS", "SPL_NOSH-SVM-TAG-DE", 
          "SPL_NOSH-SVM-TAG-OPTICS-AS-DE", "SPL_NOSH-SVM-TAG-OPTICS-AS", 
          "SPL_NOSH-SVM-TAG-OPTICS-DE", "SPL_NOSH-SVM-TAG-OPTICS", "SPL_SH-RF-TAG-AS-DE", 
          "SPL_SH-RF-TAG-AS", "SPL_SH-RF-TAG-DE", "SPL_SH-RF-TAG-OPTICS-AS-DE", 
          "SPL_SH-RF-TAG-OPTICS-AS", "SPL_SH-RF-TAG-OPTICS-DE", "SPL_SH-RF-TAG-OPTICS", 
          "SPL_SH-SVM-TAG-AS-DE", "SPL_SH-SVM-TAG-AS", "SPL_SH-SVM-TAG-DE", 
          "SPL_SH-SVM-TAG-OPTICS-AS-DE", "SPL_SH-SVM-TAG-OPTICS-AS", "SPL_SH-SVM-TAG-OPTICS-DE", 
          "SPL_SH-SVM-TAG-OPTICS", 
        """

        def concat_data(_names=None):
            if _names is None:
                _names = list(datas.keys())
            cat_data_list = [[datas[k]] for k in _names]
            cat_data = np.concatenate(cat_data_list)
            return cat_data

        data = concat_data(["SPL_SH-SVM-TAG-OPTICS-AS-DE", "SPL_SH-SVM-TAG-OPTICS-AS", "SPL_SH-SVM-TAG-OPTICS-DE",
                            "SPL_SH-SVM-TAG-OPTICS", ])

        cate_list = [("NOT_KNOW", 0), ("IS", 1), ("VEG", 2), ("SOIL", 3), ("WAT", 4), ]
        color_list = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)]
        color_map = ListedColormap(color_list)

        def same_category(_cat_data):
            out_d = np.zeros(_cat_data.shape[1:])
            for cname, c_code in cate_list:
                d = np.all(_cat_data == c_code, axis=0) * c_code
                out_d += d
            return out_d

        out_d = same_category(data)

        # r"F:\ProjectSet\Shadow\Analysis\13\cd_imdc2_count2.dat"
        def saveImdcENVI(imd, to_name):
            gr = GDALRaster(fns[0])
            gr.save(imd, to_name, fmt="ENVI", dtype=gdal.GDT_Byte, )
            hdr_fn = changext(to_name, ".hdr")
            with open(hdr_fn, "a", encoding="utf-8") as f:
                f.write("""classes = 5
class lookup = {    0,   0,   0, 255,   0,   0,   0, 255,   0, 255, 255,   0,   0,   0, 255 }
class names = { Unclassified, IS, VEG, SOIL, WAT }
                """)

        saveImdcENVI(out_d, r"F:\ProjectSet\Shadow\Analysis\13\qd_imdc2_count1.dat")

        return

    # func3()
    method_name5()

    pass


def drawRowColumn41():
    sgdgi = ShadowGeoDrawGradImage(31, 31)
    # sgdgi.color_dict = {1: (221,74,76), 2: (214,238,155), 3: (254,212,129), 4: (61,149,184)}
    sgdgi.addImdcs()
    sgdgi.isRowName(False)

    sgdgi.addColumnGoogle("Google Image", 1000, 1000)
    sgdgi.addColumnRGB("RGB")
    sgdgi.addColumnNRG("NRG")
    sgdgi.addColumnNDVI("NDVI")
    # sgdgi.addColumnNDWI("NDWI")

    # sgdgi.addColumnImdcKey("SH-AS-DE", "SPL_SH-SVM-TAG-OPTICS-AS-DE")
    # sgdgi.addColumnImdcKey("NOSH-AS-DE", "SPL_NOSH-SVM-TAG-OPTICS-AS-DE")
    # sgdgi.addColumnImdcKey("SH-AS", "SPL_SH-SVM-TAG-OPTICS-AS")
    # sgdgi.addColumnImdcKey("NOSH-AS", "SPL_NOSH-SVM-TAG-OPTICS-AS")
    # sgdgi.addColumnImdcKey("SH-DE", "SPL_SH-SVM-TAG-OPTICS-DE")
    # sgdgi.addColumnImdcKey("NOSH-DE", "SPL_NOSH-SVM-TAG-OPTICS-DE")
    # sgdgi.addColumnImdcKey("SH-OPT", "SPL_SH-SVM-TAG-OPTICS")
    # sgdgi.addColumnImdcKey("NOSH-OPT", "SPL_NOSH-SVM-TAG-OPTICS")

    sgdgi.addColumnSAR_ASVV("SAR AS")
    sgdgi.addColumnSAR_DEVV("SAR DE")

    # sgdgi.addColumnSAR_ASVH("AS VH")
    # sgdgi.addColumnSAR_DEVH("DE VH")

    # sgdgi.addColumnSAR_ASC11("AS C11")
    # sgdgi.addColumnSAR_DEC11("DE C11")
    # sgdgi.addColumnSAR_ASC22("AS C22")
    # sgdgi.addColumnSAR_DEC22("DE C22")

    def _addRowEllipse1(n_row, xy, width=8, height=6, angle=0, linewidth=1, fill=False, zorder=2,
                        edgecolor="lightgreen", is_ratio=False):
        sgdgi.addRowEllipse(n_row=n_row, xy=xy, width=width, height=height, angle=angle, linewidth=linewidth,
                            fill=fill, zorder=zorder, edgecolor=edgecolor, is_ratio=is_ratio)

    def _addEllipse1(xy, n_row=None, n_column=None, not_rows=None, not_columns=None,
                     width=8, height=6, angle=0, linewidth=1.5, fill=False, zorder=2,
                     edgecolor="lightgreen", is_ratio=False):
        sgdgi.addEllipse(xy=xy, n_row=n_row, n_column=n_column, not_rows=not_rows, not_columns=not_columns,
                         width=width, height=height, angle=angle, linewidth=linewidth,
                         fill=fill, zorder=zorder, edgecolor=edgecolor, is_ratio=is_ratio)

    def end_show():
        sgdgi.addRow("1", 116.413250, 39.907738)
        _addEllipse1((13, 6), 0, [1, 2], width=8, edgecolor="green")
        _addEllipse1((6, 16), 0, [4], edgecolor="yellow")
        _addEllipse1((26, 16), 0, [5], edgecolor="yellow")
        _addEllipse1((24, 9), 0, [4], edgecolor="red")
        _addEllipse1((10, 11), 0, [5], edgecolor="blue")
        sgdgi.addRow("2", 104.07385, 30.65005)  # select
        _addEllipse1((6, 7), 1, [1, 2, ], width=8, edgecolor="green")
        _addEllipse1((14, 13), 1, [1, 2, 3, 4, 5], edgecolor="red")
        sgdgi.addRow("3", 116.302365, 39.962880)
        _addEllipse1((21, 14), 2, [1, 2], width=8, edgecolor="green")
        _addEllipse1((6, 6), 2, [1, 2, 3, 4, 5], edgecolor="blue")
        sgdgi.addRow("4", 116.486538, 39.889220)
        _addEllipse1((23, 13), 3, [1, 2, 3, 4, 5], width=8, edgecolor="green")
        _addEllipse1((14, 13), 3, [1, 2, 3, 4, 5], edgecolor="red")
        _addEllipse1((11, 15), 3, [1, 2, 3, 4, 5], edgecolor="blue")

    # sgdgi.addRow("IS", 116.320535, 39.864142)
    # _addRowEllipse1(0, (22,9))
    # sgdgi.addRow("IS", 116.426488, 39.937361)
    # _addRowEllipse1(1, (12, 14))
    # sgdgi.addRow("VEG", 116.332767, 39.870094)
    # _addRowEllipse1(2, (14, 17))
    # sgdgi.addRow("VEG", 120.339779, 36.102030)
    # _addRowEllipse1(3, (14, 20))
    # sgdgi.addRow("VEG", 120.4062654, 36.0823765)
    # sgdgi.addRow("IS", 104.013075, 30.602504)
    # sgdgi.addRow("WAT", 104.083731, 30.618372)

    # sgdgi.addRow("IS", 116.4860893, 39.8936288)
    # _addRowEllipse1(0, (22, 9))
    # sgdgi.addRow("IS", 120.339454, 36.052821)
    # sgdgi.addRow("VEG", 116.559579, 39.946365)
    # sgdgi.addRow("VEG", 116.34989, 39.79670)
    # sgdgi.addRow("VEG", 104.07385, 30.65005) # select
    # _addRowEllipse1(4, (12, 12))
    # sgdgi.addRow("SOIL", 116.350292, 39.798293)

    def t_show():
        # sgdgi.addRow("WAT", 116.310359, 39.915394)
        # sgdgi.addRow("IS", 116.305999, 39.965146)
        # sgdgi.addRow("IS", 116.302365, 39.962880)
        # sgdgi.addRow("VEG", 116.575445, 39.935084)

        # sgdgi.addRow("IS", 120.3418499, 36.0884511)
        # sgdgi.addRow("IS", 116.4860893, 39.8936288)
        # sgdgi.addRow("IS", 120.374407, 36.064563)
        # sgdgi.addRow("IS", 120.332966, 36.118072)
        # _addEllipse1((13, 8), 3, [1, 2, 3, 4, 5], edgecolor="blue")
        # sgdgi.addRow("IS", 120.3780005, 36.1084047)
        # sgdgi.addRow("IS", 120.353551, 36.082023)

        # sgdgi.addRow("IS", 120.339454, 36.052821)
        # sgdgi.addRow("IS", 120.397521, 36.144224)
        # sgdgi.addRow("IS", 116.363178, 39.858848)
        # sgdgi.addRow("VEG", 116.34989, 39.79670)
        # sgdgi.addRow("VEG", 104.07385, 30.65005)
        # sgdgi.addRow("VEG", 104.13064, 30.62272)

        sgdgi.addRow("VEG", 116.496832, 39.912371)
        sgdgi.addRow("VEG", 116.486538, 39.889220)

    end_show()
    sgdgi.imshow(n_rows_ex=2.0, n_columns_ex=2.0)
    plt.savefig(r"F:\ProjectSet\Shadow\MkTu\4.1Details\fig_41_1.jpeg", dpi=300)
    plt.show()


def drawRowColumn42():
    sgdgi = ShadowGeoDrawGradImage(31, 31)
    # sgdgi.color_dict = {1: (221,74,76), 2: (214,238,155), 3: (254,212,129), 4: (61,149,184)}
    sgdgi.addImdcs()
    sgdgi.isRowName(True)

    sgdgi.addColumnGoogle("Google Image", 800, 800)
    sgdgi.addColumnRGB("RGB")
    sgdgi.addColumnNRG("NRG")
    # sgdgi.addColumnNDVI("NDVI")
    # sgdgi.addColumnNDWI("NDWI")

    sgdgi.addColumnSAR_ASVV("SAR AS")
    sgdgi.addColumnSAR_DEVV("SAR DE")
    # sgdgi.addColumnSAR_ASVH("AS VH")
    # sgdgi.addColumnSAR_DEVH("DE VH")

    sgdgi.addColumnImdcKey("SH-AS-DE", "SPL_SH-SVM-TAG-OPTICS-AS-DE")
    sgdgi.addColumnImdcKey("SH-AS", "SPL_SH-SVM-TAG-OPTICS-AS")
    sgdgi.addColumnImdcKey("SH-DE", "SPL_SH-SVM-TAG-OPTICS-DE")
    sgdgi.addColumnImdcKey("SH-OPT", "SPL_SH-SVM-TAG-OPTICS")
    sgdgi.column_name_map["SH-AS-DE"] = "Opt-AS-DE"
    sgdgi.column_name_map["SH-AS"] = "Opt-AS"
    sgdgi.column_name_map["SH-DE"] = "Opt-DE"
    sgdgi.column_name_map["SH-OPT"] = "Opt"

    # sgdgi.addColumnImdcKey("NOSH-AS-DE", "SPL_NOSH-SVM-TAG-OPTICS-AS-DE")
    # sgdgi.addColumnImdcKey("NOSH-AS", "SPL_NOSH-SVM-TAG-OPTICS-AS")
    # sgdgi.addColumnImdcKey("NOSH-DE", "SPL_NOSH-SVM-TAG-OPTICS-DE")
    # sgdgi.addColumnImdcKey("NOSH-OPT", "SPL_NOSH-SVM-TAG-OPTICS")

    def _addEllipse1(xy, n_row=None, n_column=None, not_rows=None, not_columns=None,
                     width=8, height=6, angle=0, linewidth=1.5, fill=False, zorder=2,
                     edgecolor="lightgreen", is_ratio=False):
        sgdgi.addEllipse(xy=xy, n_row=n_row, n_column=n_column, not_rows=not_rows, not_columns=not_columns,
                         width=width, height=height, angle=angle, linewidth=linewidth,
                         fill=fill, zorder=zorder, edgecolor=edgecolor, is_ratio=is_ratio)

    df = sdf_read_csv(r"F:\ProjectSet\Shadow\Analysis\12\mktu423_coors2.csv")
    df.asColumnType("X", float)
    df.asColumnType("Y", float)
    df.indexColumnName("Name")

    def add_row(n, row_name=None):
        if row_name is None:
            row_name = df.loc(n, "Name")
        sgdgi.addRow(row_name, df.loc(n, "X"), df.loc(n, "Y"))
        print(row_name, df.loc(n, "X"), df.loc(n, "Y"))

    find_rows = ["bj13", "bj14", "bj17", "bj20", "bj21", "qd3", "cd1", "qd4", "qd5", "cd1", "cd2", ]

    # for i in find_rows[:4]:
    #     add_row(i)
    #
    # add_row(find_rows[0], "(1)   ")
    # # _addEllipse1((19, 12), 0, None, width=8, edgecolor="green")
    # _addEllipse1((14, 5), 0, None, width=12, height=10,  edgecolor="darkred")
    #
    # def eds1(self, *args, _sgdgi: ShadowGeoDrawGradImage = None, ax=None, i_row=None, j_column=None, d=None, **kwargs):
    #     if (i_row == 0) and (_sgdgi.columns[j_column].name == "SH-AS-DE"):
    #         self.sh_as_de_data = d
    #         self.i_row = i_row
    #         self.j_column = j_column
    #
    #     if (i_row == 0) and (_sgdgi.columns[j_column].name == "SH-AS"):
    #         self.sh_as_data = d
    #         axes = kwargs["axes"]
    #         ax2 = axes[self.i_row, self.j_column]
    #         d_show = self.sh_as_de_data
    #         # d_show[8:20, 4:9, :] = self.sh_as_data[8:20, 4:9, :]
    #         d_show[4:9, 8:20, :] = self.sh_as_data[4:9, 8:20, :]
    #         ax2.imshow(d_show)
    #
    # sgdgi.ead_coll.add(eds1)

    # sgdgi.addRow("1", 120.385503, 36.090180)

    sgdgi.addRow("(1)   ", 104.079150, 30.652560)
    _addEllipse1((17, 18), 0, None, width=12, height=10, edgecolor="darkred")

    def eds3(self, *args, _sgdgi: ShadowGeoDrawGradImage = None, ax=None, i_row=None, j_column=None, d=None, **kwargs):
        if (i_row == 0) and (_sgdgi.columns[j_column].name == "SH-AS-DE"):
            self.i_row = i_row
            self.j_column = j_column
            axes = kwargs["axes"]
            ax2 = axes[self.i_row, self.j_column]
            d[15:20, 15:20, :] = np.array([1.0, 0, 0])
            ax2.imshow(d)

    sgdgi.ead_coll.add(eds3)

    sgdgi.addRow("(2)   ", 104.078183, 30.621087)
    _addEllipse1((16, 16), 1, None, width=12, height=10, edgecolor="darkred")

    add_row("qd4", "(3)   ")
    _addEllipse1((13, 14), 2, None, width=12, height=10, edgecolor="darkred")

    add_row("bj21", "(4)   ")

    def eds2(self, *args, _sgdgi: ShadowGeoDrawGradImage = None, ax=None, i_row=None, j_column=None, d=None, **kwargs):
        if (i_row == 3) and (_sgdgi.columns[j_column].name == "SH-AS-DE"):
            self.i_row = i_row
            self.j_column = j_column
            axes = kwargs["axes"]
            ax2 = axes[self.i_row, self.j_column]
            d[11:16, 23:26, :] = np.array([0, 1.0, 0])
            ax2.imshow(d)

    sgdgi.ead_coll.add(eds2)
    _addEllipse1((22, 14), 3, None, width=12, height=10, edgecolor="darkred")

    add_row("bj20", "(5)   ")
    # _addEllipse1((4, 16), 1, None, width=8, edgecolor="darkred")
    # _addEllipse1((10, 14), 1, None, width=8, edgecolor="blue")
    _addEllipse1((7, 14), 4, None, width=12, height=10, edgecolor="darkred")

    add_row("bj14", "(6)   ")
    _addEllipse1((15, 15), 5, None, width=12, height=10, edgecolor="darkred")

    # sgdgi.addRow("VEG", 104.009266,30.675382)
    # _addEllipse1((18, 10), 6, None, width=12, height=10, edgecolor="darkred")

    # "bj13", bj13 as 提取出植被，降轨没有提取出，升降轨提取错误了
    # "bj14", bj14 裸土升降轨提取的好，其他轨道提取的不好
    # "bj17", bj17 升降轨区域影像的比较效果好，
    # "bj20", bj20 升降轨可以提取出更多的不透水面
    # "bj21", bj21 降轨效果好，升轨效果差，升降轨综合和两者
    # "qd3",  qd3  是裸土，升轨区域有建筑物的特征，降轨区域有裸土的特征
    # "cd1",  cd1  是裸土
    # "qd4",  qd4  AS 和 OPT 错分为了水体
    # "qd5",  qd5  AS 和 DE  存在错分为水体
    # "cd1",  cd1  红房子和裸土
    # "cd2",  cd2  由于AS阴影错分为了水体

    # sgdgi.addRowEllipse(0, xy=(0.5, 0.2), width=10, height=6, angle=0, linewidth=2, fill=False, zorder=2,
    #                     edgecolor="lightgreen", is_ratio=True)
    # sgdgi.addRowEllipse(2, xy=(25, 20), width=10, height=6, angle=0, linewidth=2, fill=False, zorder=2,
    #                     edgecolor="lightgreen", is_ratio=False)
    # sgdgi.addRowEllipse(4, xy=(18, 9), width=10, height=6, angle=0, linewidth=2, fill=False, zorder=2,
    #                     edgecolor="lightgreen", is_ratio=False)

    sgdgi.imshow(n_rows_ex=1.6, n_columns_ex=1.6)
    plt.savefig(r"F:\ProjectSet\Shadow\MkTu\4.2NotShadow\fig_42_2.jpeg", dpi=300)
    plt.show()


def drawRowColumn43():
    sgdgi = ShadowGeoDrawGradImage(61, 61)
    # sgdgi.color_dict = {1: (221,74,76), 2: (214,238,155), 3: (254,212,129), 4: (61,149,184)}
    sgdgi.addImdcs()
    sgdgi.isRowName(True)

    sgdgi.addColumnGoogle("Google Image", 800, 800)
    sgdgi.addColumnRGB("RGB")
    sgdgi.addColumnNRG("NRG")
    # sgdgi.addColumnNDVI("NDVI")
    # sgdgi.addColumnNDWI("NDWI")

    sgdgi.addColumnSAR_ASVV("SAR AS")
    sgdgi.addColumnSAR_DEVV("SAR DE")
    # sgdgi.addColumnSAR_ASVH("AS VH")
    # sgdgi.addColumnSAR_DEVH("DE VH")

    # sgdgi.addColumnImdcKey("SH-AS-DE", "SPL_SH-SVM-TAG-OPTICS-AS-DE")
    # sgdgi.addColumnImdcKey("SH-AS", "SPL_SH-SVM-TAG-OPTICS-AS")
    # sgdgi.addColumnImdcKey("SH-DE", "SPL_SH-SVM-TAG-OPTICS-DE")
    # sgdgi.addColumnImdcKey("SH-OPT", "SPL_SH-SVM-TAG-OPTICS")
    # sgdgi.column_name_map["SH-AS-DE"] = "Opt-AS-DE"
    # sgdgi.column_name_map["SH-AS"] = "Opt-AS"
    # sgdgi.column_name_map["SH-DE"] = "Opt-DE"
    # sgdgi.column_name_map["SH-OPT"] = "Opt"

    sgdgi.addColumnImdcKey("NOSH-AS-DE", "SPL_NOSH-SVM-TAG-OPTICS-AS-DE")
    sgdgi.addColumnImdcKey("NOSH-AS", "SPL_NOSH-SVM-TAG-OPTICS-AS")
    sgdgi.addColumnImdcKey("NOSH-DE", "SPL_NOSH-SVM-TAG-OPTICS-DE")
    sgdgi.addColumnImdcKey("NOSH-OPT", "SPL_NOSH-SVM-TAG-OPTICS")
    sgdgi.column_name_map["NOSH-AS-DE"] = "Opt-AS-DE"
    sgdgi.column_name_map["NOSH-AS"] = "Opt-AS"
    sgdgi.column_name_map["NOSH-DE"] = "Opt-DE"
    sgdgi.column_name_map["NOSH-OPT"] = "Opt"

    def _addEllipse1(xy, n_row=None, n_column=None, not_rows=None, not_columns=None,
                     width=8, height=6, angle=0, linewidth=1.5, fill=False, zorder=2,
                     edgecolor="lightgreen", is_ratio=False):
        sgdgi.addEllipse(xy=xy, n_row=n_row, n_column=n_column, not_rows=not_rows, not_columns=not_columns,
                         width=width, height=height, angle=angle, linewidth=linewidth,
                         fill=fill, zorder=zorder, edgecolor=edgecolor, is_ratio=is_ratio)

    df = sdf_read_csv(r"F:\ProjectSet\Shadow\Analysis\12\mktu423_coors2.csv")
    df.asColumnType("X", float)
    df.asColumnType("Y", float)
    df.indexColumnName("Name")

    def add_row(n, row_name=None):
        if row_name is None:
            row_name = df.loc(n, "Name") + "     "
        sgdgi.addRow(row_name, df.loc(n, "X"), df.loc(n, "Y"))
        print(row_name, df.loc(n, "X"), df.loc(n, "Y"))

    find_rows = ["bj13", "bj14", "bj17", "bj20", "bj21", "qd3", "cd1", "qd4", "qd5", "cd1", "cd2", ]

    # for i in df["Name"][32:35]:
    #     add_row(i)

    sgdgi.addRow("(1)    ", 120.371130, 36.090657)
    _addEllipse1((20, 13), 0, None, width=22, height=20, edgecolor="darkred")
    sgdgi.addRow("(2)    ", 120.391223, 36.137901)
    _addEllipse1((40, 35), 1, None, width=22, height=20, edgecolor="darkred")
    add_row("bj5", "(3)    ")
    _addEllipse1((40, 25), 2, None, width=22, height=20, edgecolor="darkred")
    sgdgi.addRow("(4)    ", 120.377379, 36.090765)
    _addEllipse1((30, 26), 3, None, width=30, height=26, edgecolor="darkred")
    add_row("cd2", "(5)    ")
    _addEllipse1((30, 30), 4, None, width=36, height=16, angle=30, edgecolor="darkred")

    # add_row("qd5", "(3)    ")
    # add_row("bj17", "(1)    ")
    # add_row("cd4", "(5)    ")
    # add_row("cd7", "(6)    ")

    # "bj13", bj13 as 提取出植被，降轨没有提取出，升降轨提取错误了
    # "bj14", bj14 裸土升降轨提取的好，其他轨道提取的不好
    # "bj17", bj17 升降轨区域影像的比较效果好，
    # "bj20", bj20 升降轨可以提取出更多的不透水面
    # "bj21", bj21 降轨效果好，升轨效果差，升降轨综合和两者
    # "qd3",  qd3  是裸土，升轨区域有建筑物的特征，降轨区域有裸土的特征
    # "cd1",  cd1  是裸土
    # "qd4",  qd4  AS 和 OPT 错分为了水体
    # "qd5",  qd5  AS 和 DE  存在错分为水体
    # "cd1",  cd1  红房子和裸土
    # "cd2",  cd2  由于AS阴影错分为了水体

    sgdgi.imshow(n_rows_ex=1.6, n_columns_ex=1.6)
    plt.savefig(r"F:\ProjectSet\Shadow\MkTu\4.2NotShadow\fig_43_1.jpg", dpi=300)
    plt.show()


def method_name5():
    sgdgi = ShadowGeoDrawGradImage(61, 61)
    # sgdgi.color_dict = {1: (221,74,76), 2: (214,238,155), 3: (254,212,129), 4: (61,149,184)}
    sgdgi.addImdcs()
    sgdgi.addColumnGoogle("Google Image", 2400, 2400)
    sgdmc = sgdgi.addColumnRGB("RGB")
    sgdgi.addColumnNRG("NRG")

    # sgdgi.addColumnImdcKey("SH-AS-DE", "SPL_SH-SVM-TAG-OPTICS-AS-DE")
    # sgdgi.addColumnImdcKey("SH-AS", "SPL_SH-SVM-TAG-OPTICS-AS")
    # sgdgi.addColumnImdcKey("SH-DE", "SPL_SH-SVM-TAG-OPTICS-DE")
    # sgdgi.addColumnImdcKey("SH-OPT", "SPL_SH-SVM-TAG-OPTICS")
    sgdgi.addColumnSAR_ASVV("AS")
    sgdgi.addColumnSAR_DEVV("DE")
    # sgdgi.addColumnSAR_ASVH("AS VH")
    # sgdgi.addColumnSAR_DEVH("DE VH")
    # sgdgi.addColumnSAR_ASC11("AS C11")
    # sgdgi.addColumnSAR_DEC11("DE C11")
    # sgdgi.addColumnSAR_ASC22("AS C22")
    # sgdgi.addColumnSAR_DEC22("DE C22")
    sgdgi.addRow("(1)     ", 120.3418499, 36.0884511)
    sgdgi.addRow("(2)     ", 116.4860893, 39.8936288)
    sgdgi.addRow("(3)     ", 120.374407, 36.064563)
    # sgdgi.addRow("IS", 120.332966,36.118072)
    # sgdgi.addRow("IS", 120.3780005, 36.1084047)
    # sgdgi.addRow("IS", 120.353551, 36.082023)
    # sgdgi.addRow("IS", 120.339454, 36.052821)
    # sgdgi.addRow("IS", 120.3418499,36.0884511)
    # sgdgi.addRow("IS", 120.3418499,36.0884511)
    # sgdgi.addRow("IS", 120.3418499,36.0884511)
    # sgdgi.addRow("IS", 120.397521, 36.144224)
    # sgdgi.addRow("IS", 116.363178, 39.858848)
    # sgdgi.addRow("VEG", 116.34989, 39.79670)
    # sgdgi.addRow("VEG", 104.07385, 30.65005)
    # sgdgi.addRow("VEG", 104.13064, 30.62272)
    # sgdgi.addRow("SOIL", 104.07385, 30.65005)
    # sgdgi.addRow("SOIL", 104.13064, 30.62272)
    # sgdgi.addRow("WAT", 104.07385, 30.65005)
    # sgdgi.addRow("WAT", 104.13064, 30.62272)
    # e = sgdgi.addRowEllipse(1, xy=(17, 14), width=20, height=6, angle=0, linewidth=1, fill=False, zorder=2,
    #                         edgecolor="green")
    # fig = plt.figure()
    # axes = fig.subplots(1, 1)
    # axes.add_patch(e)
    sgdgi.imshow(n_rows_ex=2.0, n_columns_ex=2.0)
    # plt.savefig(r"F:\ProjectSet\Shadow\MkTu\4.1Details\fig2.jpeg", dpi=300)
    # fig = plt.figure(figsize=(6, 6), )
    # axes = fig.add_subplot(111, aspect='auto')
    # d = sgdmc.read(120.3418499, 36.0884511)
    # e = _mpl_patches_Ellipse( xy=(50, 60), width=20, height=30, angle=20, linewidth=2, fill=False, zorder=2).fit()
    # axes.add_patch(e)
    # # plt.imshow(d)
    # plt.xlim((0,100))
    # plt.ylim((0,100))
    # print(plt.xlim())
    # print(plt.ylim())
    plt.show()


def method_name4():
    raster_fn = r"F:\ProjectSet\Shadow\Release\ChengDuImages\SH_CD_envi.dat"
    to_dir = r"F:\OpenTitle\tu\3"
    if not os.path.isdir(to_dir):
        os.mkdir(to_dir)
    width = 0.5
    height = 0.5
    drawShadowImageSingle("NDVI", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[SFCBSM(-0.6, 0.9, True, True)])
    drawShadowImageSingle("NDWI", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[SFCBSM(-0.7, 0.8, True, True)])
    drawShadowImageSingle("AS_VV", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[_10log10, SFCBSM(-24.609674, 5.9092603, True, True)])
    drawShadowImageSingle("AS_VH", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[_10log10, SFCBSM(-31.865038, -5.2615275, True, True)])
    drawShadowImageSingle("AS_C11", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[_10log10, SFCBSM(-22.61998, 5.8634768, True, True)])
    drawShadowImageSingle("AS_C22", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[_10log10, SFCBSM(-28.579813, -5.2111626, True, True)])
    drawShadowImageSingle("AS_Lambda1", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[_10log10, SFCBSM(-21.955856, 6.124724, True, True)])
    drawShadowImageSingle("AS_Lambda2", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[_10log10, SFCBSM(-29.869734, -8.284683, True, True)])
    drawShadowImageSingle("DE_VV", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[_10log10, SFCBSM(-27.851603, 5.094706, True, True)])
    drawShadowImageSingle("DE_VH", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[_10log10, SFCBSM(-35.427082, -5.4092093, True, True)])
    drawShadowImageSingle("DE_C11", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[_10log10, SFCBSM(-26.245598, 4.9907513, True, True)])
    drawShadowImageSingle("DE_C22", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[_10log10, SFCBSM(-32.04232, -5.322515, True, True)])
    drawShadowImageSingle("DE_Lambda1", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[_10log10, SFCBSM(-25.503738, 5.2980003, True, True)])
    drawShadowImageSingle("DE_Lambda2", raster_fn, to_dir=to_dir, width=width, height=height,
                          callbacks=[_10log10, SFCBSM(-33.442368, -8.68537, True, True)])


if __name__ == "__main__":
    main()


    def main_t2():
        filelist = [
            r"F:\ProjectSet\Shadow\MkTu\Draw\SH_QD_envi.dat2.npy",
            r"F:\ProjectSet\Shadow\MkTu\Draw\SH_BJ_envi.dat2.npy",
            r"F:\ProjectSet\Shadow\MkTu\Draw\SH_CD_envi.dat2.npy",
        ]
        filelist2 = [
            r"F:\ProjectSet\Shadow\Release\QingDaoImages\SH_QD_envi.dat",
            r"F:\ProjectSet\Shadow\Release\BeiJingImages\SH_BJ_envi.dat",
            r"F:\ProjectSet\Shadow\Release\ChengDuImages\SH_CD_envi.dat",
        ]
        for i, fn in enumerate(filelist):
            GDALRasterRange(filelist2[i]).loadNPY(fn).save(fn + ".json")


    def main_t1():
        import numpy as np
        from matplotlib import patches
        import matplotlib.pyplot as plt

        # 绘制一个椭圆需要制定椭圆的中心，椭圆的长和高
        xcenter, ycenter = 1, 1
        width, height = 0.8, 0.5
        angle = -30  # 椭圆的旋转角度

        # 第一步：创建绘图对象
        fig = plt.figure()
        ax = fig.add_subplot(211, aspect='auto')
        ax.set_xbound(-1, 3)
        ax.set_ybound(-1, 3)

        # 第二步
        e1 = patches.Ellipse((xcenter, ycenter), width, height,
                             angle=angle, linewidth=2, fill=False, zorder=2)

        # 第三步
        ax.add_patch(e1)

        # 第一步
        ax = fig.add_subplot(212, aspect='equal')
        ax.set_xbound(-1, 3)
        ax.set_ybound(-1, 3)

        # 第二步
        e2 = patches.Arc((xcenter, ycenter), width, height,
                         angle=angle, linewidth=2, fill=False, zorder=2)

        # 第三步
        ax.add_patch(e2)

        plt.show()

    # main_t1()
