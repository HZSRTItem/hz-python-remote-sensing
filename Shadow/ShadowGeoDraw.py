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
from osgeo import gdal

from SRTCodes.GDALRasterIO import GDALRasterRange
from SRTCodes.GDALUtils import GDALRasterCenter
from SRTCodes.GeoRasterRW import GeoRasterWrite
from SRTCodes.SRTFeature import SRTFeatureCallBackScaleMinMax as SFCBSM
from SRTCodes.Utils import saveJson, readcsv, changext


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


def readGDALRasterCenter(x, y, win_row_size, win_column_size, channel_list, min_list=None, max_list=None,
                         callback_funcs=None, is_geo=True, no_data=0, file_list=None, geo_ranges=None):
    if file_list is None:
        file_list = []
    if callback_funcs is None:
        callback_funcs = []
    grc = None
    geo_range = None
    for i, fn in enumerate(file_list):
        grc = GDALRasterCenter(channel_list=channel_list, raster_fn=fn)
        if grc.read(x_row=x, y_column=y, win_row_size=win_row_size, win_column_size=win_column_size,
                    is_geo=is_geo, no_data=no_data, ) is not None:
            geo_range = geo_ranges[i]
            break
    for i in range(len(channel_list)):
        if min_list[i] is None:
            min_list[i] = geo_range[channel_list[i]].min
        if max_list[i] is None:
            max_list[i] = geo_range[channel_list[i]].max
    for callback_func in callback_funcs:
        grc.callBack(callback_func)
    n = min([len(min_list), len(max_list)])
    for i in range(n):
        grc.scaleMinMax(d_min=min_list[i], d_max=max_list[i], dim=i)
    return grc.d


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


class ShadowGeoDrawGradImage:

    def __init__(self, win_row_size, win_column_size):
        super().__init__()

        self.win_row_size = win_row_size
        self.win_column_size = win_column_size
        self.columns = []
        self.row_names = []
        self.xys = []

    def addColumnRGB(self, name, min_list=None, max_list=None, win_row_size=None, win_column_size=None, ):
        win_column_size, win_row_size = self.initSize(win_column_size, win_row_size)
        sgdmc = ShadowGeoDrawMultiChannel(
            name=name, win_row_size=win_row_size, win_column_size=win_column_size,
            channel_list=["Red", "Green", "Blue"], min_list=min_list, max_list=max_list,
            callback_funcs=[], is_geo=True, no_data=0,
        )
        self.columns.append(sgdmc)

    def addColumnNRG(self, name, min_list=None, max_list=None, win_row_size=None, win_column_size=None, ):
        win_column_size, win_row_size = self.initSize(win_column_size, win_row_size)
        sgdmc = ShadowGeoDrawMultiChannel(
            name=name, win_row_size=win_row_size, win_column_size=win_column_size,
            channel_list=["NIR", "Red", "Green"], min_list=min_list, max_list=max_list,
            callback_funcs=[], is_geo=True, no_data=0,
        )
        self.columns.append(sgdmc)

    def initSize(self, win_column_size, win_row_size):
        if win_row_size is None:
            win_row_size = self.win_row_size
        if win_column_size is None:
            win_column_size = self.win_column_size
        return win_column_size, win_row_size

    def addColumnSAR(self, name, channel_name, min_list=None, max_list=None, win_row_size=None, win_column_size=None, ):
        win_column_size, win_row_size = self.initSize(win_column_size, win_row_size)
        sgdsc = ShadowGeoDrawSingleChannel(
            name=name, win_row_size=win_row_size, win_column_size=win_column_size,
            channel_list=[channel_name], min_list=min_list, max_list=max_list,
            callback_funcs=[], is_geo=True, no_data=0,
        )
        self.columns.append(sgdsc)

    def addColumnSAR_ASVV(self, name, min_list=None, max_list=None,
                          win_row_size=None, win_column_size=None):
        self.addColumnSAR(name=name, channel_name="AS_VV", min_list=min_list, max_list=max_list,
                          win_row_size=win_row_size, win_column_size=win_column_size)

    def addColumnSAR_ASVH(self, name, min_list=None, max_list=None,
                          win_row_size=None, win_column_size=None):
        self.addColumnSAR(name=name, channel_name="AS_VH", min_list=min_list, max_list=max_list,
                          win_row_size=win_row_size, win_column_size=win_column_size)

    def addColumnSAR_DEVV(self, name, min_list=None, max_list=None,
                          win_row_size=None, win_column_size=None):
        self.addColumnSAR(name=name, channel_name="DE_VV", min_list=min_list, max_list=max_list,
                          win_row_size=win_row_size, win_column_size=win_column_size)

    def addColumnSAR_DEVH(self, name, min_list=None, max_list=None,
                          win_row_size=None, win_column_size=None):
        self.addColumnSAR(name=name, channel_name="DE_VH", min_list=min_list, max_list=max_list,
                          win_row_size=win_row_size, win_column_size=win_column_size)

    def addColumnImdc(self, name, imdc_fns, color_dict=None, win_row_size=None, win_column_size=None):
        win_column_size, win_row_size = self.initSize(win_column_size, win_row_size)
        if color_dict is None:
            color_dict = {}
        sgdcc = ShadowGeoDrawCategoryChannel(
            name, imdc_fns=imdc_fns, win_row_size=win_row_size, win_column_size=win_column_size, is_geo=True, no_data=0,
            color_dict=color_dict
        )
        self.columns.append(sgdcc)

    def addRow(self, name, x, y):
        self.row_names.append(name)
        self.xys.append((x, y))

    def imshow(self, n_rows_ex=1.0, n_columns_ex=1.0):
        n_rows, n_columns = len(self.row_names), len(self.columns)
        fig = plt.figure(
            figsize=(n_columns * n_columns_ex, n_rows * n_rows_ex),
            # dpi=300
        )
        axes = fig.subplots(n_rows, n_columns)
        # fig.tight_layout()
        fig.subplots_adjust(top=0.96, bottom=0.04, left=0.04, right=0.96, hspace=0.04, wspace=0.03)

        for i in range(axes.shape[0]):
            x, y = self.xys[i]
            for j in range(axes.shape[1]):
                ax = axes[i, j]
                if j == 0:
                    ax.set_ylabel(self.row_names[i], fontdict={'family': 'Times New Roman', 'size': 16})
                if i == 0:
                    ax.set_title(self.columns[j].name, fontdict={'family': 'Times New Roman', 'size': 16})
                d = self.columns[j].read(x, y)
                ax.imshow(d)
                ax.set_xticks([])
                ax.set_yticks([])
                pass


def main():
    sgdgi = ShadowGeoDrawGradImage(21, 21)
    sgdgi.addColumnRGB("RGB")
    sgdgi.addColumnNRG("NRG")
    sgdgi.addColumnImdc(
        "IMDC1", imdc_fns=[
            r"F:\ProjectSet\Shadow\Release\BeiJingMods\20231114H094632\SPL_SH-SVM-TAG-OPTICS-AS-DE_imdc.dat",
            r"F:\ProjectSet\Shadow\Release\ChengDuMods\20231117H112558\SPL_SH-SVM-TAG-OPTICS-AS-DE_imdc.dat",
            r"F:\ProjectSet\Shadow\Release\QingDaoMods\20231221H224548\SPL_SH-SVM-TAG-OPTICS-AS-DE_imdc.dat",
        ], color_dict={1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255)})
    sgdgi.addColumnSAR_ASVV("AS VV")
    sgdgi.addColumnSAR_ASVH("AS VH")
    sgdgi.addColumnSAR_DEVV("DE VV")
    sgdgi.addColumnSAR_DEVH("DE VH")

    sgdgi.addRow("ROW 1", 120.330806, 36.135239)
    sgdgi.addRow("ROW 2", 120.397521, 36.144224)
    sgdgi.addRow("ROW 3", 116.363178, 39.858848)
    sgdgi.addRow("ROW 4", 116.34989, 39.79670)
    sgdgi.addRow("ROW 5", 104.07385, 30.65005)
    sgdgi.addRow("ROW 6", 104.13064, 30.62272)

    sgdgi.imshow(n_rows_ex=1.5, n_columns_ex=1.5)

    plt.savefig(r"F:\ProjectSet\Shadow\MkTu\4.1Details\fig1.jpeg", dpi=300)
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
