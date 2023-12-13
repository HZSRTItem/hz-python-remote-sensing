# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowGeoDraw.py
@Time    : 2023/7/15 16:10
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : GEOCodes of ShadowGeoDraw
-----------------------------------------------------------------------------"""

import os.path
from inspect import isfunction

import numpy as np
from PIL import Image
from osgeo import gdal
from osgeo import gdal_array

from SRTCodes.GeoRasterRW import GeoRasterWrite
from SRTCodes.SRTFeature import SRTFeatureCallBack
from SRTCodes.SRTFeature import SRTFeatureCallBackScaleMinMax as SFCBSM
from SRTCodes.Utils import saveJson, readcsv, changext


def coorTrans(geo_trans, x, y):
    column = geo_trans[0] + x * geo_trans[1] + y * geo_trans[2]
    row = geo_trans[3] + x * geo_trans[4] + y * geo_trans[5]
    return row, column


def isCloseInt(d, eps=0.000001):
    d_int = int(d)
    return abs(d - d_int) < eps


class RasterCenter:

    def __init__(self, name, ds, channel_list, save_dict=None, raster_fn=None):
        if save_dict is None:
            save_dict = {}
        self.ds = ds
        self.name = name
        self.channel_list = channel_list
        self.raster_fn = raster_fn
        self.save_dict = {"Name": self.name}
        for k in save_dict:
            self.save_dict[k] = save_dict[k]

    def read(self, x_row=0.0, y_column=0.0, win_row_size=0, win_column_size=0, is_geo=True, no_data=0):
        ds: gdal.Dataset = self.ds
        names = []
        for i in range(ds.RasterCount):
            names.append(ds.GetRasterBand(i + 1).GetDescription())
        channel_list = self.channel_list.copy()

        for i in range(len(channel_list)):
            if isinstance(channel_list[i], str):
                channel_list[i] = names.index(channel_list[i])

        geo_trans = ds.GetGeoTransform()
        inv_geo_trans = gdal.InvGeoTransform(geo_trans)
        n_rows = ds.RasterYSize
        n_columns = ds.RasterXSize
        n_channels = ds.RasterCount
        if is_geo:
            x_row, y_column = coorTrans(inv_geo_trans, x_row, y_column)
        x_row, y_column = int(x_row), int(y_column)

        if win_row_size == 0 and win_column_size == 0:
            d = ds.ReadAsArray(interleave="pixel")
        else:
            row_off0 = x_row - int(win_row_size / 2)
            column_off0 = y_column - int(win_column_size / 2)
            if 0 <= row_off0 < n_rows - win_row_size and 0 <= column_off0 < n_columns - win_column_size:
                d = gdal_array.DatasetReadAsArray(ds, column_off0, row_off0, win_xsize=win_column_size,
                                                  win_ysize=win_row_size, interleave="pixel")
            else:
                row_size, column_size = win_row_size, win_column_size

                if row_off0 < 0:
                    row_off = 0
                    row_size = win_row_size + row_off0
                else:
                    row_off = row_off0

                if column_off0 < 0:
                    column_off = 0
                    column_size = win_column_size + column_off0
                else:
                    column_off = column_off0

                if row_off0 + win_row_size >= n_rows:
                    row_size = n_rows - row_off0

                if column_off0 + win_column_size >= n_columns:
                    column_size = n_columns - column_off0

                if row_size <= 0 or column_size <= 0:
                    raise Exception("Can not get data.")
                else:
                    d0 = gdal_array.DatasetReadAsArray(ds, column_off, row_off, column_size, row_size,
                                                       interleave="pixel")
                    d = np.ones([n_channels, win_row_size, win_column_size]) * no_data
                    x0 = column_off - column_off0
                    y0 = row_off - row_off0
                    d[:, y0:y0 + row_size, x0:x0 + column_size] = d0

        if d is not None:
            if len(d.shape) == 2:
                d = np.expand_dims(d, axis=2)
            else:
                d = d[:, :, channel_list]

        return d


def toMinMax(x):
    d_min = np.expand_dims(np.min(x, axis=(0, 1)), axis=(0, 1))
    d_max = np.expand_dims(np.max(x, axis=(0, 1)), axis=(0, 1))
    x = (x - d_min) / (d_max - d_min)
    return x


class GDALResterCenterDraw(RasterCenter):
    """ GDAL Rester Center Draw """

    GDALDatasetCollection = {}

    def __init__(self, name, raster_fn, channel_list=None):
        self.c_d = None
        if channel_list is None:
            channel_list = [name]
        raster_fn = os.path.abspath(raster_fn)
        if raster_fn not in self.GDALDatasetCollection:
            self.GDALDatasetCollection[raster_fn] = gdal.Open(raster_fn)
        self.raster_fn = raster_fn
        super().__init__(name, self.GDALDatasetCollection[raster_fn], channel_list)

        self.d = None
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

    def read(self, x_row=0.0, y_column=0.0, win_row_size=0, win_column_size=0, is_geo=True, no_data=0):
        self.d = super(GDALResterCenterDraw, self).read(
            x_row=x_row, y_column=y_column, win_row_size=win_row_size,
            win_column_size=win_column_size, is_geo=is_geo, no_data=no_data)

    def scaleMinMax(self, d_min, d_max):
        self.d = np.clip(self.d, d_min, d_max)
        self.d = (self.d - d_min) / (d_max - d_min)

    def callBackFunc(self, callback, *args, **kwargs):
        self.d = callback(self.d, args=args, kwargs=kwargs)

    def callBack(self, callback: SRTFeatureCallBack, *args, **kwargs):
        self.d = callback.fit(self.d, *args, **kwargs)

    def toCategory(self):
        self.c_d = self.d.astype("int")
        self.c_d = self.c_d[:, :, 0]
        to_shape = (self.c_d.shape[0], self.c_d.shape[1], 3)
        self.d = np.zeros(to_shape, dtype="uint8")

    def categoryColor(self, category, color: tuple = (0, 0, 0)):
        self.d[self.c_d == category, :] = np.array(color, dtype="uint8")

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
        self.raster_centers[name] = RasterCenter(name=name, ds=self.ds_dict[raster_fn], channel_list=channel_list,
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
            r_c: RasterCenter = self.raster_centers[name]
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

    def drawImdc(self, name, categorys=None):
        drawShadowImage_Imdc(name, self.x, self.y, self.rows, self.columns, self.raster_fn, to_fn=self.to_dirname,
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

    draw(60, 60, 116.461316,39.896712)


def main():
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
