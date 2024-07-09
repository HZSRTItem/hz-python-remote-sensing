# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : GDALUtils.py.py
@Time    : 2023/8/30 17:00
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of GDALUtils.py
-----------------------------------------------------------------------------"""
import os.path
import random
import xml.etree.ElementTree as ElementTree

import numpy as np
import pandas as pd
from osgeo import __version__
from osgeo import osr, gdal, gdal_array

from SRTCodes.GDALRasterIO import GDALRaster, GDALRasterChannel, GDALRasterRange, saveGTIFFImdc, NPYRaster
from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.NumpyUtils import categoryMap, NumpySampling
from SRTCodes.SRTFeature import SRTFeatureCallBack, SRTFeaturesMemory
from SRTCodes.SRTSample import SRTCategorySampleCollection, SRTSample
from SRTCodes.TrainingUtils import SRTAccuracyConfusionMatrix
from SRTCodes.Utils import readcsv, Jdt, savecsv, changext, getfilenamewithoutext, SRTDataFrame, datasCaiFen, getRandom

RESOLUTION_ANGLE = 0.0000089831529294


def splitChannelsToImages(raster_fn, to_dirname=None, is_jdt=False):
    if to_dirname is None:
        to_dirname = os.path.splitext(raster_fn)[0]
    if not os.path.isdir(to_dirname):
        os.mkdir(to_dirname)
    gr = GDALRaster(raster_fn)
    to_fns = []
    jdt = Jdt(len(gr.names), desc="Split Channels to Images")
    if is_jdt:
        jdt.start()
    for i, name in enumerate(gr.names):
        d = gr.readGDALBand(i + 1)
        if is_jdt:
            jdt.add()
        to_fn = os.path.join(to_dirname, name + ".dat")
        gr.save(d, to_fn, descriptions=[name])
        to_fns.append(to_fn)
    if is_jdt:
        jdt.end()
    return to_fns


def gdalStratifiedRandomSampling(geo_fn, numbers=None, n_max=10, is_geo=True, n_channel=0, spls_n=None):
    if numbers is None:
        numbers = []
    gr = GDALRaster(geo_fn)
    d = gr.readAsArray()
    if len(d.shape) > 2:
        d = d[n_channel, :]
    spls = {}
    if spls_n is None:
        for n in numbers:
            spls[n] = {"n": 0, "data": [], "n_max": n_max}
    else:
        for n in spls_n:
            spls[n] = {"n": 0, "data": [], "n_max": spls_n[n]}
    for i in range(d.shape[0] * d.shape[1]):
        row, column = random.randint(0, d.shape[0] - 1), random.randint(0, d.shape[1] - 1)
        n_select = 0
        for n in spls:
            if spls[n]["n"] < spls[n]["n_max"]:
                if int(d[row, column]) == n:
                    if (row, column) not in spls[n]["data"]:
                        spls[n]["data"].append((row, column))
                        spls[n]["n"] += 1
            else:
                n_select += 1
        if n_select == len(spls):
            break

    spl_ret = []
    for n in spls:
        for row, column in spls[n]["data"]:
            if is_geo:
                row, column = gr.coorRaster2Geo(row + 0.5, column + 0.5)
            spl_ret.append([row, column, n])
    return spl_ret


def samplingToCSV(csv_fn: str, gr: GDALRaster, to_csv_fn: str, x_field="X", y_field="Y",
                  coor_srs="EPSG:4326"):
    d = readcsv(csv_fn)
    x = list(map(float, d[x_field]))
    y = list(map(float, d[y_field]))

    d = samplingSingle(x, y, coor_srs, gr, data_dict=d)
    savecsv(to_csv_fn, d)


def samplingSingle(x, y, coor_srs, gr, data_dict=None, is_jdt=True):
    if data_dict is None:
        data_dict = {}
    srs = osr.SpatialReference()
    srs.SetFromUserInput(coor_srs)
    gr.setDstSrs(srs)
    n = min(len(x), len(y))
    for name in gr.names:
        data_dict[name] = []
    jdt = Jdt(total=n, desc="Sampling To CSV")
    if is_jdt:
        jdt.start()
    for i in range(n):
        d0 = gr.readAsArray(x[i], y[i], win_row_size=1, win_column_size=1, is_trans=True, is_geo=True)
        if d0 is None:
            for j, name in enumerate(gr.names):
                data_dict[name].append(0)
            continue
        d0 = d0.ravel()
        for j, name in enumerate(gr.names):
            data_dict[name].append(d0[j])
        if is_jdt:
            jdt.add()
    if is_jdt:
        jdt.end()
    return data_dict


def vrtAddDescriptions(filename, to_filename=None, descriptions=None):
    if descriptions is None:
        descriptions = []
    if to_filename is None:
        to_filename = filename
    tree = ElementTree.parse(filename)
    root = tree.getroot()
    i = 0
    for node in root.findall("VRTRasterBand"):
        element = ElementTree.Element("Description")
        element.text = descriptions[i]
        node.append(element)
        i += 1
    tree.write(to_filename, encoding='utf-8', xml_declaration=True)


class RasterToVRTS:

    def __init__(self, raster_fn):
        self.raster_fn = raster_fn
        self.gr = GDALRaster(raster_fn)
        self.to_dirname = os.path.splitext(raster_fn)[0] + "_VRTS"
        self.feat_names = self.gr.names
        self.save_fns = ["{0}.vrt".format(name) for name in self.feat_names]

    def frontStr(self, fstr):
        self.save_fns = ["{0}{1}".format(fstr, name) for name in self.save_fns]
        return self

    def save(self, to_dirname=None):
        if to_dirname is None:
            to_dirname = self.to_dirname
        if not os.path.isdir(to_dirname):
            os.mkdir(to_dirname)
        for i, fn in enumerate(self.save_fns):
            fn = os.path.join(to_dirname, fn)
            gdal.BuildVRT(fn, self.raster_fn, options=["-b", str(i + 1)])


def dictRasterToVRT(to_fn, raster_dict: dict):
    gdal.BuildVRT(to_fn, list(raster_dict.values()), options=["-separate"])
    vrtAddDescriptions(to_fn, descriptions=list(raster_dict.keys()))


class SRTRasterConcat:
    GRS = {}

    def __init__(self, to_dirname, to_fn=None, is_q=True):
        if os.path.split(to_fn)[0] == "":
            to_fn = os.path.join(to_dirname, to_fn)
        self.to_fn = to_fn
        self.datas = []
        self.to_dirname = to_dirname
        self.is_q = is_q

    def add(self, fn, c=None, des=None):
        fn = os.path.abspath(fn)
        if fn not in self.GRS:
            self.GRS[fn] = GDALRaster(fn)
        gr: GDALRaster = self.GRS[fn]
        if c is None:
            c = 1
        else:
            if isinstance(c, int):
                ...
            elif isinstance(c, str):
                if c in gr.names:
                    c = gr.names.index(c) + 1
        if des is None:
            des = gr.names[c - 1]
        self.datas.append({
            "src_fn": fn,
            "to_fn": os.path.join(self.to_dirname, des + ".vrt"),
            "channel": c,
            "description": des
        })
        return des

    def fit(self, to_fn=None):
        if to_fn is None:
            to_fn = self.to_fn
        src_fns = []
        descriptions = []
        for i, data in enumerate(self.datas):
            gdal.BuildVRT(data["to_fn"], data["src_fn"], options=["-r", "bilinear", "-b", str(data["channel"])])
            src_fns.append(data["to_fn"])
            descriptions.append(data["description"])
        if not self.is_q:
            print("Build VRT")
        gdal.BuildVRT(to_fn, src_fns, options=["-separate", "-r", "bilinear"])
        if not self.is_q:
            print("Add Descriptions")
        vrtAddDescriptions(to_fn, descriptions=descriptions)
        if not self.is_q:
            print("Split To Images")
        envi_fns = splitChannelsToImages(to_fn, self.to_dirname, is_jdt=(not self.is_q))
        if not self.is_q:
            print("Translate to ENVI")
        envi_fn = changext(to_fn, "_envi.dat")
        vrt_fn = to_fn
        gdal.BuildVRT(vrt_fn, envi_fns, options=["-separate", "-r", "bilinear"])
        vrtAddDescriptions(vrt_fn, descriptions=descriptions)
        gdal.Translate(envi_fn, vrt_fn, options=["-of", "ENVI"])

    def adds(self, fn, c_list, des_list=None):
        if des_list is None:
            des_list = [None for _ in range(len(c_list))]
        for i in range(len(c_list)):
            self.add(fn, c_list[i], des_list[i])


class GDALRasterHist:
    """ GDAL Raster Hist """

    GRS = {}

    def __init__(self, raster_fn=None):
        self.grc = GDALRasterChannel()
        """
        MIN, MAX, N, 
        """

    def show(self, field_name, raster_fn=None, d_min=None, d_max=None, n_split=256, channel=None):
        if field_name not in self.grc:
            self.grc.addGDALData(raster_fn=raster_fn, field_name=field_name, channel=channel)
        d = self.grc[field_name]
        if d_min is not None:
            d[d < d_min] = d_min
        if d_max is not None:
            d[d > d_max] = d_max
        y, x = np.histogram(d, bins=n_split)
        x = x[:-1]
        return x, y
        # plt.plot(x, y, label=field_name)


class GDALRasterCenter:
    GDALDatasetCollection = {}

    def __init__(self, name=None, ds=None, channel_list=None, save_dict=None, raster_fn=None):
        if save_dict is None:
            save_dict = {}
        if channel_list is None:
            channel_list = [0]
        self.wgs84_to_this = None
        ds, raster_fn = self.initRaster(ds, raster_fn)
        self.ds = ds
        self.name = name
        self.channel_list = channel_list
        self.raster_fn = raster_fn
        self.save_dict = {"Name": self.name}
        for k in save_dict:
            self.save_dict[k] = save_dict[k]
        self.c_d = None
        self.d = None

    def initRaster(self, ds, raster_fn):
        if ds is None:
            if raster_fn is not None:
                raster_fn = os.path.abspath(raster_fn)
                if raster_fn not in self.GDALDatasetCollection:
                    self.GDALDatasetCollection[raster_fn] = gdal.Open(raster_fn)
                ds = self.GDALDatasetCollection[raster_fn]

        wkt = ds.GetProjection()
        if wkt != "":
            dst_srs = osr.SpatialReference()
            dst_srs.ImportFromWkt(wkt)
            wgs84_srs = osr.SpatialReference()
            wgs84_srs.ImportFromEPSG(4326)
            if __version__ >= "3.0.0":
                dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                wgs84_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            self.wgs84_to_this = osr.CoordinateTransformation(wgs84_srs, dst_srs)

        self.ds = ds
        return ds, raster_fn

    def read(self, x_row=0.0, y_column=0.0, win_row_size=0, win_column_size=0, is_geo=True, no_data=0, is_trans=False,
             *args, **kwargs):
        ds: gdal.Dataset = self.ds
        names = []
        for i in range(ds.RasterCount):
            names.append(ds.GetRasterBand(i + 1).GetDescription())
        channel_list = self.channel_list.copy()

        for i in range(len(channel_list)):
            if isinstance(channel_list[i], str):
                channel_list[i] = names.index(channel_list[i])
        d = None
        geo_trans = ds.GetGeoTransform()
        inv_geo_trans = gdal.InvGeoTransform(geo_trans)
        n_rows = ds.RasterYSize
        n_columns = ds.RasterXSize
        n_channels = ds.RasterCount
        if is_geo:
            if is_trans:
                x_row, y_column, _ = self.wgs84_to_this.TransformPoint(x_row, y_column)
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
                    d = None
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
            d = d * 1.0

        self.d = d
        return d

    def initGeoRaster(self, channel_list, name, raster_fn):
        if channel_list is None:
            channel_list = [name]
        raster_fn = os.path.abspath(raster_fn)
        if raster_fn not in self.GDALDatasetCollection:
            self.GDALDatasetCollection[raster_fn] = gdal.Open(raster_fn)
        return channel_list, raster_fn

    def scaleMinMax(self, d_min, d_max, dim=None, is_01=True):
        if dim is None:
            if d_min is None:
                d_min = np.min(self.d)
            if d_max is None:
                d_max = np.max(self.d)
            self.d = np.clip(self.d, d_min, d_max)
            if is_01:
                self.d = (self.d - d_min) / (d_max - d_min)
        else:
            if d_min is None:
                d_min = np.min(self.d[:, :, dim])
            if d_max is None:
                d_max = np.max(self.d[:, :, dim])
            self.d[:, :, dim] = np.clip(self.d[:, :, dim], d_min, d_max)
            if is_01:
                self.d[:, :, dim] = (self.d[:, :, dim] - d_min) / (d_max - d_min)

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


def coorTrans(geo_trans, x, y):
    column = geo_trans[0] + x * geo_trans[1] + y * geo_trans[2]
    row = geo_trans[3] + x * geo_trans[4] + y * geo_trans[5]
    return row, column


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


class GDALRasterCenterCollection:

    def __init__(self, geo_image_fns=None, geo_ranges=None, channel_list=None, win_size=None,
                 is_min_max=False, is_01=False, min_list=None, max_list=None,
                 no_data=0.0, callback_funcs=None):

        if geo_ranges is None:
            geo_ranges = {}
        if win_size is None:
            win_size = [1, 1]
        if geo_image_fns is None:
            geo_image_fns = []

        self.win_size = win_size
        self.geo_image_fns = geo_image_fns
        self.geo_ranges = geo_ranges
        self.is_min_max = is_min_max
        self.is_01 = is_01
        self.min_list = min_list
        self.max_list = max_list
        self.channel_list = channel_list
        self.no_data = no_data
        self.callback_funcs = callback_funcs

        self.geo_fn = None
        self.grc = None
        self.geo_range = None

    def addGEOImage(self, geo_image_fn, geo_range_fn=None):
        geo_image_fn = os.path.abspath(geo_image_fn)
        if geo_range_fn is not None:
            self.geo_ranges[geo_image_fn] = GDALRasterRange(range_fn=geo_range_fn)
        if geo_range_fn not in self.geo_image_fns:
            self.geo_image_fns.append(geo_image_fn)

    def getPatch(self, x, y, win_size=None, channel_list=None,
                 is_min_max=None, is_01=None, min_list=None, max_list=None,
                 callback_funcs=None, is_geo=True, no_data=None, is_trans=False):

        if win_size is None:
            win_size = self.win_size
        if is_min_max is None:
            is_min_max = self.is_min_max
        if is_01 is None:
            is_01 = self.is_01
        if channel_list is None:
            channel_list = self.channel_list
        if no_data is None:
            no_data = self.no_data
        if callback_funcs is None:
            callback_funcs = self.callback_funcs
        self.grc = None
        self.geo_range = None

        if self.geo_fn is not None:
            self.grc = GDALRasterCenter(channel_list=channel_list, raster_fn=self.geo_fn)
            if self.geo_fn in self.geo_ranges:
                self.geo_range = self.geo_ranges[self.geo_fn]

        if self.grc is not None:
            if not (self.grc.read(x_row=x, y_column=y, win_row_size=win_size[0], win_column_size=win_size[1],
                                  is_geo=is_geo, no_data=no_data, is_trans=is_trans) is not None):
                self.grc = None
                self.geo_range = None

        if self.grc is None:
            for i, fn in enumerate(self.geo_image_fns):
                self.grc = GDALRasterCenter(channel_list=channel_list, raster_fn=fn)
                if self.grc.read(x_row=x, y_column=y, win_row_size=win_size[0], win_column_size=win_size[1],
                                 is_geo=is_geo, no_data=no_data, is_trans=is_trans) is not None:
                    if fn in self.geo_ranges:
                        self.geo_range = self.geo_ranges[fn]
                    break

        if is_min_max:
            if min_list is None:
                min_list = [None for _ in range(len(channel_list))]
            if max_list is None:
                max_list = [None for _ in range(len(channel_list))]
            for i in range(len(channel_list)):
                if min_list[i] is None:
                    if self.geo_range is not None:
                        min_list[i] = self.geo_range[channel_list[i]].min
                if max_list[i] is None:
                    if self.geo_range is not None:
                        max_list[i] = self.geo_range[channel_list[i]].max
            n = min([len(min_list), len(max_list)])
            for i in range(n):
                self.grc.scaleMinMax(d_min=min_list[i], d_max=max_list[i], dim=i, is_01=is_01)

        if callback_funcs is not None:
            for callback_func in callback_funcs:
                self.grc.callBack(callback_func)

        return self.grc.d


class GDALRasterCenterDatas:

    def __init__(self, win_size=None, is_min_max=False, is_01=False):
        # super().__init__(geo_image_fns, geo_ranges, win_size, is_min_max, is_01, min_list, max_list)

        self.data = [[None]]
        self.rasters_fns_geo_ranges = {}
        self.win_size = win_size
        self.is_min_max = is_min_max
        self.is_01 = is_01
        self.grccs = {}
        self.category_colors = {}

    def keys(self):
        return list(self.grccs.keys())

    def changeDataList(self, n_row, n_column):
        if n_row >= len(self.data):
            for i in range(n_row - len(self.data) + 1):
                self.data.append([None for _ in range(len(self.data[0]))])
        if n_column >= len(self.data[0]):
            n_column_tmp = len(self.data[0])
            for i in range(len(self.data)):
                for j in range(n_column - n_column_tmp + 1):
                    self.data[i].append(None)
        return n_row, n_column

    def addRasterCenterCollection(self, name, *raster_fns, channel_list=None, fns=None, win_size=None, is_min_max=None,
                                  is_01=None, no_data=0.0, min_list=None, max_list=None, callback_funcs=None):
        if fns is None:
            fns = []
        raster_fns = list(raster_fns) + fns
        if win_size is None:
            win_size = self.win_size
        if is_min_max is None:
            is_min_max = self.is_min_max
        if is_01 is None:
            is_01 = self.is_01

        geo_ranges = {}
        raster_fns_tmp = []
        for raster_fn in raster_fns:
            if raster_fn in self.rasters_fns_geo_ranges:
                geo_ranges[raster_fn] = self.rasters_fns_geo_ranges[raster_fn]
            else:
                for geo_ranges_fn in self.rasters_fns_geo_ranges:
                    fn = getfilenamewithoutext(geo_ranges_fn)
                    if fn == raster_fn:
                        raster_fn = geo_ranges_fn
                        geo_ranges[raster_fn] = self.rasters_fns_geo_ranges[raster_fn]
                        break
            raster_fns_tmp.append(raster_fn)

        grcc = GDALRasterCenterCollection(geo_image_fns=raster_fns_tmp, geo_ranges=geo_ranges, win_size=win_size,
                                          is_01=is_01, is_min_max=is_min_max, channel_list=channel_list,
                                          callback_funcs=callback_funcs, no_data=no_data,
                                          min_list=min_list, max_list=max_list)
        self.grccs[name] = grcc
        return name

    def addRCC(self, name, *raster_fns, channel_list=None, fns=None, win_size=None, is_min_max=None,
               is_01=None, no_data=0.0, min_list=None, max_list=None, callback_funcs=None):
        return self.addRasterCenterCollection(
            name, *raster_fns, channel_list=channel_list, fns=fns, win_size=win_size, is_min_max=is_min_max,
            is_01=is_01, no_data=no_data, min_list=min_list, max_list=max_list, callback_funcs=callback_funcs
        )

    def addGeoRange(self, raster_fn, geo_range=None):
        if geo_range is not None:
            if isinstance(geo_range, str):
                geo_range = GDALRasterRange().loadJsonFile(geo_range)
            elif isinstance(geo_range, dict):
                geo_range = GDALRasterRange().loadJsonFile(geo_range)
        raster_fn = os.path.abspath(raster_fn)
        self.rasters_fns_geo_ranges[raster_fn] = geo_range
        return getfilenamewithoutext(raster_fn)

    def addCategoryColor(self, name, *category_colors, **category_color):
        c_dict = {}
        i = 0
        c_name = ""
        for c_color in category_colors:
            if isinstance(c_color, dict):
                for k in c_color:
                    c_dict[k] = c_color[k]
            else:
                if i % 2 == 0:
                    c_name = c_color
                else:
                    c_dict[c_name] = c_color
        for k in category_color:
            c_dict[k] = category_color[k]
        self.category_colors[name] = c_dict
        return name

    def addAxisDataXY(self, n_row, n_column, grcc_name, x, y, win_size=None, is_trans=False, *args, **kwargs):
        n_row, n_column = self.changeDataList(n_row, n_column)
        grcc: GDALRasterCenterCollection = self.grccs[grcc_name]
        min_list, max_list = None, None
        if "min_list" in kwargs:
            min_list = kwargs["min_list"]
        if "max_list" in kwargs:
            max_list = kwargs["max_list"]

        d = grcc.getPatch(x, y, win_size=win_size, min_list=min_list, max_list=max_list, is_trans=is_trans)

        self.setData(n_row, n_column, d)
        return d

    def __getitem__(self, item):
        n_row, n_column = item
        return self.getData(n_column, n_row)

    def getData(self, n_row, n_column):
        return self.data[n_row][n_column]

    def setData(self, n_row, n_column, data):
        self.data[n_row][n_column] = data

    def shape(self, dim=None):
        if dim is None:
            return len(self.data), len(self.data[0])
        else:
            if dim == 0:
                return len(self.data)
            elif dim == 1:
                return len(self.data[0])


class SRTGDALCategorySampleCollection(SRTCategorySampleCollection):

    def gdalSampling(self, gdal_raster_fn, spl_size=(1, 1), is_to_field=False, no_data=0, is_jdt=False,
                     is_sampling=None, **kwargs):
        gr = GDALRaster(gdal_raster_fn)

        if is_to_field:
            spl_size = (1, 1)
        is_sampling_isin = False
        if is_sampling is not None:
            is_sampling_isin = is_sampling in self.field_names
            self.addFieldName(is_sampling)

        def is_sampling_func(_spl, is_find):
            if is_sampling is not None:
                if is_sampling_isin:
                    if _spl[is_sampling] == 0:
                        _spl[is_sampling] = is_find
                else:
                    _spl[is_sampling] = is_find

        def is_sampling_func2(_spl):
            if is_sampling is not None:
                if is_sampling_isin:
                    return _spl[is_sampling] == 1
            return False

        jdt = Jdt(len(self.samples), "ShadowHierarchicalSampleCollection::gdalSampling")

        if spl_size == (1, 1):
            if is_jdt:
                jdt.start()
            for spl in self.samples:
                if is_jdt:
                    jdt.add()

                if is_sampling_func2(spl):
                    continue

                x, y = spl.x, spl.y

                if gr.isGeoIn(x, y):
                    d = gr.readAsArray(x, y, win_row_size=1, win_column_size=1, is_geo=True).ravel()
                    is_sampling_func(spl, 1)
                else:
                    d = np.ones(gr.n_channels) * no_data
                    is_sampling_func(spl, 0)

                if is_to_field:
                    for i, name in enumerate(gr.names):
                        self.addFieldName(name)
                        spl[name] = float(d[i])
                else:
                    spl.data = d

            if is_jdt:
                jdt.end()
        else:
            if is_jdt:
                jdt.start()
            for spl in self.samples:
                if is_jdt:
                    jdt.add()

                if is_sampling_func2(spl):
                    continue

                x, y = spl.x, spl.y

                if gr.isGeoIn(x, y):
                    d = gr.readAsArrayCenter(x, y, win_row_size=spl_size[0], win_column_size=spl_size[1], is_geo=True)
                else:
                    d = np.ones((gr.n_channels, spl_size[0], spl_size[1])) * no_data
                    is_sampling_func(spl, 0)
                if d is None:
                    d = np.ones((gr.n_channels, spl_size[0], spl_size[1])) * no_data
                    is_sampling_func(spl, 0)
                    print(spl.srt, spl.x, spl.y, )
                else:
                    is_sampling_func(spl, 1)
                spl.data = d

            if is_jdt:
                jdt.end()

        return

    def gdalSamplingRasters(self, gdal_raster_fns, spl_size=(1, 1), is_to_field=False, no_data=None,
                            is_jdt=True, field_names=None, ):
        grs = [GDALRaster(fn) for fn in gdal_raster_fns]

        gr = grs[0]
        if field_names is None:
            field_names = gr.names.copy()

        jdt = Jdt(len(self.samples), "SRTGDALCategorySampleCollection::gdalSamplingRasters")

        if is_jdt:
            jdt.start()
        for i in range(len(self.samples)):
            if is_jdt:
                jdt.add()

            spl: SRTSample = self.samples[i]
            x, y = spl.x, spl.y

            if not gr.isGeoIn(x, y):
                for gr in grs:
                    if gr.isGeoIn(x, y):
                        break

            d = gr.readAsArrayCenter(x, y, spl_size[0], spl_size[1], is_geo=True)
            if d is None:
                if no_data is not None:
                    d = np.ones((gr.n_channels, spl_size[0], spl_size[1])) * no_data

            if is_to_field:
                if (d is not None) and (spl_size == (1, 1)):
                    data = d.ravel().tolist()
                    for j in range(len(data)):
                        spl[field_names[j]] = float(data[j])
                        self.addFieldName(field_names[j])

            spl.data = d

        if is_jdt:
            jdt.end()

    def copyNoSamples(self):
        scsc = SRTGDALCategorySampleCollection()
        self.copySCSC(scsc)
        return scsc


class GDALRastersSampling:

    def __init__(self, *raster_fns):
        self.rasters = {}
        self._gr = GDALRaster()
        self.add(*raster_fns)

    def toDict(self):
        to_dict = {
            "rasters": {raster: self.rasters[raster].toDict() for raster in self.rasters}
        }
        return to_dict

    def getNames(self):
        return self._gr.names.copy()

    def add(self, *raster_fns):
        raster_fns_deal = []
        for raster_fn in raster_fns:
            if isinstance(raster_fn, list) or isinstance(raster_fn, tuple):
                raster_fns_deal.extend(raster_fn)
            else:
                raster_fns_deal.append(raster_fn)
        raster_fns = raster_fns_deal
        for raster_fn in raster_fns:
            if raster_fn not in self.rasters:
                self.rasters[raster_fn] = GDALRaster(raster_fn)
                if self._gr.raster_ds is None:
                    self._gr = self.rasters[raster_fn]

    def sampling(self, x, y, win_row_size=1, win_column_size=1, interleave='band', band_list=None, no_data=0.0,
                 is_none=True):
        d = None
        if self._gr.isGeoIn(x, y):
            d = self._gr.readAsArrayCenter(
                x_row_center=x, y_column_center=y, win_row_size=win_row_size, win_column_size=win_column_size,
                interleave=interleave, band_list=band_list, no_data=no_data, is_geo=True,
            )
        else:
            for self._gr in self.rasters.values():
                if self._gr.isGeoIn(x, y):
                    d = self._gr.readAsArrayCenter(
                        x_row_center=x, y_column_center=y, win_row_size=win_row_size, win_column_size=win_column_size,
                        interleave=interleave, band_list=band_list, no_data=no_data, is_geo=True,
                    )
                    break
        if d is None:
            if not is_none:
                return np.ones([win_row_size, win_column_size, self._gr.n_channels]) * no_data
        return d

    def samplingIter(self, x_iter, y_iter, win_row_size=1, win_column_size=1, interleave='band',
                     band_list=None, no_data=0, is_none_error=False):
        d_list = []
        while True:
            try:
                x, y = next(x_iter), next(y_iter)
                d = self.sampling(x, y, win_row_size=win_row_size, win_column_size=win_column_size,
                                  interleave=interleave, band_list=band_list, no_data=no_data)
                d_list.append(d)
            except StopIteration:
                break

    @property
    def gr(self):
        return self._gr


class GDALSamplingInit:

    def __init__(self, geo_fn=None, gr=None, *args, **kwargs):
        self.gr = GDALRaster()
        if gr is not None:
            self.gr = gr
        else:
            if geo_fn is not None:
                self.gr = GDALRaster(geo_fn)
        self.n_channels = self.gr.n_channels

    def initNPYRaster(self, npy_fn):
        self.gr = NPYRaster(npy_fn)
        self.n_channels = self.gr.n_channels

    def sampling(self, x, y, field_names=None, is_jdt=True, is_trans=False):
        if field_names is None:
            field_names = self.gr.names
        to_dict = {field_name: [0 for i in range(len(x))] for field_name in field_names}
        jdt = Jdt(len(x), "sampling").start(is_jdt)
        for i in range(len(x)):
            x0, y0 = x[i], y[i]
            if is_trans:
                x0, y0, _ = self.gr.wgs84_to_this.TransformPoint(x0, y0)
            line = self.samplingOne(x0, y0)
            for k in line:
                if k in to_dict:
                    to_dict[k][i] = line[k]
            jdt.add(is_jdt=is_jdt)
        jdt.end(is_jdt=is_jdt)
        return to_dict

    def samplingOne(self, x0, y0, *args, **kwargs):
        to_dict = {field_name: 0 for field_name in self.gr.names}
        return to_dict

    def csvfile(self, csv_fn, to_csv_fn, field_names=None, is_jdt=True,
                x_field_names="X", y_field_names="Y", is_trans=False):
        df = pd.read_csv(csv_fn)
        to_df = self.samplingDF(df, field_names=field_names, is_jdt=is_jdt,
                                x_field_names=x_field_names, y_field_names=y_field_names, is_trans=is_trans, )
        to_df.to_csv(to_csv_fn, index=False)
        return to_csv_fn

    def samplingDF(self, df, field_names=None, is_jdt=True,
                   x_field_names="X", y_field_names="Y", is_trans=False):
        df_list = df.to_dict("list")
        to_dict = self.sampling(
            df_list[x_field_names], df_list[y_field_names],
            field_names=field_names, is_jdt=is_jdt, is_trans=is_trans
        )
        for k in to_dict:
            df_list[k] = to_dict[k]
        to_df = pd.DataFrame(df_list)
        return to_df

    def sampling2(self, x, y, win_row, win_column, is_jdt=True, is_trans=False):
        win_spl = [0, 0, 0, 0]
        win_spl[0] = 0 - int(win_row / 2)
        win_spl[1] = 0 + round(win_row / 2 + 0.1)
        win_spl[2] = 0 - int(win_column / 2)
        win_spl[3] = 0 + round(win_column / 2 + 0.1)

        jdt = Jdt(len(x), "sampling2").start(is_jdt)
        to_data = np.zeros((len(x), self.n_channels, win_row, win_column))
        for i in range(len(x)):
            x0, y0 = x[i], y[i]
            if is_trans:
                x0, y0, _ = self.gr.wgs84_to_this.TransformPoint(x0, y0)
            data = self.sampling2One(win_spl, x0, y0, win_row=win_row, win_column=win_column)
            if data is not None:
                to_data[i] = data
            jdt.add(is_jdt=is_jdt)
        jdt.end(is_jdt=is_jdt)
        return to_data

    def sampling2One(self, win_spl, x0, y0, win_row=1, win_column=1, *args, **kwargs):
        return None

    def csvfile2(self, csv_fn, to_npy_fn, win_row, win_column,
                 x_field_names="X", y_field_names="Y", is_jdt=True, is_trans=False):
        df = pd.read_csv(csv_fn)
        to_data = self.sampling2DF(df, win_row, win_column, x_field_names, y_field_names, is_jdt, is_trans)
        np.save(to_npy_fn, to_data.astype("float32"))
        del to_data
        return to_npy_fn

    def sampling2DF(self, df, win_row, win_column, x_field_names="X", y_field_names="Y", is_jdt=True, is_trans=False):
        return self.sampling2(
            df[x_field_names].tolist(), df[y_field_names].tolist(),
            win_row, win_column,
            is_jdt=is_jdt, is_trans=is_trans
        )


class GDALSamplingFast(GDALSamplingInit):

    def __init__(self, raster_fn, gr=None, *args, **kwargs):
        super().__init__(geo_fn=raster_fn, gr=gr, *args, **kwargs)
        self.data = self.gr.readAsArray()
        if len(self.data.shape) == 2:
            self.data = np.array([self.data])

    def samplingOne(self, x0, y0, *args, **kwargs):
        if self.gr.isGeoIn(x0, y0):
            row, column = self.gr.coorGeo2Raster(x0, y0, is_int=True)
            data = self.data[:, row, column]
            data = data.ravel()
            to_dict = {field_name: float(data[i]) for i, field_name in enumerate(self.gr.names)}
        else:
            to_dict = {field_name: 0 for field_name in self.gr.names}
        return to_dict

    def sampling2One(self, win_spl, x0, y0, *args, **kwargs):
        if self.gr.isGeoIn(x0, y0):
            row, column = self.gr.coorGeo2Raster(x0, y0, is_int=True)
            data = self.data[:, row + win_spl[0]: row + win_spl[1], column + win_spl[2]: column + win_spl[3]]
        else:
            data = None
        return data

    # def csvfile(self, csv_fn, to_csv_fn, field_names=None, is_jdt=True,
    #             x_field_names="X", y_field_names="Y", is_trans=False):
    #
    #     # def csvfile(self, csv_fn, to_csv_fn, is_jdt=True, field_names=None):
    #
    #
    #     if field_names is None:
    #         field_names = self.gr.names
    #     sdf = SRTDataFrame().read_csv(csv_fn, is_auto_type=True)
    #     sdf.addFields(field_names)
    #     jdt = Jdt(len(sdf), "GDALSamplingFast::csvfile").start(is_jdt=is_jdt)
    #     for i in range(len(sdf)):
    #         x, y = sdf["X"][i], sdf["Y"][i]
    #         if not self.gr.isGeoIn(x, y):
    #             data = np.ones(self.gr.n_channels)
    #         else:
    #             row, column = self.gr.coorGeo2Raster(x, y, is_int=True)
    #             data = self.data[:, row, column]
    #         data = data.ravel()
    #         for j, name in enumerate(field_names):
    #             sdf[name][i] = float(data[j])
    #         jdt.add(is_jdt=is_jdt)
    #     jdt.end(is_jdt=is_jdt)
    #     sdf.toCSV(to_csv_fn)
    #
    # def sampling(self, x, y, field_names=None, is_jdt=True, ):
    #     if field_names is None:
    #         field_names = self.gr.names
    #     to_dict = {field_name: [0 for i in range(len(x))] for field_name in field_names}
    #     jdt = Jdt(len(x), "GDALSamplingFast::sampling").start(is_jdt=is_jdt)
    #     for i in range(len(x)):
    #         x0, y0 = x[i], y[i]
    #         if not self.gr.isGeoIn(x0, y0):
    #             data = np.zeros(self.gr.n_channels)
    #         else:
    #             row, column = self.gr.coorGeo2Raster(x0, y0, is_int=True)
    #             data = self.data[:, row, column]
    #         data = data.ravel()
    #         for j, name in enumerate(field_names):
    #             to_dict[name][i] = float(data[j])
    #         jdt.add(is_jdt=is_jdt)
    #     jdt.end(is_jdt=is_jdt)
    #     return to_dict
    #
    # def sampling2(self, x, y, win_row, win_column, is_jdt=True, is_trans=False):
    #     win_spl = [0, 0, 0, 0]
    #     win_spl[0] = 0 - int(win_row / 2)
    #     win_spl[1] = 0 + round(win_row / 2 + 0.1)
    #     win_spl[2] = 0 - int(win_column / 2)
    #     win_spl[3] = 0 + round(win_column / 2 + 0.1)
    #     jdt = Jdt(len(x), "GDALSamplingFast::sampling2").start(is_jdt=is_jdt)
    #     to_data = np.zeros((len(x), self.data.shape[0], win_row, win_column))
    #     for i in range(len(x)):
    #         x0, y0 = x[i], y[i]
    #         if is_trans:
    #             x0, y0, _ = self.gr.wgs84_to_this.TransformPoint(x0, y0)
    #         if self.gr.isGeoIn(x0, y0):
    #             row, column = self.gr.coorGeo2Raster(x0, y0, is_int=True)
    #             data = self.data[:, row + win_spl[0]: row + win_spl[1], column + win_spl[2]: column + win_spl[3]]
    #             to_data[i] = data
    #         to_data = to_data * 0
    #         jdt.add(is_jdt=is_jdt)
    #     jdt.end(is_jdt=is_jdt)
    #     return to_data


class GDALSampling(GDALSamplingInit):

    def __init__(self, raster_fn=None, gr=None, *args, **kwargs):
        super().__init__(geo_fn=raster_fn, gr=gr, *args, **kwargs)

    def samplingOne(self, x0, y0, *args, **kwargs):
        if self.gr.isGeoIn(x0, y0):
            data = self.gr.readAsArray(x0, y0, win_row_size=1, win_column_size=1, is_geo=True)
            data = data.ravel()
            to_dict = {field_name: float(data[i]) for i, field_name in enumerate(self.gr.names)}
        else:
            to_dict = {field_name: 0 for field_name in self.gr.names}
        return to_dict

    def sampling2One(self, win_spl, x0, y0, win_row=1, win_column=1, *args, **kwargs):
        if self.gr.isGeoIn(x0, y0):
            data = self.gr.readAsArrayCenter(x0, y0, win_row, win_column, is_geo=True)
        else:
            data = None
        return data

    # def sampling(self, x, y, field_names=None, is_jdt=True, ):
    #     if field_names is None:
    #         field_names = self.gr.names
    #     to_dict = {field_name: [0 for i in range(len(x))] for field_name in field_names}
    #     jdt = Jdt(len(x), "GDALSamplingFast::sampling").start(is_jdt=is_jdt)
    #     for i in range(len(x)):
    #         x0, y0 = x[i], y[i]
    #         if not self.gr.isGeoIn(x0, y0):
    #             data = np.zeros(self.gr.n_channels)
    #         else:
    #             data = self.gr.readAsArray(x0, y0, win_row_size=1, win_column_size=1, is_geo=True)
    #         data = data.ravel()
    #         for j, name in enumerate(field_names):
    #             to_dict[name][i] = float(data[j])
    #
    #         jdt.add(is_jdt=is_jdt)
    #     jdt.end(is_jdt=is_jdt)
    #     return to_dict
    #
    # def csvfile(self, csv_fn, to_csv_fn, is_jdt=True, field_names=None):
    #     if field_names is None:
    #         field_names = self.gr.names
    #     df = pd.read_csv(csv_fn)
    #     df_list = df.to_dict("list")
    #     to_dict = self.sampling(df_list["X"], df_list["Y"], field_names=field_names, is_jdt=is_jdt)
    #     for k in to_dict:
    #         df_list[k] = to_dict[k]
    #     pd.DataFrame(df_list).to_csv(to_csv_fn, index=False)
    #     return df_list
    #
    # def sampling2(self, x, y, win_row, win_column, is_jdt=True, is_trans=False):
    #     win_spl = [0, 0, 0, 0]
    #     win_spl[0] = 0 - int(win_row / 2)
    #     win_spl[1] = 0 + round(win_row / 2 + 0.1)
    #     win_spl[2] = 0 - int(win_column / 2)
    #     win_spl[3] = 0 + round(win_column / 2 + 0.1)
    #     jdt = Jdt(len(x), "GDALSamplingFast::sampling2").start(is_jdt=is_jdt)
    #     to_data = np.zeros((len(x), self.gr.n_channels, win_row, win_column))
    #     for i in range(len(x)):
    #         x0, y0 = x[i], y[i]
    #         if is_trans:
    #             x0, y0, _ = self.gr.wgs84_to_this.TransformPoint(x0, y0)
    #         if self.gr.isGeoIn(x0, y0):
    #             data = self.gr.readAsArray(x0, y0, win_row, win_column, is_geo=True)
    #             to_data[i] = data
    #         to_data = to_data * 0
    #         jdt.add(is_jdt=is_jdt)
    #     jdt.end(is_jdt=is_jdt)
    #     return to_data


class _GSIC:

    def __init__(self, imdc_fn):
        self.gs = GDALSampling(imdc_fn)
        self.imdc_fn = imdc_fn

    def sampling(self, x, y, map_dict=None):
        to_dict = self.gs.sampling(x, y)
        y2 = list(to_dict.values())[0]
        if map_dict is not None:
            y2 = categoryMap(y2, map_dict)
        return y2


class GDALSamplingImageClassification:

    def __init__(self):
        self.gsics = {}

    def add(self, name, imdc_fn):
        self.gsics[name] = _GSIC(imdc_fn)

    def __getitem__(self, item) -> _GSIC:
        return self.gsics[item]

    def __len__(self):
        return len(self.gsics)

    def keys(self):
        return list(self.gsics.keys())

    def fit(self, x, y, *names, to_dict=None, map_dict=None):
        names = datasCaiFen(names)
        if len(names) == 0:
            names = list(self.gsics.keys())
        if to_dict is None:
            to_dict = {}
        return {**to_dict, **{name: self.gsics[name].sampling(x, y, map_dict=map_dict) for name in names}}

    def toDict(self):
        to_dict = {name: self.gsics[name].imdc_fn for name in self.gsics}
        return to_dict


class GDALAccuracyConfusionMatrix(SRTAccuracyConfusionMatrix):

    def __init__(self, n_class=0, class_names=None):
        super().__init__(n_class, class_names)

    def addImageCSV(self, imdc_fn, csv_fn, x_field_name="X", y_field_name="Y", category_field_name="CATEGORY",
                    imdc_channel=1, imdc_map_dict=None, csv_map_dict=None, filter_eqs=None):
        sdf = SRTDataFrame().read_csv(csv_fn, is_auto_type=True)
        if filter_eqs is not None:
            for k, data in filter_eqs.items():
                sdf = sdf.filterEQ(k, data)
        y1 = []
        gr = GDALRaster(imdc_fn)
        for i in range(len(sdf)):
            x, y = sdf[x_field_name][i], sdf[y_field_name][i]
            y1_tmp = gr.readAsArray(x, y, 1, 1, is_geo=True).ravel()
            y1.append(int(y1_tmp[imdc_channel - 1]))
        if imdc_map_dict is not None:
            y1 = categoryMap(y1, imdc_map_dict)
        y = sdf[category_field_name]
        if csv_map_dict is not None:
            y = categoryMap(y, csv_map_dict)
        self.add(y, y1)


def replaceCategoryImage(o_geo_fn, replace_geo_fn, to_geo_fn, o_map_dict=None, replace_map_dict=None, o_change=None,
                         color_table=None):
    if o_change is None:
        o_change = []

    def map_dict_arr(_data, _map_dict):
        to_data = np.zeros_like(_data)
        for k1, k2 in _map_dict.items():
            to_data[_data == k1] = k2
        return to_data

    gr_origin = GDALRaster(o_geo_fn)
    o_data = gr_origin.readAsArray()
    if o_map_dict is not None:
        o_data = map_dict_arr(o_data, o_map_dict)

    gr_replace = GDALRaster(replace_geo_fn)
    r_data = gr_replace.readAsArray()
    if replace_map_dict is not None:
        r_data = map_dict_arr(r_data, replace_map_dict)

    to_imdc = o_data.copy()
    for d in o_change:
        to_imdc[o_data == d] = r_data[o_data == d]

    saveGTIFFImdc(gr_replace, to_imdc, to_geo_fn, color_table=color_table)


class GDALNumpySampling(NumpySampling):

    def __init__(self, win_row, win_column, gr):
        super().__init__(win_row, win_column)
        self.gr = gr
        self.data = gr.d

    def getxy(self, x, y):
        row, column = self.gr.coorGeo2Raster(x, y, is_int=True)
        return self.get(row, column)


class GDALAccuracyImage:

    def __init__(
            self,
            imdc_geo_fn=None,
            df=pd.DataFrame(),
            csv_fn=None,
            category_field_name="CATEGORY",
            x_field_name="X",
            y_field_name="Y",
            cnames=None,
            imdc_map_dict=None,
            df_map_dict=None,
    ):
        self.cm = None
        self.imdc_data = None
        self.imdc_geo_fn = imdc_geo_fn
        self.csv_fn = csv_fn
        self.category_field_name = category_field_name
        self.x_field_name = x_field_name
        self.y_field_name = y_field_name
        self.cnames = cnames
        self.imdc_map_dict = imdc_map_dict
        self.df_map_dict = df_map_dict
        self.df = df
        self.gr = GDALRaster()
        self.init()

    def init(self):
        if self.csv_fn is not None:
            self.df = pd.read_csv(self.csv_fn)
        if self.imdc_geo_fn is not None:
            self.gr = GDALRaster(self.imdc_geo_fn)
            self.imdc_data = self.gr.readAsArray()

    def fit(self, filter_list=None):
        """ filter_list: [(eq|neq, field_name, data), ...]"""

        df = self.filterFromList(filter_list)
        category1 = df[self.category_field_name].values
        if self.df_map_dict is not None:
            category1 = categoryMap(category1, self.df_map_dict)

        x, y = df[self.x_field_name].values, df[self.y_field_name].values
        category2 = self.sampling(x, y)
        if self.imdc_map_dict is not None:
            category2 = categoryMap(category2, self.imdc_map_dict)

        cnames = self.cnames
        if cnames is None:
            cnames = [str(i) for i in range(int(np.min(category1)), int(np.max(category1)) + 1)]

        self.cm = ConfusionMatrix(len(cnames), cnames)
        self.cm.addData(category1, category2)
        return self.cm

    def sampling(self, x, y):
        to_list = []
        for x0, y0 in zip(x, y):
            row, column = self.gr.coorGeo2Raster(x0, y0, is_int=True)
            to_list.append(int(self.imdc_data[row, column]))
        return to_list

    def filterFromList(self, filter_list, df=None):
        if df is None:
            df = self.df.copy()
        if filter_list is None:
            return df
        for i, (com, field_name, data) in enumerate(filter_list):
            if com == "eq":
                df = df[df[field_name] == data]
            elif com == "neq":
                df = df[df[field_name] != data]
        return df


class _RandomCoor:

    def __init__(self, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.coors = []

    def generate(self, n, mode=None):
        if mode is None:
            self.coors = []
        for i in range(n):
            self.coors.append(self.randomXY())
        return self.coors

    def __getitem__(self, item):
        return self.coors[item]

    def __len__(self):
        return len(self.coors)

    def randomXY(self):
        return [getRandom(self.x_min, self.x_max), getRandom(self.y_min, self.y_max), ]


class GDALFeaturesMemory(SRTFeaturesMemory):

    def __init__(self, raster_fn):
        self.gr = GDALRaster(raster_fn)
        super().__init__(names=self.gr.names)

    # def read(self, *names, data=None):
    #     names = datasCaiFen(names)
    #     if len(names) == 0:
    #         names = self.gr.names
    #     if data is None:
    #         data = np.zeros((self.gr.n_channels, self.gr.n_rows, self.gr.n_columns))
    #     for name in names:
    #         data


class RasterRandomCoors:

    def __init__(self, raster_fn):
        self.gr = GDALRaster(raster_fn)
        self.coors = []
        self.random_coor = _RandomCoor(self.gr.x_min, self.gr.x_max, self.gr.y_min, self.gr.y_max)

    def fit(self, category, n, is_jdt=True):
        jdt = Jdt(n, "Raster Random Coors").start(is_jdt=is_jdt)
        n_select = 0
        for i in range(self.gr.n_rows):
            for j in range(self.gr.n_columns):
                x, y = self.random_coor.randomXY()
                d = int(self.gr.readAsLine(x, y, is_geo=True)[0])
                if d == category:
                    self.add(x, y, category)
                    jdt.add(is_jdt=is_jdt)
                    n_select += 1
                    if n_select >= n:
                        return

    def add(self, x, y, category, ):
        to_dict = {"N": len(self.coors) + 1, "X": x, "Y": y, "CATEGORY": category}
        self.coors.append(to_dict)
        return to_dict

    def saveToCSV(self, to_fn):
        df = pd.DataFrame(self.coors)
        print(df)
        df.to_csv(to_fn, index=False)

    def random(self, n):
        xy = np.array(self.random_coor.generate(n))
        df = pd.DataFrame({
            "SRT": [i + 1 for i in range(n)],
            "X": xy[:, 0],
            "Y": xy[:, 1],
        })
        return df


class GDALRasterClip:

    def __init__(self, raster_fn):
        self.gr = GDALRaster(raster_fn)

    def coorCenter(self, to_fn, x, y, rows, columns, ):
        gr = self.gr
        row, column = gr.coorGeo2Raster(x, y, is_int=True)
        data = gr.readAsArrayCenter(x, y, rows, columns, is_geo=True)

        x0, y0 = gr.coorRaster2Geo(row - int(rows / 2), column - int(columns / 2))
        x1, y1 = gr.coorRaster2Geo(row + int(rows / 2), column + int(columns / 2))

        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        geo_transform = (x0, gr.geo_transform[1], 0, y1, 0, gr.geo_transform[5])

        gr.save(
            d=data, save_geo_raster_fn=to_fn, fmt="GTiff", dtype=gdal.GDT_Float32,
            start_xy=None, descriptions=gr.names, geo_transform=geo_transform
        )
        return to_fn


def main():
    # grr = GDALRasterRange(r"F:\ProjectSet\Shadow\Release\ChengDuImages\SH_CD_look_envi.dat")
    # fn = r"F:\ProjectSet\Shadow\MkTu\Draw\SH_CD_envi.dat.npy"
    # grr.loadNPY(fn)
    # grr.save(fn + ".json")

    # grcd = GDALRasterCenterDatas()
    # grcd.changeDataList(2, 3)
    # grcd.changeDataList(1, 3)
    # grcd.changeDataList(5, 6)

    pass


if __name__ == "__main__":
    main()
