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

import matplotlib.pyplot as plt
import numpy as np
from osgeo import osr, gdal, gdal_array

from SRTCodes.GDALRasterIO import GDALRaster, GDALRasterChannel, GDALRasterRange
from SRTCodes.SRTFeature import SRTFeatureCallBack
from SRTCodes.Utils import readcsv, Jdt, savecsv, changext

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
    srs = osr.SpatialReference()
    srs.SetFromUserInput(coor_srs)
    gr.setDstSrs(srs)
    n = min(len(x), len(y))
    for name in gr.names:
        d[name] = []
    jdt = Jdt(total=n, desc="Sampling To CSV")
    jdt.start()
    for i in range(n):
        d0 = gr.readAsArray(x[i], y[i], win_row_size=1, win_column_size=1, is_trans=True, is_geo=True)
        d0 = d0.ravel()
        for j, name in enumerate(gr.names):
            d[name].append(d0[j])
        jdt.add()
    jdt.end()
    savecsv(to_csv_fn, d)


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

    def save(self, to_dirname=None):
        if to_dirname is None:
            to_dirname = self.to_dirname
        if not os.path.isdir(to_dirname):
            os.mkdir(to_dirname)
        for i, fn in enumerate(self.save_fns):
            fn = os.path.join(to_dirname, fn)
            gdal.BuildVRT(fn, self.raster_fn, options=["-b", str(i + 1)])


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
        plt.plot(x, y, label=field_name)


class GDALRasterCenter:
    GDALDatasetCollection = {}

    def __init__(self, name=None, ds=None, channel_list=None, save_dict=None, raster_fn=None):
        if save_dict is None:
            save_dict = {}
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
        self.ds = ds
        return ds, raster_fn

    def read(self, x_row=0.0, y_column=0.0, win_row_size=0, win_column_size=0, is_geo=True, no_data=0, *args, **kwargs):
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

        self.d = d
        return d

    def initGeoRaster(self, channel_list, name, raster_fn):
        if channel_list is None:
            channel_list = [name]
        raster_fn = os.path.abspath(raster_fn)
        if raster_fn not in self.GDALDatasetCollection:
            self.GDALDatasetCollection[raster_fn] = gdal.Open(raster_fn)
        return channel_list, raster_fn

    def scaleMinMax(self, d_min, d_max, dim=None):
        if dim is None:
            if d_min is None:
                d_min = np.min(self.d)
            if d_max is None:
                d_max = np.max(self.d)
            self.d = np.clip(self.d, d_min, d_max)
            self.d = (self.d - d_min) / (d_max - d_min)
        else:
            if d_min is None:
                d_min = np.min(self.d[:, :, dim])
            if d_max is None:
                d_max = np.max(self.d[:, :, dim])
            self.d[:, :, dim] = np.clip(self.d[:, :, dim], d_min, d_max)
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


def main():
    grr = GDALRasterRange(r"F:\ProjectSet\Shadow\Release\ChengDuImages\SH_CD_look_envi.dat")
    fn = r"F:\ProjectSet\Shadow\MkTu\Draw\SH_CD_envi.dat.npy"
    grr.loadNPY(fn)
    grr.save(fn + ".json")

    pass


if __name__ == "__main__":
    main()


def coorTrans(geo_trans, x, y):
    column = geo_trans[0] + x * geo_trans[1] + y * geo_trans[2]
    row = geo_trans[3] + x * geo_trans[4] + y * geo_trans[5]
    return row, column
