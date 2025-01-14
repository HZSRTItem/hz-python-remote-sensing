# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : syutils.py
@Time    : 2024/9/4 14:51
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of syutils
-----------------------------------------------------------------------------"""

import csv
import inspect
import json
import os
import random
import shutil
import sys
import time
import warnings
from datetime import datetime
from os import PathLike
from typing import Union

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from osgeo import gdal, osr, __version__, gdal_array
from osgeo_utils.gdal_merge import main as gdal_merge_main
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

StrOrBytesPath = Union[str, bytes, PathLike[str], PathLike[bytes]]

_LINUX_TYPE = "NONE"
eps = 0.000001


def W2LF(path: StrOrBytesPath):
    if _LINUX_TYPE == "UBUNTU":
        if path[1] == ":":
            path = "/mnt/" + path[0].lower() + path[2:]
        path = path.replace("\\", "/")
        return path
    return path


def printList(front_str, to_list, max_line_width=60):
    print(front_str)
    line_width = 2
    print("  ", end="")
    for name in to_list:
        str0 = "\"{0}\", ".format(name)
        line_width += len(str0)
        if line_width >= max_line_width:
            print("\n  ", end="")
            line_width = 2
        print(str0, end="")
    print()


def writeTexts(filename, *texts, mode="w", end=""):
    with open(filename, mode, encoding="utf-8") as f:
        for text in texts:
            f.write(str(text))
        f.write(end)
    return filename


class DirFileName:
    """ Directory file name """

    def __init__(self, dirname: StrOrBytesPath = None, init_week_dir=False):
        self.dirname = dirname
        if dirname is None:
            self.dirname = os.getcwd()
        self.dirname = W2LF(self.dirname)

    def mkdir(self):
        if not os.path.isdir(self.dirname):
            os.mkdir(self.dirname)
        return self

    def fn(self, *names):
        """ add directory or file name in the end of cwd
        /
        :param names:
        :return:
        """
        if len(names) == 0:
            return self.dirname
        return os.path.join(self.dirname, *names)

    def copyfile(self, filename):
        to_filename = os.path.join(self.dirname, os.path.basename(filename))
        shutil.copyfile(filename, to_filename)
        return to_filename


def datasCaiFen(datas):
    data_list = []
    for data in datas:
        if isinstance(data, list) or isinstance(data, tuple):
            data_list.extend(data)
        else:
            data_list.append(data)
    return data_list


def changext(fn, ext=""):
    fn1, ext1 = os.path.splitext(fn)
    return fn1 + ext


def changefiledirname(filename, dirname):
    filename = os.path.split(filename)[1]
    return os.path.join(dirname, filename)


def getfilenamewithoutext(fn):
    fn = os.path.splitext(fn)[0]
    return os.path.split(fn)[1]


def writeLines(filename, *lines, mode="w", sep="\n"):
    lines = datasCaiFen(lines)
    with open(filename, mode=mode, encoding="utf-8") as f:
        for line in lines:
            f.write(str(line))
            f.write(sep)
    return filename


class FRW:

    def __init__(self, filename):
        self.filename = filename

    def read(self):
        with open(self.filename, "r", encoding="utf-8") as f:
            return f.read()

    def readJson(self):
        return readJson(self.filename)

    def savecsv(self, data):
        return savecsv(self.filename, data)

    def saveJson(self, data):
        return saveJson(data, self.filename)

    def writeTextLine(self, *texts, mode="a", end="\n"):
        return writeTextLine(self.filename, *texts, mode=mode, end=end)

    def writeTexts(self, *texts, mode="w", end=""):
        return writeTexts(self.filename, *texts, mode=mode, end=end)

    def writeLines(self, *lines, mode="w", sep="\n"):
        return writeLines(self.filename, *lines, mode=mode, sep=sep)


def writeTextLine(filename, *texts, mode="a", end="\n"):
    writeTexts(filename, *texts, mode=mode, end=end)


def readJson(json_fn):
    with open(json_fn, "r", encoding="utf-8") as f:
        return json.load(f)


def saveJson(d, json_fn):
    with open(json_fn, "w", encoding="utf-8") as f:
        json.dump(d, f)
    return d


def savecsv(csv_fn, d: dict):
    with open(csv_fn, "w", encoding="utf-8", newline="") as fr:
        cw = csv.writer(fr)
        ks = list(d.keys())
        cw.writerow(ks)
        for i in range(len(d[ks[0]])):
            cw.writerow([d[k][i] for k in ks])


def scaleMinMax(d, d_min=None, d_max=None, is_01=True):
    if (d_min is None) and (d_max is None):
        is_01 = True
    if d_min is None:
        d_min = np.min(d)
    if d_max is None:
        d_max = np.max(d)
    d = np.clip(d, d_min, d_max)
    if is_01:
        d = (d - d_min) / (d_max - d_min)
    return d


def isStringInt(_str: str):
    if _str.isdigit():
        return True
    if len(_str) == 0:
        return False
    try:
        int(_str)
        return True
    except ValueError:
        return False


def isStringFloat(_str: str):
    if "." not in _str:
        return False
    if len(_str) <= 1:
        return False
    try:
        float(_str)
        return True
    except ValueError:
        return False


def listAutoType(_list):
    d_type = "int"

    for data in _list:
        if not isinstance(data, str):
            d_type = "none"
            break
        if data == "":
            d_type = "string"
            break

        if isStringFloat(data):
            if d_type == "float":
                continue
            d_type = "float"
        elif isStringInt(data):
            if d_type == "float":
                continue
            d_type = "int"
        else:
            d_type = "string"
            break

    def as_column_type(_type):
        def func(_str):
            if _str == "":
                return None
            else:
                return _type(_str)

        return list(map(func, _list))

    if d_type == "int":
        return as_column_type(int)
    elif d_type == "float":
        return as_column_type(float)

    return _list


class SRTDataFrame:

    def __init__(self):
        self._data = {}
        self._n_length = 0
        self._n_iter = 0
        self._index_name = None
        self._keys = []
        self._index = {}

        self._to_str_width = 100

    def copyEmpty(self):
        sdf = SRTDataFrame()
        sdf._data = {k: [] for k in self._data}
        return sdf

    def read_csv(self, csv_fn, index_column_name=None, is_auto_type=False):
        self._index_name = index_column_name
        self._data = {}
        self._index = {}
        with open(csv_fn, "r", encoding="utf-8-sig") as fr:
            self._n_length = 0
            cr = csv.reader(fr)
            ks = next(cr)
            for k in ks:
                self._data[self._initColumnName(k)] = []
            ks = list(self._data.keys())
            i = 0
            for line in cr:
                self._n_length += 1
                for i, k in enumerate(line):
                    self._data[ks[i]].append(k)
                self._index[i] = i
                i += 1
        if is_auto_type:
            for k in self._data:
                self.autoColumnType(k)
        for name in list(self._data.keys()):
            if not self._data[name]:
                self._data.pop(name)
        return self

    def indexColumnName(self, column_name):
        self._index_name = column_name
        self.buildIndex(d for d in self._data[column_name])

    def buildIndex(self, _index_iter):
        self._index = {}
        for i in range(self._n_length):
            _index = next(_index_iter)
            self._index[_index] = i

    def _initColumnName(self, name):
        if name == "":
            name = "_UName"
        if name not in self._data:
            return name
        new_name = "{0}_1".format(name)
        while True:
            if new_name not in self._data:
                break
        return new_name

    def autoColumnType(self, column_name):
        self._data[column_name] = listAutoType(self._data[column_name])

    def asColumnType(self, column_name, _type):
        def func(_str):
            if _str == "":
                return None
            else:
                return _type(_str)

        self._data[column_name] = list(map(func, self._data[column_name]))

    def keys(self):
        return self._data.keys()

    def data(self):
        return self._data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return self._n_length

    def __iter__(self):
        return self

    def __next__(self):
        if self._n_iter == 0:
            self._keys = list(self._data.keys())
        if self._n_iter == len(self._keys):
            self._n_iter = 0
            raise StopIteration()
        self._n_iter += 1
        return self._keys[self._n_iter - 1]

    def __contains__(self, item):
        return item in self._data

    def loc(self, _index, _column_name):
        _index = self._index[_index]
        return self._data[_column_name][_index]

    def toCSV(self, csv_fn):
        savecsv(csv_fn, self._data)

    def rowToDict(self, row):
        return {k: self._data[k][row] for k in self.data()}

    def __setitem__(self, key, value):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if self._n_length != len(value):
            warnings.warn("SRTDataFrame add field length of data [{0}] not equal this of [{1}]".format(
                len(value), self._n_length))
            if len(value) > self._n_length:
                for i in range(self._n_length, len(value)):
                    for k in self._data:
                        self._data[k].append(None)
            else:
                for i in range(len(value), self._n_length):
                    value.append(None)
        self._data[key] = value

    def addFields(self, *fields):
        to_fields = []
        for field in fields:
            if isinstance(field, list) or isinstance(field, tuple):
                to_fields += list(field)
            else:
                to_fields.append(field)
        for field in to_fields:
            self._data[field] = [None for _ in range(self._n_length)]
        return self

    def addLine(self, line=None):
        if line is None:
            line = {k: None for k in self._data}
        if isinstance(line, list) or isinstance(line, tuple):
            line = {k: line[i] for i, k in enumerate(self._data)}
        for k in self._data:
            if k in line:
                self._data[k].append(line[k])
            else:
                self._data[k].append(None)
        self._n_length += 1

    def toDict(self):
        return self._data.copy()

    def filterEQ(self, field_name, *data):
        data = list(data)
        sdf = self.copyEmpty()
        for i in range(self._n_length):
            line = self.rowToDict(i)
            if line[field_name] in data:
                sdf.addLine(line)
        return sdf


def convBnAct(in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, act=nn.ReLU()):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        act,
    )


class SRTCollection:

    def __init__(self):
        self._n_iter = 0
        self._n_next = []

    def __len__(self):
        return len(self._n_next)

    def __iter__(self):
        return self

    def __next__(self):
        if self._n_iter == len(self._n_next):
            self._n_iter = 0
            raise StopIteration()
        self._n_iter += 1
        return self._n_next[self._n_iter - 1]

    def __contains__(self, item):
        return item in self._n_next

    def __getitem__(self, item):
        return self._n_next[item]


class Jdt:
    """
    进度条
    """

    def __init__(self, total=100, desc=None, iterable=None, n_cols=20):
        """ 初始化一个进度条对象

        :param iterable: 可迭代的对象, 在手动更新时不需要进行设置
        :param desc: 字符串, 左边进度条描述文字
        :param total: 总的项目数
        :param n_cols: 调整进度条宽度, 默认是根据环境自动调节长度, 如果设置为0, 就没有进度条, 只有输出的信息
        """
        self.total = total
        self.iterable = iterable
        self.n_cols = n_cols
        self.desc = desc if desc is not None else ""

        self.n_split = float(total) / float(n_cols)
        self.n_current = 0
        self.n_print = 0
        self.is_run = False

        self.current_time = time.time()
        self.init_time = time.time()

    def start(self, is_jdt=True):
        """ 开始进度条 """
        if not is_jdt:
            return self
        self.is_run = True
        self._print()
        self.current_time = time.time()
        self.init_time = time.time()
        return self

    def add(self, n=1, is_jdt=True):
        """ 添加n个进度

        :param n: 进度的个数
        :return:
        """
        if not is_jdt:
            return
        if self.is_run:
            self.n_current += n
            self.current_time = time.time()
            if self.n_current > self.n_print * self.n_split:
                self.n_print += 1
                if self.n_print > self.n_cols:
                    self.n_print = self.n_cols
            self._print()

    def setDesc(self, desc):
        """ 添加打印信息 """
        self.desc = desc

    def fmttime(self):
        d_time = (self.current_time - self.init_time) / (self.n_current + 0.000001)

        def timestr(_time):

            tian = int(_time // (60 * 60 * 24))
            _time = _time % (60 * 60 * 24)
            shi = int(_time // (60 * 60))
            _time = _time % (60 * 60)
            fen = int(_time // 60)
            miao = int(_time % 60)

            if tian >= 1:
                return "{}D {:02d}:{:02d}:{:02d}".format(tian, shi, fen, miao)
            if shi >= 1:
                return "{:02d}:{:02d}:{:02d}".format(shi, fen, miao)
            return "{:02d}:{:02d}".format(fen, miao)

        n = 1 / (d_time + 0.000001)
        n_str = "{:.1f}".format(n)
        if len(n_str) >= 6:
            n_str = ">1000".format(n)

        fmt = "[{}<{}, {}it/s]            ".format(timestr(
            self.current_time - self.init_time), timestr(d_time * self.total), n_str)
        return fmt

    def _print(self):
        des_info = "\r{0}: {1:>3d}% |".format(self.desc, int(self.n_current / self.total * 100))
        des_info += "*" * self.n_print + "-" * (self.n_cols - self.n_print)
        des_info += "| {0}/{1}".format(self.n_current, self.total)
        des_info += " {}".format(self.fmttime())
        print(des_info, end="")

    def end(self, is_jdt=True):
        if not is_jdt:
            return
        """ 结束进度条 """
        self.n_split = float(self.total) / float(self.n_split)
        self.n_current = 0
        self.n_print = 0
        self.is_run = False
        print()


class Raster:
    """ Raster base class """

    def __init__(self, raster_fn=None):
        self.n_rows = 0  # number of rows
        self.n_columns = 0  # number of columns
        self.n_channels = 0  # number of channels
        self.d = None  # data of this raster

        self.initRaster()

    def initRaster(self):
        self.n_rows = 0  # number of rows
        self.n_columns = 0  # number of columns
        self.n_channels = 0  # number of channels
        self.d = None  # data of this raster

    def readAsArray(self, *args, **kwargs):
        return None

    def save(self, *args, **kwargs):
        return None


class GEORaster(Raster):
    """ GEORaster class """

    def __init__(self):
        super(GEORaster, self).__init__()

        self.srs = None
        self.y_size = None
        self.x_size = None
        self.y_max = None
        self.x_max = None
        self.y_min = None
        self.x_min = None

        self.initGEORaster()

    def initGEORaster(self):
        self.x_min = 0
        self.y_min = 0
        self.x_max = 0
        self.y_max = 0
        self.x_size = 0
        self.y_size = 0
        self.srs = None

    def coorRaster2Geo(self, row, column):
        x = self.x_min + column * self.x_size
        y = self.y_max - row * self.y_size
        return x, y

    def coorGeo2Raster(self, x, y):
        column = (x - self.x_min) / self.x_size
        row = (self.y_max - y) / self.y_size
        return column, row


def getGDALRasterNames(raster_fn):
    ds: gdal.Dataset = gdal.Open(raster_fn)
    names = []
    for i in range(1, ds.RasterCount + 1):
        b = ds.GetRasterBand(i)
        name = b.GetDescription()
        names.append(name)
    return names


def getArraySize(d_shape, interleave):
    wd = len(d_shape)
    if not (wd == 2 or wd == 3):
        raise Exception("The data shall be two-dimensional array single-band "
                        "data or three-dimensional multi-band data", d_shape)
    # 波段数量
    band_count = 1
    if wd == 3:
        if interleave == "band":
            band_count = d_shape[0]
            n_row = d_shape[1]
            n_column = d_shape[2]
        elif interleave == "pixel":
            band_count = d_shape[2]
            n_row = d_shape[1]
            n_column = d_shape[0]
        else:
            raise Exception("The data is organized as `band`:(b,y,x) or `pixel`:(x,y,b) not " + interleave)
    else:
        if interleave == "band":
            n_row = d_shape[0]
            n_column = d_shape[1]
        elif interleave == "pixel":
            n_row = d_shape[1]
            n_column = d_shape[0]
        else:
            raise Exception("The data is organized as `band`:(b,y,x) or `pixel`:(x,y,b) not " + interleave)
    return band_count, n_column, n_row


class GDALRasterRangeData:

    def __init__(self, d_min=0.0, d_max=0.0, t_dict=None):
        self.min = float(d_min)
        self.max = float(d_max)
        if t_dict is not None:
            self.min = float(t_dict["min"])
            self.max = float(t_dict["max"])

    def toDict(self):
        return {"min": self.min, "max": self.max}


class GDALRasterRange:

    def __init__(self, raster_fn=None, range_fn=None):
        self.raster_fn = raster_fn
        self.range_fn = ""
        self.range_dict = {}
        self.init(raster_fn, range_fn)

    def init(self, raster_fn, range_fn=None):
        if range_fn is not None:
            self.loadJsonFile(range_fn)

        if raster_fn is None:
            return
        self.raster_fn = raster_fn
        self.range_fn = changext(raster_fn, ".range")
        if os.path.isfile(self.range_fn):
            try:
                self.loadJsonFile(self.range_fn)
            except:
                pass
        return self

    def loadJsonFile(self, range_fn):
        dict_in = readJson(range_fn)
        self.loadDict(dict_in)
        return self

    def loadDict(self, dict_in):
        for k in dict_in:
            self.range_dict[k] = GDALRasterRangeData(t_dict=dict_in[k])
        return self

    def isRead(self):
        return os.path.isfile(self.range_fn)

    def loadNPY(self, npy_fn, names=None):
        if names is None:
            names = getGDALRasterNames(self.raster_fn)
        d = np.load(npy_fn)
        for i in range(len(d)):
            self.range_dict[names[i]] = GDALRasterRangeData(d[i, 0], d[i, 1])
        return self

    def save(self, range_fn=None):
        if range_fn is None:
            range_fn = self.range_fn
        save_dict = {k: self.range_dict[k].toDict() for k in self.range_dict}
        saveJson(save_dict, range_fn)

    def __getitem__(self, item) -> GDALRasterRangeData:
        if isinstance(item, str):
            grrd = self.range_dict[item]
        elif isinstance(item, int):
            ks = list(self.range_dict.keys())
            grrd = self.range_dict[ks[item]]
        else:
            grrd = None
        return grrd

    def scaleMinMax(self, name, data=None, is_01=False):
        data = scaleMinMax(data, d_min=self.range_dict[name].min, d_max=self.range_dict[name].max, is_01=is_01)
        return data


def saveGDALRaster(d, n_row, n_column, band_count, dtype, fmt, geo_transform, interleave, options, probing,
                   save_geo_raster_fn, descriptions):
    driver = gdal.GetDriverByName(fmt)  # 申请空间
    dst_ds = driver.Create(save_geo_raster_fn, n_column, n_row, band_count, dtype, options)  # 列数 行数 波段数
    dst_ds.SetGeoTransform(geo_transform)  # 设置投影信息
    dst_ds.SetProjection(probing)
    # 保存数据
    if band_count == 1:
        band: gdal.Band = dst_ds.GetRasterBand(1)
        band.WriteArray(d)
        if descriptions is not None:
            band.SetDescription(descriptions[0])
    else:
        for i in range(band_count):
            if interleave == "band":
                band: gdal.Band = dst_ds.GetRasterBand(i + 1)
                band.WriteArray(d[i, :, :])
                if descriptions is not None:
                    band.SetDescription(descriptions[i])
            elif interleave == "pixel":
                band: gdal.Band = dst_ds.GetRasterBand(i + 1)
                band.WriteArray(d[:, :, i])
                if descriptions is not None:
                    band.SetDescription(descriptions[i])
            else:
                raise Exception("The data is organized as `band`:(b,y,x) or `pixel`:(x,y,b) not " + interleave)
    del dst_ds


class GDALRasterIO(GEORaster):
    """ GDAL Raster IO"""

    # gdal type to np type
    NpType2GDALType = {
        "int8": gdal.GDT_Byte,
        "uint16": gdal.GDT_UInt16,
        "int16": gdal.GDT_Int16,
        "uint32": gdal.GDT_UInt32,
        "int32": gdal.GDT_Int32,
        "float32": gdal.GDT_Float32,
        "float64": gdal.GDT_Float64
    }

    # np type to gdal type
    GDALType2NpType = {
        gdal.GDT_Byte: "int8",
        gdal.GDT_UInt16: "uint16",
        gdal.GDT_Int16: "int16",
        gdal.GDT_UInt32: "uint32",
        gdal.GDT_Int32: "int32",
        gdal.GDT_Float32: "float32",
        gdal.GDT_Float64: "float64",
    }

    def __init__(self):
        GEORaster.__init__(self)

        self.wgs84_to_this = None
        self.gdal_raster_fn = None
        self.raster_ds: gdal.Dataset = None

        self.n_rows = None
        self.n_columns = None
        self.n_channels = None
        self.names = []

        self.geo_transform = None
        self.inv_geo_transform = None

        self.src_srs = None
        self.dst_srs = None

        self.coor_trans = None
        self.towgs84_coor_trans = None

        self.d = None
        self.interleave = "band"

        self.save_fmt = "GTiff"
        self.save_geo_raster_fn = None
        self.save_dtype = gdal.GDT_Float32
        self.save_geo_transform = None
        self.save_probing = None

        self.raster_range = None

        self.grr = GDALRasterRange()

        self.open_type = gdal.GA_ReadOnly

    def toDict(self):
        to_dict = {
            "gdal_raster_fn": self.gdal_raster_fn,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "n_channels": self.n_channels,
            "names": self.names,
        }
        return to_dict

    def _init(self):
        self.gdal_raster_fn = None
        self.raster_ds = None
        self.n_rows = None
        self.n_columns = None
        self.n_channels = None
        self.names = []
        self.geo_transform = None
        self.inv_geo_transform = None
        self.src_srs = None
        self.dst_srs = None
        self.coor_trans = None
        self.towgs84_coor_trans = None
        self.raster_range = None
        self.d = None
        self.interleave = "band"
        self.save_fmt = "GTiff"
        self.save_geo_raster_fn = None
        self.save_dtype = gdal.GDT_Float32
        self.save_geo_transform = None
        self.save_probing = None

    def ioOpen(self, *args, **kwargs):
        return gdal.Open(self.gdal_raster_fn, self.open_type)

    def initGDALRasterIO(self, gdal_raster_fn):
        self.initRaster()
        self.initGEORaster()
        self._init()

        self.gdal_raster_fn = gdal_raster_fn
        self.raster_ds: gdal.Dataset = self.ioOpen()
        if self.raster_ds is None:
            raise Exception("Input geo raster file can not open -file:" + self.gdal_raster_fn)

        self.grr.init(self.gdal_raster_fn)
        self.geo_transform = self.raster_ds.GetGeoTransform()
        self.inv_geo_transform = gdal.InvGeoTransform(self.geo_transform)
        self.n_rows = self.raster_ds.RasterYSize
        self.n_columns = self.raster_ds.RasterXSize
        self.n_channels = self.raster_ds.RasterCount

        for i in range(1, self.n_channels + 1):
            b = self.raster_ds.GetRasterBand(i)
            name = b.GetDescription()
            back_name = "FEATURE_{0}".format(i)
            name = self._initName(back_name, name)
            self.names.append(name)

        self.src_srs = osr.SpatialReference()
        self.src_srs.ImportFromEPSG(4326)
        self.dst_srs = osr.SpatialReference()
        wkt = self.raster_ds.GetProjection()
        self.dst_srs.ImportFromWkt(wkt)
        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromEPSG(4326)
        if __version__ >= "3.0.0":
            self.src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            self.dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            wgs84_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if wkt != "":
            self.coor_trans = osr.CoordinateTransformation(self.src_srs, self.dst_srs)
            self.towgs84_coor_trans = osr.CoordinateTransformation(self.dst_srs, wgs84_srs)
            self.wgs84_to_this = osr.CoordinateTransformation(wgs84_srs, self.dst_srs)

        self.save_geo_transform = self.geo_transform
        self.save_probing = self.dst_srs.ExportToWkt()

        self.getRange()
        self.x_min = self.raster_range[0]
        self.x_max = self.raster_range[1]
        self.y_min = self.raster_range[2]
        self.y_max = self.raster_range[3]
        self.x_size = self.geo_transform[1]
        self.y_size = self.geo_transform[5]

    def _initName(self, back_name, name):
        if name == "":
            name = back_name
        if name in self.names:
            name_idx = 1
            while True:
                name_tmp = "{0}_{1}".format(name, name_idx)
                if name_tmp not in self.names:
                    name = name_tmp
                    break
                name_idx += 1
        return name

    def setDstSrs(self, dst_srs: osr.SpatialReference = None):
        if dst_srs is not None:
            if __version__ >= "3.0.0":
                dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            self.dst_srs = dst_srs
        self.coor_trans = osr.CoordinateTransformation(self.src_srs, self.dst_srs)

    def coorGeo2Raster(self, x, y, is_int=False):
        """ Geographical coordinates to image coordinates
        \
        :param is_int:
        :param x: Geographic North Coordinates / Latitude
        :param y: Geographical East Coordinates / Longitude
        :return: Image coordinates
        """
        column = self.inv_geo_transform[0] + x * self.inv_geo_transform[1] + y * self.inv_geo_transform[2]
        row = self.inv_geo_transform[3] + x * self.inv_geo_transform[4] + y * self.inv_geo_transform[5]
        if is_int:
            return int(row), int(column)
        else:
            return row, column

    def coorRaster2Geo(self, row, column):
        """ image coordinates to Geographical coordinates
        \
        :param row: row
        :param column: column
        :return: Geographical coordinates
        """
        x = self.geo_transform[0] + column * self.geo_transform[1] + row * self.geo_transform[2]
        y = self.geo_transform[3] + column * self.geo_transform[4] + row * self.geo_transform[5]
        return x, y

    def getRange(self):
        """ x_min, x_max, y_min, y_max """
        if self.raster_ds is None:
            return None
        x0, y0 = self.coorRaster2Geo(0, 0)
        x1, y1 = self.coorRaster2Geo(self.n_rows, self.n_columns)
        if x0 > x1:
            x1, x0 = x0, x1
        if y0 > y1:
            y1, y0 = y0, y1
        self.raster_range = [x0, x1, y0, y1]
        return [x0, x1, y0, y1]

    def _save(self, d: np.array = None, save_geo_raster_fn=None, fmt="ENVI", dtype=None, geo_transform=None,
              probing=None, start_xy=None, interleave='band', options=None, descriptions=None):

        if d is None:
            d = self.readAsArray()
        if options is None:
            options = []

        if save_geo_raster_fn is None:
            save_geo_raster_fn = self.save_geo_raster_fn
        if save_geo_raster_fn is None:
            raise Exception("Saved geo raster file name not found")

        if fmt is None:
            fmt = self.save_fmt
        if dtype is None:
            dtype = self.save_dtype
        if geo_transform is None:
            geo_transform = self.save_geo_transform
        if probing is None:
            probing = self.save_probing

        band_count, n_column, n_row = getArraySize(d.shape, interleave)

        if start_xy is not None:
            x0, y0 = start_xy[0], start_xy[1]
            x1, y1 = self.coorRaster2Geo(n_row + 1, n_column + 1)
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0
            geo_transform = (x0, geo_transform[1], 0, y1, 0, geo_transform[5])

        self.save_geo_raster_fn = save_geo_raster_fn
        self.save_fmt = fmt
        self.save_dtype = dtype
        self.save_geo_transform = geo_transform
        self.save_probing = probing
        self.save_geo_transform = geo_transform

        saveGDALRaster(d, n_row, n_column, band_count, dtype, fmt, geo_transform, interleave, options, probing,
                       save_geo_raster_fn, descriptions)

    def toWgs84(self, x, y):
        x1, y1, _ = self.towgs84_coor_trans.TransformPoint(x, y)
        return x1, y1

    def readGDALBand(self, n_band, ds: gdal.Dataset = None, is_range=False, is_01=False):
        """ n_band start at 1 """
        if ds is None:
            ds = self.raster_ds
        if isinstance(n_band, str):
            n_band = self.names.index(n_band) + 1
        band = ds.GetRasterBand(n_band)
        d = gdal_array.BandReadAsArray(band)
        d = self._GRR(d, is_01, is_range, n_band)
        return d

    def readGDALBands(self, *bands, ds: gdal.Dataset = None, ):
        if ds is None:
            ds = self.raster_ds
        band_list = []
        for n_band in bands:
            if isinstance(n_band, str):
                n_band = self.names.index(n_band) + 1
            band_list.append(n_band)
        data = gdal_array.DatasetReadAsArray(ds, band_list=band_list)
        return data

    def getGDALBand(self, n_band, ds: gdal.Dataset = None, ) -> gdal.Band:
        if ds is None:
            ds = self.raster_ds
        if isinstance(n_band, str):
            n_band = self.names.index(n_band) + 1
        band = ds.GetRasterBand(n_band)
        return band

    def _GRR(self, d, is_01, is_range, n_band):
        if is_range:
            name = self.names[n_band - 1]
            d = self.grr.scaleMinMax(name, d, is_01=is_01)
        return d

    def readAsArray(self, x_row_off=0.0, y_column_off=0.0, win_row_size=None, win_column_size=None, interleave='band',
                    band_list=None, is_geo=False, is_trans=False, is_range=False, is_01=False):
        """ Read geographic raster data as numpy arrays by location
        \
        :param is_trans: whether Coordinate Translate default:WGS84
        :param x_row_off: rows or geographic X coordinate
        :param y_column_off: columns or geographic Y coordinate
        :param win_row_size: The number of columns of the window, data type int
        :param win_column_size: The number of rows of the window, data type int
        :param interleave: The data is organized as `band`:(b,y,x) or `pixel`:(x,y,b)
        :param band_list: List of bands, default is all bands
        :param is_geo: Is it a geographic coordinate, the default is `False`
        :return: A numpy array of size win
        """
        if is_geo:
            if is_trans:
                x_row_off, y_column_off, _ = self.coor_trans.TransformPoint(x_row_off, y_column_off)
            x_row_off, y_column_off = self.coorGeo2Raster(x_row_off, y_column_off)
        x_row_off, y_column_off = int(x_row_off), int(y_column_off)
        self.d = gdal_array.DatasetReadAsArray(self.raster_ds, y_column_off, x_row_off, win_xsize=win_column_size,
                                               win_ysize=win_row_size, interleave=interleave)
        self.interleave = interleave
        # self._GRRS(interleave, is_01, is_range)
        return self.d

    def readAsLine(self, x_row_off=0.0, y_column_off=0.0, is_geo=False, is_trans=False, ):
        if is_geo:
            if is_trans:
                x_row_off, y_column_off, _ = self.coor_trans.TransformPoint(x_row_off, y_column_off)
            x_row_off, y_column_off = self.coorGeo2Raster(x_row_off, y_column_off)
        x_row_off, y_column_off = int(x_row_off), int(y_column_off)
        data = gdal_array.DatasetReadAsArray(self.raster_ds, y_column_off, x_row_off, win_xsize=1, win_ysize=1)
        return data.ravel()

    def readAsDict(self, x_row_off=0.0, y_column_off=0.0, is_geo=False, is_trans=False, ):
        data = self.readAsLine(
            x_row_off=x_row_off, y_column_off=y_column_off, is_geo=is_geo, is_trans=is_trans, ).ravel()
        return {self.names[i]: float(data[i]) for i in range(len(data))}

    def _GRRS(self, interleave, is_01, is_range):
        if interleave == "band":
            for i in range(self.d.shape[0]):
                self.d[i, :, :] = self._GRR(self.d[i, :, :], is_01, is_range, i + 1)
        elif interleave == "pixel":
            for i in range(self.d.shape[2]):
                self.d[:, :, i] = self._GRR(self.d[:, :, i], is_01, is_range, i + 1)

    def isGeoIn(self, x, y):
        if not (self.x_min < x < self.x_max):
            return False
        if not (self.y_min < y < self.y_max):
            return False
        return True

    def coorWGS84ToThis(self, x, y):
        x1, y1, _ = self.wgs84_to_this.TransformPoint(x, y)
        return x1, y1

    def updateData(self, name, data):
        band: gdal.Band = self.getGDALBand(name)
        band.WriteArray(data)


class GDALRaster(GDALRasterIO, SRTCollection):
    """ GDALRaster """

    def __init__(self, gdal_raster_fn="", open_type=gdal.GA_ReadOnly):
        GDALRasterIO.__init__(self)
        SRTCollection.__init__(self)
        self.open_type = open_type

        if os.path.isfile(gdal_raster_fn):
            self.initGDALRaster(gdal_raster_fn)

    def toDict(self):
        to_dict_front = super(GDALRaster, self).toDict()
        to_dict = {
            **to_dict_front,
        }
        return to_dict

    def initGDALRaster(self, gdal_raster_fn):
        self.initGDALRasterIO(gdal_raster_fn)
        for name in self.names:
            self._n_next.append(name)

    def sampleCenter(self, x_row, y_column, band_list=None, is_geo=False, no_data=0, is_trans=False):
        n = len(x_row)
        d = np.ones([n, self.n_channels]) * no_data
        for i in range(n):
            d0 = self.readAsArray(x_row[i], y_column[i], win_row_size=1, win_column_size=1, band_list=band_list,
                                  is_trans=is_trans, is_geo=is_geo)
            d[i] = d0.ravel()
            # print(i)
        return d

    def readAsArrayCenter(self, x_row_center=0, y_column_center=0, win_row_size=1, win_column_size=1,
                          interleave='band', band_list=None, is_geo=False, no_data=0, is_trans=False):
        """ Read geographic raster data as numpy arrays by location
        \
        :param is_trans: whether Coordinate Translate default:WGS84
        :param no_data: not data
        :param x_row_center: rows or geographic X coordinate
        :param y_column_center: columns or geographic Y coordinate
        :param win_row_size: The number of columns of the window, data type int
        :param win_column_size: The number of rows of the window, data type int
        :param interleave: The data is organized as `band`:(b,y,x) or `pixel`:(x,y,b)
        :param band_list: List of bands, default is all bands
        :param is_geo: Is it a geographic coordinate, the default is `False`
        :return: A numpy array of size win
        """
        if is_geo:
            if is_trans:
                x_row_center, y_column_center, _ = self.coor_trans.TransformPoint(x_row_center, y_column_center)
            x_row_center, y_column_center = self.coorGeo2Raster(x_row_center, y_column_center)
        x_row_center, y_column_center = int(x_row_center), int(y_column_center)
        if (win_row_size == 1) and (win_column_size == 1):
            return gdal_array.DatasetReadAsArray(self.raster_ds, y_column_center, x_row_center,
                                                 win_xsize=win_column_size,
                                                 win_ysize=win_row_size, interleave=interleave)

        row_off0 = x_row_center - int(win_row_size / 2)
        column_off0 = y_column_center - int(win_column_size / 2)

        if 0 <= row_off0 < self.n_rows - win_row_size and 0 <= column_off0 < self.n_columns - win_column_size:
            return gdal_array.DatasetReadAsArray(self.raster_ds, column_off0, row_off0, win_xsize=win_column_size,
                                                 win_ysize=win_row_size, interleave=interleave)

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

        if row_off0 + win_row_size >= self.n_rows:
            row_size = self.n_rows - row_off0

        if column_off0 + win_column_size >= self.n_columns:
            column_size = self.n_columns - column_off0

        if row_size <= 0 or column_size <= 0:
            return None

        d0 = gdal_array.DatasetReadAsArray(self.raster_ds, column_off, row_off, column_size, row_size,
                                           interleave=interleave)

        if interleave == "band":
            if band_list is not None:
                d = np.ones([len(band_list), win_row_size, win_column_size]) * no_data
            else:
                d = np.ones([self.n_channels, win_row_size, win_column_size]) * no_data
            x0 = column_off - column_off0
            y0 = row_off - row_off0
            d[:, y0:y0 + row_size, x0:x0 + column_size] = d0
        else:
            if band_list is not None:
                d = np.ones([win_row_size, win_column_size, len(band_list)]) * no_data
            else:
                d = np.ones([win_row_size, win_column_size, self.n_channels]) * no_data
            x0 = column_off - column_off0
            y0 = row_off - row_off0
            d[y0:y0 + row_size, x0:x0 + column_size, :] = d0
        return d

    def save(self, d: np.array = None, save_geo_raster_fn: StrOrBytesPath = None, fmt="ENVI", dtype=None,
             geo_transform=None,
             probing=None, start_xy=None, interleave='band', options=None, descriptions=None):
        """ Save geo image
        \
        :param descriptions: descriptions
        :param options: save options list
        :param interleave: The data is organized as `band`:(b,y,x) or `pixel`:(x,y,b)
        :param start_xy: Coordinates of the upper left corner of the image
        :param probing: projection information
        :param geo_transform: projection transformation information
        :param d: data
        :param save_geo_raster_fn: saved image path
        :param fmt: save type
        :param dtype: save data type default:gdal.GDT_Float32
        :return: None
        """
        if d is None:
            if self.d is None:
                self.d = self.readAsArray()
            d = self.d
        self._save(d=d, save_geo_raster_fn=save_geo_raster_fn, fmt=fmt, dtype=dtype, geo_transform=geo_transform,
                   probing=probing, start_xy=start_xy, interleave=interleave, options=options,
                   descriptions=descriptions)

    def __getitem__(self, feat_name_or_number):
        if self.d is None:
            raise Exception("Can not readAsArray.")
        if isinstance(feat_name_or_number, str):
            feat_name_or_number = self._n_next.index(feat_name_or_number)
        if self.interleave == "band":
            return self.d[feat_name_or_number, :, :]
        if self.interleave == "pixel":
            return self.d[:, :, feat_name_or_number]

    def __setitem__(self, feat_name_or_number, arr):
        if self.d is None:
            raise Exception("Can not readAsArray.")
        if isinstance(feat_name_or_number, str):
            feat_name_or_number = self._n_next.index(feat_name_or_number)
        if self.interleave == "band":
            self.d[feat_name_or_number, :, :] = arr
        if self.interleave == "pixel":
            self.d[:, :, feat_name_or_number] = arr


class NPYRaster(GDALRasterIO):

    def __init__(self, npy_fn=None):
        super().__init__()
        self.geo_json_fn = None
        self.data = None
        self.initNPYRaster(npy_fn)

    def initNPYRaster(self, npy_fn):
        if npy_fn is None:
            return
        self.initRaster()
        self.initGEORaster()

        geo_json_fn = changext(npy_fn, ".geonpy")
        json_dict = readJson(geo_json_fn)
        self.raster_ds = json_dict
        self.gdal_raster_fn = npy_fn
        self.geo_json_fn = geo_json_fn
        self.grr.init(self.gdal_raster_fn)
        self.geo_transform = json_dict["geo_transform"]
        self.inv_geo_transform = json_dict["inv_geo_transform"]
        self.n_rows = json_dict["n_rows"]
        self.n_columns = json_dict["n_columns"]
        self.n_channels = json_dict["n_channels"]
        self.names = json_dict["names"]

        self.src_srs = osr.SpatialReference()
        self.src_srs.ImportFromEPSG(4326)
        self.dst_srs = osr.SpatialReference()
        wkt = json_dict["wkt"]
        self.dst_srs.ImportFromWkt(wkt)
        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromEPSG(4326)
        if __version__ >= "3.0.0":
            self.src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            self.dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            wgs84_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if wkt != "":
            self.coor_trans = osr.CoordinateTransformation(self.src_srs, self.dst_srs)
            self.towgs84_coor_trans = osr.CoordinateTransformation(self.dst_srs, wgs84_srs)
            self.wgs84_to_this = osr.CoordinateTransformation(wgs84_srs, self.dst_srs)

        self.save_geo_transform = self.geo_transform
        self.save_probing = self.dst_srs.ExportToWkt()

        self.getRange()
        self.x_min = self.raster_range[0]
        self.x_max = self.raster_range[1]
        self.y_min = self.raster_range[2]
        self.y_max = self.raster_range[3]
        self.x_size = self.geo_transform[1]
        self.y_size = self.geo_transform[5]

        return

    def readAsArrayCenter(self, x_row_off=0.0, y_column_off=0.0, win_row_size=None, win_column_size=None,
                          interleave='band',
                          band_list=None, is_geo=False, is_trans=False, is_range=False, is_01=False):
        if self.data is None:
            self.readNPYData()
        if is_geo:
            if is_trans:
                x_row_off, y_column_off, _ = self.coor_trans.TransformPoint(x_row_off, y_column_off)
            x_row_off, y_column_off = self.coorGeo2Raster(x_row_off, y_column_off)
        x_row_off, y_column_off = int(x_row_off), int(y_column_off)
        row, column = x_row_off, y_column_off
        win_spl = [0, 0, 0, 0]
        win_spl[0] = 0 - int(win_row_size / 2)
        win_spl[1] = 0 + round(win_row_size / 2 + 0.1)
        win_spl[2] = 0 - int(win_column_size / 2)
        win_spl[3] = 0 + round(win_column_size / 2 + 0.1)
        self.d = self.data[:, row + win_spl[0]: row + win_spl[1], column + win_spl[2]: column + win_spl[3]]
        self.interleave = interleave
        return self.d

    def readAsArray(self, x_row_off=0.0, y_column_off=0.0, win_row_size=None, win_column_size=None, interleave='band',
                    band_list=None, is_geo=False, is_trans=False, is_range=False, is_01=False):
        if self.data is None:
            self.readNPYData()
        if is_geo:
            if is_trans:
                x_row_off, y_column_off, _ = self.coor_trans.TransformPoint(x_row_off, y_column_off)
            x_row_off, y_column_off = self.coorGeo2Raster(x_row_off, y_column_off)
        x_row_off, y_column_off = int(x_row_off), int(y_column_off)
        self.d = self.data[:, x_row_off: x_row_off + win_row_size, y_column_off: y_column_off + win_column_size]
        self.interleave = interleave
        return self.d

    def readNPYData(self):
        self.data = np.load(self.gdal_raster_fn)
        if len(self.data.shape) == 2:
            self.data = np.array([self.data])


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

    def sampling2(self, x, y, win_row, win_column, is_jdt=True, is_trans=False, is_ret_index=False):
        win_spl = [0, 0, 0, 0]
        win_spl[0] = 0 - int(win_row / 2)
        win_spl[1] = 0 + round(win_row / 2 + 0.1)
        win_spl[2] = 0 - int(win_column / 2)
        win_spl[3] = 0 + round(win_column / 2 + 0.1)
        to_list = []

        jdt = Jdt(len(x), "sampling2").start(is_jdt)
        to_data = np.zeros((len(x), self.n_channels, win_row, win_column))
        for i in range(len(x)):
            x0, y0 = x[i], y[i]
            if is_trans:
                x0, y0, _ = self.gr.wgs84_to_this.TransformPoint(x0, y0)
            data = self.sampling2One(win_spl, x0, y0, win_row=win_row, win_column=win_column)
            if data is not None:
                if data.shape == to_data.shape[1:]:
                    to_data[i] = data
                    to_list.append(1)
                else:
                    to_list.append(0)
            else:
                to_list.append(0)
            jdt.add(is_jdt=is_jdt)
        jdt.end(is_jdt=is_jdt)
        if is_ret_index:
            return to_data, to_list
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

    def samplingToData(self, x, y, is_jdt=True, is_trans=False):
        data = self.sampling2(x, y, 1, 1, is_jdt, is_trans)
        return data[:, :, 0, 0]


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


class NumpyDataCenter:

    def __init__(self, dim=2, win_size=None, spl_size=None):
        self.dim = dim
        self.spl_size = spl_size
        self.win_size = win_size
        self.win_spl = [0, 0, 0, 0]
        self.row = 0
        self.column = 0
        self.range = [0, 0, 0, 0]

        self.initWinSize(win_size)
        self.initSampleSize(spl_size)
        self.initRange()

    def copy(self):
        return NumpyDataCenter(self.dim, self.win_size, self.spl_size)

    def initWinSize(self, win_size=None):
        self.win_size = win_size
        if self.win_size is None:
            return
        row_size, column_size = win_size[0], win_size[1]
        self.win_spl[0] = 0 - int(row_size / 2)
        self.win_spl[1] = 0 + round(row_size / 2 + 0.1)
        self.win_spl[2] = 0 - int(column_size / 2)
        self.win_spl[3] = 0 + round(column_size / 2 + 0.1)

    def initSampleSize(self, spl_size=None):
        self.spl_size = spl_size
        if self.spl_size is None:
            return
        self.row = int(self.spl_size[0] / 2.0)
        self.column = int(self.spl_size[1] / 2.0)

    def initRange(self):

        self.range[0] = self.row + self.win_spl[0]
        self.range[1] = self.row + self.win_spl[1]
        self.range[2] = self.column + self.win_spl[2]
        self.range[3] = self.column + self.win_spl[3]

    def fit(self, x):
        if self.win_size is None:
            return x
        if self.spl_size is None:
            if self.dim == 2:
                self.spl_size = x.shape
            elif self.dim == 3:
                self.spl_size = x.shape[1:]
            elif self.dim == 4:
                self.spl_size = x.shape[2:]
            self.initSampleSize(self.spl_size)
            self.initRange()

        if self.dim == 2:
            out_x = x[self.range[0]:self.range[1], self.range[2]:self.range[3]]
        elif self.dim == 3:
            out_x = x[:, self.range[0]:self.range[1], self.range[2]:self.range[3]]
        elif self.dim == 4:
            out_x = x[:, :, self.range[0]:self.range[1], self.range[2]:self.range[3]]
        else:
            return x

        return out_x

    def fit2(self, x):
        if self.win_size is None:
            return x
        if self.spl_size is None:
            if self.dim == 2:
                self.spl_size = x.shape
            elif self.dim == 3:
                self.spl_size = x.shape[1:]
            self.initSampleSize(self.spl_size)
            self.initRange()

        out_x = x[:, :, self.range[0]:self.range[1], self.range[2]:self.range[3]]
        return out_x


def initCallBack(x):
    return x


class SRTFeatureCallBack:

    def __init__(self, callback_func=None, is_trans=False, feat_name=None):
        if callback_func is None:
            callback_func = initCallBack
        self.call_back = callback_func
        self.is_trans = is_trans
        self.feat_name = feat_name

    def fit(self, x: np.ndarray, *args, **kwargs):
        if self.is_trans:
            x = self.call_back(x)
        return x


class SRTFeatureCallBackScaleMinMax(SRTFeatureCallBack):

    def __init__(self, x_min=None, x_max=None, is_trans=False, is_to_01=False):
        super(SRTFeatureCallBackScaleMinMax, self).__init__(is_trans=is_trans)
        self.x_min = x_min
        self.x_max = x_max
        self.is_to_01 = is_to_01

    def fit(self, x: np.ndarray, *args, **kwargs):
        if self.is_trans:
            x_min, x_max = self.x_min, self.x_max
            if x_min is None:
                x_min = np.min(x)
            if x_max is None:
                x_max = np.max(x)
            x = np.clip(x, x_min, x_max)
            if self.is_to_01:
                x = (x - x_min) / (x_max - x_min)
        return x


class SRTFeatureCallBackList:

    def __init__(self):
        self._callback_list = []
        self._n_iter = 0

    def addScaleMinMax(self, x_min=None, x_max=None, is_trans=False, is_to_01=False):
        self._callback_list.append(SRTFeatureCallBackScaleMinMax(
            x_min=x_min, x_max=x_max, is_trans=is_trans, is_to_01=is_to_01))
        return self

    def add(self, callback_func, is_trans=False):
        self._callback_list.append(SRTFeatureCallBack(callback_func=callback_func, is_trans=is_trans))

    def __len__(self):
        return len(self._callback_list)

    def __getitem__(self, item):
        return self._callback_list[item]

    def __iter__(self):
        return self

    def __next__(self):
        if self._n_iter == len(self._callback_list):
            self._n_iter = 0
            raise StopIteration()
        self._n_iter += 1
        return self._callback_list[self._n_iter - 1]

    def __contains__(self, item):
        return item in self._callback_list

    def fit(self, x):
        for callback in self._callback_list:
            x = callback.fit(x)
        return x


class SRTFeaturesMemory:

    def __init__(self, length=None, names=None):
        self.names = []
        self._callbacks = []
        self.length = length
        if length is None:
            if names is not None:
                self.length = len(names)
                self.names = names
        else:
            self.length = length
            self.initNames()

    def __len__(self):
        return self.length

    def initNames(self, names=None):
        if names is None:
            names = ["Name{}".format(i) for i in range(self.length)]
        else:
            if len(names) != self.length:
                warnings.warn("length of names not eq data. {} not eq {}".format(len(names), self.length))
        self.names = names
        return self

    def initCallBacks(self):
        self._callbacks = [SRTFeatureCallBackList() for _ in range(self.length)]
        return self

    def callbacks(self, name_number) -> SRTFeatureCallBackList:
        return self._callbacks[self._name_number(name_number)]

    def _name_number(self, name_number):
        if isinstance(name_number, str):
            return self.names.index(name_number)
        else:
            return name_number


def tiffAddColorTable(gtif_fn, band_count=1, code_colors=None):
    if code_colors is None:
        code_colors = {}
    if len(code_colors) == 0:
        return

    color_table = gdal.ColorTable()
    for c_code, color in code_colors.items():
        color_table.SetColorEntry(c_code, color)

    input_ds = gdal.Open(gtif_fn, gdal.GA_Update)
    band: gdal.Band = input_ds.GetRasterBand(band_count)
    band.SetColorTable(color_table)

    del input_ds


def dataModelPredict(data, data_deal, is_jdt, model):
    data_c = np.zeros((data.shape[1], data.shape[2]))
    jdt = Jdt(data.shape[1], "dataModelPredict").start(is_jdt=is_jdt)
    for i in range(data.shape[1]):
        jdt.add(is_jdt=is_jdt)
        if data_deal is not None:
            x = data_deal(data[:, i, :].T)
        else:
            x = data[:, i, :].T
        y = model.predict(x)
        data_c[i, :] = y
    jdt.end(is_jdt=is_jdt)
    return data_c


def imdc1(model, data, to_geo_fn, gr, data_deal=None, is_jdt=True, color_table=None):
    imdc = dataModelPredict(data, data_deal=data_deal, is_jdt=is_jdt, model=model)
    gr.save(imdc.astype("int8"), to_geo_fn, fmt="GTiff", dtype=gdal.GDT_Byte, options=["COMPRESS=PACKBITS"])
    if color_table is not None:
        tiffAddColorTable(to_geo_fn, 1, color_table)


class SRTFeaturesCalculation:
    """ Features calculate self, extraction

    type: data type as df|np
    """

    DATA_TYPES = ["df", "np"]

    def __init__(self, *init_names):
        self.init_names = list(init_names)
        self.type = None
        self.data = None
        self.calculation = []

    def initData(self, _type, _data):
        if _type not in self.DATA_TYPES:
            raise Exception("Can not support data type of \"{}\". Not in {}.".format(_type, self.DATA_TYPES))
        self.type = _type
        self.data = _data

    def __getitem__(self, item):
        if self.type == "np":
            return self.data[self.init_names.index(item)]
        elif self.type == "df":
            return self.data[item]
        else:
            return None

    def __setitem__(self, key, value):
        if self.type == "np":
            self.data[self.init_names.index(key)] = value
        elif self.type == "df":
            self.data[key] = value

    def __contains__(self, item):
        return item in self.init_names

    def add(self, init_name, fit_names, func):
        if init_name not in self.init_names:
            self.init_names.append(init_name)
            warnings.warn("Init name \"{}\" not in list and add.".format(init_name))
        self.calculation.append((init_name, fit_names, func))

    def fit(self, ):
        for init_name, fit_names, func in self.calculation:
            datas = {}
            for name in fit_names:
                if self.type == "df":
                    datas[name] = self.__getitem__(name).values
                else:
                    datas[name] = self.__getitem__(name)
            data = func(datas)
            self.__setitem__(init_name, data)


def dataPredictPatch(image_data, win_size, predict_func, is_jdt=True, **kwargs):
    imdc = np.zeros(image_data.shape[1:])
    win_row, win_column = win_size
    image_data_view = sliding_window_view(image_data, (image_data.shape[0], win_row, win_column), writeable=True)
    n_rows, n_columns = image_data_view.shape[1], image_data_view.shape[2]
    row_start, column_start = int(win_row / 2), int(win_column / 2)
    jdt = Jdt(int(image_data_view.shape[1]), "Raster Predict Patch").start(is_jdt=is_jdt)
    for i in range(0, image_data_view.shape[1]):
        x_data = image_data_view[0, i]
        y = predict_func(x_data)
        imdc[row_start + i, column_start: column_start + n_columns] = y
        jdt.add(is_jdt=is_jdt)
    jdt.end(is_jdt=is_jdt)
    return imdc


def dataPredictPatch2(image_data, win_size, predict_func, n=1000, is_jdt=True, **kwargs):
    imdc = np.zeros(image_data.shape[1:])
    win_row, win_column = win_size

    image_data_view = sliding_window_view(image_data, (image_data.shape[0], win_row, win_column), writeable=True)
    n_rows, n_columns = image_data_view.shape[1], image_data_view.shape[2]
    # image_data_view = np.reshape(image_data_view, (n_rows * n_columns, image_data.shape[0], win_row, win_column))

    row_start, column_start = int(win_row / 2), int(win_column / 2)

    rows, columns = [], []
    rows_run, columns_run = [], []

    jdt = Jdt(int(n_rows * n_columns / n), "Raster Predict Patch").start(is_jdt=is_jdt)

    class _time_save:

        def __init__(self):
            self.time_del = [0, 0, 0]
            self.n_time = 0

    _time_save_data = _time_save()

    def calculate():
        time1 = time.time()
        x_data = image_data_view[n_select_start:n_select_end].copy()
        time2 = time.time()
        y = predict_func(x_data)
        imdc[rows, columns] = y
        time3 = time.time()

        rows.clear()
        columns.clear()
        rows_run.clear()
        columns_run.clear()

        _time_save_data.n_time += 1
        _time_save_data.time_del[0] += time10
        _time_save_data.time_del[1] += time2 - time1
        _time_save_data.time_del[2] += time3 - time2

        if _time_save_data.n_time >= 10:
            _time_save_data.time_del = [0, 0, 0]
            _time_save_data.n_time = 0

    time_0 = time.time()
    n_select_start, n_select_end = 0, 0
    n_select = 0
    for i in range(row_start, row_start + n_rows):
        for j in range(column_start, row_start + n_columns):
            rows.append(i)
            columns.append(j)
            n_select += 1
            if n_select >= n:
                time10 = time.time() - time_0
                n_select_end = n_select
                calculate()
                time_0 = time.time()
                n_select = 0
                n_select_start = n_select_end
                jdt.add(is_jdt=is_jdt)
    n_select_end = n_select
    calculate()
    jdt.end(is_jdt=is_jdt)
    return imdc


def imdc2(func_predict, data, win_size, to_geo_fn, gr, data_deal=None, is_jdt=True, color_table=None, func_run=None,
          n=1000, ):
    if data_deal is not None:
        data = data_deal(data)
    if func_run is None:
        func_run = dataPredictPatch
    imdc = func_run(data, win_size, func_predict, n=n, is_jdt=is_jdt)
    gr.save(imdc.astype("int8"), to_geo_fn, fmt="GTiff", dtype=gdal.GDT_Byte, options=["COMPRESS=PACKBITS"])
    if color_table is not None:
        tiffAddColorTable(to_geo_fn, 1, color_table)


class GDALImdc:

    def __init__(self, *raster_fns, is_sfm=True, sfc=None):
        self.raster_fns = datasCaiFen(raster_fns)
        self.color_table = None
        self.sfm = SRTFeaturesMemory()
        self.sfc = sfc
        self.is_sfm = is_sfm
        if len(self.raster_fns) >= 1:
            self.sfm = SRTFeaturesMemory(names=GDALRaster(raster_fns[0]).names).initCallBacks()

    def imdc1(self, model, to_imdc_fn, fit_names=None, data_deal=None, is_jdt=True, color_table=None):
        color_table, fit_names = self._initImdc(color_table, fit_names)

        if len(self.raster_fns) == 1:
            raster_fn = self.raster_fns[0]
            self._imdc1(model, raster_fn, to_imdc_fn, fit_names, data_deal, is_jdt, color_table)
        else:
            to_imdc_dirname = changext(to_imdc_fn, "_tiles")
            if not os.path.isdir(to_imdc_dirname):
                os.mkdir(to_imdc_dirname)

            to_fn_tmps = []
            for fn in self.raster_fns:
                to_imdc_fn_tmp = os.path.join(to_imdc_dirname, changext(os.path.split(fn)[1], "_imdc.tif"))
                to_fn_tmps.append(to_imdc_fn_tmp)
                self._imdc1(model, fn, to_imdc_fn_tmp, fit_names, data_deal, is_jdt, color_table)

            gdal_merge_main(["gdal_merge_main", "-of", "GTiff", "-n", "0", "-ot", "Byte", "-co", "COMPRESS=PACKBITS",
                             "-o", to_imdc_fn, *to_fn_tmps, ])

            if color_table is not None:
                tiffAddColorTable(to_imdc_fn, 1, code_colors=color_table)

    def _initImdc(self, color_table, fit_names):
        if fit_names is None:
            fit_names = self.sfm.names
        if color_table is None:
            color_table = self.color_table
        if len(self.raster_fns) == 0:
            raise Exception("Can not find raster")
        return color_table, fit_names

    def _imdc1(self, model, raster_fn, to_geo_fn, fit_names, data_deal, is_jdt, color_table):
        gr = GDALRaster(raster_fn)
        data = np.zeros((len(fit_names), gr.n_rows, gr.n_columns))
        self.readRaster(data, fit_names, gr, is_jdt)
        imdc1(model, data, to_geo_fn, gr, data_deal=data_deal, is_jdt=is_jdt, color_table=color_table)
        data = None

    def readRaster(self, data, fit_names, gr, is_jdt, *args, **kwargs):
        read_names, sfc = self.getReadNames(data, fit_names, gr)
        jdt = Jdt(len(read_names), "Read Raster").start(is_jdt)
        for name in read_names:
            data_i = gr.readGDALBand(name)
            data_i[np.isnan(data_i)] = 0
            if self.is_sfm:
                data_i = self.sfm.callbacks(name).fit(data_i)
            sfc[name] = data_i
            jdt.add(is_jdt)
        jdt.end(is_jdt)
        sfc.fit()

        return

    def getReadNames(self, data, fit_names, gr):
        sfc = self.sfc
        if sfc is None:
            sfc = SRTFeaturesCalculation(*fit_names)
        sfc.initData("np", data)
        read_names = []
        for name in sfc.init_names:
            if name in gr.names:
                read_names.append(name)
        return read_names, sfc

    def _imdc2(self, func_predict, raster_fn, win_size, to_geo_fn, fit_names, data_deal, is_jdt, color_table, n=1000):
        gr = GDALRaster(raster_fn)
        data = np.zeros((len(fit_names), gr.n_rows, gr.n_columns), dtype="float32")
        self.readRaster(data, fit_names, gr, is_jdt)
        func_run = dataPredictPatch
        if n != -1:
            func_run = dataPredictPatch2
        imdc2(func_predict, data, win_size=win_size, to_geo_fn=to_geo_fn, gr=gr, data_deal=data_deal, is_jdt=is_jdt,
              color_table=color_table, func_run=func_run, n=n)
        data = None

    def imdc2(self, func_predict, win_size, to_imdc_fn, fit_names, data_deal=None, is_jdt=True, color_table=None, n=-1):
        color_table, fit_names = self._initImdc(color_table, fit_names)

        if len(self.raster_fns) == 1:
            raster_fn = self.raster_fns[0]
            self._imdc2(func_predict, raster_fn, win_size, to_imdc_fn, fit_names, data_deal, is_jdt, color_table, n=n)
        else:
            to_imdc_dirname = changext(to_imdc_fn, "_tiles")
            if not os.path.isdir(to_imdc_dirname):
                os.mkdir(to_imdc_dirname)

            to_fn_tmps = []
            for fn in self.raster_fns:
                to_imdc_fn_tmp = os.path.join(to_imdc_dirname, changext(os.path.split(fn)[1], "_imdc.tif"))
                to_fn_tmps.append(to_imdc_fn_tmp)
                self._imdc2(func_predict, fn, win_size, to_imdc_fn_tmp, fit_names, data_deal, is_jdt, color_table, n=n)

            gdal_merge_main(["gdal_merge_main", "-of", "GTiff", "-n", "0", "-ot", "Byte", "-co", "COMPRESS=PACKBITS",
                             "-o", to_imdc_fn, *to_fn_tmps, ])

            if color_table is not None:
                tiffAddColorTable(to_imdc_fn, 1, code_colors=color_table)


class GRCNR_featExt:

    def __init__(self, data_dict, func_ext, *args, **kwargs):
        self.data_dict = data_dict
        self.func_ext = func_ext
        self.args = args
        self.kwargs = kwargs
        self.data = None

    def fit(self):
        if self.data is None:
            self.data = self.func_ext(self.data_dict, *self.args, **self.kwargs)
        return self.data


class GDALRasterChannel:
    """ GDAL Raster Channel """

    GRS = {}

    def __init__(self):
        self.data = {}
        self._n_iter = 0
        self.shape = ()

    def addGDALData(self, raster_fn, field_name, channel=None):
        gr = self.addGR(raster_fn)
        if channel is None:
            channel = field_name
        self.data[field_name] = gr.readGDALBand(channel)
        if self.data[field_name] is None:
            print("Warning: can not read data from {0}:{1}".format(field_name, channel))
        if self.shape == ():
            self.shape = self.data[field_name].shape
        return field_name

    def addGR(self, raster_fn):
        raster_fn = os.path.abspath(raster_fn)
        if raster_fn not in self.GRS:
            self.GRS[raster_fn] = GDALRaster(raster_fn)
        gr: GDALRaster = self.GRS[raster_fn]
        return gr

    def _getGR(self, geo_fn) -> GDALRaster:
        if geo_fn is None:
            ks = list(self.GRS.keys())
            geo_fn = ks[0]
        else:
            if geo_fn not in self.GRS:
                self.GRS[geo_fn] = GDALRaster(geo_fn)
        gr: GDALRaster = self.GRS[geo_fn]
        return gr

    def addGDALDatas(self, raster_fn, names=None):
        gr = self.addGR(raster_fn)
        if names is None:
            names = gr.names
        for name in names:
            if name in gr.names:
                self.addGDALData(raster_fn, name)
            else:
                warnings.warn("name of \"{0}\" not in this raster names. raster_fn:{1}".format(name, raster_fn))

    def saveRasterToFile(self, raster_fn, *this_key, geo_fn=None, **kwargs):
        gr = self._getGR(geo_fn)
        d = []
        for k in this_key:
            d.append(self.data[k])
        d = np.array(d)
        gr.save(d=d, save_geo_raster_fn=raster_fn, **kwargs)

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, item):
        return self.data[item]

    def __contains__(self, item):
        return item in self.data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        if self._n_iter == len(self.data):
            self._n_iter = 0
            raise StopIteration()
        self._n_iter += 1
        ks = list(self.data.keys())
        return ks[self._n_iter - 1]

    def getRasterNames(self, raster_fn):
        gr = self.addGR(raster_fn)
        return gr.names

    def fieldNamesToData(self, *field_names):
        data = np.zeros((len(field_names), self.shape[0], self.shape[1]))
        for i, field_name in enumerate(field_names):
            data[i] = self.data[field_name]
        return data

    def addFeatExt(self, field_name, func_ext, *args, **kwargs):
        self.data[field_name] = GRCNR_featExt(self.data, func_ext, *args, **kwargs).fit()
        return field_name


class ConfusionMatrix:

    def __init__(self, n_class=0, class_names=None):
        """
                预测
            |  0  |  1  |
             -----------    制
        真 0 |    |    |    图
        实 1 |    |    |    精
        ------------------  度
              用户精度

        :param n_class: number of category
        :param class_names: names of category
        """
        self._n_class = n_class
        self._class_names = class_names
        if n_class == 0:
            if class_names is not None:
                n_class = len(class_names)
            else:
                return
        self._n_class = n_class
        self._cm = np.zeros((n_class, n_class))
        self._cm_accuracy = self.calCM()
        if self._class_names is None:
            self._class_names = ["CATEGORY_" + str(i + 1) for i in range(n_class)]
        elif len(self._class_names) != n_class:
            raise Exception("The number of category names is different from the number of input categories.")
        self._it_count = 0

    def toDict(self):
        to_dict = {
            "_n_class": self._n_class,
            "_class_names": self._class_names,
        }
        return to_dict

    def CNAMES(self):
        return self._class_names

    def addData(self, y_true, y_pred):
        for i in range(len(y_true)):
            if int(y_true[i]) > 0 or int(y_pred[i]) > 0:
                if (int(y_pred[i]) - 1) == 40:
                    continue
                if (int(y_true[i]) <= 0) or (int(y_pred[i]) <= 0):
                    continue
                if (int(y_true[i]) > self._n_class) or (int(y_pred[i]) > self._n_class):
                    continue
                self._cm[int(y_true[i]) - 1, int(y_pred[i]) - 1] += 1
        self._cm_accuracy = self.calCM()

    def UA(self, idx_name=None):
        acc = self._cm_accuracy[self._n_class + 1, :self._n_class]
        if idx_name is None:
            return acc
        elif isinstance(idx_name, int):
            return acc[idx_name - 1]
        elif isinstance(idx_name, str):
            return acc[self._class_names.index(idx_name)]

    def PA(self, idx_name=None):
        acc = self._cm_accuracy[:self._n_class, self._n_class + 1]
        if idx_name is None:
            return acc
        elif isinstance(idx_name, int):
            return acc[idx_name - 1]
        elif isinstance(idx_name, str):
            return acc[self._class_names.index(idx_name)]

    def OA(self):
        return self._cm_accuracy[-1, -1]

    def getKappa(self):
        pe = np.sum(np.sum(self._cm, axis=0) * np.sum(self._cm, axis=1)) / (self._cm.sum() * self._cm.sum() + eps)
        return (self.OA() / 100 - pe) / (1 - pe + eps)

    #
    # def getKappa(self):
    #     pe = np.sum(self.OA() * self.PA()) / (self._cm.sum() * self._cm.sum() + eps)
    #     return (self.OA() - pe) / (1 - pe + eps)

    def printCM(self):
        print(self.fmtCM())

    def clear(self):
        self._cm = np.zeros((self._n_class, self._n_class))
        self._cm_accuracy = self.calCM()

    def fmtCM(self, cm: np.array = None, cate_names=None):
        if cm is None:
            cm = self._cm_accuracy
        if cate_names is None:
            cate_names = self._class_names
        fmt_row0 = "{:>8}"
        fmt_column0 = "{:>8}"
        fmt_number = "{:>8d}"
        fmt_float = "{:>8.2f}"
        n_cate = len(cate_names)
        out_s = ""
        out_s += fmt_column0.format("CM")
        for i in range(n_cate):
            out_s += " " + fmt_row0.format(cate_names[i])
        out_s += " " + fmt_row0.format("SUM")
        out_s += " " + fmt_row0.format("PA") + "\n"
        for i in range(n_cate):
            out_s += fmt_column0.format(cate_names[i])
            for j in range(n_cate):
                out_s += " " + fmt_number.format(int(cm[i, j]))
            out_s += " " + fmt_number.format(int(cm[i, n_cate]))
            out_s += " " + fmt_float.format(cm[i, n_cate + 1]) + "\n"
        out_s += fmt_column0.format("SUM")
        for i in range(n_cate):
            out_s += " " + fmt_number.format(int(cm[n_cate, i]))
        out_s += " " + fmt_number.format(int(cm[n_cate, n_cate]))
        out_s += " " + fmt_float.format(cm[n_cate, n_cate + 1]) + "\n"
        out_s += fmt_column0.format("UA")
        for i in range(n_cate):
            out_s += " " + fmt_float.format(cm[n_cate + 1, i])
        out_s += " " + fmt_float.format(cm[n_cate + 1, n_cate])
        out_s += " " + fmt_float.format(cm[n_cate + 1, n_cate + 1]) + "\n"
        return out_s

    def calCM(self, cm: np.array = None):
        if cm is None:
            cm = self._cm
        n_class = cm.shape[0]
        out_cm = np.zeros([n_class + 2, n_class + 2])
        out_cm[:n_class, :n_class] = cm
        out_cm[n_class, :] = np.sum(out_cm, axis=0)
        out_cm[:, n_class] = np.sum(out_cm, axis=1)
        out_cm[n_class + 1, :] = np.diag(out_cm) * 1.0 / (out_cm[n_class, :] + 0.00001) * 100
        out_cm[:, n_class + 1] = np.diag(out_cm) * 1.0 / (out_cm[:, n_class] + 0.00001) * 100
        out_cm[n_class + 1, n_class + 1] = (np.sum(np.diag(cm))) / (out_cm[n_class, n_class] + 0.00001) * 100
        return out_cm

    def categoryNames(self):
        return self._class_names

    def accuracy(self):
        """ OA PA UA """
        cm = self._cm_accuracy
        cate_names = self._class_names
        pa_d = self.PA().tolist()
        ua_d = self.UA().tolist()
        oa_d = np.diag(self._cm_accuracy)
        # oa_d = oa_d[:len(cate_names)] / (oa_d[-2] + 0.00001)
        # oa_d = oa_d.tolist()
        to_dict = {}
        for i, cate_name in enumerate(cate_names):
            to_dict[cate_name] = [float(oa_d[-1]), float(pa_d[i]), float(ua_d[i])]
        return to_dict

    def accuracyCategory(self, category):
        cname = category
        if isinstance(category, str):
            category = self._class_names.index(category)
        else:
            category = category - 1

        cm_category = np.zeros((2, 2))
        cm_category[0, 0] = int(self._cm[category, category])
        cm_category[0, 1] = int(np.sum(np.diag(self._cm[category, :]))) - int(self._cm[category, category])
        cm_category[1, 0] = int(np.sum(np.diag(self._cm[:, category]))) - int(self._cm[category, category])
        cm_category[1, 1] = int(np.sum(self._cm)) - int(np.sum(cm_category))

        cm = ConfusionMatrix(2, [str(cname), "NOT_KNOW"])
        cm._cm = cm_category
        cm._cm_accuracy = cm.calCM()
        return cm

    def __iter__(self):
        return self

    def __next__(self):
        # 获取下一个数
        if self._it_count < len(self._class_names):
            result = self._class_names[self._it_count]
            self._it_count += 1
            return result
        else:
            self._it_count = 0
            raise StopIteration

    def toList(self):
        return self._cm.tolist()


def numberfilename(init_fn, sep="", start_number=1):
    dirname = os.path.dirname(init_fn)
    if not os.path.isdir(dirname):
        raise Exception("Can not find dirname: {0}".format(dirname))
    fn, ext = os.path.splitext(init_fn)
    while True:
        to_fn = "{0}{1}{2}{3}".format(fn, sep, start_number, ext)
        if not os.path.isfile(to_fn):
            return to_fn
        start_number += 1


class SRTModelImageInit:

    def __init__(
            self,
            model_name=None,
            model_dirname=None,
            category_names=None,
            category_colors=None,
    ):
        self.mod_fn = None
        if category_names is None:
            category_names = ["NOT_KNOW"]

        self.model_name = model_name
        self.model_dirname = model_dirname
        self.category_names = category_names
        self.category_colors = category_colors

        self.ext_funcs = {}
        self.color_table = {}

        self.train_cm = ConfusionMatrix()
        self.test_cm = ConfusionMatrix()

    def initColorTable(self, color_table_dict, *code_colors, **kwargs):
        for c_code, color in color_table_dict.items():
            self.color_table[c_code] = color
        for c_code, color in code_colors:
            self.color_table[c_code] = color
        for c_code, color in kwargs.items():
            self.color_table[c_code] = color

    def addFeatExt(self, to_field_name, ext_func, *args, **kwargs):
        self.ext_funcs[to_field_name] = (ext_func, args, kwargs)

    def train(self, *args, **kwargs):
        return

    def imdc(self, *args, **kwargs):
        return

    def initImdc1(self, geo_fns, grc):
        if geo_fns is None:
            geo_fns = []
        if grc is None:
            grc = GDALRasterChannel()
        if geo_fns is not None:
            if isinstance(geo_fns, str):
                geo_fns = [geo_fns]
            for geo_fn in geo_fns:
                grc.addGDALDatas(geo_fn)
        if self.ext_funcs:
            for field_name, (ext_func, args, kwargs) in self.ext_funcs.items():
                grc.data[field_name] = ext_func(grc, *args, **kwargs)
        return grc

    def predict(self, *args, **kwargs):
        return []

    def saveCodeFile(self, *args, **kwargs, ):
        if ("code_fn" in kwargs) and ("to_code_fn" in kwargs):
            shutil.copyfile(kwargs["code_fn"], kwargs["to_code_fn"])

    def getToGeoFn(self, to_geo_fn, ext="_imdc.tif"):
        if to_geo_fn is None:
            if self.mod_fn is None:
                if len(sys.argv) >= 2:
                    self.mod_fn = sys.argv[1]
            if self.mod_fn is not None:
                to_geo_fn = changext(self.mod_fn, ext)
                to_geo_fn = numberfilename(to_geo_fn)
        if to_geo_fn is None:
            raise Exception("Can not get to geo filename.")
        return to_geo_fn


class TrainLog(SRTCollection):

    def __init__(self, save_csv_file="train_save.csv", log_filename="train_log.txt"):
        super(TrainLog, self).__init__()

        self.field_names = []
        self.field_types = []
        self.field_index = {}
        self.field_datas = []
        self.n_datas = 0

        self.is_save_log = False
        self.log_filename = log_filename
        # "trainlog" + time.strftime("%Y%m%d") + ".txt"

        self.print_type = "table"  # table or keyword
        self.print_type_fmts = {"table": "{0}", "keyword": "{1}: {0}"}
        self.print_sep = "\t"
        self.print_field_names = []
        self.print_column_fmt = []
        self.print_float_decimal = 3
        self.print_type_column_width = {"int": 6, "float": 12, "string": 26}
        self.print_type_init_v = {"int": 0, "float": 0.0, "string": ""}
        self.print_type_column_fmt = {}
        self._getTypeColumnFmt()

        self.save_csv_file = save_csv_file

    def toDict(self):
        to_dict = {
            "_n_iter": self._n_iter,
            "_n_next": self._n_next,
            "field_names": self.field_names,
            "field_types": self.field_types,
            "field_index": self.field_index,
            "field_datas": self.field_datas,
            "n_datas": self.n_datas,
            "is_save_log": self.is_save_log,
            "log_filename": self.log_filename,
            "print_type": self.print_type,
            "print_type_fmts": self.print_type_fmts,
            "print_sep": self.print_sep,
            "print_field_names": self.print_field_names,
            "print_column_fmt": self.print_column_fmt,
            "print_float_decimal": self.print_float_decimal,
            "print_type_column_width": self.print_type_column_width,
            "print_type_init_v": self.print_type_init_v,
            "print_type_column_fmt": self.print_type_column_fmt,
            "save_csv_file": self.save_csv_file,
        }
        return to_dict

    def loadDict(self, to_dict):
        self._n_iter = to_dict["_n_iter"]
        self._n_next = to_dict["_n_next"]
        self.field_names = to_dict["field_names"]
        self.field_types = to_dict["field_types"]
        self.field_index = to_dict["field_index"]
        self.field_datas = to_dict["field_datas"]
        self.n_datas = to_dict["n_datas"]
        self.is_save_log = to_dict["is_save_log"]
        self.log_filename = to_dict["log_filename"]
        self.print_type = to_dict["print_type"]
        self.print_type_fmts = to_dict["print_type_fmts"]
        self.print_sep = to_dict["print_sep"]
        self.print_field_names = to_dict["print_field_names"]
        self.print_column_fmt = to_dict["print_column_fmt"]
        self.print_float_decimal = to_dict["print_float_decimal"]
        self.print_type_column_width = to_dict["print_type_column_width"]
        self.print_type_init_v = to_dict["print_type_init_v"]
        self.print_type_column_fmt = to_dict["print_type_column_fmt"]
        self.save_csv_file = to_dict["save_csv_file"]

    def _getTypeColumnFmt(self):
        self.print_type_column_fmt = {
            "int": "{:>" + str(self.print_type_column_width["int"]) + "d}",
            "float": "{:>" + str(self.print_type_column_width["float"]) + "." + str(self.print_float_decimal) + "f}",
            "string": "{:>" + str(self.print_type_column_width["string"]) + "}"
        }

    def addField(self, field_name, field_type="string", init_v=None):
        if field_name in self.field_names:
            raise Exception("Error: field name \"{0}\" have in field names.".format(field_name))
        self.field_names.append(field_name)
        self.field_types.append(field_type)
        if init_v is not None:
            self.print_type_init_v[field_type] = init_v
        self.field_index[field_name] = len(self.field_names) - 1
        self._n_next.append(field_name)

    def getFieldName(self, idx):
        return self.field_names[idx]

    def getFieldIndex(self, name):
        return self.field_index[name]

    def printOptions(self, print_type="table", print_sep="\t", print_field_names=None, print_column_fmt=None,
                     print_float_decimal=3):
        """ print options
        /
        :param print_float_decimal: float decimal
        :param print_type: table or keyword default "table"
        :param print_sep: sep default "\t"
        :param print_field_names: default all
        :param print_column_fmt: default get from field type
        :return: None
        """
        self.print_type = print_type  # table or keyword
        self.print_sep = print_sep
        self.print_float_decimal = print_float_decimal

        if print_field_names is None:
            print_field_names = []
        if not print_field_names:
            print_field_names = self.field_names.copy()
        self.print_field_names = print_field_names

        if print_column_fmt is None:
            print_column_fmt = []
        if not print_column_fmt:
            for name in self.print_field_names:
                ft = self.field_types[self.getFieldIndex(name)]
                print_column_fmt.append(self.print_type_column_fmt[ft])
        self.print_column_fmt = print_column_fmt

    def print(self, front_str=None, is_to_file=False, end="\n", line_idx=-1):
        if front_str is not None:
            print(front_str, end="")
        lines = self.field_datas[line_idx]
        self._printLine(lines, end=end)
        if is_to_file:
            with open(self.log_filename, "a", encoding="utf-8") as fw:
                self._printLine(lines, end=end, file=fw)

    def _printLine(self, lines, end="\n", file=None):
        for i, name in enumerate(self.print_field_names):
            d = self.print_column_fmt[i].format(lines[self.getFieldIndex(name)])
            d = d.strip()
            fmt_d = self.print_type_fmts[self.print_type].format(d, name)
            print(fmt_d, end=self.print_sep, file=file)
        if self.print_sep != "\n":
            print(end=end, file=file)

    def updateField(self, field_name, field_data, idx_field_data=-1, newline=False):
        if len(self.field_datas) == 0:
            newline = True
        if newline:
            self.field_datas.append(self._initFieldDataLine())
        self.field_datas[idx_field_data][self.getFieldIndex(field_name)] = field_data

    def newLine(self):
        self.field_datas.append(self._initFieldDataLine())

    def _initFieldDataLine(self):
        line = []
        for i in range(len(self.field_names)):
            line.append(self.print_type_init_v[self.field_types[i]])
        return line

    def printFirstLine(self, end="\n", is_to_file=False):
        for name in self.print_field_names:
            print(name, end=self.print_sep)
        print(end=end)
        if is_to_file:
            with open(self.log_filename, "a", encoding="utf-8") as fw:
                for name in self.print_field_names:
                    print(name, end=self.print_sep, file=fw)
                print(end=end, file=fw)

    def saveLine(self, n_line=-1):
        if self.save_csv_file is not None:
            with open(self.save_csv_file, "a", encoding="utf-8", newline="") as fw:
                cw = csv.writer(fw)
                cw.writerow(self.field_datas[n_line])

    def saveHeader(self):
        if self.save_csv_file is not None:
            with open(self.save_csv_file, "w", encoding="utf-8", newline="") as fw:
                cw = csv.writer(fw)
                cw.writerow(self.field_names)

    def __getitem__(self, field_name):
        field_name_idx = self.getFieldIndex(field_name)
        return self.field_datas[-1][field_name_idx]

    def __setitem__(self, field_name, value):
        field_name_idx = self.getFieldIndex(field_name)
        self.field_datas[-1][field_name_idx] = value


class Training:

    def __init__(self, model_dir, model_name):
        if model_dir is None:
            return

        self.model_dir = model_dir
        self.model = None
        self.models = []
        self.model_name = model_name

        if self.model_dir is None:
            raise Exception("Model directory is None.")
        self.model_dir = os.path.abspath(self.model_dir)
        if not os.path.isdir(self.model_dir):
            raise Exception("Can not find model directory " + self.model_dir)

        self._log: TrainLog = None

    def toDict(self):
        to_dict = {
            "model_dir": self.model_dir,
            "model": str(self.model),
            "models": [str(model) for model in self.models],
            "model_name": self.model_name,
            "_log": self._log.toDict(),
        }
        return to_dict

    def _initLog(self):
        self._log = TrainLog(log_filename=os.path.join(self.model_dir, "train_log.txt"))
        self._log.addField("ModelName", "string")

    def addModel(self, model):
        self.model = model
        self.models.append(model)

    def train(self, *args, **kwargs):
        return self.model.train()

    def saveModel(self, model_name, *args, **kwargs):
        return self.model

    def timeModelDir(self):
        dir_name = time.strftime("%Y%m%dH%H%M%S")
        self.model_dir = os.path.join(self.model_dir, dir_name)
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
        self._log.save_csv_file = os.path.join(self.model_dir, "train_save_" + dir_name + ".csv")
        return dir_name

    def testAccuracy(self):
        return 0

    def print(self):
        print("Model", self.model)


def pytorchModelCodeString(model):
    model_str = ""
    if model is not None:
        try:
            import inspect
            model_str = inspect.getsource(model.__class__)
        except Exception as ex:
            model_str = str(ex)
    return model_str


def dataLoaderString(data_loader):
    model_str = {}
    if data_loader is not None:
        model_str = {"len": len(data_loader)}
    return model_str


class PytorchTraining(Training):

    def __init__(self, epochs=10, device=None, n_test=100):
        Training.__init__(self, None, None)
        self.epochs = epochs
        self.device = device
        self.n_test = n_test

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.train_loader = None
        self.test_loader = None
        self.optimizer = None
        self.criterion = None
        self.loss = None
        self.scheduler = None

        if self._log is None:
            return

        self._log.addField("Epoch", "int")
        self._log.addField("Batch", "int")
        self._log.addField("Loss", "float")
        self._log.printOptions(print_float_decimal=3)

        return

    def toDict(self):
        to_dict_1 = super(PytorchTraining, self).toDict()
        to_dict_1["model"] = pytorchModelCodeString(self.model)
        to_dict_1["models"] = [pytorchModelCodeString(model) for model in self.models]

        to_dict = {
            **to_dict_1,
            "epochs": self.epochs,
            "device": self.device,
            "n_test": self.n_test,
            "train_loader": dataLoaderString(self.train_loader),
            "test_loader": dataLoaderString(self.test_loader),
            "optimizer": str(self.optimizer),
            "criterion": pytorchModelCodeString(self.criterion),
            "scheduler": str(self.scheduler),
        }

        return to_dict

    def saveModelCodeFile(self, model_code_file):
        shutil.copyfile(model_code_file, os.path.join(self.model_dir, os.path.split(model_code_file)[1]))

    def trainLoader(self, ds: Dataset, batch_size=128, shuffle=True, sampler=None, batch_sampler=None, num_workers=0):
        self.train_loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                                       batch_sampler=batch_sampler, num_workers=num_workers)

    def testLoader(self, ds: Dataset, batch_size=128, shuffle=True, sampler=None, batch_sampler=None, num_workers=0):
        self.test_loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                                      batch_sampler=batch_sampler, num_workers=num_workers)

    def addCriterion(self, criterion):
        self.criterion = criterion

    def addOptimizer(self, optim_func="adam", lr=0.001, eps=0.00001, optimizer=None, ):
        if optim_func == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=eps)
        else:
            self.optimizer = optimizer

    def addScheduler(self, scheduler):
        self.scheduler = scheduler

    def _initTrain(self):
        if self.model is None:
            raise Exception("Model can not find.")
        if self.criterion is None:
            raise Exception("Criterion can not find.")
        if self.optimizer is None:
            raise Exception("Optimizer can not find.")
        self.model.to(self.device)
        self.criterion.to(self.device)

    def _printModel(self):
        print("model:\n", self.model)
        print("criterion:\n", self.criterion)
        print("optimizer:\n", self.optimizer)

    def lossDeal(self, loss=None):
        if loss is None:
            loss = self.loss
        # L1_reg = 0  # 为Loss添加L1正则化项
        # for param in model.parameters():
        #     L1_reg += torch.sum(torch.abs(param))
        # loss += 0.001 * L1_reg  # lambda=0.001
        return loss

    def logBefore(self, batch, epoch):
        self._log.updateField("Epoch", epoch + 1)
        self._log.updateField("Batch", batch + 1)
        self._log.updateField("Loss", self.loss.item())
        # model_name = "{0}_{1}.pth".format(self.model_name, str(len(self._log.field_datas)))
        model_name = "{0}_Epoch{1}_Batch{2}.pth".format(self.model_name, batch, epoch)
        self._log.updateField("ModelName", model_name)
        return model_name

    def saveModel(self, model_name, *args, **kwargs):
        mod_fn = os.path.join(self.model_dir, model_name)
        torch.save(self.model.state_dict(), mod_fn)
        return mod_fn

    def train(self, batch_save=False, epoch_save=True, *args, **kwargs):
        self.initTrain()

        # for epoch in range(self.epochs):
        #     self.model.train()
        #     for batchix, (x, y) in enumerate(self.train_loader):
        #         x, y = x.to(self.device), y.to(self.device)
        #         x, y = x.float(), y.float()
        #         logts = self.model(x)  # 模型训练
        #         self.loss = self.criterion(logts, y)  # 损失函数
        #         self.loss = self.lossDeal(self.loss)  # loss处理
        #         self.optimizer.zero_grad()  # 梯度清零
        #         self.loss.backward()  # 反向传播
        #         self.optimizer.step()  # 优化迭代
        #
        #         # 测试 ------------------------------------------------------------------
        #         if self.test_loader is not None:
        #             if batchix % self.n_test == 0:
        #                 self.testAccuracy()
        #                 modname = self.log(batchix, epoch)
        #                 if batch_save:
        #                     self.saveModel(modname)
        #
        #     print("-" * 73)
        #     self.testAccuracy()
        #     modname = self.log(-1, epoch)
        #     modname = self.model_name + "_epoch_{0}.pth".format(epoch)
        #     print("*" * 73)
        #
        #     if epoch_save:
        #         self.saveModel(modname)

        for epoch in range(self.epochs):

            for batchix, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                x, y = x.float(), y.long()

                self.model.train()

                logts = self.model(x)
                self.loss = self.criterion(logts, y)
                self.loss.backward()
                self.optimizer.zero_grad()
                self.optimizer.step()

                self.batchTAcc(batch_save, batchix, epoch)

            self.epochTAcc(epoch, epoch_save)

            if self.scheduler is not None:
                self.scheduler.step()

    def initTrain(self):
        self._initTrain()
        self._printModel()
        self._log.saveHeader()

    def log(self, batch, epoch):
        model_name = self.logBefore(batch, epoch)
        return model_name

    def batchTAcc(self, batch_save, batchix, epoch):
        if self.test_loader is not None:
            if batchix % self.n_test == 0:
                self.testAccuracy()
                modname = self.log(batchix, epoch)
                if batch_save:
                    self.saveModel(modname)

    def epochTAcc(self, epoch, epoch_save):
        print("-" * 80)
        self.testAccuracy()
        self.log(-1, epoch)
        modname = self.model_name + "_epoch_{0}.pth".format(epoch)
        if epoch_save:
            mod_fn = self.saveModel(modname)
            print("MODEL:", mod_fn)
        print("*" * 80)


class SRTCollectionDict:

    def __init__(self):
        self._n_iter = 0
        self.n_next = {}
        self._keys = None

    def __len__(self):
        return len(self.n_next)

    def __iter__(self):
        return self

    def __next__(self):
        if self._n_iter == len(self.n_next):
            self._n_iter = 0
            raise StopIteration()
        if self._n_iter == 0:
            self._keys = iter(self.n_next.keys())
        self._n_iter += 1
        return next(self._keys)

    def __contains__(self, item):
        return item in self.n_next

    def __getitem__(self, item):
        return self.n_next[item]


class ConfusionMatrixCollection(SRTCollectionDict):

    def __init__(self, n_class=0, class_names=None):
        super().__init__()
        self.n_class = n_class
        self.class_names = class_names

    def toDict(self):
        to_dict = {
            "cms": {cm: self.n_next[cm].toDict() for cm in self.n_next}
        }
        return to_dict

    def addCM(self, name, n_class=0, class_names=None, cm: ConfusionMatrix = None) -> ConfusionMatrix:
        if cm is not None:
            self.n_next[name] = cm
            return self.n_next[name]
        if (n_class == 0) and (class_names is None):
            n_class = self.n_class
            class_names = self.class_names
        self.n_next[name] = ConfusionMatrix(n_class=n_class, class_names=class_names)
        return self.n_next[name]

    def __getitem__(self, item) -> ConfusionMatrix:
        return self.n_next[item]


class ConfusionMatrixLog:

    def __init__(self, n_category=2, category_names=None):
        self.cms = ConfusionMatrixCollection(n_class=n_category, class_names=category_names)
        self.log = None

    def toDict(self):
        to_dict = {
            "cms": self.cms.toDict(),
            "log": self.log.toDict() if self.log is not None else None,
        }
        return to_dict

    def addCM(self, name, cm=None, ):
        self.cms.addCM(name, cm=cm)

    def initLog(self, log_type: str, cm: ConfusionMatrix = None, log: TrainLog = None, ):
        cm, log = self.initlogcm(cm, log, log_type)
        log.addField("OA{}".format(log_type), "float")
        log.addField("Kappa{}".format(log_type), "float")
        for name in cm:
            log.addField(name + " UA{}".format(log_type), "float")
            log.addField(name + " PA{}".format(log_type), "float")

    def initlogcm(self, cm, log, log_type):
        if log is None:
            log = self.log
        else:
            self.log = log
        if cm is None:
            cm = self.cms[log_type]
        return cm, log

    def initThisLogs(self, log: TrainLog = None, ):
        for name in self.cms:
            self.initLog(name, cm=self.cms[name], log=log)

    def updateLog(self, log_type: str, cm: ConfusionMatrix = None, log: TrainLog = None, ):
        cm, log = self.initlogcm(cm, log, log_type)
        log.updateField("OA{}".format(log_type), cm.OA())
        log.updateField("Kappa{}".format(log_type), cm.getKappa())
        for name in cm:
            log.updateField(name + " UA{}".format(log_type), cm.UA(name))
            log.updateField(name + " PA{}".format(log_type), cm.PA(name))


def funcPredict(model, x: torch.Tensor):
    logit = model(x)
    y = torch.argmax(logit, dim=1)
    return y


def funcCodeString(func):
    if func is None:
        return ""
    try:
        return inspect.getsource(func)
    except Exception as ex:
        return str(ex)


class MI_PytorchTraining(PytorchTraining):

    def __init__(self, model_dir=None, model_name="PytorchModel", epochs=10, device=None, n_test=100):
        self.model_dir = model_dir
        self.model_name = model_name
        self.model = None
        self.models = []
        self._log = None
        super().__init__(epochs, device, n_test)
        self.cm_log = ConfusionMatrixLog()

        self.func_logit_category = funcPredict
        self.func_loss_deal = lambda loss: loss
        self.func_xy_deal = lambda x, y: (x.float(), y.long())
        self.func_y_deal = lambda y: y

        self.func_batch = None
        self.epoch_batch = None

    def toDict(self):
        to_dict_1 = super(MI_PytorchTraining, self).toDict()
        to_dict = {
            **to_dict_1,
            "cm_log": self.cm_log.toDict(),
            "func_logit_category": funcCodeString(self.func_logit_category),
            "func_loss_deal": funcCodeString(self.func_loss_deal),
            "func_xy_deal": funcCodeString(self.func_xy_deal),
            "func_y_deal": funcCodeString(self.func_y_deal),
        }
        return to_dict

    def initLog(self, log: TrainLog):
        self.cm_log.log = log
        self._log = log
        self._log.addField("ModelName", "string")
        self._log.addField("Epoch", "int")
        self._log.addField("Batch", "int")
        self._log.addField("Loss", "float")
        self._log.printOptions(print_float_decimal=3)

    def train(self, batch_save=False, epoch_save=True, *args, **kwargs):
        self.initTrain()

        for epoch in range(self.epochs):

            if self.epoch_batch is not None:
                self.epoch_batch()

            for batchix, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                x, y = self.func_xy_deal(x, y)

                if self.func_batch is not None:
                    self.func_batch()

                self.model.train()
                logts = self.model(x)
                self.loss = self.criterion(logts, y)
                self.loss = self.func_loss_deal(self.loss)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                self.batchTAcc(batch_save, batchix, epoch)

            self.epochTAcc(epoch, epoch_save)
            if self.scheduler is not None:
                self.scheduler.step()

    def log(self, batch, epoch):
        self._log.updateField("Epoch", epoch + 1)
        self._log.updateField("Batch", batch + 1)
        self._log.updateField("Loss", self.loss.item())
        if batch == -1:
            model_name = "{0}_epoch{1}.pth".format(self.model_name, epoch)
        else:
            model_name = "{0}_epoch{1}_batch{2}.pth".format(self.model_name, epoch, batch)
        self._log.updateField("ModelName", model_name)

        if self.test_loader is not None:
            self.cm_log.updateLog("Test")

        self._log.saveLine()
        self._log.print(is_to_file=True)
        self._log.newLine()

        return model_name

    def tlogsave(self, is_save, batchix, epoch, is_print=False):
        if self.test_loader is not None:
            self.testAccuracy()
        modname = self.log(batchix, epoch)
        if is_save:
            mod_fn = self.saveModel(modname)
            if is_print:
                print("MODEL:", mod_fn)

    def batchTAcc(self, batch_save, batchix, epoch):
        if batchix % self.n_test == 0:
            self.tlogsave(batch_save, batchix, epoch, False)

    def epochTAcc(self, epoch, epoch_save):
        print("-" * 80)
        self.tlogsave(epoch_save, -1, epoch, True)
        print("*" * 80)

    def testAccuracy(self):
        self.cm_log.cms["Test"].clear()
        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                x = x.to(self.device).float()
                y = y.numpy()
                y = self.func_y_deal(y)
                y1 = self.func_logit_category(self.model, x)
                y1 = y1.cpu().numpy()
                self.cm_log.cms["Test"].addData(y, y1)
        self.model.train()
        self.cm_log.updateLog("Test")
        return self.cm_log.cms["Test"].OA()


def timeDirName(dirname=None, is_mk=False):
    current_time = datetime.now()
    save_dirname = current_time.strftime("%Y%m%dH%H%M%S")
    if dirname is not None:
        save_dirname = os.path.join(dirname, save_dirname)
    if is_mk:
        if not os.path.isdir(save_dirname):
            os.mkdir(save_dirname)
    return save_dirname


class FN:

    def __init__(self, filename=""):
        self._filename = filename

    def fn(self):
        return self._filename

    def changext(self, ext):
        return changext(self._filename, ext)

    def changedirname(self, dirname):
        return changefiledirname(self._filename, dirname)

    def basename(self):
        return os.path.basename(self._filename)

    def dirname(self):
        return os.path.dirname(self._filename)

    def exists(self):
        return os.path.exists(self._filename)

    def getatime(self):
        return os.path.getatime(self._filename)

    def isfile(self):
        return os.path.isfile(self._filename)

    def realpath(self):
        return os.path.realpath(self._filename)

    def getfilenamewithoutext(self):
        return getfilenamewithoutext(self._filename)


def saveGTIFFImdc(gr: GDALRaster, data, to_fn, color_table=None, description="Category"):
    if color_table is None:
        color_table = {}
    gr.save(data.astype("int8"), to_fn, fmt="GTiff", dtype=gdal.GDT_Byte, descriptions=[description],
            options=["COMPRESS=PACKBITS"])
    tiffAddColorTable(to_fn, code_colors=color_table)


def listfilename(fn_list):
    return [os.path.split(fn)[1] for fn in fn_list]


def filterFileExt(dirname=".", ext="", is_full=True):
    filelist = []
    for f in os.listdir(dirname):
        if os.path.splitext(f)[1] == ext:
            filelist.append(os.path.join(dirname, f))

    if not is_full:
        return listfilename(filelist)
    return filelist


class SRTModImPytorch(SRTModelImageInit):

    def __init__(
            self,
            model_dir=None,
            model_name="PytorchModel",
            epochs=100,
            device="cuda",
            n_test=100,
            batch_size=32,
            n_class=2,
            class_names=None,
            win_size=(),
    ):
        super().__init__()

        self.n = None
        self.model_dirname = model_dir
        self.model_name = model_name
        self.epochs = epochs
        self.device = device
        self.n_test = n_test
        self.batch_size = batch_size
        self.n_class = n_class
        self.class_names = class_names
        self.win_size = win_size

        self.pt = MI_PytorchTraining()
        self.log = TrainLog()

        self.train_ds = None
        self.test_ds = None

        self.model = None

        self.func_predict = funcPredict
        self.func_y_deal = lambda y: y + 1

    def print(self):
        print(
            "model_dirname", self.model_dirname,
            "\nmodel_name", self.model_name,
            "\nepochs", self.epochs,
            "\ndevice", self.device,
            "\nn_test", self.n_test,
            "\nbatch_size", self.batch_size,
            "\nn_class", self.n_class,
            "\nclass_names", self.class_names,
            "\nwin_size", self.win_size,
        )
        if self.train_ds is not None:
            print("length of train_ds:", len(self.train_ds))
        if self.test_ds is not None:
            print("length of test_ds:", len(self.test_ds))
        self.test_ds = None

    def toDict(self):
        to_dict = {
            "model_name": self.model_name,
            "model_dirname": self.model_dirname,
            "category_names": self.category_names,
            "category_colors": self.category_colors,
            "ext_funcs": str(self.ext_funcs),
            "color_table": self.color_table,
            "train_cm": self.train_cm.toDict(),
            "test_cm": self.test_cm.toDict(),
            "epochs": self.epochs,
            "device": self.device,
            "n_test": self.n_test,
            "batch_size": self.batch_size,
            "n_class": self.n_class,
            "class_names": self.class_names,
            "win_size": self.win_size,
            "pt": self.pt.toDict(),
            "log": self.log.toDict(),
            "train_ds": pytorchModelCodeString(self.train_ds),
            "test_ds": pytorchModelCodeString(self.test_ds),
            "model": pytorchModelCodeString(self.model),
            "mod_fn": self.mod_fn,
            "func_predict": funcCodeString(self.func_predict),
            "func_y_deal": funcCodeString(self.func_y_deal),
        }
        return to_dict

    def initTrainLog(self):
        self.log.log_filename = os.path.join(self.model_dirname, "{0}_log.txt".format(self.model_name))
        self.log.save_csv_file = os.path.join(self.model_dirname, "{0}_log.csv".format(self.model_name))

    def initPytorchTraining(self):
        self.pt.__init__(
            model_dir=self.model_dirname,
            model_name=self.model_name,
            epochs=self.epochs,
            device=self.device,
            n_test=self.n_test,
        )
        self.pt.initLog(self.log)
        self.pt.func_logit_category = self.func_predict

    def initDataLoader(self, train_ds=None, test_ds=None):
        if train_ds is None:
            train_ds = self.train_ds
        if test_ds is None:
            test_ds = self.test_ds
        self.pt.trainLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.pt.testLoader(test_ds, batch_size=self.batch_size, shuffle=False)

    def initModel(self, model=None):
        if model is None:
            model = self.model
        else:
            self.model = model
        model.to(self.device)
        self.pt.addModel(model)

    def initCriterion(self, criterion):
        self.pt.addCriterion(criterion)

    def initOptimizer(self, optimizer_class, *args, **kwargs):
        optimizer = optimizer_class(self.pt.model.parameters(), *args, **kwargs)
        self.pt.addOptimizer(optimizer=optimizer)

    def timeDirName(self, save_dirname=None):
        if save_dirname is not None:
            if self.model_dirname is not None:
                save_dirname = os.path.join(self.model_dirname, save_dirname)
            if not os.path.isdir(save_dirname):
                os.mkdir(save_dirname)
            self.model_dirname = save_dirname
        else:
            self.model_dirname = timeDirName(self.model_dirname, is_mk=True)

    def toCSVFN(self):
        return os.path.join(self.model_dirname, "train_data.csv")

    def copyFile(self, fn):
        to_fn = FN(fn).changedirname(self.model_dirname)
        shutil.copyfile(fn, to_fn)

    def train(self, *args, **kwargs):
        self.test_cm = ConfusionMatrix(n_class=self.n_class, class_names=self.class_names)
        self.pt.cm_log.addCM("Test", self.test_cm)
        self.pt.cm_log.initLog("Test")
        self.pt.cm_log.log.printOptions(print_type="keyword", print_field_names=["Epoch", "Batch", "Loss", "OATest"])
        smip_json_fn = os.path.join(self.model_dirname, "smip.json")
        print("smip_json_fn:", smip_json_fn)
        saveJson(self.toDict(), smip_json_fn, )
        self.pt.train()

    def imdc(self, to_geo_fn=None, geo_fns=None, grc: GDALRasterChannel = None, is_jdt=True, data_deal=None,
             is_print=True, data=None, gr=None, description="Category"):

        to_geo_fn = self.getToGeoFn(to_geo_fn)

        if is_print:
            print("to_geo_fn:", to_geo_fn)
        grc = self.initImdc1(geo_fns, grc)
        data = grc.fieldNamesToData(*list(grc.data.keys()))
        gr = list(grc.GRS.values())[0]

        imdc = self.imdcData(data, data_deal, is_jdt)
        self.saveImdc(to_geo_fn, gr, imdc, description)

    def saveImdc(self, to_geo_fn, gr, imdc, description):
        saveGTIFFImdc(gr, imdc, to_geo_fn, color_table=self.color_table, description=description)

    def imdcData(self, data, data_deal=None, is_jdt=True):
        if data_deal is not None:
            data = data_deal(data)
        self.model.eval()

        def func_predict(x):
            with torch.no_grad():
                x = torch.from_numpy(x).float().to(self.device)
                y = self.func_predict(self.model, x)
            y = y.cpu().numpy()
            return y

        imdc = dataPredictPatch(data, self.win_size, predict_func=func_predict, is_jdt=is_jdt)
        return imdc

    def imdcGDALFile(self, fn, to_fn=None, data_deal=None, is_jdt=True, description="CATEGORY", is_print=False):
        to_fn = self.getToGeoFn(to_fn)
        if is_print:
            print("to_fn", to_fn)
            print("read data ... ", end="")
        gr = GDALRaster(fn)
        data = gr.readAsArray()
        if is_print:
            print("end")
        imdc = self.imdcData(data, data_deal=data_deal, is_jdt=is_jdt)
        self.saveImdc(to_fn, gr, imdc, description=description)
        return to_fn

    def imdcTiles(self, to_fn=None, tiles_dirname=None, tiles_fns=None, data_deal=None, is_jdt=True,
                  description="CATEGORY"):

        to_fn = self.getToGeoFn(to_fn)
        if tiles_fns is None:
            tiles_fns = []
        tiles_fns = list(tiles_fns)
        if tiles_dirname is not None:
            tiles_fns.extend(filterFileExt(tiles_dirname, ext=".tif"))
        to_tiles_dirname = os.path.splitext(to_fn)[0] + "_imdctiles"
        if not os.path.isdir(to_tiles_dirname):
            os.mkdir(to_tiles_dirname)

        to_fn_tmps = self.imdcGDALFiles(
            to_dirname=to_tiles_dirname, fns=tiles_fns,
            data_deal=data_deal, description=description, is_jdt=is_jdt
        )

        print("Merge:", to_fn)
        gdal_merge_main(["gdal_merge_main",
                         "-of", "GTiff",
                         "-n", "0",
                         "-ot", "Byte",
                         "-co", "COMPRESS=PACKBITS",
                         "-o", to_fn,
                         *to_fn_tmps, ])
        tiffAddColorTable(to_fn, code_colors=self.color_table)

        return to_fn

    def imdcGDALFiles(self, to_dirname, fns, data_deal=None, is_jdt=True, description="CATEGORY"):
        to_fn_tmps = []
        for fn in fns:
            to_fn_tmp = changext(fn, "_imdc.tif")
            to_fn_tmp = changefiledirname(to_fn_tmp, to_dirname)
            to_fn_tmps.append(to_fn_tmp)
            print("Image:", fn)
            print("Imdc :", to_fn_tmp)
            if os.path.isfile(to_fn_tmp):
                print("Imdc 100%")
                continue
            self.imdcGDALFile(fn, to_fn_tmp, data_deal=data_deal, is_jdt=is_jdt, description=description)
        return to_fn_tmps

    def predict(self, x, *args, **kwargs):
        return self.func_predict(self.model, x)

    def loadPTH(self, mod_fn):
        if mod_fn is None:
            mod_fn = sys.argv[1]
        data = torch.load(mod_fn)
        self.model.load_state_dict(data)
        self.mod_fn = mod_fn


def is_in_poly(p, poly):
    px, py = p
    is_in = False
    for i, corner in enumerate(poly):
        next_i = i + 1 if i + 1 < len(poly) else 0
        x1, y1 = corner[0], corner[1],
        x2, y2 = poly[next_i][0], poly[next_i][1]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in


def getRandom(x0, x1):
    return x0 + random.random() * (x1 - x0)


class GeoJsonPolygonCoor:

    def __init__(self, geo_json_fn):
        self.data = readJson(geo_json_fn)
        self.type = self.data["type"]
        self.name = self.data["name"]
        self.crs = self.data["crs"]
        self.features = self.data["features"]
        self.feature = self.features[0] if len(self.features) > 0 else None
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.getRange()

    def coor(self, x, y, field_names=None, **kwargs):
        to_dict = self.filterFeature(x, y, field_names=field_names, **kwargs)
        if to_dict is None:
            for feat in self.features:
                to_dict = self.filterFeature(x, y, feat=feat, field_names=field_names, **kwargs)
                if to_dict is not None:
                    return to_dict
        return None

    def filterFeature(self, x, y, feat=None, field_names=None, **kwargs):
        if feat is None:
            feat = self.feature
        else:
            self.feature = feat
        if not is_in_poly((x, y), feat["geometry"]["coordinates"][0]):
            return None
        if field_names is None:
            field_names = {}
        return {"X": x, "Y": y, **feat["properties"], **field_names, **kwargs}

    def coors(self, x, y, column_dicts=None, field_names=None, **kwargs):
        to_list = []
        if column_dicts is None:
            column_dicts = {}
        if field_names is None:
            field_names = {}
        for i in range(len(x)):
            to_dict = self.coor(x[i], y[i])
            if to_dict is None:
                continue
            line = {**to_dict, **field_names, **{k: column_dicts[k][i] for k in column_dicts}, **kwargs}
            to_list.append(line)
        return to_list

    def coorDF(self, df, column_dicts=None, field_names=None, x_field_name="X", y_field_name="Y", **kwargs):
        df_keys = list(df.keys())
        df_keys.remove(x_field_name)
        df_keys.remove(y_field_name)
        if column_dicts is None:
            column_dicts = {}
        df_columns = df[df_keys].to_dict("list")
        column_dicts = {**df_columns, **column_dicts, }
        to_list = self.coors(df[x_field_name], df[y_field_name],
                             column_dicts=column_dicts, field_names=field_names, **kwargs)
        return to_list

    def random(self, n_samples, field_names=None, **kwargs):
        n_spl = 0
        to_list = []
        jdt = Jdt(n_samples, "GeoJsonPolygonCoor::random").start()
        while n_spl <= n_samples:
            x, y = self.randomXY()
            to_dict = self.coor(x, y, field_names=field_names, **kwargs)
            if to_dict is not None:
                to_list.append(to_dict)
                n_spl += 1
                jdt.add()
        jdt.end()
        return to_list

    def randomXY(self, n_samples=1):

        if n_samples == 1:
            return getRandom(self.x_min, self.x_max), getRandom(self.y_min, self.y_max)
        return [getRandom(self.x_min, self.x_max) for i in range(n_samples)], \
               [getRandom(self.y_min, self.y_max) for i in range(n_samples)]

    def getRange(self):
        x_min, x_max, y_min, y_max = None, None, None, None
        for feat in self.features:
            for coor in feat["geometry"]["coordinates"][0]:
                x, y = coor[0], coor[1]
                if x_min is None:
                    x_min, x_max = x, y
                    y_min, y_max = x, y
                if x_min > x:
                    x_min = x
                if x_max < x:
                    x_max = x
                if y_min > y:
                    y_min = y
                if y_max < y:
                    y_max = y
        self.x_min, self.x_max, self.y_min, self.y_max = x_min, x_max, y_min, y_max
        return x_min, x_max, y_min, y_max


def geojsonAddField(geojson_fn, to_fn, func, name=None):
    json_dict = readJson(geojson_fn)
    if name is not None:
        json_dict["name"] = name
    for feat in json_dict["features"]:
        func(feat)
    saveJson(json_dict, to_fn)


def main():
    print("")
    pass


if __name__ == "__main__":
    main()
