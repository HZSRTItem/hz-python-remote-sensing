# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : GDALRasterIO.py
@Time    : 2023/6/22 14:20
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PytorchGeo of GDALRasterIO
-----------------------------------------------------------------------------"""
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from osgeo import __version__
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr

from SRTCodes.NumpyUtils import scaleMinMax
from SRTCodes.RasterIO import GEORaster
from SRTCodes.SRTCollection import SRTCollection
from SRTCodes.SRTFeature import SRTFeatures
from SRTCodes.Utils import readcsv, savecsv, Jdt, changext, readJson, saveJson


def _cheng(n1, n2, arr):
    n11 = arr[0] + n1 * arr[1] + n2 * arr[2]
    n21 = arr[3] + n1 * arr[4] + n2 * arr[5]
    return n11, n21


def gdalTypeStr2GDAL(type_str):
    type_gdal = gdal.GDT_Unknown
    if type_str == "int8":
        type_gdal = gdal.GDT_Unknown


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


def getGDALRasterNames(raster_fn):
    ds: gdal.Dataset = gdal.Open(raster_fn)
    names = []
    for i in range(1, ds.RasterCount + 1):
        b = ds.GetRasterBand(i)
        name = b.GetDescription()
        names.append(name)
    return names


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

    def save(self, d: np.array = None, save_geo_raster_fn=None, fmt="ENVI", dtype=None, geo_transform=None,
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


class GDALRasterFeatures(GDALRasterIO, SRTFeatures):

    def __init__(self, gdal_raster_fn=""):
        GDALRasterIO.__init__(self)
        SRTFeatures.__init__(self)

        if gdal_raster_fn != "":
            self.initGDALRaster(gdal_raster_fn)

    def initGDALRaster(self, gdal_raster_fn):
        self.initGDALRasterIO(gdal_raster_fn)
        for i, name in enumerate(self.names):
            self.addFeature(name)

    def save(self, d: np.array, save_geo_raster_fn=None, fmt="ENVI", dtype=None, geo_transform=None,
             probing=None, start_xy=None, interleave='band', options=None, descriptions=None):
        if d is None:
            d = np.concatenate(list(self.features.values()))
            descriptions = list(self.features.keys())
        self._save(d=d, save_geo_raster_fn=save_geo_raster_fn, fmt=fmt, dtype=dtype, geo_transform=geo_transform,
                   probing=probing, start_xy=start_xy, interleave=interleave, options=options,
                   descriptions=descriptions)

    def saveFeatureToGDALRaster(self, feat_name, save_geo_raster_fn=None, save_geo_raster_dir=None, ext=".dat",
                                fmt="ENVI", dtype=None, geo_transform=None, probing=None,
                                start_xy=None, interleave='band', options=None):
        d = self.features[feat_name]
        if save_geo_raster_fn is None:
            save_geo_raster_fn = os.path.join(save_geo_raster_dir, feat_name)
        fns = os.path.splitext(save_geo_raster_fn)
        save_geo_raster_fn = fns[0] + ext
        self._save(d=d, save_geo_raster_fn=save_geo_raster_fn, fmt=fmt, dtype=dtype, geo_transform=geo_transform,
                   probing=probing, start_xy=start_xy, interleave=interleave, options=options,
                   descriptions=[feat_name])

    def __getitem__(self, feat):
        if isinstance(feat, list):
            d = self.getFeaturesData(feat)
        else:
            d = self.getFeatureData(feat)
        return d

    def getFeaturesData(self, feat):
        d = np.zeros([len(feat), self.n_rows, self.n_columns])
        for i, f in enumerate(feat):
            d[i] = self.getFeatureData(f)
        return d

    def getFeatureData(self, feat_name_or_number):
        if isinstance(feat_name_or_number, int):
            feat_name_or_number = self._n_next[feat_name_or_number]
        n_feat = self._n_next.index(feat_name_or_number) + 1
        if self.features[feat_name_or_number] is None:
            self.features[feat_name_or_number] = self.readGDALBand(n_feat)
        d = self.features[feat_name_or_number]
        return d


class GDALRasterCollection(GDALRaster):

    def __init__(self, file_list=None, dirname=None, ext=None):
        super(GDALRasterCollection, self).__init__()

        self.dirname = os.getcwd() if dirname is None else dirname
        self.rds = {}
        self._n_iter = 0
        self._n_next = []

        self.geo_transform = None
        self.inv_geo_transform = None
        self.n_rows = None
        self.n_columns = None
        self.n_channels = None

        self.save_geo_raster_fn = None
        self.save_fmt = "GTiff"
        self.save_dtype = gdal.GDT_Float32
        self.save_geo_transform = None
        self.save_probing = None

        self.adds(file_list, dirname=dirname, ext=ext)

    def _initCollection(self, gr: GDALRaster):
        self.geo_transform = gr.geo_transform
        self.inv_geo_transform = gr.inv_geo_transform
        self.n_rows = gr.n_rows
        self.n_columns = gr.n_columns
        self.n_channels = gr.n_channels
        self.save_geo_transform = gr.save_geo_transform
        self.save_probing = gr.save_probing

    def add(self, filename, dirname=None, ext=None):
        if dirname is None:
            dirname = self.dirname
        filename = os.path.join(dirname, os.path.split(filename)[1])
        if ext is not None:
            filename = os.path.splitext(filename)[0] + ext
        if not os.path.isfile(filename):
            print("Warning: can not find file " + filename)
            return None

        fn = os.path.split(filename)[1]
        fn = os.path.splitext(fn)[0]
        self.rds[fn] = GDALRaster(filename)

        if self.geo_transform is None:
            self._initCollection(self.rds[fn])

        self._n_next.append(fn)
        return fn

    def adds(self, file_list, dirname=None, ext=None):
        for f in file_list:
            self.add(f, dirname, ext)

    def __iter__(self):
        return self

    def __next__(self) -> GDALRaster:
        if self._n_iter == len(self._n_next):
            self._n_iter = 0
            raise StopIteration()
        self._n_iter += 1
        return self._n_next[self._n_iter - 1]

    def __getitem__(self, item) -> GDALRaster:
        return self.rds[item]

    def __len__(self):
        return len(self._n_next)

    def keys(self):
        return self.rds.keys()


class GDALRasterFeatureCollection(SRTCollection):
    """ GDAL Raster Feature Collection """

    GRS = {}

    def __init__(self):
        super().__init__()
        self.data = {}

    def addGDALData(self, raster_fn, feat_name=None, feat_index=None, is_all=False):
        raster_fn = os.path.abspath(raster_fn)
        if raster_fn not in self.GRS:
            self.GRS[raster_fn] = GDALRaster(raster_fn)
        gr = self.GRS[raster_fn]


class GDALRasterDraw(GDALRaster):

    def __init__(self, geo_raster_fn):
        super(GDALRasterDraw, self).__init__(geo_raster_fn)
        self.win_row_size = 0
        self.win_column_size = 0

    def image(self, x_row, y_column, to_fn=None, win_row_size=None, win_column_size=None, band_list=None, is_geo=False,
              no_data=0, is_trans=False):
        if win_row_size is None:
            win_row_size = self.win_row_size
        if win_column_size is None:
            win_column_size = self.win_column_size
        if band_list is None:
            band_list = [1]
        d = self.readAsArrayCenter(x_row, y_column, win_row_size=win_row_size, win_column_size=win_column_size,
                                   band_list=band_list, is_geo=is_geo, no_data=no_data, is_trans=is_trans)
        if to_fn is not None:
            np.save(to_fn, d)
        return d


class GDALRasterWarp(GDALRaster):
    """ GDAL Raster Warp

    # destNameOrDestDS --- 输出数据集路径或对象
    # srcDSOrSrcDSTab --- 数据集对象或文件名or数据集对象或文件名的数组
    # 关键字参数是gdal.WarpOptions()的返回值，或者直接定义gdal.WarpOptions()

    gdal.WarpOptions(options=[], format='GTiff', outputBounds=None,
                     outputBoundsSRS=one, xRes=None, yRes=None,
                     targetAlignedPixels=False, width=0, height=0, srcSRS=None,
                     dstSRS=None, srcAlpha=False, dstAlpha=False, warpOptions=None,
                     errorThreshold=None, warpMemoryLimit=None, creationOptions=None,
                     outputType=GDT_Unknown, workingType=GDT_Unknown, resampleAlg=None,
                     srcNodata=None, dstNodata=None, multithread=False, tps=False,
                     rpc=False, geoloc=False, polynomialOrder=None,
                     transformerOptions=None, cutlineDSName=None, cutlineLayer=None,
                     cutlineWhere=None, cutlineSQL=None, cutlineBlend=None,
                     ropToCutline=False, copyMetadata=True, metadataConflictValue=None,
                     setColorInterpretation=False, callback=None, callback_data=None):
    options --- 字符串数组, 字符串或者空值
    format --- 输出格式 ("GTiff", etc...)
    outputBounds --- 结果在目标空间参考的边界范围(minX, minY, maxX, maxY)
    outputBoundsSRS --- 结果边界范围的空间参考, 如果在dstSRS中没有指定的话，采用此参数
    xRes, yRes --- 输出分辨率，即像素的大小
    targetAlignedPixels --- 是否强制输出边界是输出分辨率的倍数
    width --- 输出栅格的列数
    height --- 输出栅格的行数
    srcSRS --- 输入数据集的空间参考
    dstSRS --- 输出数据集的空间参考
    srcAlpha --- 是否将输入数据集的最后一个波段作为alpha波段
    dstAlpha --- 是否强制创建输出
    outputType --- 输出栅格的变量类型 (gdal.GDT_Byte, etc...)
    workingType --- working type (gdal.GDT_Byte, etc...)
    warpOptions --- list of warping options
    errorThreshold --- 近似转换的误差阈值(误差像素数目)
    warpMemoryLimit --- 工作内存限制 Bytes
    resampleAlg --- 重采样方法
    creationOptions --- list of creation options
    srcNodata --- 输入栅格的无效值
    dstNodata --- 输出栅格的无效值
    multithread --- 是否多线程和I/O操作
    tps --- 是否使用Thin Plate Spline校正方法
    rpc --- 是否使用RPC校正
    geoloc --- 是否使用地理查找表校正
    polynomialOrder --- 几何多项式校正次数
    transformerOptions --- list of transformer options
    cutlineDSName --- cutline数据集名称
    cutlineLayer --- cutline图层名称
    cutlineWhere --- cutline WHERE 子句
    cutlineSQL --- cutline SQL语句
    cutlineBlend --- cutline blend distance in pixels
    cropToCutline --- 是否使用切割线范围作为输出边界
    copyMetadata --- 是否复制元数据
    metadataConflictValue --- 元数据冲突值
    setColorInterpretation --- 是否强制将输入栅格颜色表给输出栅格
    callback --- 回调函数
    callback_data --- 用于回调的用户数据
    """

    RESOLUTION_ANGLE = 0.000089831529294

    def __init__(self):
        super().__init__()

        gdal.GCP()
        self.gcps = []
        self.gcp_srs = osr.SpatialReference()
        self.gcp_srs.ImportFromEPSG(4326)

    def addGCP(self, x=0.0, y=0.0, z=0.0, pixel=0.0, line=0.0, text="", idx=""):
        """
        x、y、z 是控制点对应的投影坐标，默认为0;
        pixel、line 是控制点在图像上的列、行位置，默认为0;
        text、idx 是用于说明控制点的描述和点号的可选字符串，默认为空.
        """
        self.gcps.append(gdal.GCP(x, y, z, pixel, line, text, idx))

    def warp(self, to_fn, geo_fmt='GTiff', xres=None, yres=None, dst_nodata=0, src_nodata=0, dtype="float32"):
        if xres is None:
            xres = abs(self.x_size)
        if yres is None:
            yres = abs(self.y_size)

        self.raster_ds.SetGCPs(self.gcps, self.gcp_srs.ExportToWkt())

        warp_ds = gdal.Warp(to_fn, self.raster_ds, format=geo_fmt, tps=True,
                            xRes=xres, yRes=yres, dstNodata=dst_nodata, srcNodata=src_nodata,
                            resampleAlg=gdal.GRIORA_NearestNeighbour,
                            outputType=GDALRasterIO.NpType2GDALType[dtype])
        return warp_ds

    def addGCPImageGround(self, x_image, y_image, x_ground, y_ground):
        row, column = self.coorGeo2Raster(x_image, y_image)

        self.addGCP(x_ground, y_ground, 0, column, row)


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


_GR_RW: gdal.Dataset  # GDALRaster as read and write


def readGEORaster(geo_fn, x_row_off=0.0, y_column_off=0.0, win_row_size=None, win_column_size=None, interleave='band',
                  band_list=None):
    """ Read geographic raster data as numpy arrays by location
    \
    :param band_list: number
    :param geo_fn: geo raster filename
    :param x_row_off: rows or geographic X coordinate
    :param y_column_off: columns or geographic Y coordinate
    :param win_row_size: The number of columns of the window, data type int
    :param win_column_size: The number of rows of the window, data type int
    :param interleave: The data is organized as `band`:(b,y,x) or `pixel`:(x,y,b)
    :return: A numpy array of size win
    """
    ds: gdal.Dataset = gdal.Open(geo_fn)
    if ds is not None:
        global _GR_RW
        _GR_RW = ds
    names = [ds.GetRasterBand(i + 1).GetDescription() for i in range(ds.RasterCount)]
    if band_list is not None:
        for i, channel in enumerate(band_list):
            if isinstance(channel, str):
                band_list[i] = names.index(channel)
    return gdal_array.DatasetReadAsArray(ds, y_column_off, x_row_off, win_xsize=win_column_size,
                                         win_ysize=win_row_size, interleave=interleave, band_list=band_list)


def saveGEORaster(d, geo_fn=None, copy_geo_fn=None, fmt="ENVI", dtype="float32", geo_transform=None,
                  probing=None, interleave='band', options=None, descriptions=None):
    """ Save geo image
    \
    :param geo_fn: save geo file name
    :param copy_geo_fn: get geo_transform probing in this geo file
    :param descriptions: descriptions
    :param options: save options list
    :param interleave: The data is organized as `band`:(b,y,x) or `pixel`:(x,y,b)
    :param probing: projection information
    :param geo_transform: projection transformation information
    :param d: data
    :param fmt: save type
    :param dtype: save data type default:gdal.GDT_Float32
    :return: None
    """
    ds: gdal.Dataset = _GR_RW
    if copy_geo_fn is not None:
        ds = gdal.Open(copy_geo_fn)
    if options is None:
        options = []
    if isinstance(dtype, str):
        dtype = GDALRasterIO.NpType2GDALType[dtype]
    if geo_transform is None:
        geo_transform = ds.GetGeoTransform()
    if probing is None:
        probing = ds.GetProjection()
    band_count, n_column, n_row = getArraySize(d.shape, interleave)
    saveGDALRaster(d, n_row, n_column, band_count, dtype, fmt, geo_transform, interleave, options, probing,
                   geo_fn, descriptions)


class GRCNR_read:

    def __init__(self, gr: GDALRaster, channel):
        self.gr = gr
        self.channel = channel
        self.data = None

    def fit(self):
        if self.data is None:
            self.data = self.gr.readGDALBand(self.channel)
        return self.data


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


class GDALRasterChannelNotRead(GDALRasterChannel):
    """ GDAL Raster Channel Not Read """

    def __init__(self):
        super().__init__()

    def addGDALData(self, raster_fn, field_name, channel=None):
        gr = self.addGR(raster_fn)
        if channel is None:
            channel = field_name
        self.data[field_name] = GRCNR_read(gr, channel)
        if self.shape == ():
            self.shape = (gr.n_rows, gr.n_columns)
        return field_name

    def addFeatExt(self, field_name, func_ext, *args, **kwargs):
        self.data[field_name] = GRCNR_featExt(self.data, func_ext, *args, **kwargs)
        return field_name

    def __setitem__(self, key, value):
        grcnr_read = GRCNR_read(GDALRaster(), None)
        grcnr_read.data = value
        if self.shape == ():
            self.shape = value.shape
        self.data[key] = value

    def __getitem__(self, item):
        if isinstance(item, list) or isinstance(item, tuple):
            return self.fieldNamesToData(*item)
        else:
            return self.get(item)

    def get(self, key):
        data = self.data[key].fit()
        if self.shape == ():
            self.shape = data.shape
        return data

    def fieldNamesToData(self, *field_names):
        data = np.zeros((len(field_names), self.shape[0], self.shape[1]))
        for i, field_name in enumerate(field_names):
            data[i] = self.get(field_name)
        return data


def samplingGDALRastersToCSV(raster_fns: list, csv_fn, to_csv_fn):
    grs = [GDALRaster(raster_fn) for raster_fn in raster_fns]
    gr = grs[0]
    df = readcsv(csv_fn)
    jdt = Jdt(total=len(df["X"]), desc="Sampling GDAL Rasters to CSV")
    jdt.start()
    to_df = {k: [] for k in df}
    for k in gr.names:
        to_df[k] = []
    for i in range(len(df["X"])):
        x, y = float(df["X"][i]), float(df["Y"][i])
        if not gr.isGeoIn(x, y):
            for gr in grs:
                if gr.isGeoIn(x, y):
                    break
        if gr.isGeoIn(x, y):
            d = gr.readAsArray(x_row_off=x, y_column_off=y, win_row_size=1, win_column_size=1, is_geo=True)
            d = d.ravel()
            for k in df:
                to_df[k].append(df[k][i])
            for j, k in enumerate(gr.names):
                to_df[k].append(d[j])
        jdt.add()
    jdt.end()
    savecsv(to_csv_fn, to_df)


class GDALMBTiles(GDALRaster):

    def __init__(self, mb_tiles_fn=""):
        super().__init__(gdal_raster_fn=mb_tiles_fn)

    def getCenterImage(self, x_center, y_center, win_size=(10, 10), is_to_wgs84=True):
        if is_to_wgs84:
            x_center, y_center = self.coorWGS84ToThis(x_center, y_center)
        d = self.readAsArrayCenter(x_center, y_center, win_row_size=win_size[0], win_column_size=win_size[1],
                                   is_geo=True, interleave="pixel")
        return d


def saveGTIFFImdc(gr: GDALRaster, data, to_fn, color_table=None, description="Category"):
    if color_table is None:
        color_table = {}
    gr.save(data.astype("int8"), to_fn, fmt="GTiff", dtype=gdal.GDT_Byte, descriptions=[description],
            options=["COMPRESS=PACKBITS"])
    tiffAddColorTable(to_fn, code_colors=color_table)


def main():
    # gr = GDALRaster(r"F:\ProjectSet\ASDEShadow_T1\ImageDeal\qd_rgbn_si_asdeC_raw.dat")
    # gr.readAsArray()
    # print(gr.d.shape)
    # print(gr.readAsArrayCenter(0, 0, 5, 5))

    # bj3_vrt_fn = r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3.vrt"
    # grfc = GDALRasterFeatureCollection()
    # grfc.addGDALData(bj3_vrt_fn, "Blue")

    gmbt = GDALMBTiles(r"F:\ProjectSet\Shadow\MkTu\4.1Details\BingImages\qd\qd_googleimage2.mbtiles")
    x, y = 120.37623005, 36.07113323
    d = gmbt.getCenterImage(x, y, (100, 100))
    plt.imshow(d)
    plt.show()
    print(x, y)

    pass


if __name__ == "__main__":
    main()


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
