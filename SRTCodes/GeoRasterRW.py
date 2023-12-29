# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : GeoRasterRW.py
@Time    : 2022/12/18 11:42
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2022, ZhengHan. All rights reserved.
@Desc    : PyGdal of GeoRasterRW_1
-----------------------------------------------------------------------------"""
import os
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
from osgeo import __version__
import numpy as np

np.set_printoptions(linewidth=500)


class GeoRaster:
    """
    geographic raster
    """

    def __init__(self, geo_raster_file=None):
        self.geo_raster_file = geo_raster_file
        self.geo_transform = None
        self.n_rows = None
        self.inv_geo_transform = None
        self.n_columns = None
        self.n_bands = None
        self.src_srs = None
        self.coor_trans = None
        self.dst_srs = None
        self.raster_ds = None
        if os.path.isfile(geo_raster_file):
            self.initFromExistingFile(geo_raster_file)

    def initFromExistingFile(self, geo_raster_file):
        self.geo_raster_file = geo_raster_file
        self.raster_ds: gdal.Dataset = gdal.Open(self.geo_raster_file)
        if self.raster_ds is None:
            raise Exception("Input geo raster file can not open -file:" + self.geo_raster_file)
        self.geo_transform = self.raster_ds.GetGeoTransform()
        self.inv_geo_transform = gdal.InvGeoTransform(self.geo_transform)
        self.n_rows = self.raster_ds.RasterYSize
        self.n_columns = self.raster_ds.RasterXSize
        self.n_bands = self.raster_ds.RasterCount
        self.src_srs = osr.SpatialReference()
        self.src_srs.ImportFromEPSG(4326)
        self.dst_srs = osr.SpatialReference()
        self.dst_srs.ImportFromWkt(self.raster_ds.GetProjection())
        if __version__ >= "3.0.0":
            self.src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            self.dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        self.coor_trans = osr.CoordinateTransformation(self.src_srs, self.dst_srs)

    def setDstSrs(self, dst_srs: osr.SpatialReference = None):
        if dst_srs is not None:
            self.dst_srs = dst_srs
        self.coor_trans = osr.CoordinateTransformation(self.src_srs, self.dst_srs)

    def coorGeoToIm(self, x, y):
        """ Geographical coordinates to image coordinates
        \
        :param x: Geographic North Coordinates / Latitude
        :param y: Geographical East Coordinates / Longitude
        :return: Image coordinates
        """
        column = self.inv_geo_transform[0] + x * self.inv_geo_transform[1] + y * self.inv_geo_transform[2]
        row = self.inv_geo_transform[3] + x * self.inv_geo_transform[4] + y * self.inv_geo_transform[5]
        return row, column

    def coorImToGeo(self, row, column):
        """ image coordinates to Geographical coordinates
        \
        :param row: row
        :param column: column
        :return: Geographical coordinates
        """
        x = self.geo_transform[0] + column * self.geo_transform[1] + row * self.geo_transform[2]
        y = self.geo_transform[3] + column * self.geo_transform[4] + row * self.geo_transform[5]
        return x, y


class GeoRasterWrite(GeoRaster):
    """
    Write a numpy array as geographic raster
    """

    def __init__(self, info_geo_raster_fn):
        super().__init__(info_geo_raster_fn)
        self.save_geo_raster_fn = None
        self.save_fmt = "GTiff"
        self.save_dtype = gdal.GDT_Float32
        self.save_geo_transform = self.geo_transform
        self.save_probing = None if self.dst_srs is None else self.dst_srs.ExportToWkt()

    def save(self, d: np.array, save_geo_raster_fn=None, fmt="ENVI",
             dtype=None, geo_transform=None, probing=None, start_xy=None,
             interleave='band'):
        """ Save geo image
        \
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
        if save_geo_raster_fn is not None:
            self.save_geo_raster_fn = save_geo_raster_fn
        if self.save_geo_raster_fn is None:
            raise Exception("Saved geo raster file name not found")
        if fmt is not None:
            self.save_fmt = fmt
        if dtype is not None:
            self.save_dtype = dtype
        if geo_transform is not None:
            self.save_geo_transform = geo_transform
        if probing is not None:
            self.save_probing = probing
        wd = len(d.shape)
        if not (wd == 2 or wd == 3):
            raise Exception("The data shall be two-dimensional array single-band "
                            "data or three-dimensional multi-band data", d)
        # 波段数量
        band_count = 1
        n_column = 1
        n_row = 1
        if wd == 3:
            if interleave == "band":
                band_count = d.shape[0]
                n_row = d.shape[1]
                n_column = d.shape[2]
            elif interleave == "pixel":
                band_count = d.shape[2]
                n_row = d.shape[1]
                n_column = d.shape[0]
            else:
                raise Exception("The data is organized as `band`:(b,y,x) or `pixel`:(x,y,b) not " + interleave)
        else:
            if interleave == "band":
                n_row = d.shape[0]
                n_column = d.shape[1]
            elif interleave == "pixel":
                n_row = d.shape[1]
                n_column = d.shape[0]
            else:
                raise Exception("The data is organized as `band`:(b,y,x) or `pixel`:(x,y,b) not " + interleave)
        # resultDS.SetGeoTransform([minx, self.scaleX, 0, maxy, 0, self.scaleY])
        if start_xy is not None:
            x0, y0 = start_xy[0], start_xy[1]
            x1, y1 = self.coorImToGeo(n_row + 1, n_column + 1)
            if x0 > x1:
                tmp = x0
                x0 = x1
                x1 = tmp
            if y0 > y1:
                tmp = y0
                y0 = y1
                y1 = tmp
            geo_transform0 = (x0, self.save_geo_transform[1], 0, y1, 0, self.save_geo_transform[5])
            self.save_geo_transform = geo_transform0
        # 申请空间
        driver = gdal.GetDriverByName(fmt)
        # 列数 行数 波段数
        dst_ds = driver.Create(self.save_geo_raster_fn, n_column, n_row, band_count, self.save_dtype)
        # 设置投影信息
        dst_ds.SetGeoTransform(self.save_geo_transform)
        dst_ds.SetProjection(self.save_probing)
        # 保存数据
        if band_count == 1:
            dst_ds.GetRasterBand(1).WriteArray(d)
        else:
            for i in range(band_count):
                if interleave == "band":
                    dst_ds.GetRasterBand(i + 1).WriteArray(d[i, :, :])
                elif interleave == "pixel":
                    dst_ds.GetRasterBand(i + 1).WriteArray(d[:, :, i])
                else:
                    raise Exception("The data is organized as `band`:(b,y,x) or `pixel`:(x,y,b) not " + interleave)
        del dst_ds


class GeoRasterRead(GeoRaster):
    """
    Go read geographic raster as a numpy array
    """

    def __init__(self, geo_raster_file):
        """
        /
        :param geo_raster_file: Geographic raster files
        """
        super().__init__(geo_raster_file)

    def readAsArray(self, x_row_off=0, y_column_off=0, win_row_size=None, win_column_size=None, interleave='band'
                    , band_list=None, is_geo=False, is_trans=False):
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
            x_row_off, y_column_off = self.coorGeoToIm(x_row_off, y_column_off)
        x_row_off, y_column_off = int(x_row_off), int(y_column_off)
        return gdal_array.DatasetReadAsArray(self.raster_ds, y_column_off, x_row_off, win_xsize=win_column_size,
                                             win_ysize=win_row_size,
                                             interleave=interleave)

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
            x_row_center, y_column_center = self.coorGeoToIm(x_row_center, y_column_center)
        x_row_center, y_column_center = int(x_row_center), int(y_column_center)

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
                d = np.ones([self.n_bands, win_row_size, win_column_size]) * no_data
            x0 = column_off - column_off0
            y0 = row_off - row_off0
            d[:, y0:y0 + row_size, x0:x0 + column_size] = d0
        else:
            if band_list is not None:
                d = np.ones([win_row_size, win_column_size, len(band_list)]) * no_data
            else:
                d = np.ones([win_row_size, win_column_size, self.n_bands]) * no_data
            x0 = column_off - column_off0
            y0 = row_off - row_off0
            d[y0:y0 + row_size, x0:x0 + column_size, :] = d0

        return d

    def getRange(self):
        """ x_min, x_max, y_min, y_max"""
        if self.raster_ds is None:
            return None
        x0, y0 = self.coorImToGeo(0, 0)
        x1, y1 = self.coorImToGeo(self.n_rows, self.n_columns)
        self.raster_range = [0, 0, 0, 0]
        if x0 > x1:
            x1, x0 = x0, x1
        if y0 > y1:
            y1, y0 = y0, y1
        return x0, x1, y0, y1


def main():
    # resultDS.SetGeoTransform([minx, self.scaleX, 0, maxy, 0, self.scaleY])
    rds = GeoRasterRead(r"F:\ProjectSet\ASDEShadow_T1\ImageDeal\qd_rgbn_si_asdeC_raw.dat")
    d = rds.readAsArrayCenter(0, 0, 5, 5)
    pass


if __name__ == "__main__":
    main()
