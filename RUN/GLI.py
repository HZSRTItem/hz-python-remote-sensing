# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : GLI.py
@Time    : 2023/2/27 20:45
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyGdal of GLI
-----------------------------------------------------------------------------"""
import csv
import os
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
from osgeo import ogr
from osgeo import __version__
import numpy as np
import argparse

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
            raise Exception(
                "Input geo raster file can not open -file:" + self.geo_raster_file)
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
        self.coor_trans = osr.CoordinateTransformation(
            self.src_srs, self.dst_srs)

    def setDstSrs(self, dst_srs: osr.SpatialReference = None):
        if dst_srs is not None:
            self.dst_srs = dst_srs
        self.coor_trans = osr.CoordinateTransformation(
            self.src_srs, self.dst_srs)

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
        x = self.geo_transform[0] + column * \
            self.geo_transform[1] + row * self.geo_transform[2]
        y = self.geo_transform[3] + column * \
            self.geo_transform[4] + row * self.geo_transform[5]
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
                raise Exception(
                    "The data is organized as `band`:(b,y,x) or `pixel`:(x,y,b) not " + interleave)
        else:
            if interleave == "band":
                n_row = d.shape[0]
                n_column = d.shape[1]
            elif interleave == "pixel":
                n_row = d.shape[1]
                n_column = d.shape[0]
            else:
                raise Exception(
                    "The data is organized as `band`:(b,y,x) or `pixel`:(x,y,b) not " + interleave)
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
            geo_transform0 = (
                x0, self.save_geo_transform[1], 0, y1, 0, self.save_geo_transform[5])
            self.save_geo_transform = geo_transform0
        # 申请空间
        driver = gdal.GetDriverByName(fmt)
        # 列数 行数 波段数
        dst_ds = driver.Create(self.save_geo_raster_fn,
                               n_column, n_row, band_count, self.save_dtype)
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
                    raise Exception(
                        "The data is organized as `band`:(b,y,x) or `pixel`:(x,y,b) not " + interleave)
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

    def readAsArray(self, x_row_off=0.0, y_column_off=0.0, win_row_size=None, win_column_size=None, interleave='band',
                    band_list=None, is_geo=False, is_trans=False):
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
                x_row_off, y_column_off, _ = self.coor_trans.TransformPoint(
                    x_row_off, y_column_off)
            x_row_off, y_column_off = self.coorGeoToIm(x_row_off, y_column_off)
        x_row_off, y_column_off = int(x_row_off), int(y_column_off)
        return gdal_array.DatasetReadAsArray(self.raster_ds, y_column_off, x_row_off, win_xsize=win_column_size,
                                             win_ysize=win_row_size, interleave=interleave, band_list=band_list)

    def readAsArrayCenter(self, x_row_center=0.0, y_column_center=0.0, win_row_size=1, win_column_size=1,
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
                x_row_center, y_column_center, _ = self.coor_trans.TransformPoint(
                    x_row_center, y_column_center)
            x_row_center, y_column_center = self.coorGeoToIm(
                x_row_center, y_column_center)
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
                d = np.ones([len(band_list), win_row_size,
                             win_column_size]) * no_data
            else:
                d = np.ones([self.n_bands, win_row_size,
                             win_column_size]) * no_data
            x0 = column_off - column_off0
            y0 = row_off - row_off0
            d[:, y0:y0 + row_size, x0:x0 + column_size] = d0
        else:
            if band_list is not None:
                d = np.ones([win_row_size, win_column_size,
                             len(band_list)]) * no_data
            else:
                d = np.ones([win_row_size, win_column_size,
                             self.n_bands]) * no_data
            x0 = column_off - column_off0
            y0 = row_off - row_off0
            d[y0:y0 + row_size, x0:x0 + column_size, :] = d0

        return d


class ShapeRead:

    def __init__(self, shape_file_name):
        self.shape_file_name = shape_file_name
        self.shp_ds: ogr.DataSource = ogr.Open(self.shape_file_name)
        self.shp_layer: ogr.Layer = self.shp_ds.GetLayer(0)
        self.n_feature = self.shp_layer.GetFeatureCount()
        self.o_defn = self.shp_layer.GetLayerDefn()
        self.n_field = self.o_defn.GetFieldCount()
        self.field_names = [self.o_defn.GetFieldDefn(i).GetNameRef() for i in range(self.n_field)]

    def getXYs(self):
        xys = []
        for feat in self.shp_layer:
            geom = feat.geometry()
            xys.append([geom.GetX(), geom.GetY()])
        return np.array(xys)

    def getXYFeild(self):
        infos = []
        for feat in self.shp_layer:
            feat: ogr.Feature
            geom = feat.geometry()
            info = [geom.GetX(), geom.GetY()]
            for i in range(self.n_field):
                info.append(feat.GetField(i))
            infos.append(info)
        return infos


def readShape(shp_fn, driver_name='ESRI Shapefile', is_ret_srs=False):
    driver = ogr.GetDriverByName(driver_name)
    ds: ogr.DataSource = driver.Open(shp_fn)
    inlayer: ogr.Layer = ds.GetLayer()
    if inlayer.GetGeomType() != ogr.wkbPoint:
        raise Exception("Input vector data source is not point vector\n")
    layer_def = inlayer.GetLayerDefn()
    layer_def_list = ["X", "Y"]
    i_fields = layer_def.GetFieldCount()
    for i in range(i_fields):
        layer_defi = layer_def.GetFieldDefn(i)
        layer_def_list.append(layer_defi.GetName())
    d = []
    feat: ogr.Feature
    for feat in inlayer:
        geom = feat.geometry()
        pt = geom.GetPoint()
        d0 = [pt[0], pt[1]]
        for i in range(i_fields):
            d0.append(feat.GetField(i))
        d.append(d0)
    if is_ret_srs:
        return layer_def_list, d, inlayer.GetSpatialRef()
    else:
        return layer_def_list, d


def readCSV(csv_fn):
    with open(csv_fn, 'rt') as f:
        cr = csv.reader(f)
        crs = [row for row in cr]
        return crs[0], crs[1:]


def getXYIndex(fields):
    i_x, i_y = -1, -1
    for i in range(len(fields)):
        if "x" == fields[i].lower():
            i_x = i
        if "y" == fields[i].lower():
            i_y = i
    return i_x, i_y


def saveToCSV(fields, d, fn):
    fn = os.path.splitext(fn)[0] + ".csv"
    with open(fn, "w", encoding="utf-8", newline="") as fw:
        w = csv.writer(fw)
        w.writerow(fields)
        for line in d:
            w.writerow(line)
    return fn


def gli1(input_im_fn, input_fn, out_fn, input_fmt=None, out_fmt="csv", qianzui="FEAT"):
    if not os.path.isfile(input_fn):
        raise Exception("Can not find vector file: " + input_fn)
    if not os.path.isfile(input_im_fn):
        raise Exception("Can not find raster file: " + input_im_fn)
    raster_ds = GeoRasterRead(input_im_fn)
    ext = os.path.splitext(input_fn)
    if input_fmt is None:
        if ext[1] == ".shp":
            fields, d, raster_ds.src_srs = readShape(input_fn, is_ret_srs=True)
        elif ext[1] == ".csv":
            fields, d = readCSV(input_fn)
        else:
            raise Exception("Unable to determine the file format of input vector file: " + input_fn)
    else:
        fields, d = readShape(input_fn, input_fmt)
    i_x, i_y = getXYIndex(fields)
    if i_x == -1:
        raise Exception("Can not find X field in [{0}]".format(", ".join(fields)))
    if i_y == -1:
        raise Exception("Can not find Y field in [{0}]".format(", ".join(fields)))
    fields = fields + ["{0}_{1}".format(qianzui, i + 1) for i in range(raster_ds.n_bands)]
    for i in range(len(d)):
        d0 = raster_ds.readAsArray(x_row_off=float(d[i][i_x]), y_column_off=float(d[i][i_y]),
                                   win_row_size=1, win_column_size=1, is_trans=True, is_geo=True)
        d0 = d0.ravel().tolist()
        d[i] = d[i] + d0
    if out_fmt == "csv":
        saveToCSV(fields, d, out_fn)
    else:
        print("Not yet supported format: " + out_fmt)
    pass


def usage():
    print("srt_glitocsv in_raster_file in_vector_file [opt:-o out_csv_file]\n"
          "    in_raster_file: input raster file support by GDAL\n"
          "    in_vector_file: input vector file support by OGR\n"
          "    [opt:-o out_csv_file]: output csv file default:`out.csv`")


def main(argv):
    in_raster_file = None
    in_vector_file = None
    out_csv_file = "out.csv"

    for i in range(1, len(argv)):
        if argv[i] == "-o" and i < len(argv) - 1:
            out_csv_file = argv[i + 1]
            i += 1
        elif in_raster_file is None:
            in_raster_file = argv[i]
        elif in_vector_file is None:
            in_vector_file = argv[i]
    if in_raster_file is None:
        raise Exception("Can not get `in_raster_file`.")
    if in_vector_file is None:
        raise Exception("Can not get `in_vector_file`.")
    try:
        gli1(in_raster_file, in_vector_file, out_csv_file)
    except Exception as ex:
        print(ex)
    pass


class GLI_main:

    def __init__(self):
        self.name = "gli"
        self.description = "Use shapefile to sample raster"
        self.argv = []

    def run(self, argv):
        self.argv = argv
        try:
            main(argv)
        except Exception as ex:
            print(ex)
            usage()

    def usage(self):
        usage()
        pass

# if __name__ == "__main__":
#     # resultDS.SetGeoTransform([minx, self.scaleX, 0, maxy, 0, self.scaleY])
#     # rds = GeoRasterRead(r"D:\FiveMeterImper\Data\GeoImage\lm_rgbn_esa_1.tif")
#     # d = rds.readAsArrayCenter(0, 0, 5, 5)
#     main(["tt", r"D:\RemoteShadow\Samples\Test1\ImageDeal\qd_rgbn_si_asdeC_raw.dat",
#           r"D:\RemoteShadow\Samples\Test1\sh_qd_cc1_spl1.shp",
#           "-o", r"D:\CodeProjects\PyGdal\Data\tt1"])
