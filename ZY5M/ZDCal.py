# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ZDCal.py
@Time    : 2023/12/5 18:49
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of ZDCal
-----------------------------------------------------------------------------"""
import os.path

import numpy as np
from osgeo import gdal, gdal_array

from SRTCodes.Utils import readJson, saveJson

TEMP_DIR_NAME = r"K:\zhongdianyanfa\ZDCityCal\tmp"
PIXEL_AREA = 100
LMLY_GEO_FNS = {
    2015: r"M:\Dataset\AOIS_JPZ_DS\AOIS\BJST_15_gtif.tif",
    2016: r"M:\Dataset\AOIS_JPZ_DS\AOIS\BJST_16_gtif.tif",
    2017: r"M:\Dataset\AOIS_JPZ_DS\AOIS\BJST_17_gtif.tif",
    2018: r"M:\Dataset\AOIS_JPZ_DS\AOIS\BJST_18_gtif.tif",
    2019: r"M:\Dataset\AOIS_JPZ_DS\AOIS\BJST_19_gtif.tif",
    2020: r"M:\Dataset\AOIS_JPZ_DS\AOIS\BJST_20_gtif.tif",
    2021: r"M:\Dataset\AOIS_JPZ_DS\AOIS\BJST_21_gtif.tif",
    2022: r"M:\Dataset\AOIS_JPZ_DS\AOIS\BJST_22_gtif.tif"
}
BJST_GEO_FNS = {
    2015: r"M:\Dataset\AOIS_JPZ_DS\AOIS\LMLY_15_gtif.tif",
    2016: r"M:\Dataset\AOIS_JPZ_DS\AOIS\LMLY_16_gtif.tif",
    2017: r"M:\Dataset\AOIS_JPZ_DS\AOIS\LMLY_17_gtif.tif",
    2018: r"M:\Dataset\AOIS_JPZ_DS\AOIS\LMLY_18_gtif.tif",
    2019: r"M:\Dataset\AOIS_JPZ_DS\AOIS\LMLY_19_gtif.tif",
    2020: r"M:\Dataset\AOIS_JPZ_DS\AOIS\LMLY_20_gtif.tif",
    2021: r"M:\Dataset\AOIS_JPZ_DS\AOIS\LMLY_21_gtif.tif",
    2022: r"M:\Dataset\AOIS_JPZ_DS\AOIS\LMLY_22_gtif.tif"
}
DATA_DICT = {}


def readGEORaster(geo_fn, x_row_off=0.0, y_column_off=0.0,
                  win_row_size=None, win_column_size=None,
                  interleave='band', band_list=None):
    ds: gdal.Dataset = gdal.Open(geo_fn)
    return gdal_array.DatasetReadAsArray(
        ds, y_column_off, x_row_off, win_xsize=win_column_size,
        win_ysize=win_row_size, interleave=interleave, band_list=band_list)


def warpImage(region_fn, geo_fn, to_geo_fn):
    """
    gdalwarp -overwrite -if ENVI -of ENVI -r near -cutline region.shp -cl region c -crop_to_cutline srcfile* dstfile

    format='GTiff',
    cutlineDSName=region_fn,
    cutlineLayer=name(region_fn),
    cropToCutline = True
    """
    region_name = os.path.split(region_fn)[1]
    region_name = os.path.splitext(region_name)[0]
    gdal.Warp(destNameOrDestDS=to_geo_fn, srcDSOrSrcDSTab=geo_fn, format='GTiff', cutlineDSName=region_fn,
              cutlineLayer=region_name, cropToCutline=True, dstNodata=6)


def calArea(geo_fn, im_code):
    geo_fn = os.path.abspath(geo_fn)
    if geo_fn not in DATA_DICT:
        DATA_DICT[geo_fn] = readGEORaster(geo_fn, band_list=[1])
    d = DATA_DICT[geo_fn]
    d_n_0 = np.sum((d == im_code) * 0)
    d_n_1 = np.sum((d == im_code) * 1)
    d_area_0 = d_n_0 * PIXEL_AREA
    d_area_1 = d_n_1 * PIXEL_AREA
    return d_area_1, d_area_0 + d_area_1


class ZDCityCalArea:

    def __init__(self, region_name, year, region_fn, geo_fn, tmp_dirname=TEMP_DIR_NAME):
        self.region_name = region_name
        self.region_fn = region_fn
        self.year = year
        self.geo_fn = geo_fn
        self.tmp_dirname = tmp_dirname

    def cal(self, im_code):
        to_fn = os.path.join(self.tmp_dirname, "{0}_{1}_image.tif".format(self.region_name, self.year))
        if not os.path.isfile(to_fn):
            warpImage(region_fn=self.region_fn, geo_fn=self.geo_fn, to_geo_fn=to_fn)
        return calArea(to_fn, im_code)


def calExpansion(province_name, years, region_fn, geo_fns):
    zd_cca1 = ZDCityCalArea(province_name, years[0], region_fn, geo_fns[0])
    area1, area12 = zd_cca1.cal(1)
    zd_cca2 = ZDCityCalArea(province_name, years[1], region_fn, geo_fns[1])
    area2, area12 = zd_cca2.cal(1)

    sulv = float(area2) - float(area1)
    qiangdu = sulv / float(area12)

    return sulv, qiangdu


class ZDCityCalExpansion:

    def __init__(self, province_name, years, region_fn, geo_fns):
        self.province_name = province_name
        self.years = years
        self.region_fn = region_fn
        self.geo_fns = geo_fns

    def cal(self):
        return calExpansion(self.province_name, self.years, self.region_fn, self.geo_fns)


class ZDCityCalExpansionCollection:

    def __init__(self, geo_fns=None):
        if geo_fns is None:
            geo_fns = LMLY_GEO_FNS
        self.coll = {}
        self.GEO_FNS = geo_fns

    def add(self, province_name: str, year1: int, year2: int, region_fn):
        self.coll[province_name] = ZDCityCalExpansion(
            province_name=province_name,
            years=[year1, year2],
            region_fn=region_fn,
            geo_fns=[self.GEO_FNS[year1], self.GEO_FNS[year2]]
        )

    def addGeoFile(self, year, filename):
        self.GEO_FNS[year] = filename


class City_Cambodia_Province_2015_2021(ZDCityCalExpansionCollection):

    def __init__(self):
        super(City_Cambodia_Province_2015_2021, self).__init__()

    def add_2015_2021(self, province_name: str, region_fn):
        self.add(province_name, 2015, 2021, region_fn)

    def cal(self, fs=None):
        print("年份  植被类型  省份  扩张速率 扩张强度   ", file=fs)
        for name in self.coll:
            sulv, qiangdu = self.coll[name].cal()
            print("2016-2021 建设用地 {0} {1} {2}".format(name, sulv, qiangdu))


class ZDCityCalMain:

    def __init__(self):
        self.name = "ZDCityCal"
        self.tmp_dirname = r"K:\zhongdianyanfa\ZDCityCal"
        self.regions = {}

    def addRegionImage(self, name, region_filename, geo_fn):
        self.regions[name] = region_filename


def main():
    d = readJson(r"K:\zhongdianyanfa\ZDCityCal\gadm41_KHM_1.json\gadm41_KHM_1.json")
    d_save = {'type': d['type'], 'name': d['name'], 'crs': d['crs'], 'features': []}
    for feat in d["features"]:
        d_save['features'] = [feat]
        name = feat["properties"]["VARNAME_1"].split("|")[0]
        to_fn = os.path.join(r"K:\zhongdianyanfa\ZDCityCal\regions", name + "_region.geojson")
        print(name)
        saveJson(d_save, to_fn)

    pass


if __name__ == "__main__":
    main()
