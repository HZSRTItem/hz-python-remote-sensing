# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : OGRUtils.py
@Time    : 2023/10/8 19:44
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of OGRUtils
-----------------------------------------------------------------------------"""
import random

import numpy as np
from osgeo import ogr

from SRTCodes.GDALUtils import RESOLUTION_ANGLE
from SRTCodes.Utils import readJson, saveJson, Jdt


class SRTGeoField:

    def __init__(self, field_name="field_name", d_type=None, data=None):
        d_type = self._dataToType(d_type, data)
        self.name = field_name
        self.d_type = d_type
        self.data = data
        self.init()

    @staticmethod
    def _dataToType(d_type, data):
        if data is not None:
            if d_type is None:
                if isinstance(data, str):
                    d_type = "string"
                elif isinstance(data, int):
                    d_type = "int"
                elif isinstance(data, float):
                    d_type = "float"
        return d_type

    def init(self):
        if self.d_type == "string":
            self.data = ""
        elif self.d_type == "int":
            self.data = 0
        elif self.d_type == "float":
            self.data = 0.0
        else:
            self.data = None

    def d(self, data):
        if self.d_type == "string":
            if not isinstance(data, str):
                raise Exception("Data type can not equal string.")
            self.data = data
        elif self.d_type == "int":
            if not isinstance(data, int):
                raise Exception("Data type can not equal int.")
            self.data = data
        elif self.d_type == "float":
            if not isinstance(data, float):
                raise Exception("Data type can not equal float.")
            self.data = data
        else:
            self.data = None


class SRTGeoFeature:

    def __init__(self, geom_type="Point"):
        self.geom_type = geom_type
        self.fields = {}
        self.coors = []

    def addField(self, field_name="field_name", d_type="string", data=""):
        if field_name in self.fields:
            raise Exception("Filed {0} have in {1}".format(field_name, self.fields))
        self.fields[field_name] = SRTGeoField(field_name=field_name, d_type=d_type, data=data)

    def __setitem__(self, key, data):
        if key in self.fields:
            return self.fields[key].d(data)
        self.fields[key] = SRTGeoField(field_name=key, data=data)

    def __getitem__(self, item):
        return self.fields[item].data

    def coordinates(self, coors: list):
        self.coors = coors

    def __contains__(self, name):
        return name in self.fields

    def init(self):
        self.coors = []
        for k in self.fields:
            self.fields[k].init()

    def toDict(self):
        return {
            "type": "Feature",
            "properties": {key: self.fields[key].data for key in self.fields},
            "geometry": {
                "type": self.geom_type,
                "coordinates": self.geom_type
            }
        }


class SRTGeoJson:

    def __init__(self, geojson_fn=None):
        self.geojson_fn = geojson_fn
        self.geojson = {}
        self.geo_feature = SRTGeoFeature()
        self.readJson()

    def readJson(self, geojson_fn=None):
        if geojson_fn is None:
            geojson_fn = self.geojson_fn
        if geojson_fn is not None:
            self.geojson_fn = geojson_fn
            self.geojson = readJson(geojson_fn)
            self.geo_feature.geom_type = self.geojson["features"][0]["geometry"]["type"]

    def __getitem__(self, item):
        return self.geojson[item]

    def __setitem__(self, key, value):
        self.geojson[key] = value

    def addField(self, field_name="field_name", d_type="string"):
        self.geo_feature.addField(field_name=field_name, d_type=d_type)

    def addFeature(self, *fields, coors=None, **kw_fields):
        if coors is None:
            coors = []
        self.geo_feature.init()
        for i in range(0, 2, len(fields)):
            if fields[i] not in self.geo_feature:
                raise Exception("Field {0} not in this.")
            self.geo_feature[fields[i]] = fields[i + 1]
        for k in kw_fields:
            if k not in self.geo_feature:
                raise Exception("Field {0} not in this.")
            self.geo_feature[k] = kw_fields[k]
        self.geo_feature.coordinates(coors)
        self.geojson["features"].append(self.geo_feature.toDict())

    def __contains__(self, name):
        return name in self.geo_feature

    def saveToJson(self, geojson_fn=None):
        if geojson_fn is None:
            geojson_fn = self.geojson_fn
        if geojson_fn is None:
            saveJson(self.geojson, geojson_fn)


def initFromGeoJson(geojson: SRTGeoJson, name="geojson_name") -> SRTGeoJson:
    d = SRTGeoJson()
    d["type"] = geojson["type"]
    d["name"] = name
    d["crs"] = geojson["crs"]
    d["features"] = []
    return d


def sampleSpaceUniform(coors: list, x_len: float, y_len: float, is_trans_jiaodu=False, is_jdt=False, ret_index=False):
    if is_trans_jiaodu:
        x_len = x_len * RESOLUTION_ANGLE
        y_len = y_len * RESOLUTION_ANGLE
    index_list = [i for i in range(len(coors))]
    random.shuffle(index_list)

    d = np.array(coors)
    d_min = np.min(d, axis=1)
    x0, y0 = d_min[0], d_min[1]

    out_index_list = []
    coors2 = []
    grid = []
    jdt = Jdt(len(coors), "sampleSpaceUniform")
    if is_jdt:
        jdt.start()
    for i in index_list:
        coor = coors[i]
        grid0 = (int((coor[0] - x0) / x_len), int((coor[1] - y0) / y_len))
        if grid0 not in grid:
            coors2.append(coor)
            grid.append(grid0)
            out_index_list.append(i)

        if is_jdt:
            jdt.add()
    if is_jdt:
        jdt.end()
    if ret_index:
        return coors2, out_index_list
    return coors2


class SRTOGRSampleSpaceUniform:

    def __init__(self, x_len: float, y_len: float, x0=0.0, y0=0.0, is_trans_jiaodu=False):
        if is_trans_jiaodu:
            x_len = x_len * RESOLUTION_ANGLE
            y_len = y_len * RESOLUTION_ANGLE
        self.x_len = x_len
        self.y_len = y_len
        self.x0 = x0
        self.y0 = y0
        self.grids = []

    def coor(self, x, y):
        grid = (int((x - self.x0) / self.x_len), int((y - self.y0) / self.y_len))
        if grid not in self.grids:
            self.grids.append(grid)
            return True
        else:
            return False


class SRTESRIShapeFileRead:

    def __init__(self, shp_fn=None):
        self.filename = shp_fn
        self.feature_list = []
        self.readShape()

    def readShape(self, shp_fn=None):
        if shp_fn is None:
            shp_fn = self.filename
        if shp_fn is None:
            return None
        driver = ogr.GetDriverByName("ESRI Shapefile")
        data_source = driver.Open(shp_fn, 0)  # read only
        if data_source is None:
            raise Exception("Can not open file " + shp_fn)
        layer = data_source.GetLayer(0)
        self.feature_list = []
        for feature in layer:
            attributes = {}
            for field in layer.schema:
                attributes[field.name] = feature.GetField(field.name)
            geometry: ogr.Geometry = feature.GetGeometryRef()
            feature_data = {
                "geometry": geometry.GetPoints(),
                "attributes": attributes
            }
            self.feature_list.append(feature_data)

        return self.feature_list

    def getXList(self):
        if self.feature_list:
            return [feat["geometry"][0][0] for feat in self.feature_list]
        else:
            return []

    def getYList(self):
        if self.feature_list:
            return [feat["geometry"][0][1] for feat in self.feature_list]
        else:
            return []

    def getCoorList(self):
        if self.feature_list:
            return [feat["geometry"][0] for feat in self.feature_list]
        else:
            return []


def main():
    pass


if __name__ == "__main__":
    main()
