# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SampleUtils.py
@Time    : 2024/12/20 15:43
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of RemoteSensingSamples

SRTSampleSelect 抽样的时候用
SRTSample 单个样本
SRTSampleCollection 样本集合
SamplesManage 样本管理添加类型的样本
GDALSampleUpdate 采样
-----------------------------------------------------------------------------"""
from collections import OrderedDict

import pandas as pd
from osgeo import gdal_array

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.SRTSample import SRTSampleCollectionInit, readQJYTxt, GeoJsonPolygonCoor
from SRTCodes.Utils import datasCaiFen, Jdt


def _COM(name, f, data, com_data):
    if f == "==":
        return com_data == data
    elif f == ">":
        return com_data > data
    elif f == "<":
        return com_data < data
    elif f == ">=":
        return com_data >= data
    elif f == "<=":
        return com_data <= data
    elif f == "!=":
        return com_data != data
    else:
        raise Exception("Can not format \"{}\"".format(f))


class _Sample(OrderedDict):

    def __init__(self, other=(), **kwds):
        super().__init__(other, **kwds)

    def isfilter_or(self, _filters):
        if _filters is None:
            return True
        for name, f, data in _filters:
            if not self.__contains__(name):
                return False
            if _COM(name, f, data, self.__getitem__(name)):
                return True
        return False

    def isfilter_and(self, _filters):
        if _filters is None:
            return True
        for name, f, data in _filters:
            if not self.__contains__(name):
                return False
            if not _COM(name, f, data, self.__getitem__(name)):
                return False
        return True


class SamplesUtil(SRTSampleCollectionInit):

    def __init__(self):
        super(SamplesUtil, self).__init__()

    def addSample(self, spl, field_datas=None, _func=None, _filters_and=None, _filters_or=None):
        if field_datas is not None:
            for name in field_datas:
                spl[name] = field_datas[name]
        spl = _Sample(spl)
        if spl.isfilter_and(_filters_and):
            if _func is not None:
                _func(spl)
            return super(SamplesUtil, self).addSample(spl)
        elif spl.isfilter_or(_filters_or):
            if _func is not None:
                _func(spl)
            return super(SamplesUtil, self).addSample(spl)
        else:
            return None

    def addSamples(self, *spls, field_datas=None, _func=None, _filters_and=None, _filters_or=None):
        to_spls = []
        spls = datasCaiFen(*spls)
        for spl in spls:
            spl = self.addSample(
                spl, field_datas=field_datas, _func=_func,
                _filters_and=_filters_and, _filters_or=_filters_or
            )
            if spl is not None:
                to_spls.append(spl)
        return to_spls

    def addDF(self, df: pd.DataFrame, field_datas=None, _func=None, _filters_and=None, _filters_or=None):
        df_list = df.to_dict("records")
        return self.addSamples(df_list, field_datas=field_datas, _func=_func,
                               _filters_and=_filters_and, _filters_or=_filters_or)

    def addCSV(self, csv_fn, field_datas=None, _func=None, _filters_and=None, _filters_or=None):
        return self.addDF(pd.read_csv(csv_fn), field_datas=field_datas, _func=_func,
                          _filters_and=_filters_and, _filters_or=_filters_or)

    def addQJY(self, txt_fn, field_datas=None, _func=None, _filters_and=None, _filters_or=None):
        df_dict = readQJYTxt(txt_fn)
        df_dict["X"] = df_dict.pop("__X")
        df_dict["Y"] = df_dict.pop("__Y")
        df_dict["CNAME"] = df_dict.pop("__CNAME")
        return self.addDF(pd.DataFrame(df_dict), field_datas=field_datas, _func=_func,
                          _filters_and=_filters_and, _filters_or=_filters_or)

    def addGeoJSONRange(self, json_fn, sampling_type=None, field_datas=None, _func=None, _filters_and=None,
                        _filters_or=None):
        if sampling_type is None:
            sampling_type = {"random": 1}
        geo_json = GeoJsonPolygonCoor(json_fn)
        if "random" in sampling_type:
            spls = geo_json.random(sampling_type["random"])
            return self.addSamples(spls, field_datas=field_datas, _func=_func,
                                   _filters_and=_filters_and, _filters_or=_filters_or)

    def filterSamples(self, _filters_and=None, _filters_or=None):
        spls = []
        if _filters_and is not None:
            for spl in self.samples:
                if spl.isfilter_and(_filters_and):
                    spls.append(spl)
        elif _filters_or is not None:
            for spl in self.samples:
                if spl.isfilter_or(_filters_or):
                    spls.append(spl)
        else:
            return self.samples
        return spls

    def sampling1(self, name, raster_fns, _func=None, _filters_and=None, _filters_or=None,
                  x_field_name="X", y_field_name="Y", is_jdt=True):
        spls = self.filterSamples(_filters_and=_filters_and, _filters_or=_filters_or)
        grs = [GDALRaster(fn) for fn in raster_fns]
        gr_channels = [gr.getGDALBand(1) for gr in grs]
        channel = gr_channels[0]
        gr = grs[0]
        jdt = Jdt(len(spls), "SamplesUtil::sampling1(name={})".format(name)).start(is_jdt=is_jdt)
        for spl in spls:
            x, y = float(spl[x_field_name]), float(spl[y_field_name])
            if not gr.isGeoIn(x, y):
                gr.readAsArray()
                is_find = False
                for i, gr in enumerate(grs):
                    if gr.isGeoIn(x, y):
                        channel = gr_channels[i]
                        is_find = True
                        break
                if not is_find:
                    continue
            x_row_off, y_column_off = gr.coorGeo2Raster(x, y, is_int=True)
            data = gdal_array.BandReadAsArray(channel, y_column_off, x_row_off, win_xsize=1, win_ysize=1, )
            spl[name] = data[0][0]
            jdt.add(is_jdt=is_jdt)
        jdt.end(is_jdt=is_jdt)
        return spls

    def toDF(self) -> pd.DataFrame:
        return pd.DataFrame(self.samples)

    def toCSV(self, csv_fn, ):
        self.toDF().to_csv(csv_fn, index=False)
        return csv_fn

    def __next__(self) -> _Sample:
        return super(SamplesUtil, self).__next__()

    def __getitem__(self, item) -> _Sample:
        return super(SamplesUtil, self).__getitem__(item)


def main():
    # spl = _Sample({"N": 1, "M": 2})
    # print(pd.DataFrame([_Sample({"N": 1, "M": 2}), _Sample({"N": 3, "M": 5})]))

    # gr = GDALRaster(r"G:\SHImages\QD_NDVI.tif")
    # data = gr.getGDALBand(1)
    # print(data)
    # to_data = gdal_array.BandReadAsArray(band=data, xoff=3, yoff=2, win_xsize=1000, win_ysize=2000, )
    # print(to_data.shape, "\n", to_data)
    #
    # import matplotlib.pyplot as plt
    # from SRTCodes.NumpyUtils import scaleMinMax
    #
    # plt.imshow(scaleMinMax(to_data))
    # plt.show()

    su = SamplesUtil()
    print(len(su.addCSV(r"F:\GraduationDesign\Result\run\Samples\sh2_spl30_bj1_spl.csv", field_datas={"CITY": "bj"})))
    print(len(su.addCSV(r"F:\GraduationDesign\Result\run\Samples\sh2_spl30_cd6_spl.csv", field_datas={"CITY": "cd"})))
    print(len(su.addCSV(r"F:\GraduationDesign\Result\run\Samples\sh2_spl30_qd6_spl.csv", field_datas={"CITY": "qd"})))
    print(len(su))

    su.sampling1("NDVI_DA", [
        r"G:\SHImages\QD_NDVI.tif",
        r"G:\SHImages\BJ_NDVI.tif",
        r"G:\SHImages\CD_NDVI.tif",
    ])

    df = su.toDF()
    print(df[['X', 'Y', 'CNAME', "CITY", "NDVI_DA"]])

    return


if __name__ == "__main__":
    main()
