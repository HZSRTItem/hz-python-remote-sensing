# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowHSample.py
@Time    : 2024/2/27 17:10
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of ShadowHSample
-----------------------------------------------------------------------------"""
import time

import numpy as np

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.GDALUtils import SRTGDALCategorySampleCollection, GDALRastersSampling
from SRTCodes.NumpyUtils import NumpyDataCenter
from SRTCodes.Utils import ofn, SRTFilter, DirFileName, Jdt, changext, SRTDataFrame
from Shadow.Hierarchical import SHHConfig
from Shadow.Hierarchical.SHHFuncs import initSHHGRS


class ShadowHierarchicalSampleCollection(SRTGDALCategorySampleCollection):

    def __init__(self):
        super(ShadowHierarchicalSampleCollection, self).__init__()

        self.ndc = NumpyDataCenter()
        self.category_coll_map = {}
        self.rasters_sampling = GDALRastersSampling()

    def initSHHCategory(self):
        self.addCategory()
        self.addCategory("IS", 1, (255, 0, 0))
        self.addCategory("VEG", 2, (0, 255, 0))
        self.addCategory("SOIL", 3, (255, 255, 0))
        self.addCategory("WAT", 4, (0, 0, 255))
        self.addCategory("IS_SH", 5, (128, 0, 0))
        self.addCategory("VEG_SH", 6, (0, 128, 0))
        self.addCategory("SOIL_SH", 7, (128, 128, 0))
        self.addCategory("WAT_SH", 8, (0, 0, 128))

    def copyNoSamples(self):
        scsc = ShadowHierarchicalSampleCollection()
        self.copySCSC(scsc)
        return scsc

    def copySCSC(self, scsc):
        super(ShadowHierarchicalSampleCollection, self).copySCSC(scsc)
        scsc.ndc = self.ndc.copy()
        scsc.category_coll_map = self.category_coll_map.copy()

    def data(self, i, is_center=False):
        x = self.samples[i].data
        if is_center:
            x = self.ndc.fit(x)
        return x

    def initVegHighLowCategoryCollMap(self):
        self.category_coll_map = {
            "IS": 2,
            "VEG": 1,
            "SOIL": 2,
            "WAT": 3,
            "IS_SH": 3,
            "VEG_SH": 3,
            "SOIL_SH": 3,
            "WAT_SH": 3,
        }

    def category(self, i):
        spl = self.samples[i]
        cate = spl.code
        if spl.name in self.category_coll_map:
            cate = self.category_coll_map[spl.name]
        return cate

    def shhSampling1(self):
        self.gdalSamplingRasters([
            r"F:\ProjectSet\Shadow\Hierarchical\Images\bj_sh2_1.tif",
            r"F:\ProjectSet\Shadow\Hierarchical\Images\cd_sh2_1.tif",
            r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\qd_sh2_1.tif",
        ], spl_size=(1, 1), is_to_field=True, no_data=0, is_jdt=True, field_names=None, )
        self.gdalSamplingRasters([
            r"F:\ProjectSet\Shadow\Hierarchical\Images\bj_sh2_esa.tif",
            r"F:\ProjectSet\Shadow\Hierarchical\Images\cd_sh2_esa.tif",
            r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\qd_sh2_esa.tif",
        ], spl_size=(1, 1), is_to_field=True, no_data=0, is_jdt=True, field_names=["ESA21"], )

    def initGDALRastersSampling(self, grs_type, raster_fns=None):
        self.rasters_sampling = initSHHGRS(grs_type, raster_fns)

    def filterFuncGRS(self, filter_func=None, win_row_size=1, win_column_size=1, is_jdt=True,
                      _scc=None, *args, **kwargs):
        if _scc is None:
            _scc = self.copyNoSamples()
        jdt = Jdt(len(self.samples), "ShadowHierarchicalSampleCollection::filterFuncGRS")
        if is_jdt:
            jdt.start()
        for spl in self.samples:
            d = self.rasters_sampling.sampling(
                spl.x, spl.y, win_row_size=win_row_size, win_column_size=win_column_size)
            if filter_func(spl, d, *args, **kwargs):
                _scc.addSample(spl)
            if is_jdt:
                jdt.add()
        if is_jdt:
            jdt.end()
        return _scc


class SHHSplColl(ShadowHierarchicalSampleCollection):

    def __init__(self):
        super(SHHSplColl, self).__init__()

    def copyNoSamples(self):
        scsc = SHHSplColl()
        self.copySCSC(scsc)
        return scsc


class SHHDataFrameSampleCollection(ShadowHierarchicalSampleCollection):

    def __init__(self):
        super(SHHDataFrameSampleCollection, self).__init__()


def loadSHHSamples(name):
    """ name: ["sample1", "sample1[21,21]", "qd_sample1[21,21]"] "qd_sample2[21,21]"]"""

    def split_t(_shh_sc: ShadowHierarchicalSampleCollection):
        _shh_sc_train = _shh_sc.filterCompare("eq", "TEST", 1)
        _shh_sc_test = _shh_sc.filterCompare("eq", "TEST", 0)
        return _shh_sc_train, _shh_sc_test

    shh_sc = ShadowHierarchicalSampleCollection()
    shh_sc_train = ShadowHierarchicalSampleCollection()
    shh_sc_test = ShadowHierarchicalSampleCollection()
    dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Samples")

    if name == "sample1":
        shh_sc.readJson(dfn.fn("SHHSample1_y.json"))
        shh_sc_train, shh_sc_test = split_t(shh_sc)
    elif name == "sample1[21,21]":
        # shh_sc.readJson(dfn.fn("SHHSample1_y.json"))
        # shh_sc.loadDataFromNPY(dfn.fn("SHHSample1_npy.npy"))
        # shh_sc_train, shh_sc_test = split_t(shh_sc)
        # r"F:\ProjectSet\Shadow\Hierarchical\Samples\7\shh2_spl7_2_data.npy"
        # r"F:\ProjectSet\Shadow\Hierarchical\Samples\7\shh2_spl7_2_label.csv"
        shh_sc.readJson(dfn.fn(r"7\shh2_spl7_2_spl.json"))
        shh_sc.loadDataFromNPY(dfn.fn(r"7\shh2_spl7_2_data.npy"))
        shh_sc_train, shh_sc_test = split_t(shh_sc)
    elif name == "qd_sample1[21,21]":
        shh_sc.readJson(dfn.fn(r"2\sh2_spl2_1_y.json"))
        shh_sc.loadDataFromNPY(dfn.fn(r"2\sh2_spl2_1_x.npy"))
        shh_sc_train, shh_sc_test = split_t(shh_sc)
    elif name == "qd_sample2[21,21]":
        shh_sc.readJson(dfn.fn(r"2\sh2_spl2_2_y.json"))
        shh_sc.loadDataFromNPY(dfn.fn(r"2\sh2_spl2_2_x.npy"))
        shh_sc_train, shh_sc_test = split_t(shh_sc)
    elif name == "qd_sample3[21,21]":
        shh_sc.readJson(dfn.fn(r"2\sh2_spl2_3_y.json"))
        shh_sc.loadDataFromNPY(dfn.fn(r"2\sh2_spl2_3_x.npy"))
        shh_sc_train, shh_sc_test = split_t(shh_sc)
    elif name == "qd_sample4[21,21]":
        shh_sc.readJson(dfn.fn(r"2\sh2_spl2_4_y.json"))
        shh_sc.loadDataFromNPY(dfn.fn(r"2\sh2_spl2_4_x.npy"))
        shh_sc_train, shh_sc_test = split_t(shh_sc)
    else:
        raise Exception("Can not find ShadowHierarchicalSampleCollection for \"{0}\".".format(name))

    return shh_sc_train, shh_sc_test


def samplingKnow():
    csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\7\shh2_spl7_2.csv"
    to_npy_fn = changext(csv_fn, "_data.npy")
    to_csv_fn = changext(csv_fn, "_label.csv")
    to_json_fn = changext(csv_fn, "_spl.json")
    win_size = [21, 21]

    sdf = SRTDataFrame().read_csv(csv_fn, is_auto_type=True)
    sdf.toCSV(to_csv_fn)
    geo_fns = SHHConfig.SHH2_IMAGE1_FNS
    grs = [GDALRaster(fn) for fn in geo_fns]
    gr = grs[0]
    npy_indexs = []
    out_d = np.zeros((len(sdf), gr.n_channels, win_size[0], win_size[1]))

    jdt = Jdt(len(sdf), "samplingKnow").start()
    for i in range(len(sdf)):
        x, y = sdf["X"][i], sdf["Y"][i]
        npy_indexs.append(i)
        if not gr.isGeoIn(x, y):
            for gr in grs:
                if gr.isGeoIn(x, y):
                    break
        jdt.add()
        out_d[i] = gr.readAsArrayCenter(x, y, win_row_size=win_size[0], win_column_size=win_size[1], is_geo=True)
    jdt.end()

    sdf["__DATA_N__"] = npy_indexs
    sdf.toCSV(to_csv_fn)
    np.save(to_npy_fn, out_d)

    shh_sc = ShadowHierarchicalSampleCollection()
    shh_sc.initSHHCategory()
    shh_sc.read_csv(csv_fn)
    shh_sc.toJson(to_json_fn, is_save_data=False,is_jdt=True)



def main():
    def func1():
        shh_sc = SHHSplColl()
        shh_sc.initSHHCategory()
        shh_sc.read_csv(r"F:\Week\20240303\Data\tmp2.csv")
        shh_sc.toJson(ofn.ddir("tmp2.json"))

    def func2():
        shh_sc = SHHSplColl()
        t1 = time.time()
        shh_sc.readJson(ofn.ddir("tmp5.json"))
        # shh_sc.gdalSampling(r"F:\ProjectSet\Shadow\Hierarchical\Images\qd_sh2_1.tif",
        #                     is_to_field=True, is_jdt=True)
        # shh_sc.toJson(ofn.ddir("tmp5.json"), is_save_data=True, is_jdt=True)
        shh_sc.getFields()
        shh_sc2 = shh_sc.filter(SRTFilter.eq("TEST", 1))
        print(time.time() - t1)
        return

    def func3():
        # shh_sc = SHHSplColl()
        # # shh_sc.readJson(r"F:\ProjectSet\Shadow\Hierarchical\Samples\images1_spl1.json")
        # shh_sc.initSHHCategory()
        # shh_sc.read_csv(r"F:\ProjectSet\Shadow\Release\BeiJingSamples\sh_bj_sample.csv")
        # shh_sc.read_csv(r"F:\ProjectSet\Shadow\Release\ChengDuSamples\sh_cd_sample.csv")
        # shh_sc.read_csv(r"F:\ProjectSet\Shadow\Release\QingDaoSamples\sh_qd_sample.csv")
        # shh_sc.gdalSampling(r"F:\ProjectSet\Shadow\Hierarchical\Images\qd_sh2_1.tif", is_to_field=True, is_jdt=True,
        #                     is_sampling="SAMPLING")
        # shh_sc.gdalSampling(r"F:\ProjectSet\Shadow\Hierarchical\Images\bj_sh2_1.tif", is_to_field=True, is_jdt=True,
        #                     is_sampling="SAMPLING")
        # shh_sc.gdalSampling(r"F:\ProjectSet\Shadow\Hierarchical\Images\cd_sh2_1.tif", is_to_field=True, is_jdt=True,
        #                     is_sampling="SAMPLING")
        # shh_sc = shh_sc.filter(SRTFilter.eq("SAMPLING", 1))
        # shh_sc.toJson(r"F:\ProjectSet\Shadow\Hierarchical\Samples\images1_spl2.json")
        # shh_sc.toCSV(r"F:\ProjectSet\Shadow\Hierarchical\Samples\images1_spl2.csv")

        def func3_sampling(csv_fn, tif_fn, to_fn):
            shh_sc = SHHSplColl()
            shh_sc.initSHHCategory()
            shh_sc.read_csv(csv_fn)
            shh_sc.gdalSampling(tif_fn, is_to_field=True, is_jdt=True, is_sampling="SAMPLING")
            shh_sc = shh_sc.filter(SRTFilter.eq("SAMPLING", 1))
            shh_sc.toJson(to_fn + ".json")
            shh_sc.toCSV(to_fn + ".csv")

        # func3_sampling(
        #     csv_fn=r"F:\ProjectSet\Shadow\Release\BeiJingSamples\sh_bj_sample.csv",
        #     tif_fn=r"F:\ProjectSet\Shadow\Hierarchical\Images\bj_sh2_1.tif",
        #     to_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\bj_sh1_spl1",
        # )

        # func3_sampling(
        #     csv_fn=r"F:\ProjectSet\Shadow\Release\ChengDuSamples\sh_cd_sample.csv",
        #     tif_fn=r"F:\ProjectSet\Shadow\Hierarchical\Images\cd_sh2_1.tif",
        #     to_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\cd_sh1_spl1",
        # )

        func3_sampling(
            csv_fn=r"F:\ProjectSet\Shadow\Release\QingDaoSamples\sh_qd_sample.csv",
            tif_fn=r"F:\ProjectSet\Shadow\Hierarchical\Images\qd_sh2_1.tif",
            to_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\qd_sh1_spl1",
        )

        return

    def func4():
        shh_sc = SHHSplColl()
        shh_sc.readJson(r"F:\ProjectSet\Shadow\Hierarchical\Samples\SHHSample1.json")
        shh_sc.loadDataFromNPY(r"F:\ProjectSet\Shadow\Hierarchical\Samples\SHHSample2.npy")

        # shh_sc.initSHHCategory()
        # shh_sc.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\SHHSample1.csv")

        # dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Images")
        # shh_sc.gdalSampling(dfn.fn("qd_sh2_1.tif"), spl_size=(21, 21), is_jdt=True, is_sampling="SAMPLING")
        # shh_sc.gdalSampling(dfn.fn("cd_sh2_1.tif"), spl_size=(21, 21), is_jdt=True, is_sampling="SAMPLING")
        # shh_sc.gdalSampling(dfn.fn("bj_sh2_1.tif"), spl_size=(21, 21), is_jdt=True, is_sampling="SAMPLING")
        # shh_sc.saveDataToNPY(r"F:\ProjectSet\Shadow\Hierarchical\Samples\SHHSample2.npy")
        # shh_sc.toJson(r"F:\ProjectSet\Shadow\Hierarchical\Samples\SHHSample2.json", is_save_data=True, is_jdt=True)

        return

    def qd_sampling():
        shh_sc = SHHSplColl()
        shh_sc.addCategory("NOT_KNOW", 0, (0, 255, 0))
        shh_sc.addCategory("VEG", 1, (0, 255, 0))
        shh_sc.addCategory("HIGH", 2, (200, 200, 200))
        shh_sc.addCategory("LOW", 3, (60, 60, 60))
        shh_sc.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\2\sh2_spl2_4.csv")
        dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Images")
        shh_sc.gdalSampling(dfn.fn(r"QingDao\qd_sh2_1.tif"), spl_size=(21, 21), is_jdt=True)
        shh_sc.saveDataToNPY(r"F:\ProjectSet\Shadow\Hierarchical\Samples\2\sh2_spl2_4_x.npy")
        shh_sc.toJson(r"F:\ProjectSet\Shadow\Hierarchical\Samples\2\sh2_spl2_4_y.json", is_save_data=False, is_jdt=True)

    def sampling_3():
        shh_sc = SHHSplColl()
        shh_sc.readJson(r"F:\ProjectSet\Shadow\Hierarchical\Samples\SHHSample1.json")
        dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Images")
        shh_sc.gdalSamplingRasters(
            gdal_raster_fns=[
                r"F:\ProjectSet\Shadow\Hierarchical\Images\bj_sh2_1.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\cd_sh2_1.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\qd_sh2_1.tif",
            ],
            spl_size=(1, 1), is_to_field=True,
            no_data=None,
            is_jdt=True, field_names=None,
        )
        # shh_sc.saveDataToNPY(r"F:\ProjectSet\Shadow\Hierarchical\Samples\SHHSample1_npy.npy")
        shh_sc.toJson(r"F:\ProjectSet\Shadow\Hierarchical\Samples\SHHSample1_2_y.json", is_save_data=False, is_jdt=True)
        shh_sc.toCSV(r"F:\ProjectSet\Shadow\Hierarchical\Samples\SHHSample1_2_y.csv")

    def sh1_fc_sampling():
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\6\sh2_spl6_1_212.csv"
        to_csv_fn = changext(csv_fn, "_spl.csv")

        shh_sc = SHHSplColl()
        shh_sc.FN_CNAME = "CNAME2"
        shh_sc.read_csv(csv_fn)
        shh_sc.gdalSamplingRasters(
            gdal_raster_fns=SHHConfig.SHH2_IMAGE1_FNS, spl_size=(1, 1), is_to_field=True, is_jdt=True, )
        shh_sc.gdalSamplingRasters(
            gdal_raster_fns=SHHConfig.SHH2_IMAGE1_GLCM_FNS, spl_size=(1, 1), is_to_field=True, is_jdt=True, )
        shh_sc.gdalSamplingRasters(
            gdal_raster_fns=SHHConfig.SHH2_IMAGE1_ESA21_FNS, spl_size=(1, 1), is_to_field=True, is_jdt=True, )
        shh_sc.toCSV(to_csv_fn)

    sh1_fc_sampling()
    # shh_sc_train, shh_sc_test = loadSHHSamples("qd_sample1[21,21]")
    return


if __name__ == "__main__":
    samplingKnow()
