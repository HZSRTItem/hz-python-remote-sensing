# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowHSample.py
@Time    : 2024/2/27 17:10
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of ShadowHSample
-----------------------------------------------------------------------------"""
import csv
import os
import random
import time
import warnings
from shutil import copyfile

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.GDALUtils import SRTGDALCategorySampleCollection, GDALRastersSampling
from SRTCodes.NumpyUtils import NumpyDataCenter
from SRTCodes.SRTSample import SRTSample
from SRTCodes.Utils import ofn, SRTFilter, DirFileName, Jdt, changext, SRTDataFrame, numberfilename, FN, mkdir, \
    datasCaiFen, saveJson
from Shadow.Hierarchical import SHHConfig


class ShadowHierarchicalSampleCollection(SRTGDALCategorySampleCollection):
    """
    添加样本的时候，以类别名为主
    """

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


class SHH2Sample(SRTSample):

    def __init__(self, *field_datas, data=None, **field_kws):
        super().__init__(*field_datas, data=data, **field_kws)


class SHH2Samples:

    def __init__(self):
        self.y_field_name = None
        self.x_field_name = None
        self.samples = []
        self.field_names = []
        self.filenames = {}
        self.FN_FIELD = "__FN_FIELD__"
        self._n_iter = 0
        self.filter_fns = {}
        self.ndc = NumpyDataCenter()

        self.is_init_xy = False
        self.is_init_category = False

        self.category_field_name = "CATEGORY"
        self.cname_field_name = "CNAME"
        self.x_field_name = "X"
        self.y_field_name = "Y"

    def addSamples(self, spls):
        for spl in spls:
            self.addSample(spl)
        return self

    def filterEQ(self, field_name, data):
        return [spl for spl in self.samples if spl[field_name] == data]

    def filterNotEQ(self, field_name, data):
        return [spl for spl in self.samples if spl[field_name] != data]

    def filterEQ_FN(self, fn_key):
        return self.filterEQ(self.FN_FIELD, self.filter_fns[fn_key])

    def addFileName(self, fn):
        self.filenames[fn] = len(self.filenames)
        self.filenames[self.filenames[fn]] = fn
        return self.filenames[fn]

    def addFieldName(self, name):
        if name not in self.field_names:
            self.field_names.append(name)
            for spl in self.samples:
                spl[name] = None
        return name

    def addSample(self, spl: SHH2Sample):
        for name in spl:
            self.addFieldName(name)
        for name in self.field_names:
            if name not in spl:
                spl[name] = None
        self.samples.append(spl)

    def addCSV(self, csv_fn, npy_fn=None, field_datas=None, *args, **kwargs):
        if field_datas is None:
            field_datas = {}
        sdf = SRTDataFrame().read_csv(csv_fn, is_auto_type=True)
        n = self.addFileName(csv_fn)
        for i in range(len(sdf)):
            spl = SHH2Sample()
            for k in sdf:
                spl[k] = sdf[k][i]
            for k in field_datas:
                spl[k] = field_datas[k]
            spl[self.FN_FIELD] = n
            self.addSample(spl)
        if npy_fn is not None:
            self.loadNpy(npy_fn, csv_fn)

    def addField(self, field_name):
        if field_name in self.field_names:
            warnings.warn("field name \"{0}\" have in this.".format(field_name))
            return
        self.field_names.append(field_name)
        for spl in self.samples:
            spl[field_name] = None

    def sampling(self, grs: GDALRastersSampling, win_size=(1, 1), fn=None, is_jdt=True, is_to_field=False):
        spls = self._filterFNSpls(fn)
        channel_names = grs.getNames()

        jdt = Jdt(len(spls), "SHH2Samples::sampling").start(is_jdt=is_jdt)
        for spl in spls:
            d = grs.sampling(spl.x, spl.y, win_row_size=win_size[0], win_column_size=win_size[1])
            spl.data = d
            if is_to_field:
                to_d = d.ravel()
                for i, name in enumerate(channel_names):
                    if name not in spl:
                        self.addField(name)
                    spl[name] = float(to_d[i])
            jdt.add(is_jdt=is_jdt)
        jdt.end(is_jdt=is_jdt)

    def _filterFNSpls(self, fn):
        if fn is None:
            spls = self.samples
        else:
            spls = self.filterEQ(self.FN_FIELD, self.filenames[fn])
        return spls

    def loadNpy(self, npy_fn, fn=None):
        spls = self._filterFNSpls(fn)
        d = np.load(npy_fn)
        for i, spl in enumerate(spls):
            spl.data = d[i]
        return len(d)

    def toNpy(self, npy_fn, fn=None):
        spls = self._filterFNSpls(fn)
        to_d = None
        for i, spl in enumerate(spls):
            if to_d is None:
                to_d = np.zeros((len(spls),) + spl.data.shape)
            to_d[i] = spl.data
        np.save(npy_fn, to_d.astype("float32"))
        return len(to_d)

    def toCSV(self, to_csv_fn, **kwargs):
        with open(to_csv_fn, "w", encoding="utf-8", newline="") as f:
            cw = csv.writer(f)
            first_line = []
            if "x_field_name" in kwargs:
                first_line.append(kwargs["x_field_name"])
            if "y_field_name" in kwargs:
                first_line.append(kwargs["y_field_name"])
            if "category_field_name" in kwargs:
                first_line.append(kwargs["category_field_name"])
            if "cname_field_name" in kwargs:
                first_line.append(kwargs["cname_field_name"])
            first_line.extend(self.field_names)
            cw.writerow(first_line)
            for i in range(len(self.samples)):
                spl: SHH2Sample = self.samples[i]
                line = []
                if "x_field_name" in kwargs:
                    line.append(spl.x)
                if "y_field_name" in kwargs:
                    line.append(spl.y)
                if "category_field_name" in kwargs:
                    line.append(spl.code)
                if "cname_field_name" in kwargs:
                    line.append(spl.name)
                line.extend([spl[k] for k in self.field_names])
                cw.writerow(line)

    def toDF(self):
        return pd.DataFrame(self.toDict())

    def toDict(self):
        to_dict = {name: [] for name in self.field_names}
        if self.is_init_category:
            to_dict[self.category_field_name] = []
            to_dict[self.cname_field_name] = []
        if self.is_init_xy:
            to_dict[self.x_field_name] = []
            to_dict[self.y_field_name] = []
        for i in range(len(self.samples)):
            spl: SHH2Sample = self.samples[i]
            for k in self.field_names:
                to_dict[k].append(spl[k])
            if self.is_init_category:
                to_dict[self.category_field_name].append(spl.code)
                to_dict[self.cname_field_name].append(spl.name)
            if self.is_init_xy:
                to_dict[self.x_field_name].append(spl.x)
                to_dict[self.y_field_name].append(spl.y)
        return to_dict

    def initXY(self, x_field_name="X", y_field_name="Y", fn=None):
        spls = self._filterFNSpls(fn)
        self.is_init_xy = True
        self.x_field_name = x_field_name
        self.y_field_name = y_field_name
        for spl in spls:
            spl.x = float(spl[x_field_name])
            spl.y = float(spl[y_field_name])

    def initCategory(self, field_name, map_dict, fn=None, others=None):
        spls = self._filterFNSpls(fn)
        self.is_init_category = True
        for i, spl in enumerate(spls):
            spl: SHH2Sample
            spl.name = spl[field_name]
            if others is not None:
                if spl.name in map_dict:
                    spl.code = map_dict[spl[field_name]]
                else:
                    spl.code = others
            else:
                spl.code = map_dict[spl[field_name]]

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return self

    def __next__(self) -> SHH2Sample:
        if self._n_iter == len(self.samples):
            self._n_iter = 0
            raise StopIteration()
        self._n_iter += 1
        return self.samples[self._n_iter - 1]

    def __getitem__(self, item) -> SHH2Sample:
        return self.samples[item]

    def filterFunc(self, func):
        return filter(func, self.samples)

    def filterCODEContain(self, *datas):
        data_list = datasCaiFen(datas)

        def func(spl):
            return spl.code in data_list

        return self.filterFunc(func)

    def filterCNAMEContain(self, *datas):
        data_list = datasCaiFen(datas)

        def func(spl):
            return spl.name in data_list

        return self.filterFunc(func)

    def mapFiled(self, field_name, map_dict):
        for spl in self.samples:
            spl[field_name] = map_dict[spl[field_name]]

    def randomSelect(self, selects, fn=None):
        spls = self._filterFNSpls(fn)
        category_counts = {}
        for spl in spls:
            if spl.code not in category_counts:
                category_counts[spl.code] = 0
            category_counts[spl.code] += 1
        for k in selects:
            if k not in category_counts:
                warnings.warn("\"{0}\" in selects can not in code.".format(k))
            else:
                if selects[k] is None:
                    selects[k] = category_counts[k]
        selects_counts = {k: 0 for k in selects}
        random_list = [i for i in range(len(spls))]
        random.shuffle(random_list)
        if fn is not None:
            samples = self.filterNotEQ(self.FN_FIELD, self.filenames[fn])
        else:
            samples = []
        for i in random_list:
            spl = spls[i]
            if spl.code in selects_counts:
                if selects_counts[spl.code] < selects[spl.code]:
                    samples.append(spl)
                    selects_counts[spl.code] += 1
        return samples

    def vLookUp(self, field_name, to_field_name, df, look_field_name, look_index_field_name=None, ):
        self.addFieldName(to_field_name)
        if look_index_field_name is None:
            look_field_name = field_name
        look_dict = {df[look_index_field_name][i]: df[look_field_name][i] for i in range(len(df))}
        for spl in self.samples:
            spl[to_field_name] = look_dict[spl[field_name]]

    def fmtStrColumns(self, *field_names):
        field_names = datasCaiFen(field_names)


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
    shh_sc.toJson(to_json_fn, is_save_data=False, is_jdt=True)


def loadSHH2Samples():
    dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Samples\Release")
    shh_spl = SHH2Samples()

    shh_spl.addCSV(dfn.fn(r"sh2_spl6_1_2_spl1_1.csv"), field_datas={"CITY": "qd", "TEST": 1})
    shh_spl.loadNpy(dfn.fn(r"sh2_spl6_1_2_spl1_1_data.npy"), dfn.fn(r"sh2_spl6_1_2_spl1_1.csv"))
    shh_spl.initCategory("CATEGORY_NAME", SHHConfig.CNAME_MAP_SH881, fn=dfn.fn(r"sh2_spl6_1_2_spl1_1.csv"))

    shh_spl.addCSV(dfn.fn(r"shh2_spl9_bj1210_spl.csv"), field_datas={"CITY": "bj", "TEST": 1})
    shh_spl.loadNpy(dfn.fn(r"shh2_spl9_bj1210_spl_data.npy"), dfn.fn(r"shh2_spl9_bj1210_spl.csv"))
    shh_spl.initCategory("CATEGORY_NAME", SHHConfig.CNAME_MAP_SH881, fn=dfn.fn(r"sh2_spl6_1_2_spl1_1.csv"))

    shh_spl.addCSV(dfn.fn(r"shh2_spl7_2_spl.csv"))
    shh_spl.loadNpy(dfn.fn(r"shh2_spl7_2_spl_data.npy"), dfn.fn(r"shh2_spl7_2_spl.csv"))
    shh_spl.initCategory("CNAME", SHHConfig.CNAME_MAP_SH881, fn=dfn.fn(r"sh2_spl6_1_2_spl1_1.csv"))

    shh_spl.initXY()

    shh_spl.filter_fns["qd_random1_1000"] = 0
    shh_spl.filter_fns["bj_random1_1000"] = 2
    shh_spl.filter_fns["shadow1"] = 4

    return shh_spl


def copySHH2Samples(shh2_spl: SHH2Samples):
    shh2_spl_copy = SHH2Samples()
    shh2_spl_copy.y_field_name = shh2_spl.y_field_name
    shh2_spl_copy.x_field_name = shh2_spl.x_field_name
    shh2_spl_copy.filenames = shh2_spl.filenames.copy()
    shh2_spl_copy.FN_FIELD = shh2_spl.FN_FIELD
    shh2_spl_copy.filter_fns = shh2_spl.filter_fns
    shh2_spl_copy.ndc = shh2_spl.ndc
    return shh2_spl_copy


class SHH2Samples_Dataset(Dataset):

    def __init__(self, shh_spl: SHH2Samples, data_deal=None):
        super(SHH2Samples_Dataset, self).__init__()
        self.shh_spl = shh_spl
        if data_deal is None:
            def data_deal_tmp(x, y=None):
                return x, y

            data_deal = data_deal_tmp
        self.data_deal = data_deal
        self.ndc = self.shh_spl.ndc

        self.is_deal = False

    def deal(self):
        for spl in self.shh_spl:
            x = self.ndc.fit(spl.data)
            y = int(spl.code)
            x, y = self.data_deal(x, y)
            spl.data = x
            spl.code = y
        self.is_deal = True

    def __getitem__(self, index):
        if self.is_deal:
            spl = self.shh_spl[index]
            return spl.data, spl.code
        else:
            x = self.shh_spl[index].data
            x = self.ndc.fit(x)
            y = int(self.shh_spl[index].code)
            x, y = self.data_deal(x, y)
            return x, y

    def __len__(self):
        return len(self.shh_spl)


class SHH2Samples_Dataset2(Dataset):

    def __init__(self, shh_spl: SHH2Samples, data_deal=None):
        super(SHH2Samples_Dataset2, self).__init__()
        self.shh_spl = shh_spl
        if data_deal is None:
            def data_deal_tmp(x, y=None):
                return x, y

            data_deal = data_deal_tmp
        self.data_deal = data_deal
        self.ndc = self.shh_spl.ndc

    def __getitem__(self, index):
        x = self.shh_spl[index].data
        x = self.ndc.fit(x)
        y = int(self.shh_spl[index].code)
        x, y = self.data_deal(x, y)
        return x, y

    def __len__(self):
        return len(self.shh_spl)


def loadSHH2SamplesDataset(shh_spl: SHH2Samples, is_test=True, data_deal=None):
    if is_test:
        train_shh_spl = copySHH2Samples(shh_spl).addSamples(shh_spl.filterEQ("TEST", 1))
        test_shh_spl = copySHH2Samples(shh_spl).addSamples(shh_spl.filterEQ("TEST", 0))
        train_ds = SHH2Samples_Dataset(train_shh_spl, data_deal=data_deal)
        test_ds = SHH2Samples_Dataset(test_shh_spl, data_deal=data_deal)
        return train_ds, test_ds
    else:
        return SHH2Samples_Dataset(shh_spl, data_deal=data_deal)


class SHH2_SPL:

    def __init__(self, category_field_name=None, map_dict: dict = None, others=None, is_npy=True):
        self.shh2_spl = SHH2Samples()
        self.dfn_release = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Samples\Release")

        self.category_field_name = category_field_name
        self.map_dict = map_dict
        self.others = others
        self.is_npy = is_npy

    def __len__(self):
        return len(self.shh2_spl)

    def initAddFunc(self, category_field_name=None, map_dict: dict = None, others=None):
        if category_field_name is None:
            category_field_name = self.category_field_name
        if map_dict is None:
            map_dict = self.map_dict
        if others is None:
            others = self.others
        return category_field_name, map_dict, others

    def csvToRelease(self, csv_fn, to_name=None, npy_fn=None, grs: GDALRastersSampling = None, win_size=(1, 1)):
        if to_name is None:
            to_name = FN(csv_fn).getfilenamewithoutext()
        to_dict = {"csv_fn": csv_fn, "to_name": csv_fn, "win_size": win_size}
        print("To Name:", to_name)
        to_dirname = self.dfn_release.fn(to_name)
        to_dirname = mkdir(to_dirname)
        print("To Dirname:", to_dirname)
        dfn = DirFileName(to_dirname)
        to_csv_fn = dfn.fn("spl.csv")
        to_npy_fn = dfn.fn("data.npy")
        if os.path.isfile(to_csv_fn):
            is_y = input("CSV file {0} exist.\nWhether rewrite or not [y/n]: ".format(to_csv_fn))
            if not is_y.startswith("y"):
                return to_name
        copyfile(csv_fn, to_csv_fn)
        if npy_fn is not None:
            copyfile(npy_fn, to_npy_fn)
            to_dict["npy_fn"] = npy_fn
        else:
            if grs is not None:
                to_dict["grs"] = grs.toDict()
                shh2_spl = SHH2Samples()
                shh2_spl.addCSV(csv_fn)
                shh2_spl.initXY()
                shh2_spl.sampling(grs, fn=csv_fn, win_size=win_size, )
                shh2_spl.toNpy(to_npy_fn, fn=csv_fn)
                shh2_spl.toCSV(to_csv_fn)
        saveJson(to_dict, os.path.join(to_dirname, "csvtorelease.json"))
        return to_name

    def add(self, csv_fn, npy_fn=None, field_datas: dict = None,
            category_field_name=None, map_dict: dict = None, others=None, **kwargs):
        category_field_name, map_dict, others = self.initAddFunc(category_field_name, map_dict, others)
        self.shh2_spl.addCSV(csv_fn=csv_fn, npy_fn=npy_fn, field_datas=field_datas)
        self.shh2_spl.initCategory(field_name=category_field_name, map_dict=map_dict, fn=csv_fn, others=others)
        if "selects" in kwargs:
            spls = self.shh2_spl.randomSelect(kwargs["selects"], fn=csv_fn)
            self.shh2_spl = copySHH2Samples(self.shh2_spl).addSamples(spls)
        return self

    def addRelease(self, name, is_npy=None, field_datas: dict = None, category_field_name=None, map_dict: dict = None,
                   others=None, **kwargs):
        csv_fn = self.dfn_release.fn(name, "spl.csv")
        npy_fn = self.dfn_release.fn(name, "data.npy")
        if is_npy is None:
            is_npy = self.is_npy
        if not is_npy:
            npy_fn = None
        return self.add(csv_fn=csv_fn, npy_fn=npy_fn, field_datas=field_datas, category_field_name=category_field_name,
                        map_dict=map_dict, others=others, **kwargs)

    def add_shadow1samples(self, is_npy=None, field_datas: dict = None, category_field_name=None, map_dict: dict = None,
                           others=None, **kwargs):
        if category_field_name is None:
            category_field_name = "CATEGORY"
        return self.addRelease("shadow1samples", is_npy=is_npy, field_datas=field_datas,
                               category_field_name=category_field_name,
                               map_dict=map_dict, others=others, **kwargs)

    def add_qd_random1000_1(self, is_npy=None, field_datas: dict = None, category_field_name=None,
                            map_dict: dict = None, others=None, **kwargs):
        return self.addRelease("qd_random1000_1", is_npy=is_npy, field_datas=field_datas,
                               category_field_name=category_field_name, map_dict=map_dict, others=others, **kwargs)

    def add_bj_random1000_1(self, is_npy=None, field_datas: dict = None, category_field_name=None,
                            map_dict: dict = None, others=None, **kwargs):
        return self.addRelease("bj_random1000_1", is_npy=is_npy, field_datas=field_datas,
                               category_field_name=category_field_name, map_dict=map_dict, others=others, **kwargs)

    def add_qd_VHL_random2000(self, is_npy=None, field_datas: dict = None, category_field_name=None,
                              map_dict: dict = None, others=None, **kwargs):
        return self.addRelease("qd_VHL_random2000_1", is_npy=is_npy, field_datas=field_datas,
                               category_field_name=category_field_name, map_dict=map_dict, others=others, **kwargs)

    def add_qd_VHL_random10000(self, is_npy=None, field_datas: dict = None, category_field_name=None,
                               map_dict: dict = None, others=None, **kwargs):
        return self.addRelease("qd_VHL_random10000_1", is_npy=is_npy, field_datas=field_datas,
                               category_field_name=category_field_name, map_dict=map_dict, others=others, **kwargs)

    def add_qd_roads_shouhua_tp1(self, is_npy=None, field_datas: dict = None, category_field_name=None,
                                 map_dict: dict = None, others=None, **kwargs):
        return self.addRelease("qd_roads_shouhua_tp1", is_npy=is_npy, field_datas=field_datas,
                               category_field_name=category_field_name, map_dict=map_dict, others=others, **kwargs)

    def add_bj_vhl_random2000(self, is_npy=None, field_datas: dict = None, category_field_name="CATEGORY_CODE",
                              map_dict: dict = None, others=None, **kwargs):

        return self.addRelease("bj_vhl_random2000", is_npy=is_npy, field_datas=field_datas,
                               category_field_name=category_field_name, map_dict=map_dict, others=others, **kwargs)

    def add_bj_vhl_random10000(self, is_npy=None, field_datas: dict = None, category_field_name="CATEGORY",
                               map_dict: dict = None, others=None, **kwargs):
        return self.addRelease("bj_vhl_random10000", is_npy=is_npy, field_datas=field_datas,
                               category_field_name=category_field_name, map_dict=map_dict, others=others, **kwargs)

    def add_cd_vhl_random2000(self, is_npy=None, field_datas: dict = None, category_field_name="CATEGORY_CODE",
                              map_dict: dict = None, others=None, **kwargs):
        return self.addRelease("cd_vhl_random2000", is_npy=is_npy, field_datas=field_datas,
                               category_field_name=category_field_name, map_dict=map_dict, others=others, **kwargs)

    def add_qd_is_random2000(self, is_npy=None, field_datas: dict = None, category_field_name="CATEGORY_CODE",
                             map_dict: dict = None, others=None, **kwargs):
        return self.addRelease("qd_is_random2000", is_npy=is_npy, field_datas=field_datas,
                               category_field_name=category_field_name, map_dict=map_dict, others=others, **kwargs)

    def filterCODEContain(self, *code):
        self.shh2_spl = copySHH2Samples(self.shh2_spl).addSamples(self.shh2_spl.filterCODEContain(*code))
        return self.shh2_spl

    def filterEq(self, field_name, data):
        self.shh2_spl = copySHH2Samples(self.shh2_spl).addSamples(self.shh2_spl.filterEQ(field_name, data))
        return self

    def filterNotEq(self, field_name, data):
        self.shh2_spl = copySHH2Samples(self.shh2_spl).addSamples(self.shh2_spl.filterNotEQ(field_name, data))
        return self

    def loadSHH2SamplesDataset(self, is_test=True, data_deal=None):
        return loadSHH2SamplesDataset(self.shh2_spl, is_test=is_test, data_deal=data_deal)

    @staticmethod
    def qd_random_1000_1(d_type="shh2_spl"):
        """ d_type: shh2_spl|df """
        dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Samples\Release")
        shh_spl = SHH2Samples()

        shh_spl.addCSV(dfn.fn(r"sh2_spl6_1_2_spl1_1.csv"), field_datas={"CITY": "qd", "TEST": 1})
        if d_type == "df":
            return shh_spl.toDF()

        shh_spl.loadNpy(dfn.fn(r"sh2_spl6_1_2_spl1_1_data.npy"), dfn.fn(r"sh2_spl6_1_2_spl1_1.csv"))
        shh_spl.initCategory("CATEGORY_NAME", SHHConfig.CNAME_MAP_SH881, fn=dfn.fn(r"sh2_spl6_1_2_spl1_1.csv"))

        if d_type == "shh2_spl":
            return shh_spl

    @staticmethod
    def shh1(d_type="shh2_spl"):
        dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Samples\Release")
        shh_spl = SHH2Samples()

        shh_spl.addCSV(dfn.fn(r"shh2_spl7_2_spl.csv"))
        if d_type == "df":
            return shh_spl.toDF()

        shh_spl.loadNpy(dfn.fn(r"shh2_spl7_2_spl_data.npy"), dfn.fn(r"shh2_spl7_2_spl.csv"))
        shh_spl.initCategory("CNAME", SHHConfig.CNAME_MAP_SH881, fn=dfn.fn(r"sh2_spl6_1_2_spl1_1.csv"))

        if d_type == "shh2_spl":
            return shh_spl


def main():
    # shh_sc = ShadowHierarchicalSampleCollection()
    # shh_sc.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\Release\shh2_spl9_bj1210.csv")
    # shh_sc.shhSampling1()
    # shh_sc.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\6\sh2_spl6_1_215_spl.csv")
    # shh_sc.toCSV()

    def func1():
        # "F:\ProjectSet\Shadow\Hierarchical\Samples\Release\sh2_spl6_1_2_spl1_1.csv"
        # "F:\ProjectSet\Shadow\Hierarchical\Samples\Release\shh2_spl9_bj1210_spl.csv"
        dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Samples\Release")
        shh_sc = SHH2Samples()
        shh_sc.addCSV(dfn.fn(r"sh2_spl6_1_2_spl1_1.csv"), field_datas={"CITY": "qd"})
        shh_sc.addCSV(dfn.fn(r"shh2_spl9_bj1210_spl.csv"), field_datas={"CITY": "bj"})
        shh_sc.addCSV(dfn.fn(r"shh2_spl7_2_spl.csv"))
        shh_sc.initXY()

        # shh_sc.sampling(SHHConfig.GRS_SHH2_IMAGE1_FNS(), (21, 21))
        # shh_sc.toNpy(dfn.fn(r"sh2_spl6_1_2_spl1_1_data.npy"), dfn.fn(r"sh2_spl6_1_2_spl1_1.csv"), )
        # shh_sc.toNpy(dfn.fn(r"shh2_spl9_bj1210_spl_data.npy"), dfn.fn(r"shh2_spl9_bj1210_spl.csv"), )
        # shh_sc.toNpy(dfn.fn(r"shh2_spl7_2_spl_data.npy"))

        # shh_sc.sampling(SHHConfig.GRS_SHH2_IMAGE1_FNS(), (1, 1), is_to_field=True)
        # shh_sc.sampling(SHHConfig.GRS_SHH2_IMAGE1_GLCM_FNS(), (1, 1), is_to_field=True)
        # shh_sc.sampling(SHHConfig.GRS_SHH2_IMAGE1_ESA21_FNS(), (1, 1), is_to_field=True)

        shh_sc.toCSV(numberfilename(dfn.fn(r"shh2_spl7_2_spl.csv")))

        # dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Samples")
        # shh_sc = ShadowHierarchicalSampleCollection()
        # shh_sc.readJson(dfn.fn(r"7\shh2_spl7_2_spl.json"))
        # # shh_sc.loadDataFromNPY(dfn.fn(r"7\shh2_spl7_2_data.npy"))
        # shh_sc.toCSV(r"F:\ProjectSet\Shadow\Hierarchical\Samples\Release\shh2_spl7_2_spl.csv")

    shh_spl = loadSHH2Samples()
    shh_spl.toCSV(r"F:\ProjectSet\Shadow\Hierarchical\Samples\Release\shh2_spl.csv")
    return


def method_name():
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

    def func5():
        df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\Release\shh2_spl.csv")
        grs = GDALRastersSampling(SHHConfig.SHH2_IMAGE1_FNS)
        win_r, win_c = 21, 21
        print((len(df), grs._gr.n_channels, 21, 21))
        data = np.ones((len(df), grs._gr.n_channels, 21, 21))
        jdt = Jdt(len(df)).start()
        for i in range(len(df)):
            x, y = float(df["X"][i]), float(df["Y"][i])
            d = grs.sampling(x, y, win_r, win_c, )
            if d is None:
                print(i, x, y)
            else:
                data[i] = d
            jdt.add()
        jdt.end()
        np.save(r"F:\ProjectSet\Shadow\Hierarchical\Samples\SampleData\shh2_spl\shh2_spl.npy", data)
        df.to_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\SampleData\shh2_spl\shh2_spl.csv")

    def func6():
        SHH2_SPL().csvToRelease(
            csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\24\sh2_spl24_is_spl2.csv",
            to_name="qd_is_random2000",
            grs=SHHConfig.GRS_SHH2_IMAGE1_FNS(),
            win_size=(21, 21),
        )

    def func7():
        s2spl = SHH2_SPL(map_dict=SHHConfig.CATE_MAP_SH881, others=0, is_npy=False)
        s2spl.add_shadow1samples(category_field_name="CATEGORY")
        s2spl.add_qd_random1000_1(category_field_name="CATEGORY_CODE")
        s2spl.add_bj_random1000_1(category_field_name="CATEGORY_CODE")

        s2spl.filterEq("CITY", "qd")
        print(len(s2spl))
        s2spl.shh2_spl.toCSV(r"F:\ProjectSet\Shadow\Hierarchical\Samples\14\shh2_spl14_1.csv",
                             category_field_name="CATEGORY2", cname_field_name="CNAME2")

    func6()
    # shh_sc_train, shh_sc_test = loadSHHSamples("qd_sample1[21,21]")


def initSHHGRS(grs_type, raster_fns=None):
    if raster_fns is None:
        if grs_type == "qd_sh1":
            rasters_sampling = GDALRastersSampling(
                r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\qd_sh2_1.tif")
        elif grs_type == "qd_esa21":
            rasters_sampling = GDALRastersSampling(
                r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\qd_sh2_esa.tif")
        else:
            raise Exception("Can not find grs type \"{0}\"".format(grs_type))
    else:
        rasters_sampling = GDALRastersSampling(*tuple(raster_fns))
    return rasters_sampling


def samplingSHH21OptSarGLCM(csv_fn, to_csv_fn=None):
    shh_spl = SHH2Samples()
    if to_csv_fn is None:
        to_csv_fn = numberfilename(csv_fn, sep="_spl")
    print("csv_fn:", csv_fn)
    print("to_csv_fn:", to_csv_fn)
    shh_spl.addCSV(csv_fn)
    shh_spl.initXY()
    shh_spl.sampling(grs=SHHConfig.GRS_SHH2_IMAGE1_FNS2(), is_to_field=True)
    shh_spl.toCSV(to_csv_fn)


if __name__ == "__main__":
    method_name()
