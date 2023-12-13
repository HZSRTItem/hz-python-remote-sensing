# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHDLData.py
@Time    : 2023/11/29 11:27
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHDLData
-----------------------------------------------------------------------------"""
import csv
import os.path
import time

import numpy as np
import pandas as pd
from osgeo import gdal_array

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.Utils import Jdt, saveJson, changext

NOT_KNOW_CNAME = "NOT_KNOW"
NOT_KNOW_CODE = 0


def dataDeal(x):
    return x


class SHDLDataSample:

    def __init__(self, cname=NOT_KNOW_CNAME, category=NOT_KNOW_CODE, x=0.0, y=0.0,
                 oid=-1, tag="", c_tag="train", fields: dict = None, data=None, **kwargs):
        self.cname = cname
        self.category = category
        self.x = x
        self.y = y
        self.oid = oid
        self.tag = tag
        self.c_tag = c_tag
        self.fields = {}
        self.addFields(fields, **kwargs)
        self.data = None
        self.setData(data)

    def addFields(self, fields: dict = None, **kwargs):
        for k in fields:
            self.fields[k] = fields[k]
        for k in kwargs:
            self.fields[k] = kwargs[k]

    def setData(self, data):
        self.data = data

    def getData(self, is_data_deal=True):
        if is_data_deal:
            return dataDeal(self.data)
        else:
            return self.data

    def __getitem__(self, item):
        return self.fields[item]

    def __len__(self):
        return self.fields

    def __contains__(self, item):
        return item in self.fields

    def toDict(self):
        to_dict = self.fields.copy()
        to_dict["CNAME"] = self.cname
        to_dict["CATEGORY"] = self.category
        to_dict["X"] = self.x
        to_dict["Y"] = self.y
        to_dict["SRT"] = self.oid
        to_dict["TAG"] = self.tag
        to_dict["CTAG"] = self.c_tag
        return to_dict


class SHDLDataCategorys:

    def __init__(self, cname=NOT_KNOW_CNAME, category=NOT_KNOW_CODE):
        self.cnames = [cname]
        self.categorys = [category]

    def addCategory(self, cname=NOT_KNOW_CNAME, category=NOT_KNOW_CODE):
        if cname in self.cnames:
            return cname, self.get(cname)
        self.cnames.append(cname)
        if category in self.categorys:
            raise Exception("category: {0} have in categorys.".format(category))
        self.categorys.append(category)
        return cname, category

    def __getitem__(self, item):
        return self.get(item)

    def get(self, item):
        ret = None
        if isinstance(item, int):
            if item in self.categorys:
                ret = self.cnames[self.categorys.index(item)]
        elif isinstance(item, str):
            if item in self.cnames:
                ret = self.categorys[self.cnames.index(item)]
        return ret

    def toString(self):
        to_str = ""
        n = len(self.cnames[0])
        for cname in self.cnames:
            if n < len(cname):
                n = len(cname)
        fmt = "{" + ":" + str(n) + "}"
        for i in range(len(self.cnames)):
            to_str += fmt.format(self.cnames[i]) + " " + str(self.categorys[i]) + "\n"
        return to_str

    def getSaveDict(self):
        return {"cnames": self.cnames, "category": self.categorys}

    def index(self, cname_category):
        ret = None
        if isinstance(cname_category, int):
            if cname_category in self.categorys:
                ret = self.categorys.index(cname_category)
        elif isinstance(cname_category, str):
            if cname_category in self.cnames:
                ret = self.cnames.index(cname_category)
        return ret


class SHDLDataSampleCollection:

    def __init__(self):
        super().__init__()
        self.samples = []
        self.field_names = []
        self._init_oid = 1
        self.categorys = SHDLDataCategorys()
        self.init_fields = ["NUMBER", "CNAME", "CATEGORY", "X", "Y", "SRT", "TAG", "CTAG"]
        self._n_iter = 0
        self.save_dict = {"filenames": [], "init_fields": self.init_fields,
                          "field_names": self.field_names, "categorys": self.categorys.getSaveDict()}

    def addExcel(self, xlsx_fn, city_name):
        def add_excel(sheet_name):
            df = pd.read_excel(xlsx_fn, sheet_name=sheet_name)
            for i in range(len(df)):
                line = df.loc[i].to_dict()
                line["CITY"] = city_name
                for k in line:
                    if k not in self.field_names:
                        self.field_names.append(k)
                cname, category = self.categorys.addCategory(str(line["CNAME"]), int(line["CATEGORY"]))
                self.addSample(
                    cname=cname,
                    category=category,
                    x=float(line["X"]),
                    y=float(line["Y"]),
                    oid=int(line["SRT"]),
                    tag=str(line["TAG"]),
                    c_tag=sheet_name,
                    fields=line,
                )

        add_excel("Train")
        add_excel("Test")
        add_excel("ShadowTest")

        self.save_dict["filenames"].append(xlsx_fn)

    def addSample(self, cname=NOT_KNOW_CNAME, category=NOT_KNOW_CODE, x=0.0, y=0.0,
                  oid=-1, tag="", c_tag="train", fields: dict = None, data=None, **kwargs):
        if oid == -1:
            oid = self._init_oid
            self._init_oid += 1
        self.samples.append(SHDLDataSample(cname=cname, category=category, x=x, y=y,
                                           oid=oid, tag=tag, c_tag=c_tag, fields=fields, data=data, **kwargs))

    def addCategory(self, cname=NOT_KNOW_CNAME, category=NOT_KNOW_CODE):
        self.categorys.addCategory(cname, category)

    def __getitem__(self, item) -> SHDLDataSample:
        return self.samples[item]

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return self

    def __next__(self) -> SHDLDataSample:
        if self._n_iter == len(self.samples):
            self._n_iter = 0
            raise StopIteration()
        self._n_iter += 1
        return self.samples[self._n_iter - 1]

    def __contains__(self, item):
        return item in self.samples

    def filterCTAG(self, c_tag):
        samples = []
        for i, spl in enumerate(self.samples):
            if spl.c_tag == c_tag:
                samples.append(spl)
        shdk_dsc = self.newFromSamples(samples)
        return shdk_dsc

    def newFromSamples(self, samples):
        shdk_dsc = SHDLDataSampleCollection()
        shdk_dsc.samples = samples
        shdk_dsc.field_names = self.field_names
        shdk_dsc._init_oid = self._init_oid
        shdk_dsc.categorys = self.categorys
        return shdk_dsc

    def getSaveDict(self):
        self.save_dict["init_fields"] = self.init_fields
        self.save_dict["field_names"] = self.field_names
        self.save_dict["categorys"] = self.categorys.getSaveDict()
        return self.save_dict

    def addCSV(self, csv_fn, data_fn=None):
        df = pd.read_csv(csv_fn)
        data = None
        if data_fn is not None:
            data = np.load(data_fn)
        for i in range(len(df)):
            line = df.loc[i].to_dict()
            for k in line:
                if k not in self.field_names:
                    self.field_names.append(k)
            cname, category = self.categorys.addCategory(str(line["CNAME"]), int(line["CATEGORY"]))
            if data is None:
                d = None
            else:
                d = data[int(line["NUMBER"])]
            self.addSample(
                cname=cname,
                category=category,
                x=float(line["X"]),
                y=float(line["Y"]),
                oid=int(line["SRT"]),
                tag=str(line["TAG"]),
                c_tag=str(line["CTAG"]),
                fields=line,
                data=d
            )

        self.save_dict["filenames"].append(csv_fn)


class SHDLGDALRaster(GDALRaster):

    def __init__(self, gdal_raster_fn=""):
        super(SHDLGDALRaster, self).__init__(gdal_raster_fn)

    def readAsArrayCenter(self, x_row_center=0.0, y_column_center=0.0, win_row_size=1, win_column_size=1,
                          interleave='band', band_list=None, is_geo=False, no_data=0, is_trans=False):
        if is_geo:
            if is_trans:
                x_row_center, y_column_center, _ = self.coor_trans.TransformPoint(x_row_center, y_column_center)
            x_row_center, y_column_center = self.coorGeo2Raster(x_row_center, y_column_center)
        x_row_center, y_column_center = int(x_row_center), int(y_column_center)

        row_off0 = x_row_center - int(win_row_size / 2)
        column_off0 = y_column_center - int(win_column_size / 2)

        band_list_int = []
        for c_name in band_list:
            if isinstance(c_name, int):
                band_list_int.append(c_name)
            elif isinstance(c_name, str):
                band_list_int.append(self.names.index(c_name) + 1)

        if 0 <= row_off0 < self.n_rows - win_row_size and 0 <= column_off0 < self.n_columns - win_column_size:
            return gdal_array.DatasetReadAsArray(self.raster_ds, column_off0, row_off0, win_xsize=win_column_size,
                                                 win_ysize=win_row_size, interleave=interleave, band_list=band_list_int)
        return None


class SHDLGDALDataSampling:

    def __init__(self):
        self.samples = SHDLDataSampleCollection()
        self.grs = {}
        self.gr = SHDLGDALRaster()
        self.spl_dirname = os.path.join(r"F:\ProjectSet\Shadow\DeepLearn\Samples", time.strftime("%Y%m%dH%H%M%S"))
        if not os.path.isdir(self.spl_dirname):
            os.mkdir(self.spl_dirname)
        self.save_dict = {"spl_dirname": self.spl_dirname, "geo_filenames": list(self.grs.keys()),
                          "samples": self.samples.getSaveDict()}

    def initSamples(self, samples: SHDLDataSampleCollection):
        self.samples = samples

    def addExcel(self, xlsx_fn, city_name):
        self.samples.addExcel(xlsx_fn, city_name)

    def addGDALFile(self, geo_fn):
        geo_fn = os.path.abspath(geo_fn)
        if geo_fn not in self.grs:
            self.grs[geo_fn] = SHDLGDALRaster(geo_fn)
        self.gr = self.grs[geo_fn]

    def addCategory(self, cname=NOT_KNOW_CNAME, category=NOT_KNOW_CODE):
        self.samples.categorys.addCategory(cname, category)

    def sampling(self, to_csv_fn, to_npy_fn, spl_size=(1, 1), channel_list=None):
        data_list = []
        to_csv_fn = os.path.join(self.spl_dirname, to_csv_fn)
        to_npy_fn = os.path.join(self.spl_dirname, to_npy_fn)
        n = 0
        with open(to_csv_fn, "w", encoding="utf-8", newline="") as fw:
            cw = csv.writer(fw)
            first_line = self.samples.init_fields.copy()
            for k in self.samples.field_names:
                if k not in first_line:
                    first_line.append(k)
            cw.writerow(first_line)
            jdt = Jdt(total=len(self.samples), desc="SHDLGDALDataSampling Sample")
            jdt.start()
            for i, spl in enumerate(self.samples):
                spl: SHDLDataSample
                d = self.gr.readAsArrayCenter(
                    x_row_center=spl.x, y_column_center=spl.y, win_row_size=spl_size[0], win_column_size=spl_size[1],
                    interleave='band', band_list=channel_list, is_geo=True, no_data=0, is_trans=False)
                if d is None:
                    for k in self.grs:
                        self.gr = self.grs[k]
                        d = self.gr.readAsArrayCenter(
                            x_row_center=spl.x, y_column_center=spl.y, win_row_size=spl_size[0],
                            win_column_size=spl_size[1],
                            interleave='band', band_list=channel_list, is_geo=True, no_data=0, is_trans=False)
                        if d is not None:
                            break
                if d is not None:
                    data_list.append(d)
                    line = [n, spl.cname, spl.category, spl.x, spl.y, spl.oid, spl.tag, spl.c_tag]
                    n += 1
                    for k in first_line:
                        if k not in self.samples.init_fields:
                            line.append(spl[k])
                    cw.writerow(line)
                jdt.add()
            jdt.end()
        data = np.array(data_list)
        np.save(to_npy_fn, data)
        save_dict = self.getSaveDict()
        save_dict["spl_size"] = list(spl_size)
        save_dict["channel_list"] = list(channel_list)
        saveJson(self.getSaveDict(), changext(to_csv_fn, ".json"))

    def getSaveDict(self):
        self.save_dict["geo_filenames"] = list(self.grs.keys())
        self.save_dict["samples"] = self.samples.getSaveDict()
        return self.save_dict


# class SHDLDataSet(Dataset):
#
#     def __init__(self, csv_fn, data_fn, ):
#         super(SHDLDataSet, self).__init__()
#
#         self.samples = SHDLDataSampleCollection()
#         self.samples.addCSV(csv_fn=csv_fn, data_fn=data_fn)
#         self.train_spl = self.samples.filterCTAG("Train")
#         self.test_spl = self.samples.filterCTAG("Test")
#         self.test_sh_spl = self.samples.filterCTAG("ShadowTest")
#
#     def __getitem__(self, index):
#         return self.get(index)
#
#     def get(self, index):
#         spl = self.samples[index]
#         x = dataDeal(spl.data)
#         y = self.samples.categorys.index(spl.category) - 1
#         return x, y


def shdlSampling():
    samples = SHDLDataSampleCollection()
    samples.addCategory()
    samples.addCategory("IS", 11)
    samples.addCategory("IS_SH", 12)
    samples.addCategory("VEG", 21)
    samples.addCategory("VEG_SH", 22)
    samples.addCategory("SOIL", 31)
    samples.addCategory("SOIL_SH", 32)
    samples.addCategory("WAT", 41)
    samples.addCategory("WAT_SH", 42)
    samples.addExcel(r"F:\ProjectSet\Shadow\Release\QingDaoSamples\QingDaoSamples.xlsx", "QingDao")
    samples.addExcel(r"F:\ProjectSet\Shadow\Release\BeiJingSamples\BeiJingSamples.xlsx", "BeiJing")
    samples.addExcel(r"F:\ProjectSet\Shadow\Release\ChengDuSamples\ChengDuSamples.xlsx", "ChengDu")
    shdl_gds = SHDLGDALDataSampling()
    shdl_gds.initSamples(samples)
    shdl_gds.addGDALFile(r"F:\ProjectSet\Shadow\Release\QingDaoImages\SH_QD_envi.dat")
    shdl_gds.addGDALFile(r"F:\ProjectSet\Shadow\Release\BeiJingImages\SH_BJ_envi.dat")
    shdl_gds.addGDALFile(r"F:\ProjectSet\Shadow\Release\ChengDuImages\SH_CD_envi.dat")
    shdl_gds.sampling(
        "SHDL_THREE.csv", "SHDL_THREE.npy", (9, 9),
        channel_list=["Blue", "Green", "Red", "NIR", "AS_VV", "AS_VH", "DE_VV", "DE_VH"])


def main():
    # shdlSampling()

    # shdl_ds = SHDLDataSet(
    #     csv_fn=r"F:\ProjectSet\Shadow\DeepLearn\Samples\20231201H094911\SHDL_THREE.csv",
    #     data_fn=r"F:\ProjectSet\Shadow\DeepLearn\Samples\20231201H094911\SHDL_THREE.npy"
    # )
    # x, y = shdl_ds[0]

    pass


if __name__ == "__main__":
    main()
