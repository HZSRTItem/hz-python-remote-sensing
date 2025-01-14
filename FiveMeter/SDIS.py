# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SDIS.py
@Time    : 2024/9/15 14:52
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SDIS
-----------------------------------------------------------------------------"""
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from osgeo import gdal
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tabulate import tabulate

from SRTCodes.GDALRasterClassification import tilesRasterImdc
from SRTCodes.GDALRasterIO import GDALRaster, tiffAddColorTable
from SRTCodes.GDALUtils import GDALSampling, vrtAddDescriptions
from SRTCodes.ModelTraining import dataModelPredict
from SRTCodes.NumpyUtils import reHist0
from SRTCodes.SRTModel import RF_RGS
from SRTCodes.Utils import FRW, SRTWriteText, saveJson, DirFileName, readJson, printList, samplesFilterOR, \
    samplesFilterAnd, Jdt, TableLinePrint, numberfilename, getfilenme

S2_SELECT_NAMES = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12",
    "ND1", "NDVI", "NDWI", "ND4", "ND5", "ND6", "ND7", "ND8", "ND9", "ND10", "ND11", "ND12", "ND13", "ND14", "ND15",
    "Gray_asm", "Gray_contrast", "Gray_corr", "Gray_var", "Gray_idm", "Gray_ent", "Gray_diss",
    "NDWI_stdDev", "NDVI_stdDev", "MNDWI_stdDev", "NDBI_stdDev", "NDVI_max"
]
S1_AS_SELECT_NAMES = [
    "AS_VV", "AS_VH", "AS_VH_VV",
    "AS_VV_stdDev", "AS_VH_stdDev",
    "AS_VV_asm", "AS_VV_contrast", "AS_VV_corr", "AS_VV_var", "AS_VV_idm", "AS_VV_ent", "AS_VV_diss",
    "AS_VH_asm", "AS_VH_contrast", "AS_VH_corr", "AS_VH_var", "AS_VH_idm", "AS_VH_ent", "AS_VH_diss",
]
S1_DE_SELECT_NAMES = [
    "DE_VV", "DE_VH", "DE_VH_VV",
    "DE_VV_stdDev", "DE_VH_stdDev",
    "DE_VV_asm", "DE_VV_contrast", "DE_VV_corr", "DE_VV_var", "DE_VV_idm", "DE_VV_ent", "DE_VV_diss",
    "DE_VH_asm", "DE_VH_contrast", "DE_VH_corr", "DE_VH_var", "DE_VH_idm", "DE_VH_ent", "DE_VH_diss",
]

_DS_GISA30m = r"F:\ChinaNorthIS\Run\SDDatasets\GISA30m.tif"
_DS_GISA10m = r"F:\ChinaNorthIS\Run\SDDatasets\GISA10m.tif"
_DS_GAIA = r"F:\ChinaNorthIS\Run\SDDatasets\GAIA.tif"
_DS_FROM_GLC10 = r"F:\ChinaNorthIS\Run\SDDatasets\FROM_GLC10.tif"
_DS_ESRILC23 = r"F:\ChinaNorthIS\Run\SDDatasets\ESRILC23.tif"
_DS_ESRILC22 = r"F:\ChinaNorthIS\Run\SDDatasets\ESRILC22.tif"
_DS_ESRILC21 = r"F:\ChinaNorthIS\Run\SDDatasets\ESRILC21.tif"
_DS_ESRILC20 = r"F:\ChinaNorthIS\Run\SDDatasets\ESRILC20.tif"
_DS_ESRILC19 = r"F:\ChinaNorthIS\Run\SDDatasets\ESRILC19.tif"
_DS_ESRILC18 = r"F:\ChinaNorthIS\Run\SDDatasets\ESRILC18.tif"
_DS_ESRILC17 = r"F:\ChinaNorthIS\Run\SDDatasets\ESRILC17.tif"
_DS_ESA21 = r"F:\ChinaNorthIS\Run\SDDatasets\ESA21.tif"
_DS_ESA20 = r"F:\ChinaNorthIS\Run\SDDatasets\ESA20.tif"
_DS_DW24 = r"F:\ChinaNorthIS\Run\SDDatasets\DW24.tif"
_DS_DW23 = r"F:\ChinaNorthIS\Run\SDDatasets\DW23.tif"
_DS_DW22 = r"F:\ChinaNorthIS\Run\SDDatasets\DW22.tif"
_DS_DW21 = r"F:\ChinaNorthIS\Run\SDDatasets\DW21.tif"
_DS_DW20 = r"F:\ChinaNorthIS\Run\SDDatasets\DW20.tif"
_DS_DW19 = r"F:\ChinaNorthIS\Run\SDDatasets\DW19.tif"
_DS_DW18 = r"F:\ChinaNorthIS\Run\SDDatasets\DW18.tif"
_DS_DW17 = r"F:\ChinaNorthIS\Run\SDDatasets\DW17.tif"
_DS_DW16 = r"F:\ChinaNorthIS\Run\SDDatasets\DW16.tif"

_DS_ALL = r"F:\ChinaNorthIS\Run\SDDatasets\sd_datasets2.vrt"

_SAMPLES_INIT_FN = r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_random10w_INIT.csv"
_HB_SAMPLES_INIT_FN = r"F:\ChinaNorthIS\Run\Samples\hb\hb_spl_random_INIT.csv"


class _SDIS_Sampling:

    def __init__(self, df=None):
        if df is None:
            df = pd.read_csv(_SAMPLES_INIT_FN)
        self.df = df

    def sampling(self, name, raster_fn):
        gs = GDALSampling(raster_fn)
        spl = gs.sampling(self.df["X"].tolist(), self.df["Y"].tolist(), is_jdt=name)
        spl = list(spl.items())
        self.df[name] = spl[0][1]
        return self

    def samplingYears(self, name="ESA", years=None, years_fns=None):
        for year in years:
            self.sampling("{}_{}".format(name, year), years_fns[year])
        return self

    def GISA10m(self, name="GISA10m"):
        return self.sampling(name, _DS_GISA10m)

    def GISA30m(self, name="GISA30m"):
        return self.sampling(name, _DS_GISA30m)

    def GAIA(self, name="GAIA"):
        return self.sampling(name, _DS_GAIA)

    def FROM_GLC10(self, name="FROM_GLC10"):
        return self.sampling(name, _DS_FROM_GLC10)

    def ESA(self, name="ESA", years=None):
        if years is None:
            years = [20, 21]
        return self.samplingYears(name, years, {20: _DS_ESA20, 21: _DS_ESA21, })

    def ESRILC10(self, name="ESRILC10", years=None):
        if years is None:
            years = [17, 18, 19, 20, 21, 22, 23]
        return self.samplingYears(name, years, {
            17: _DS_ESRILC17, 18: _DS_ESRILC18, 19: _DS_ESRILC19, 20: _DS_ESRILC20,
            21: _DS_ESRILC21, 22: _DS_ESRILC22, 23: _DS_ESRILC23,
        })

    def DW(self, name="DW", years=None):
        if years is None:
            years = [16, 17, 18, 19, 20, 21, 22, 23, 24]
        return self.samplingYears(name, years, {
            16: _DS_DW16, 17: _DS_DW17, 18: _DS_DW18, 19: _DS_DW19, 20: _DS_DW20,
            21: _DS_DW21, 22: _DS_DW22, 23: _DS_DW23, 24: _DS_DW24,
        })

    def DSALL(self):
        self.GISA10m()
        self.GISA30m()
        self.GAIA()
        self.FROM_GLC10()
        self.ESA()
        self.ESRILC10()
        self.DW()


class _SDIS_Samples:

    def __init__(self):
        self.samples = {}
        self.field_names = []

    def addSample(self, spl):
        for name in spl:
            if name not in self.field_names:
                self.field_names.append(name)

        if spl["SRT"] in self.samples:
            for name in spl:
                self.samples[spl["SRT"]][name] = spl[name]
        else:
            for name in self.field_names:
                if name not in spl:
                    spl[name] = None
            self.samples[spl["SRT"]] = spl
        return self.samples[spl["SRT"]]

    def addGeoJson(self, json_fn):
        data = FRW(json_fn).readJson()
        features = data["features"]
        for feat in features:
            spl = feat["properties"]
            spl["X"] = feat["geometry"]["coordinates"][0]
            spl["Y"] = feat["geometry"]["coordinates"][1]
            self.addSample(spl)

    def addGeoJsons(self, *json_fns):
        for json_fn in json_fns:
            self.addGeoJson(json_fn)

    def df(self):
        return pd.DataFrame(self.samples.values())

    def checkNone(self):
        print("checkNone")
        for n, spl in self.samples.items():
            is_none = []
            for name in spl:
                if spl[name] is None:
                    is_none.append(name)
            if is_none:
                print("{:<20d}: {}".format(n, is_none))

    def addTest(self, ratio=0.9):
        if "TEST" not in self.field_names:
            self.field_names.append("TEST")
        for n in self.samples:
            self.samples[n]["TEST"] = int(random.random() < ratio)

    def __getitem__(self, item):
        return self.samples[item]


class _SDIS_SamplesAcquisition:

    def __init__(self, samples=None):
        if samples is None:
            samples = pd.read_csv(_SAMPLES_INIT_FN).to_dict("records")
        if isinstance(samples, str):
            samples = pd.read_csv(samples).to_dict("records")
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]

    def fileterAnd(self, *filters):
        self.samples = samplesFilterAnd(self.samples, *filters)

    def fileterOR(self, *filters):
        self.samples = samplesFilterOR(self.samples, *filters)

    def df(self):
        return pd.DataFrame(self.samples)

    def addISField(self, name, _func):
        for spl in self.samples:
            spl[name] = _func(spl)

    def map(self, _map_func):
        for i, spl in enumerate(self.samples):
            self.samples[i] = _map_func(spl)

    def addFieldSum(self, name, *field_names):
        def _sum_func(_spl):
            _spl[name] = sum(_spl[field_name] for field_name in field_names)
            return _spl

        self.map(_sum_func)

    def update(self, samples, update_field_names, _find_field_name="SRT", is_jdt=True):
        jdt = Jdt(len(samples), "_SDIS_SamplesAcquisition::update").start(is_jdt=is_jdt)
        for spl in samples:
            for spl_tmp in self.samples:
                if spl_tmp["SRT"] == spl["SRT"]:
                    if update_field_names is None:
                        for name in spl:
                            spl_tmp[name] = spl[name]
                    else:
                        for name in update_field_names:
                            spl_tmp[name] = spl[name]
                    break
            jdt.add()
        jdt.end()

    def hist(self, field_name, *args, bins=256, filters_and=None, filters_or=None, **kwargs):
        samples = self._filterSamples(filters_and, filters_or)
        data = np.array([_spl[field_name] for _spl in samples])
        print("_SDIS_SamplesAcquisition::hist - length of data", len(data))
        _y, _x = np.histogram(data, bins=bins, density=True)
        plt.plot(_x[:-1] + (_x[0] + _x[1]) / 2, _y, *args, **kwargs)

    def _filterSamples(self, filters_and, filters_or):
        samples = self.samples
        if filters_and is not None:
            samples = samplesFilterAnd(samples, *filters_and)
        if filters_or is not None:
            samples = samplesFilterOR(samples, *filters_or)
        return samples

    def calculate(self, field_names, _func, filters_and=None, filters_or=None, ):
        samples = self._filterSamples(filters_and, filters_or)
        if isinstance(field_names, str):
            return _func([_spl[field_names] for _spl in samples])
        to_dict = {}
        for field_name in field_names:
            to_dict[field_name] = _func([_spl[field_name] for _spl in samples])
        return to_dict

    def mean(self, field_names, filters_and=None, filters_or=None, ):
        return self.calculate(field_names, np.mean, filters_and, filters_or)

    def min(self, field_names, filters_and=None, filters_or=None, ):
        return self.calculate(field_names, np.min, filters_and, filters_or)

    def max(self, field_names, filters_and=None, filters_or=None, ):
        return self.calculate(field_names, np.max, filters_and, filters_or)

    def std(self, field_names, filters_and=None, filters_or=None, ):
        return self.calculate(field_names, np.std, filters_and, filters_or)

    def groupStatusCounts(self, row_field_name, column_field_name, is_show=True):
        row_names = list(set(_spl[row_field_name] for _spl in self.samples))
        row_names.sort()
        column_names = list(set(_spl[column_field_name] for _spl in self.samples))
        column_names.sort()
        to_dict = {column_name: {row_name: 0 for row_name in row_names} for column_name in column_names}
        for _spl in self.samples:
            to_dict[_spl[column_field_name]][_spl[row_field_name]] += 1
        if is_show:
            print(tabulate(pd.DataFrame(to_dict), headers="keys", tablefmt="simple", ))
        return to_dict

    def randomSampling(self, n=-1, filters_and=None, filters_or=None, ):
        samples = self._filterSamples(filters_and, filters_or)
        if n < 0:
            return samples
        if len(samples) <= n:
            warnings.warn("Number of samples is {} <= number of input {}".format(len(samples), n))
            return samples
        else:
            return random.sample(samples, n)


class _SDIS_DatasetsImage:

    def __init__(self, raster_fn=_DS_ALL, read_rows=10000, read_columns=10000, dtype="float32", to_dtype="int8", ):
        self.raster_fn = raster_fn
        self.read_rows = read_rows
        self.read_columns = read_columns
        self.to_dtype = to_dtype
        self.dtype = dtype

    def fit(self, to_imdc_fn, predict_func):
        return tilesRasterImdc(
            self.raster_fn, to_imdc_fn=to_imdc_fn, predict_func=predict_func,
            read_size=(self.read_rows, self.read_columns),
            interval_size=(-60, -60),
            channels=None, tiles_dirname=None, dtype=self.dtype, color_table=None,
            is_jdt=True, to_dtype=self.to_dtype
        )


def sampling():
    def func1():
        spls = _SDIS_Samples()
        spls.addGeoJson(
            r"F:\ChinaNorthIS\Run\Samples\2\drive-download-20240915T051441Z-001\sdspl2_S2_2016-01-01.geojson")
        spls.addGeoJson(
            r"F:\ChinaNorthIS\Run\Samples\2\drive-download-20240915T051441Z-001\sdspl2_S1AS_2016-01-01.geojson")
        print(spls.field_names)
        df = spls.df()
        print(df)
        spls.checkNone()
        # df.to_csv(r"F:\ChinaNorthIS\Run\Samples\2\drive-download-20240915T051441Z-001\sdspl2_2016_S2_S1AS.csv",
        #           index=False)

    def func2():
        GDALSampling(r"F:\ChinaNorthIS\Run\GUB\GUB_Global_2020_sd_10m.tif").csvfile(
            r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_random10w.csv",
            r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_random10w_spl.csv"
        )

    return func2()


def run(*args, **kwargs):
    def func1(year):
        dfn_spl = DirFileName(r"F:\ChinaNorthIS\Run\Samples\2\drive-download-20240915T051441Z-001")
        sw = SRTWriteText(r"F:\ChinaNorthIS\Run\Models\20240916H161229\sdmod2_{}_S2_S1ASDE.txt".format(year), mode="a",
                          is_show=True)
        csv_fn = dfn_spl.fn(r"sdspl2_{}_S2_S1ASDE.csv".format(year))
        print(year)

        if not os.path.isfile(csv_fn):
            spls = _SDIS_Samples()
            spls.addGeoJson(dfn_spl.fn(r"sdspl2_S2_{}-01-01.geojson".format(year)))
            spls.addGeoJson(dfn_spl.fn(r"sdspl2_S1AS_{}-01-01.geojson".format(year)))
            spls.addGeoJson(dfn_spl.fn(r"sdspl2_S1DE_{}-01-01.geojson".format(year)))
            spls.checkNone()
            spls.addTest()

            print(spls.field_names)
            df = spls.df()
            print(df)
            df.to_csv(csv_fn, index=False)

        df = pd.read_csv(csv_fn)

        feature_names = S2_SELECT_NAMES + S1_AS_SELECT_NAMES + S1_DE_SELECT_NAMES

        def _train_test_data(_test):
            _df = df[df["TEST"] == _test]

            _df = _df.dropna(axis=0, how="any")
            _x = _df[feature_names]
            _y = _df["GAIA10m"].tolist()
            return _x, _y

        x_train, y_train = _train_test_data(1)
        x_test, y_test = _train_test_data(0)

        sw.write("x_train", x_train)
        sw.write("x_test", x_test)

        bast_kwargs_list = []
        for i in range(5):
            clf = RF_RGS()
            clf.fit(x_train, y_train)

            sw.write(clf.bast_model)
            sw.write(clf.bast_kwargs)
            sw.write(clf.bast_accuracy)

            acc = clf.score(x_test, y_test)
            sw.write()

            clf.bast_kwargs["accuracy"] = acc
            bast_kwargs_list.append(clf.bast_kwargs)

        sw.write(pd.DataFrame(bast_kwargs_list))

        return

    def func2(years):
        for year in years:
            func1(year)

    def func3(*_args, **_kwargs):
        year = 16
        csv_fn = r"F:\ChinaNorthIS\Run\Samples\hb\update2\hb_update2_16_spl1.csv"

        sw = SRTWriteText(
            r"F:\ChinaNorthIS\Run\Models\20240928H202156\hbmod1_{}_S2_S1ASDE.txt".format(year),
            mode="a", is_show=True
        )

        feature_names = S2_SELECT_NAMES + S1_AS_SELECT_NAMES + S1_DE_SELECT_NAMES

        df = pd.read_csv(csv_fn)

        def _train_test_data(_test):
            _df = df[df["TEST"] == _test]
            _df = _df[feature_names + ["CATEGORY{}".format(year)]]
            _df = _df.dropna(axis=0, how="any")
            _x = _df[feature_names]
            _y = _df["CATEGORY{}".format(year)].tolist()
            return _x, _y

        x_train, y_train = _train_test_data(1)
        x_test, y_test = _train_test_data(0)

        sw.write("x_train", x_train)
        sw.write("x_test", x_test)

        bast_kwargs_list = []

        for i in range(5):
            clf = RF_RGS()
            clf.fit(x_train, y_train)

            sw.write(clf.bast_model)
            sw.write(clf.bast_kwargs)
            sw.write(clf.bast_accuracy)

            acc = clf.score(x_test, y_test)
            sw.write()

            clf.bast_kwargs["accuracy"] = acc
            bast_kwargs_list.append(clf.bast_kwargs)

        sw.write(pd.DataFrame(bast_kwargs_list))

    return func3(*args, **kwargs)


def concat_geojson(json_fn_list, to_json_fn):
    json_list = [readJson(fn) for fn in json_fn_list]
    json_out = json_list[0]
    for json0 in json_list[1:]:
        json_out["features"] += json0["features"]
    saveJson(json_out, to_json_fn)


def funcs():
    def func1():
        dfn = DirFileName(r"F:\ChinaNorthIS\Run\Models\20240916H161229")
        find_dirnames = [
            # dfn.fn("drive-download-20240916T021857Z-001"),
            # dfn.fn("drive-download-20240916T021933Z-001"),
            # dfn.fn("drive-download-20240916T082712Z-001"),
            # dfn.fn("drive-download-20240916T082859Z-001"),
            # dfn.fn("drive-download-20240919T012422Z-001"),
            dfn.fn("imdc1")
        ]
        find_names = []
        for dirname in find_dirnames:
            find_names.extend([os.path.join(dirname, fn) for fn in os.listdir(dirname) if fn.endswith(".tif")])
        for year in range(16, 25):
            dfn_year = DirFileName(dfn.fn("sdimdc2_{}".format(year))).mkdir()
            init_name = "sdimdc{}".format(year)
            for fn in find_names:
                if init_name in fn:
                    to_fn = dfn_year.fn(os.path.basename(fn))
                    if not os.path.isfile(to_fn):
                        to_fn = dfn_year.copyfile(fn)
                        print("gdaladdo {}".format(to_fn))
        return

    def func2():
        init_fmt = "cd {0}\n" \
                   "where /r . *.tif > filelist.txt\n" \
                   "gdalbuildvrt -input_file_list filelist.txt {1}.vrt\n" \
                   "gdal_translate  -ot Byte -of GTiff -co COMPRESS=LZW {1}.vrt G:\\Downloads\\hbimdc2\\{1}_gtif.tif\n" \
                   "gdaladdo G:\\Downloads\\hbimdc2\\{1}_gtif.tif\n"

        dfn = DirFileName(r"G:\Downloads\hbimdc2")
        for year in range(16, 25):
            dfn_year = DirFileName(dfn.fn("hbimdc2_{}".format(year)))
            print(init_fmt.format(dfn_year.fn(), "hbimdc2_{}".format(year)))
            print()

    def func3():
        # tiffAddColorTable(
        #     r"G:\Downloads\drive-download-20240921T012932Z-001\FROM_GLC10_gtif.tif",
        #     code_colors={0: (0.0, 0.0, 0.0),
        #                  1: (163.0, 255.0, 115.0),
        #                  2: (38.0, 115.0, 0.0),
        #                  3: (76.0, 230.0, 0.0),
        #                  4: (112.0, 168.0, 0.0),
        #                  5: (0.0, 92.0, 255.0),
        #                  6: (197.0, 0.0, 255.0),
        #                  7: (255.0, 170.0, 0.0),
        #                  8: (0.0, 255.0, 197.0),
        #                  9: (255.0, 255.0, 255.0)}
        # )
        gr = GDALRaster(r"G:\Downloads\drive-download-20240921T012932Z-001\DW_0.tif")
        init_dfn = DirFileName(r"G:\Downloads\drive-download-20240921T012932Z-001")
        print(gr.names)
        # GDALRaster(init_dfn.fn(fn))
        grs = [GDALRaster(init_dfn.fn(fn)) for fn in os.listdir(init_dfn.fn()) if
               (fn.startswith("DW") and os.path.isfile(init_dfn.fn(fn)))]
        printList("grs", grs)

        init_fmt = "cd {0}\n" \
                   "where /r . *.tif > filelist.txt\n" \
                   "gdalbuildvrt -input_file_list filelist.txt {1}.vrt\n" \
                   "gdal_translate  -ot Byte -of GTiff -co COMPRESS=LZW {1}.vrt {1}_gtif.tif\n" \
                   "gdaladdo {1}_gtif.tif\n"

        to_fmts = []
        names = ['ESRILC17', 'ESRILC18', 'ESRILC19', 'ESRILC20', 'ESRILC21', 'ESRILC22', 'ESRILC23']
        names = ['DW_16', 'DW_17', 'DW_18', 'DW_19', 'DW_20', 'DW_21', 'DW_22', 'DW_23', 'DW_24']
        for name in names:
            dfn = DirFileName(init_dfn.fn(name)).mkdir()
            # for gr in grs:
            #     data = gr.readGDALBand(name)
            #     to_fn = dfn.fn(os.path.split(gr.gdal_raster_fn)[1])
            #     gr.save(data, save_geo_raster_fn=to_fn, fmt="GTiff", dtype=gdal.GDT_Byte, descriptions=[name],
            #             options=["COMPRESS=LZW"])
            #     print(to_fn)
            # to_fmts.append(init_fmt.format(dfn.fn(), name))
            print("copy", dfn.fn("{0}_gtif.tif".format(name)), "{0}.tif".format(name.replace("_", "")))
            tiffAddColorTable(
                dfn.fn("{0}_gtif.tif".format(name)),
                code_colors={0: (65.0, 155.0, 223.0),
                             1: (57.0, 125.0, 73.0),
                             2: (136.0, 176.0, 83.0),
                             3: (122.0, 135.0, 198.0),
                             4: (228.0, 150.0, 53.0),
                             5: (223.0, 195.0, 90.0),
                             6: (196.0, 40.0, 27.0),
                             7: (165.0, 155.0, 143.0),
                             8: (179.0, 159.0, 225.0)}
            )

        for to_fmt in to_fmts:
            print(to_fmt)
            print()

    def func4():
        dfn = DirFileName(r"F:\ChinaNorthIS\Run\SDImdc\1")
        sdis_spl = _SDIS_Sampling(pd.read_csv(r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_random10w_DS2.csv"))
        # years = [16, 17, 18, 19, 20, 21, 22, 23, 24]
        # sdis_spl.samplingYears("SDImdc1", years=years, years_fns={
        #     year: dfn.fn("sdimdc1_{}.tif".format(year)) for year in years
        # })
        sdis_spl.sampling("GUB", r"F:\ChinaNorthIS\Run\GUB\GUB_Global_2020_sd_10m.tif")
        print(sdis_spl.df)
        sdis_spl.df.to_csv(r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_random10w_DS3.csv", index=False)

    def func5():
        sdis_sa = _SDIS_SamplesAcquisition(pd.read_csv(
            r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_random10w_DS5.csv",
        ).to_dict("records"))
        sdis_sa.fileterAnd(("GUB", "==", 1))
        sdis_sa.addISField("GISA10m_IS", lambda _spl: int(_spl["GISA10m"] == 1))
        sdis_sa.addISField("FROM_GLC10_IS", lambda _spl: int(_spl["FROM_GLC10"] == 6))
        sdis_sa.addISField("ESA_20_IS", lambda _spl: int(_spl["ESA_20"] == 50))
        sdis_sa.addISField("ESA_21_IS", lambda _spl: int(_spl["ESA_21"] == 50))
        sdis_sa.addFieldSum("CITY_SUM", "GISA10m_IS", "FROM_GLC10_IS", "ESA_20_IS", "ESA_21_IS")
        df = sdis_sa.df()
        print(df[["SRT", "GISA10m_IS", "FROM_GLC10_IS", "ESA_20_IS", "ESA_21_IS", "CITY_SUM"]])
        df.to_csv(r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_random10w_DS4.csv")
        df = df[["SRT", "X", "Y", "GUB"]]
        df.to_csv(r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_random10w_GUB_City.csv", index=False)

    def func6():
        sdis_sa = _SDIS_SamplesAcquisition(pd.read_csv(
            r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_random10w_DS3.csv",
        ).to_dict("records"))
        # sdis_sa.fileterAnd(("GUB", "==", 0))

        field_names = [
            # "SRT", "X", "Y", "id", "GAIA10m",
            "GISA10m", "GISA30m", "GAIA", "FROM_GLC10", "ESA_20", "ESA_21",
            "ESRILC10_17", "ESRILC10_18", "ESRILC10_19", "ESRILC10_20", "ESRILC10_21", "ESRILC10_22", "ESRILC10_23",
            "DW_16", "DW_17", "DW_18", "DW_19", "DW_20", "DW_21", "DW_22", "DW_23", "DW_24",
            # "SDImdc1_16", "SDImdc1_17",
            # "SDImdc1_18", "SDImdc1_19", "SDImdc1_20", "SDImdc1_21", "SDImdc1_22", "SDImdc1_23", "SDImdc1_24",
            # "GUB"
        ]

        sdis_sa.addISField("GISA10m_IS", lambda _spl: int(_spl["GISA10m"] == 1))
        sdis_sa.addISField("GISA30m_IS", lambda _spl: int(_spl["GISA30m"] != 0))
        sdis_sa.addISField("GAIA_IS", lambda _spl: int(_spl["GAIA"] != 0))
        sdis_sa.addISField("FROM_GLC10_IS", lambda _spl: int(_spl["FROM_GLC10"] == 6))
        sdis_sa.addISField("ESA_20_IS", lambda _spl: int(_spl["ESA_20"] == 50))
        sdis_sa.addISField("ESA_21_IS", lambda _spl: int(_spl["ESA_21"] == 50))
        print("1")
        for i in range(17, 24):
            sdis_sa.addISField("ESRILC10_{}_IS".format(i), lambda _spl: int(_spl["ESRILC10_{}".format(i)] == 5))
        print("2")
        for i in range(16, 25):
            sdis_sa.addISField("DW_{}_IS".format(i), lambda _spl: int(_spl["DW_{}".format(i)] == 6))
        field_names_is = ["{}_IS".format(field_name) for field_name in field_names]
        printList("field_names_is", field_names_is)
        sdis_sa.addFieldSum("CITY_SUM_ALL", *["{}_IS".format(field_name) for field_name in field_names])
        sdis_sa.addISField("IS_OR", lambda _spl: int(_spl["CITY_SUM_ALL"] != 0))

        # sdis_sa.fileterAnd(("NoCITY_SUM", "!=", 0))

        df = sdis_sa.df()
        df.to_csv(r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_random10w_DS5.csv", index=False)

        print(df[["SRT", "CITY_SUM_ALL"]])
        # df = df[["SRT", "X", "Y", "GUB", "NoCITY_SUM", "IS_OR"]]
        # df.to_csv(r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_random10w_GUB_NoCity.csv")

    def func7():
        vrtAddDescriptions(
            r"F:\ChinaNorthIS\Run\SDDatasets\sd_datasets.vrt",
            r"F:\ChinaNorthIS\Run\SDDatasets\sd_datasets2.vrt",
            ["DW16", "DW17", "DW18", "DW19", "DW20", "DW21", "DW22", "DW23", "DW24", "ESA20", "ESA21", "ESRILC17",
             "ESRILC18", "ESRILC19", "ESRILC20", "ESRILC21", "ESRILC22", "ESRILC23", "FROM_GLC10", "GAIA",
             "GISA10m", "GISA30m", ]
        )

    def func8():
        tlp = TableLinePrint()

        names = [
            "DW16", "DW17", "DW18", "DW19", "DW20", "DW21", "DW22", "DW23", "DW24",
            "ESA20", "ESA21",
            "ESRILC17", "ESRILC18", "ESRILC19", "ESRILC20", "ESRILC21", "ESRILC22", "ESRILC23",
            "FROM_GLC10", "GAIA", "GISA10m", "GISA30m",
        ]
        name_dict = {
            0: "DW16", 1: "DW17", 2: "DW18", 3: "DW19", 4: "DW20", 5: "DW21", 6: "DW22", 7: "DW23", 8: "DW24",
            9: "ESA20", 10: "ESA21",
            11: "ESRILC17", 12: "ESRILC18", 13: "ESRILC19", 14: "ESRILC20", 15: "ESRILC21", 16: "ESRILC22",
            17: "ESRILC23",
            18: "FROM_GLC10", 19: "GAIA", 20: "GISA10m", 21: "GISA30m",
        }

        # for i, name in enumerate(names):
        #     print("{}:\"{}\",".format(i, name), end=" ")

        def _predict_func(data):
            data_esa20 = data[9]
            data_esa21 = data[10]
            data_from_glc10 = data[18]
            data_gisa10m = data[20]

            data_is = (data_esa20 == 50) & (data_esa21 == 50) & (data_from_glc10 == 6) & (data_gisa10m == 1)
            data_water = (data_esa20 == 80) & (data_esa21 == 80) & (data_from_glc10 == 5)
            data_farm = (data_esa20 == 40) & (data_esa21 == 40) & (data_from_glc10 == 1)
            data_soil = (data_esa20 == 60) & (data_esa21 == 60) & (data_from_glc10 == 7)
            data_veg = ((data_esa20 == 10) | (data_esa20 == 20) | (data_esa20 == 30) | (data_esa20 == 90) | (
                    data_esa20 == 95) | (data_esa20 == 100)) & \
                       ((data_esa21 == 10) | (data_esa21 == 20) | (data_esa21 == 30) | (data_esa21 == 90) | (
                               data_esa21 == 95) | (data_esa21 == 100)) & \
                       ((data_from_glc10 == 2) | (data_from_glc10 == 3) | (data_from_glc10 == 4))

            to_data = np.zeros((data.shape[1], data.shape[2]), dtype="int8")
            to_data[data_is] = 1
            to_data[data_water] = 4
            to_data[data_farm] = 2
            to_data[data_soil] = 3
            to_data[data_veg] = 5

            return to_data

        def _predict_func2(data):
            data_esa20 = data[9]
            data_esa21 = data[10]
            data_from_glc10 = data[18]
            data_gisa10m = data[20]

            data_is = (data_esa20 == 50) | (data_esa21 == 50) | (data_from_glc10 == 6) | (data_gisa10m == 1)
            data_is = data_is * 1
            to_data = sliding_window_view(data_is, win_size)
            to_data = to_data.sum(axis=(2, 3))
            to_data2 = np.zeros((data.shape[1], data.shape[2]), dtype="int16")
            win_size1 = int(win_size[0] / 2)
            win_size2 = int(win_size[1] / 2)
            to_data2[win_size1:win_size1 + to_data.shape[0], win_size2:win_size2 + to_data.shape[1]] = to_data

            # to_data = conv2dDim1(data_is, kernel=np.ones(win_size), is_jdt=True)

            return to_data2

        _SDIS_DatasetsImage().fit(r"F:\ChinaNorthIS\Run\Images\1\sd_dataset_im2.tif", _predict_func)

        win_size = [21, 21]

        # _SDIS_DatasetsImage(read_rows=5000, read_columns=5000, dtype="int8", to_dtype="int16").fit(
        #     r"F:\ChinaNorthIS\Run\Images\1\sd_dataset_im3_isdensity.tif", _predict_func2)

    def func9():
        sdis_sa = _SDIS_SamplesAcquisition()
        ndvi_std_names = [
            'NDVIStd16', 'NDVIStd17', 'NDVIStd18', 'NDVIStd19', 'NDVIStd20',
            'NDVIStd21', 'NDVIStd22', 'NDVIStd23', 'NDVIStd24',
        ]

        # City ndvistd 0.112

        sdis_sa.addFieldSum("CITY_SUM", "GISA10m_IS", "FROM_GLC10_IS", "ESA_20_IS", "ESA_21_IS")

        def _draw_hist(_name, _n=256):
            sdis_sa.hist(_name, label="{} City 0".format(_name), filters_and=[("GUB", "==", 1), ("CITY_SUM", "==", 0)])
            sdis_sa.hist(_name, label="{} City 1".format(_name), filters_and=[("GUB", "==", 1), ("CITY_SUM", "==", 1)])
            sdis_sa.hist(_name, label="{} City 2".format(_name), filters_and=[("GUB", "==", 1), ("CITY_SUM", "==", 2)])
            sdis_sa.hist(_name, label="{} City 3".format(_name), filters_and=[("GUB", "==", 1), ("CITY_SUM", "==", 3)])
            sdis_sa.hist(_name, label="{} City 4".format(_name), filters_and=[("GUB", "==", 1), ("CITY_SUM", "==", 4)])
            # sdsi_sa.hist(_name, label="{} NoCity".format(_name), filters_and=[("GUB", "==", 0)])

        _draw_hist("NDVIStd16")
        # _draw_hist("NDVIStd17")
        # _draw_hist("NDVIStd18")
        # _draw_hist("NDVIStd19")
        # _draw_hist("NDVIStd20")
        # _draw_hist("NDVIStd21")
        # _draw_hist("NDVIStd22")
        # _draw_hist("NDVIStd23")
        # _draw_hist("NDVIStd24")

        plt.legend()
        plt.show()

    def func10(csv_fn=_HB_SAMPLES_INIT_FN, to_csv_fn=None):
        print("#", "-" * 30, "Samples City", "-" * 30, "#")
        tlp = TableLinePrint()
        tlp.print("Name", "Mean", "Std", "+2std", "-2std")
        tlp.separationLine()

        # 城市样本
        sdis_sa = _SDIS_SamplesAcquisition(csv_fn)
        sdis_sa.fileterAnd(("GUB", "==", 1))
        # sdis_sa.addFieldSum("IS4_CATE", "GISA10m_IS", "FROM_GLC10_IS", "ESA_20_IS", "ESA_21_IS")

        c_name_list = []

        def func10_1(c_name, year_name):
            c_name_list.append(c_name)

            mean0 = sdis_sa.mean(year_name, filters_and=[("IS4_CATE", "==", 0)])
            std0 = sdis_sa.std(year_name, filters_and=[("IS4_CATE", "==", 0)])
            mean4 = sdis_sa.mean(year_name, filters_and=[("IS4_CATE", "==", 4)])
            std4 = sdis_sa.std(year_name, filters_and=[("IS4_CATE", "==", 4)])

            tlp.print("{} 0".format(year_name), mean0, std0, mean0 + 2 * std0, mean0 - 2 * std0)
            tlp.print("{} 4".format(year_name), mean4, std4, mean4 + 2 * std4, mean4 - 2 * std4)
            tlp.separationLine()

            def _is_veg(_spl):
                _is_veg_n = 0
                if _spl["FROM_GLC10"] in [1, 2, 4, 5, 7, ]:
                    _is_veg_n += 1
                if _spl["ESA_20"] in [10, 20, 30, 40, 60, ]:
                    _is_veg_n += 1
                if _spl["ESA_21"] in [10, 20, 30, 40, 60, ]:
                    _is_veg_n += 1
                _spl["IS_VEG"] = _is_veg_n
                return _spl

            def _is_water(_spl):
                _is_water_n = 0
                if _spl["FROM_GLC10"] == 5:
                    _is_water_n += 1
                if _spl["ESA_20"] == 80:
                    _is_water_n += 1
                if _spl["ESA_21"] == 80:
                    _is_water_n += 1
                _spl["IS_WATER"] = _is_water_n
                return _spl

            def _func_category(_spl):
                _spl = _is_veg(_spl)
                _spl = _is_water(_spl)

                category = 0

                if _spl["IS4_CATE"] in [0, 1]:
                    category = 0
                    # if _spl[year_name] < (mean0 - 2 * std0):
                    #     category = 1
                elif _spl["IS4_CATE"] in [3, 4]:
                    category = 1
                    # if _spl[year_name] > (mean4 + 2 * std4):
                    #     category = 0
                else:
                    # TODO: 确定是植被类别还是裸土类别还是什么水体类别
                    # 植被类别就使用NDVI过滤
                    # 水体就NDWI过滤确定类别
                    if _spl["IS_WATER"] != 0:
                        category = 0
                    else:
                        if _spl["IS_VEG"] != 0:
                            if _spl[year_name] > 0.11:
                                category = 0
                            else:
                                category = 1
                        else:
                            category = 0

                _spl[c_name] = category
                return _spl

            sdis_sa.map(_func_category)

            return c_name

        for i in range(16, 25):
            func10_1("CATEGORY{}".format(i), "NDVIStd{}".format(i))

        df = sdis_sa.df()
        print(df[c_name_list])
        if to_csv_fn is not None:
            df.to_csv(to_csv_fn, index=False)

        return to_csv_fn

    def func11():
        sdis_sa = _SDIS_SamplesAcquisition()
        sdis_sa.update(
            pd.read_csv(r"F:\ChinaNorthIS\Run\Samples\3\sdspl2_10w_ndwi_1.csv").to_dict("records"),
            update_field_names=[
                "NDWIMax16", "NDWIMax17", "NDWIMax18", "NDWIMax19", "NDWIMax20", "NDWIMax21", "NDWIMax22", "NDWIMax23",
                "NDWIMax24",
            ])
        sdis_sa.df().to_csv(r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_random10w_DS.csv")

    def func12():
        sdis_sa = _SDIS_Sampling()
        for year in range(16, 25):
            sdis_sa.sampling("SDIS_{}".format(year), r"F:\ChinaNorthIS\Run\SDImdc\1\sdimdc1_{}.tif".format(year))
        print(sdis_sa.df)
        sdis_sa.df.to_csv(r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_random10w_DS7.csv", index=False)

    def func13():
        sdis_sa = _SDIS_Sampling()
        sdis_sa.sampling("ISDENSITY21", r"F:\ChinaNorthIS\Run\Images\1\sd_dataset_im3_isdensity.tif")
        sdis_sa.sampling("Union1", r"F:\ChinaNorthIS\Run\Images\1\sd_dataset_im2.tif")
        sdis_sa.df.to_csv(r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_random10w_DS8.csv", index=False)

    def func14():
        sdis_sa = _SDIS_SamplesAcquisition()
        sdis_sa.addFieldSum("IS4_CATE", "GISA10m_IS", "FROM_GLC10_IS", "ESA_20_IS", "ESA_21_IS")
        sdis_sa.df().to_csv(r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_random10w_DS6.csv", index=False)

    def func15():
        sdis_sa = _SDIS_SamplesAcquisition()
        sdis_sa.fileterAnd(("GUB", "==", 0))
        print(len(sdis_sa))

        # df = sdis_sa.df()
        # plt.subplot(121)
        # df["ISDENSITY21"].hist(cumulative=False, bins=441, range=[1, 441])
        # plt.subplot(122)
        # df["ISDENSITY21"].hist(cumulative=True, bins=441)

        # print((df["ISDENSITY21"] >= 100).sum())

        sdis_sa.addISField("ISDENSITY21_CATE", lambda _spl: int(_spl["ISDENSITY21"] > 100))
        sdis_sa.groupStatusCounts("ISDENSITY21_CATE", "IS4_CATE")
        sdis_sa.fileterOR(("ISDENSITY21_CATE", "!=", 0), ("IS4_CATE", "!=", 0), )
        print(len(sdis_sa), len(sdis_sa) + 67159)
        sdis_sa.groupStatusCounts("ESA_21", "GUB")

        def _map_func(_spl):
            category = int(_spl["IS4_CATE"] != 0)
            for i in range(16, 25):
                _spl["CATEGORY{}".format(i)] = category
            return _spl

        sdis_sa.map(_map_func)
        sdis_sa.df().to_csv(r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_random10w_villis1.csv", index=False)

        # sdis_sa.hist("NDVIStd16", label="NDVIStd16 0", filters_and=[("ISDENSITY21_CATE", "==", 1), ("IS4_CATE", "==", 0)])
        # sdis_sa.hist("NDVIStd16", label="NDVIStd16 1", filters_and=[("ISDENSITY21_CATE", "==", 1), ("IS4_CATE", "==", 1)])
        # sdis_sa.hist("NDVIStd16", label="NDVIStd16 2", filters_and=[("ISDENSITY21_CATE", "==", 1), ("IS4_CATE", "==", 2)])
        # sdis_sa.hist("NDVIStd16", label="NDVIStd16 3", filters_and=[("ISDENSITY21_CATE", "==", 1), ("IS4_CATE", "==", 3)])
        # sdis_sa.hist("NDVIStd16", label="NDVIStd16 4", filters_and=[("ISDENSITY21_CATE", "==", 1), ("IS4_CATE", "==", 4)])
        # sdis_sa.hist("NDVIStd16", label="NDVIStd16 0", filters_and=[("ISDENSITY21_CATE", "==", 0), ("IS4_CATE", "==", 0)])

        # sdis_sa.df().to_csv(r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_random10w_DS7.csv", index=False)
        # plt.legend()
        # plt.show()

    def func16(csv_fn=_HB_SAMPLES_INIT_FN, to_csv_fn=None, n_category_dict=None):
        print("#", "-" * 30, "Samples No City 2", "-" * 30, "#")

        if n_category_dict is None:
            n_category_dict = {"WATER": -1, "SOIL": -1, "VEG": -1, "FARM": -1}

        sdis_sa = _SDIS_SamplesAcquisition(csv_fn)
        sdis_sa.fileterAnd(("GUB", "==", 0))
        print(len(sdis_sa))

        sdis_sa.addISField("ISDENSITY21_CATE", lambda _spl: int(_spl["ISDENSITY21"] > 100))
        sdis_sa.groupStatusCounts("ISDENSITY21_CATE", "IS4_CATE")
        sdis_sa.fileterAnd(("ISDENSITY21_CATE", "==", 0), ("IS4_CATE", "==", 0), )
        print(len(sdis_sa), len(sdis_sa) + 67159)
        sdis_sa.groupStatusCounts("ESA_21", "GUB")

        category_dict = {"WATER": 1, "SOIL": 2, "VEG": 3, "FARM": 4}

        def _map_func(_spl):
            is_water = (_spl["ESA_20"] == 80) | (_spl["ESA_21"] == 80) | (_spl["FROM_GLC10"] == 5)
            if is_water:
                _spl["NOIS_CATE"] = 1
                return _spl

            is_soil = (_spl["ESA_20"] == 60) | (_spl["ESA_21"] == 60) | (_spl["FROM_GLC10"] == 7)
            if is_soil:
                _spl["NOIS_CATE"] = 2
                return _spl

            is_veg = (_spl["ESA_20"] == 10) | (_spl["ESA_20"] == 20) | (_spl["ESA_20"] == 30) | (
                    _spl["ESA_20"] == 90) | (_spl["ESA_20"] == 95) | (_spl["ESA_20"] == 100) | \
                     (_spl["ESA_21"] == 10) | (_spl["ESA_21"] == 20) | (_spl["ESA_21"] == 30) | (
                             _spl["ESA_21"] == 90) | (_spl["ESA_21"] == 95) | (_spl["ESA_21"] == 100) | \
                     (_spl["FROM_GLC10"] == 2) | (_spl["FROM_GLC10"] == 3) | (_spl["FROM_GLC10"] == 4)
            if is_veg:
                _spl["NOIS_CATE"] = 3
                return _spl

            _spl["NOIS_CATE"] = 4
            return _spl

        sdis_sa.map(_map_func)

        sdis_sa.groupStatusCounts("NOIS_CATE", "GUB")

        to_spls = [
            *sdis_sa.randomSampling(n_category_dict["WATER"], filters_and=[("NOIS_CATE", "==", 1)]),
            *sdis_sa.randomSampling(n_category_dict["SOIL"], filters_and=[("NOIS_CATE", "==", 2)]),
            *sdis_sa.randomSampling(n_category_dict["VEG"], filters_and=[("NOIS_CATE", "==", 3)]),
            *sdis_sa.randomSampling(n_category_dict["FARM"], filters_and=[("NOIS_CATE", "==", 4)]),
        ]

        category_dict_2 = {"CATEGORY{}".format(i): 0 for i in range(16, 25)}
        for _spl in to_spls:
            _spl.update(category_dict_2)

        to_df = pd.DataFrame(to_spls)
        print(to_df)
        if to_csv_fn is not None:
            to_df.to_csv(to_csv_fn, index=False)

        return to_csv_fn

    def func17():
        to_spls = [
            *pd.read_csv(r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_random10w_city4.csv").to_dict("records"),
            *pd.read_csv(r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_random10w_villis1.csv").to_dict("records"),
            *pd.read_csv(r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_random10w_villnois1.csv").to_dict("records"),
        ]

        to_df = pd.DataFrame(to_spls)
        print(to_df)
        to_df.to_csv(r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_random10w_release1.csv", index=False)

    def func18():
        tlp = TableLinePrint()
        tlp.print("Image Type", "Number", "File Name", "Exists")
        tlp.separationLine()

        year = 19
        print(year)
        dfn_spl = DirFileName(r"F:\ChinaNorthIS\Run\Samples\3\random10wrelease1")
        csv_fn = dfn_spl.fn(r"sdspl3_{}_S2_S1ASDE.csv".format(year))
        print("csv_fn", csv_fn)

        spls = _SDIS_Samples()

        for image_type in ["S2", "S1AS", "S1DE"]:
            for i in range(21):
                fn = dfn_spl.fn(r"sdspl3_{}_20{}-01-01_{}.geojson".format(image_type, year, i))
                tlp.print(image_type, i, fn, os.path.isfile(fn))
                spls.addGeoJson(fn)

        spls.checkNone()
        spls.addTest(0.95)

        print(spls.field_names)

        df_init = pd.read_csv(dfn_spl.fn("sdspl2_random10w_release1.csv"))
        df_init_list = df_init[list(["SRT"] + ["CATEGORY{}".format(i) for i in range(16, 25)])].to_dict("records")
        df_init_dict = {_spl["SRT"]: _spl for _spl in df_init_list}
        find_list = df_init["SRT"].tolist()
        to_spls = [spls[i] for i in find_list]
        for _spl in to_spls:
            _spl.update(df_init_dict[_spl["SRT"]])

        df = pd.DataFrame(to_spls)
        print(df)
        df.to_csv(csv_fn, index=False)

        return

    def func19():
        _SDIS_Sampling(pd.read_csv(r"F:\ChinaNorthIS\Run\Samples\hb\hb_spl_random.csv")).sampling(
            "GISA10m", r"H:\HBGISA10m\HBGISA10m_gtif.tif"
        ).df.to_csv(
            r"F:\ChinaNorthIS\Run\Samples\hb\hb_spl_random_GISA10m.csv"
        )

    def func20():
        sdis_sa = _SDIS_SamplesAcquisition(pd.read_csv(_HB_SAMPLES_INIT_FN).to_dict("records"))
        print("sdis_sa init")

        sdis_sa.addISField("GISA10m_IS", lambda _spl: int(_spl["GISA10m"] == 1))
        print("GISA10m_IS")
        sdis_sa.addISField("FROM_GLC10_IS", lambda _spl: int(_spl["FROM_GLC10"] == 6))
        print("FROM_GLC10_IS")
        sdis_sa.addISField("ESA_20_IS", lambda _spl: int(_spl["ESA_20"] == 50))
        print("ESA_20_IS")
        sdis_sa.addISField("ESA_21_IS", lambda _spl: int(_spl["ESA_21"] == 50))
        print("ESA_21_IS")
        sdis_sa.addFieldSum("CITY_SUM", "GISA10m_IS", "FROM_GLC10_IS", "ESA_20_IS", "ESA_21_IS")

        df = sdis_sa.df()
        print(df[["SRT", "CITY_SUM"]])
        df.to_csv(r"F:\ChinaNorthIS\Run\Samples\hb\hb_spl_random_INIT_tmp1.csv", index=False)

    def func21():
        print([[1 for i in range(21)] for j in range(21)])

        # [0, 10, 20, 30, 40, 60, 80, 90, 100, 120]
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def func22(csv_fn=_HB_SAMPLES_INIT_FN, to_csv_fn=None):
        print("#", "-" * 30, "Samples No City 1", "-" * 30, "#")
        sdis_sa = _SDIS_SamplesAcquisition(csv_fn)
        sdis_sa.fileterAnd(("GUB", "==", 0))
        print(len(sdis_sa))

        # df = sdis_sa.df()
        # plt.subplot(121)
        # df["ISDENSITY21"].hist(cumulative=False, bins=441, range=[1, 441])
        # plt.subplot(122)
        # df["ISDENSITY21"].hist(cumulative=True, bins=441)

        sdis_sa.addISField("ISDENSITY21_CATE", lambda _spl: int(_spl["ISDENSITY21"] > 100))
        sdis_sa.groupStatusCounts("ISDENSITY21_CATE", "IS4_CATE")

        sdis_sa.fileterOR(("ISDENSITY21_CATE", "!=", 0), ("IS4_CATE", "!=", 0), )
        print(len(sdis_sa))
        sdis_sa.groupStatusCounts("ESA_21", "GUB")

        def _map_func(_spl):
            category = int(_spl["IS4_CATE"] > 0)
            for i in range(16, 25):
                _spl["CATEGORY{}".format(i)] = category
            return _spl

        sdis_sa.map(_map_func)

        if to_csv_fn is not None:
            sdis_sa.df().to_csv(to_csv_fn, index=False)

        # plt.legend()
        # plt.show()

        return to_csv_fn

    def func23():
        # City func10
        city_csv_fn = func10(to_csv_fn=r"F:\ChinaNorthIS\Run\Samples\hb\hb_spl_random_INIT_city_tmp.csv")
        # No city 1
        nocity1_csv_fn = func22(to_csv_fn=r"F:\ChinaNorthIS\Run\Samples\hb\hb_spl_random_INIT_nocity1_tmp.csv")
        # No city 2
        nocity2_csv_fn = func16(
            to_csv_fn=r"F:\ChinaNorthIS\Run\Samples\hb\hb_spl_random_INIT_nocity2_tmp.csv",
            n_category_dict={"WATER": 10000, "SOIL": -1, "VEG": 10000, "FARM": 20000}
        )

        to_spls = [
            *pd.read_csv(city_csv_fn).to_dict("records"),
            *pd.read_csv(nocity1_csv_fn).to_dict("records"),
            *pd.read_csv(nocity2_csv_fn).to_dict("records"),
        ]

        to_df = pd.DataFrame(to_spls)
        print(to_df)
        to_df.to_csv(r"F:\ChinaNorthIS\Run\Samples\hb\hb_spl_random_INIT_release1.csv", index=False)

    def func24():
        spls = pd.read_csv(r"F:\ChinaNorthIS\Run\Samples\hb\hb_spl_random_INIT_release1.csv").to_dict("records")
        to_spls = random.sample(spls, 100000)
        to_df = pd.DataFrame(to_spls)
        print(to_df)
        to_df.to_csv(r"F:\ChinaNorthIS\Run\Samples\hb\hb_spl_random_INIT_release1_select10w.csv", index=False)
        to_df[["SRT", "X", "Y", *["CATEGORY{}".format(i) for i in range(16, 25)]]].to_csv(
            r"F:\ChinaNorthIS\Run\Samples\hb\hb_spl_random_INIT_release1_select10w_update.csv", index=False)

    def func25():
        df = pd.read_csv(r"F:\ChinaNorthIS\Run\Samples\hb\hb_spl_random_INIT_release1_select10w_update.csv")
        spls = df.to_dict("records")
        x_min, x_max, y_min, y_max = df["X"].min(), df["X"].max(), df["Y"].min(), df["Y"].max()
        print(x_min, x_max, y_min, y_max)

        class _spls_grids:

            def __init__(self, _x_n, _y_n):
                def _get_list(_n, _min, _max):
                    _d = (_max - _min) / _n
                    _list_min = [_min + i * _d for i in range(_n)]
                    _list_max = [_min + (i + 1) * _d for i in range(_n)]
                    return _list_min, _list_max

                _x_list_min, _x_list_max = _get_list(_x_n, x_min, x_max)
                _y_list_min, _y_list_max = _get_list(_y_n, y_min, y_max)

                self.grids = []
                for i in range(_x_n):
                    for j in range(_y_n):
                        self.grids.append([_x_list_min[i], _x_list_max[i], _y_list_min[j], _y_list_max[j]])

            def get_n(self, _x, _y):
                for i in range(len(self.grids)):
                    if self.isin(i, _x, _y):
                        return i
                return -1

            def isin(self, i, _x, _y):
                return (self.grids[i][0] <= _x <= self.grids[i][1]) and (self.grids[i][2] <= _y <= self.grids[i][3])

        spls_grids = _spls_grids(6, 5)

        for spl in spls:
            spl["GRID"] = spls_grids.get_n(spl["X"], spl["Y"])

        to_df = pd.DataFrame(spls)
        print(to_df)
        to_df.to_csv(r"F:\ChinaNorthIS\Run\Samples\hb\hb_spl_random_INIT_release1_select10w_update2.csv", index=False)

    def func26():
        dfn = DirFileName(r"G:\Downloads\drive-download-20240929T112342Z-001")
        data_dict = {}
        fns = list(os.listdir(dfn.fn()))
        jdt = Jdt(len(fns), "concat samples").start()
        year = "2024"
        fn_list = []
        for fn in fns:
            if fn.endswith(".geojson") and (year in fn):
                json_data = readJson(dfn.fn(fn))
                fn_list.append(fn)
                for feat in json_data["features"]:
                    select_id = feat["properties"]["SRT"]
                    if select_id not in data_dict:
                        data_dict[select_id] = {
                            "SRT": select_id,
                            "X": feat["geometry"]["coordinates"][0],
                            "Y": feat["geometry"]["coordinates"][1],
                        }
                    for name in feat["properties"]:
                        data_dict[select_id][name] = feat["properties"][name]
            jdt.add()
        jdt.end()
        printList("fn_list", fn_list)

        category_data_list = pd.read_csv(
            r"F:\ChinaNorthIS\Run\Samples\hb\update2\hb_spl_random_INIT_release1_select10w_update2.csv"
        ).to_dict("records")

        jdt = Jdt(len(category_data_list), "update category").start()
        for spl in category_data_list:
            data_dict[spl["SRT"]].update(spl)
            jdt.add()
        jdt.end()

        for i in data_dict:
            data_dict[i]["TEST"] = int(random.random() < 0.95)
        df = pd.DataFrame(list(data_dict.values()))
        df.to_csv(r"F:\ChinaNorthIS\Run\Samples\hb\update2\hb_update2_24_spl1.csv", index=False)
        print(df)

    def func27():
        df_list = pd.read_csv(
            r"F:\ChinaNorthIS\Run\Samples\hb\update2\hb_spl_random_INIT_release1_select10w_update2.csv")
        df_data_list = pd.read_csv(r"F:\ChinaNorthIS\Run\Samples\hb\hb_spl_random_INIT_release1.csv")
        df = df_cat(df_data_list, df_list)
        df.to_csv(r"F:\ChinaNorthIS\Run\Samples\hb\update2\hb_spl_random_INIT_release1_select10w_update2_data.csv",
                  index=False)

    def df_cat(df_data, df, is_update=False):
        df_list = df_data.sort_values("SRT").to_dict("records")
        df_data_list = df.sort_values("SRT").to_dict("records")
        jdt = Jdt(len(df_list), "func27").start()
        i_spl_tmp, n_tmp = 0, 0
        for spl in df_list:
            for i_spl_tmp in range(n_tmp, len(df_data_list)):
                n_tmp = i_spl_tmp
                if spl["SRT"] == df_data_list[i_spl_tmp]["SRT"]:
                    spl_tmp = df_data_list[i_spl_tmp]
                    for name in spl_tmp:
                        if not is_update:
                            if name not in spl:
                                spl[name] = spl_tmp[name]
                        else:
                            spl[name] = spl_tmp[name]
                    break
            if n_tmp == (len(df_data_list) - 1):
                n_tmp = 0
            jdt.add()
        jdt.end()
        df = pd.DataFrame(df_list)
        return df

    def func28():
        df_list = pd.read_csv(
            r"F:\ChinaNorthIS\Run\Samples\hb\update2\hb_spl_random_INIT_release1_select10w_update2_data.csv").to_dict(
            "records")
        for spl in df_list:
            if spl["GUB"] == 0:
                category = int(spl["IS4_CATE"] > 0)
                for i in range(16, 25):
                    spl["CATEGORY2_{}".format(i)] = category
            else:
                for i in range(16, 25):
                    spl["CATEGORY2_{}".format(i)] = spl["CATEGORY{}".format(i)]

        df = pd.DataFrame(df_list)
        print(df)
        df.to_csv(r"F:\ChinaNorthIS\Run\Samples\hb\update2\hb_spl_random_INIT_release1_select10w_update2_data2.csv",
                  index=False)

    def func29():
        feature_names, x_test, x_train, y_test, y_train = func_get_data1(24, S2_SELECT_NAMES + S1_AS_SELECT_NAMES)

        clf = RandomForestClassifier(
            **{'n_estimators': 120, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2})
        clf.fit(x_train, y_train)
        print(clf.score(x_test, y_test))

        gr = GDALRaster(r"F:\ChinaNorthIS\Run\Images\2\hb_im24_1.tif")
        data = np.zeros((len(feature_names), gr.n_rows, gr.n_columns))

        for i, name in enumerate(feature_names):
            data[i] = gr.readGDALBand(name)

        print("imdc ...")
        imdc = dataModelPredict(data, data_deal=None, is_jdt=True, model=clf)
        to_geo_fn = numberfilename(r"F:\ChinaNorthIS\Run\Images\2\hb_im24_1_imdc.tif")
        print("to_geo_fn", to_geo_fn)
        gr.save(imdc.astype("int8"), to_geo_fn, fmt="GTiff", dtype=gdal.GDT_Byte, options=["COMPRESS=LZW"])
        # color_table = {0: (0, 255, 0), 1: (255, 0, 0)}
        # tiffAddColorTable(to_geo_fn, 1, color_table)

    def func_get_data1(year, feature_names, category_name=None, data_deal=None,
                       random_select=None, data_coll=None, filter_func=None):
        if category_name is None:
            category_name = "CATEGORY2_{}".format(year)
        df_category = pd.read_csv(
            r"F:\ChinaNorthIS\Run\Samples\hb\update2\hb_spl_random_INIT_release1_select10w_update2_data2.csv")
        df_category = df_category[["SRT", "CATEGORY2_{}".format(year), "IS4_CATE", "GUB"]]
        df_category["CATEGORY{}".format(year)] = df_category["CATEGORY2_{}".format(year)]
        csv_fn_fmt = r"F:\ChinaNorthIS\Run\Samples\hb\update2\hb_update2_{}_spl1.csv"
        csv_fn = csv_fn_fmt.format(year)
        df = pd.read_csv(csv_fn)
        df = df_cat(df_category, df, is_update=True)

        if filter_func is not None:
            _list = df.to_dict("records")
            _list_2 = []
            for _spl in _list:
                if filter_func(_spl):
                    _list_2.append(_spl)
            df = pd.DataFrame(_list_2)

        print(df["CATEGORY{}".format(year)].sum(), df["CATEGORY2_{}".format(year)].sum())
        print(df)

        if data_deal is not None:
            if data_deal == "minmax":
                for feat_name in feature_names:
                    df[feat_name] = (df[feat_name] - df[feat_name].min()) / (df[feat_name].max() - df[feat_name].min())
            elif data_deal == "z-score":
                for feat_name in feature_names:
                    df[feat_name] = (df[feat_name] - df[feat_name].mean()) / df[feat_name].std()
            elif data_deal.startswith("minmax_"):
                ratio = float(data_deal.split("_")[-1])
                for feat_name in feature_names:
                    data = df[feat_name].values
                    d0, d1 = reHist0(data, ratio=ratio, is_print=False)
                    df[feat_name] = (np.clip(data, d0, d1) - d0) / (d1 - d0)
                    if data_coll is not None:
                        data_coll.append([d0, d1])

        def _train_test_data(_test):
            _df = df[df["TEST"] == _test]
            _df = _df[feature_names + [category_name]]
            _df = _df.dropna(axis=0, how="any")
            if random_select is not None:
                if _test == 1:
                    _df_list = _df.to_dict("records")
                    n = random_select
                    if 0 < random_select < 1:
                        n = random_select * len(_df_list)
                    _df_list = random.sample(_df_list, int(n))
                    _df = pd.DataFrame(_df_list)
            _x = _df[feature_names].values
            _y = _df[category_name].tolist()
            return _x, _y

        x_train, y_train = _train_test_data(1)
        x_test, y_test = _train_test_data(0)
        print("x_train", len(x_train))
        print("x_test", len(x_test))
        return feature_names, x_test, x_train, y_test, y_train

    def func30():
        tlp = TableLinePrint()
        tlp.print("N", "NAME", "MIN", "MAX", "D0", "D1")
        tlp.separationLine()

        year = 24
        csv_fn_fmt = r"F:\ChinaNorthIS\Run\Samples\hb\update2\hb_update2_{}_spl1.csv"
        df = pd.read_csv(csv_fn_fmt.format(year))
        dfn = DirFileName(r"F:\ChinaNorthIS\Run\Samples\hb\update2\hist{}".format(year)).mkdir()
        feature_names = S2_SELECT_NAMES + S1_AS_SELECT_NAMES + S1_DE_SELECT_NAMES
        n = 1
        for feat_name in feature_names:
            if feat_name in df:
                data = df[feat_name].values
                d0, d1 = reHist0(data, ratio=0.01, is_print=False)
                df[feat_name] = np.clip(data, d0, d1)
                # y, x = np.histogram(
                #     data, bins=256, density=True,
                #     range=[d0, d1],
                # )
                # plt.plot(x[:-1], y)
                # plt.title("{} {} {}".format(n, feat_name, year))
                # plt.savefig(dfn.fn("{:0>2d}_{}_{}_2.jpg".format(n, feat_name, year)))
                # plt.close()
                tlp.print(n, feat_name, data.min(), data.max(), d0, d1)
                n += 1

    def func31():
        range_list = []
        feature_names, x_test, x_train, y_test, y_train = func_get_data1(
            24, S2_SELECT_NAMES + S1_AS_SELECT_NAMES,
            data_deal="minmax_0.01",
            # random_select=0.1
            data_coll=range_list
        )
        tlp = TableLinePrint()
        tlp.print("No.", "NAME", "MIN", "MAX")
        tlp.separationLine()
        for i, feat_name in enumerate(feature_names):
            tlp.print(i + 1, feat_name, x_train[:, i].min(), x_train[:, i].max(), )

        # clf = SVM_RGS()
        clf = SVC(C=10, gamma=0.1, cache_size=5000)
        clf.fit(x_train, y_train)
        print(clf.score(x_test, y_test))

        gr = GDALRaster(r"F:\ChinaNorthIS\Run\Images\2\hb_im24_1.tif")
        data = np.zeros((len(feature_names), gr.n_rows, gr.n_columns))

        for i, name in enumerate(feature_names):
            d0, d1 = tuple(range_list[i])
            data[i] = (gr.readGDALBand(name) - d0) / (d1 - d0)

        print("imdc ...")
        imdc = dataModelPredict(data, data_deal=None, is_jdt=True, model=clf)
        to_geo_fn = numberfilename(r"F:\ChinaNorthIS\Run\Images\2\hb_im24_1_imdc.tif")
        print("to_geo_fn", to_geo_fn)
        gr.save(imdc.astype("int8"), to_geo_fn, fmt="GTiff", dtype=gdal.GDT_Byte, options=["COMPRESS=LZW"])
        # color_table = {0: (0, 255, 0), 1: (255, 0, 0)}
        # tiffAddColorTable(to_geo_fn, 1, color_table)

    def func32():
        def _filter_fun(_spl):
            if _spl["GUB"] == 0:
                if _spl["IS4_CATE"] == 1:
                    return False
            return True

        feature_names, x_test, x_train, y_test, y_train = func_get_data1(
            24, S2_SELECT_NAMES + S1_AS_SELECT_NAMES,
            # data_deal="minmax_0.01",
            filter_func=_filter_fun
        )

        tlp = TableLinePrint()
        tlp.print("No.", "NAME", "MIN", "MAX")
        tlp.separationLine()
        for i, feat_name in enumerate(feature_names):
            tlp.print(i + 1, feat_name, x_train[:, i].min(), x_train[:, i].max(), )

        clf = RandomForestClassifier(
            **{'n_estimators': 120, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2})
        clf.fit(x_train, y_train)
        print(clf.score(x_test, y_test))

        gr = GDALRaster(r"F:\ChinaNorthIS\Run\Images\2\hb_im24_1.tif")
        data = np.zeros((len(feature_names), gr.n_rows, gr.n_columns))

        for i, name in enumerate(feature_names):
            data[i] = gr.readGDALBand(name)

        print("imdc ...")
        imdc = dataModelPredict(data, data_deal=None, is_jdt=True, model=clf)
        to_geo_fn = numberfilename(r"F:\ChinaNorthIS\Run\Images\2\hb_im24_1_imdc.tif")
        print("to_geo_fn", to_geo_fn)
        gr.save(imdc.astype("int8"), to_geo_fn, fmt="GTiff", dtype=gdal.GDT_Byte, options=["COMPRESS=LZW"])
        # color_table = {0: (0, 255, 0), 1: (255, 0, 0)}
        # tiffAddColorTable(to_geo_fn, 1, color_table)

    def func33():
        dfn = DirFileName(r"G:\Downloads\hbimdc2")
        for dirname in dfn.listdirnames(is_join=False):
            dfn2 = dfn.dfn(dirname)
            is_print = 0
            n = len(dfn2.listfiles(_glob="*.tif"))
            print(dirname, n)
            if n >= 276:
                continue
            for i in range(300):
                fn = dfn2.fn("{}_ASDE{}.tif".format(dirname, i))
                if not os.path.isfile(fn):
                    print("  * {}".format(fn))
                    is_print += 1
                else:
                    is_print = 0
                if is_print >= (276 - n):
                    break
            for i in range(300):
                fn = dfn2.fn("{}_AS{}.tif".format(dirname, i))
                if not os.path.isfile(fn):
                    print("  * {}".format(fn))
                    is_print += 1
                else:
                    is_print = 0
                if is_print >= (276 - n):
                    break

    def func34():

        def _conv(_win_size, _func, _data):
            _row, _column = int(_win_size[0] / 2), int(_win_size[1] / 2)
            data_tmp = sliding_window_view(_data, _win_size)
            _imdc = np.zeros_like(_data, dtype="int8")
            _jdt = Jdt(data_tmp.shape[0], "_conv").start()
            for i in range(0, data_tmp.shape[0]):
                for j in range(0, data_tmp.shape[1]):
                    _imdc[i + _row, j + _column] = _func(data_tmp[i, j])
                _jdt.add()
            _jdt.end()
            return _imdc

        def func34_1(x):
            if x[0, 1] == x[1, 0] == x[1, 2] == x[2, 1] == x[0, 0] == x[2, 0] == x[0, 2] == x[2, 2]:
                return x[0, 1]
            return x[1, 1]

        def _predict_func(data):
            if data.sum() != 0:
                return _conv((3, 3), func34_1, data[0])
            return data[0]

        def _run(raster_fn):
            if not os.path.isfile(raster_fn):
                print("Can not find {}".format(raster_fn))
                return
            to_fn = r"F:\ChinaNorthIS\Run\Models\20240928H202156\1\{}".format(getfilenme(raster_fn))
            if os.path.isfile(to_fn):
                os.remove(to_fn)
            print("to_fn", to_fn)

            _SDIS_DatasetsImage(
                raster_fn=raster_fn,
                read_rows=10000, read_columns=10000, dtype="int8", to_dtype="int8",
            ).fit(to_fn, _predict_func)

        for i in range(16, 26):
            _run(r"G:\Downloads\hbimdc2\hbimdc2_{}_gtif.tif".format(i))

    return func34()


def main():
    dfn = DirFileName(r"F:\ChinaNorthIS\Run\Samples\hb\hbspl3-20240928T012532Z-001\hbspl3")
    concat_geojson([dfn.fn(fn) for fn in os.listdir(dfn.fn()) if fn.endswith(".geojson")],
                   dfn.fn("hbspl2_ndvistd.json"))
    return


if __name__ == "__main__":
    funcs()

    r"""
python -c "import sys; sys.path.append(r'F:\PyCodes'); from FiveMeter.SDIS import run; run()"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from FiveMeter.SDIS import funcs; funcs()"
    """
