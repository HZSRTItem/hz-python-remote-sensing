# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2Sample.py
@Time    : 2024/6/8 16:56
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2Sample
-----------------------------------------------------------------------------"""
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.GDALUtils import GDALSampling, GDALSamplingFast, GDALNumpySampling
from SRTCodes.OGRUtils import sampleSpaceUniform
from SRTCodes.SRTSample import GEOJsonWriteWGS84
from SRTCodes.Utils import getRandom, getfilenamewithoutext, DirFileName, FN, Jdt
from Shadow.Hierarchical import SHH2Config

RESOLUTION_ANGLE = 0.000089831529294


def _GS_NPY(npy_fn):
    gs = GDALSampling()
    gs.initNPYRaster(npy_fn)
    return gs


def QD_GS_NPY():
    return _GS_NPY(SHH2Config.QD_NPY_FN)


def BJ_GS_NPY():
    return _GS_NPY(SHH2Config.BJ_NPY_FN)


def CD_GS_NPY():
    return _GS_NPY(SHH2Config.CD_NPY_FN)


def sampling():
    def qd():
        GDALSamplingFast(SHH2Config.QD_ENVI_FN).csvfile(
            csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\vhl\sh2_spl25_vhl_2_spl.csv",
            to_csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\vhl\sh2_spl25_vhl_2_spl2.csv",
        )

    def bj():
        GDALSamplingFast(SHH2Config.BJ_ENVI_FN).csvfile(
            csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\3\sh2_spl273_5_spl.csv",
            to_csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\3\sh2_spl273_5_spl2.csv",
        )

    def cd():
        GDALSamplingFast(SHH2Config.CD_ENVI_FN).csvfile(
            csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\2\sh2_spl26_4_spl2.csv",
            to_csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\2\sh2_spl26_4_spl3.csv",
        )

    bj()
    cd()


class SHH2Sampling:

    def __init__(self, csv_fn):
        self.csv_fn = None
        self.dirname1 = r"F:\ProjectSet\Shadow\Hierarchical\Samples\ML"
        self.dirname2 = r"F:\ProjectSet\Shadow\Hierarchical\Samples\DL"

    def get(self):

        return

    def get2(self, win_rows, win_columns):
        csv_fn = self.csv_fn

        def getFileName():
            dfn = DirFileName(self.dirname2)
            _to_fn = dfn.fn("{}-{}_{}.csv".format(FN(csv_fn).getfilenamewithoutext(), win_rows, win_columns))
            _to_npy_fn = FN(to_fn).changext("-data.npy")
            return to_fn, to_npy_fn

        class sample:

            def __init__(self, _line):
                self.x = _line["X"]
                self.y = _line["Y"]
                self.city = None

        to_fn, to_npy_fn = getFileName()

        gr = {"qd": SHH2Config.QD_GR(), "bj": SHH2Config.BJ_GR(), "cd": SHH2Config.CD_GR(), }

        n_channels = gr["qd"].n_channels

        df = pd.read_csv(csv_fn)
        samples = []
        numbers = {"qd": 0, "bj": 0, "cd": 0}
        data = np.zeros((len(df), n_channels, win_rows, win_columns))
        data_shape = data[0].shape

        jdt = Jdt(len(df), "SHH2Sampling Sampling").start()
        for i in range(len(df)):
            line = df.loc[i]
            spl = sample(line)
            for k in gr:
                if gr[k].isGeoIn(spl.x, spl.y):
                    spl.city = k
                    break
            if spl.city is None:
                warnings.warn("{}, {} not in raster.".format(spl.x, spl.y))
                jdt.add()
            else:
                numbers[spl.city] += 1
            samples.append(spl)
        print(numbers)
        for k in numbers:
            if numbers[k] != 0:
                gr[k].readAsArray()
                # gr[k].d = np.zeros((3, gr[k].n_rows, gr[k].n_columns))
                gns = GDALNumpySampling(win_rows, win_columns, gr[k])
                for i, spl in enumerate(samples):
                    data_i = gns.getxy(spl.x, spl.y)
                    if data_shape == data_i.shape:
                        data[i] = gns.getxy(spl.x, spl.y)
                    else:
                        spl.city = None
                        warnings.warn("{}, {} not in raster.".format(spl.x, spl.y))
                    jdt.add()
                gns.data = None
                gr[k].d = None

        jdt.end()
        df["city"] = [str(spl.city) for spl in samples]
        df.to_csv(to_fn, index=False)
        print(to_fn)
        np.save(to_npy_fn, data.astype("float16"))

        return


def samplingTest():
    gs = GDALSamplingFast(SHH2Config.QD_LOOK_FN)
    gs.csvfile(
        csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\2\sh2_spl252_test_random1000.csv",
        to_csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\2\sh2_spl252_test_random1000_spl.csv",
    )


def randomSamples(n, x0, x1, y0, y1):
    return {
        "X": [getRandom(x0, x1) for _ in range(n)],
        "Y": [getRandom(y0, y1) for _ in range(n)],
        "SRT": [i + 1 for i in range(n)],
    }


def samplingImdc(df, dirname, fns=None):
    if fns is None:
        fns = ['OPT', 'OPT+AS', 'OPT+DE', 'OPT+AS+DE', 'OPT+GLCM', 'OPT+BS', 'OPT+C2', 'OPT+HA', 'OPT+SARGLCM']
    images = {fn: os.path.join(dirname, fn + "_imdc.tif") for fn in fns}
    gss = {fn: GDALSampling(os.path.join(dirname, fn + "_imdc.tif")) for fn in fns}
    for fn in images:
        gs = gss[fn]
        categorys = gs.sampling(df["X"].tolist(), df["Y"].tolist())
        for name in categorys:
            df[fn] = categorys[name]
            break
    return fns


def main():
    dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Samples\29")
    raster_fn = r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240626H135521\Three_epoch88_imdc1.tif"
    csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\vhl\sh2_spl25_vhl_2_spl.csv"
    to_fn = dfn.fn("sh2_spl29_qd1.csv")
    df = pd.read_csv(csv_fn)
    df = GDALSampling(raster_fn).samplingDF(df)

    map_dict = {}
    for i in range(len(df)):
        line = df.loc[i]
        data1 = df["FEATURE_1"]

    df.to_csv(to_fn, index=False)

    return


def method_name1():
    def func1():
        df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\2\sh2_spl27_1.csv")
        coors2, out_index_list = sampleSpaceUniform(df[["X", "Y"]].values.tolist(), x_len=1200, y_len=1200,
                                                    is_trans_jiaodu=True, ret_index=True)
        is_save = np.zeros(len(df))
        is_save[out_index_list] = 1
        df["IS_SAVE"] = is_save
        df = df[df["IS_SAVE"] == 1]
        print(len(out_index_list))
        df.to_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\2\sh2_spl27_1_ssu.csv", index=False)

        return

    def func2():
        # beijing 116.265589506,116.766938820,39.614856934,39.966812062
        # chengdu 103.766340133, 104.184577287, 30.537178820, 30.842317360
        df = pd.DataFrame(randomSamples(1000, 116.265589506, 116.766938820, 39.614856934, 39.966812062))
        print(df)
        df.to_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\2\sh2_spl271_1.csv", index=False)

    def func3():
        df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\2\sh2_spl271_1.csv")
        to_csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\2\sh2_spl271_2.csv"
        dirname = r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240609H091804"

        fns = samplingImdc(df, dirname)

        print(df.keys())
        data = df[fns].values
        category = []
        for i in range(len(df)):
            if len(np.unique(data[i])) == 1:
                category.append(data[i, 0])
            else:
                category.append(0)
        df["CATEGORY"] = category

        df.to_csv(to_csv_fn, index=False)
        BJ_GS_NPY().csvfile(csv_fn=to_csv_fn, to_csv_fn=to_csv_fn, )
        print(df)

    def func4():
        df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\is_soil_test\spl1.csv")

        def scatter_name(c_field_name, cname, color):
            _df = df[df[c_field_name] == cname]
            plt.scatter(_df[x_field_name], _df[y_field_name], label=cname, color=color)

        x_field_name, y_field_name = "NDVI", "NDWI"
        scatter_name("CNAME", "VEG", "green")
        # scatter_name("CNAME", "WAT", "blue")
        scatter_name("CNAME", "SOIL", "yellow")
        # scatter_name("CNAME", "IS", "red")

        plt.legend()
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.xlabel(x_field_name)
        plt.ylabel(y_field_name)
        plt.show()

    def func5():
        gr = GDALRaster(SHH2Config.CD_ENVI_FN)
        data = gr.readGDALBand("NDWI")
        gr.save(data, r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\is_soil_test\data\ndwi.dat")

    def func6():
        fns = [
            r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240614H170915\OPT_imdc.tif"
            , r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240614H170915\OPT+AS_imdc.tif"
            , r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240614H170915\OPT+AS+DE_imdc.tif"
            , r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240614H170915\OPT+BS_imdc.tif"
            , r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240614H170915\OPT+C2_imdc.tif"
            , r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240614H170915\OPT+DE_imdc.tif"
            , r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240614H170915\OPT+GLCM_imdc.tif"
            , r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240614H170915\OPT+HA_imdc.tif"
            , r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240614H170915\OPT+SARGLCM_imdc.tif"
        ]

        df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\is_soil_test\sh2_spl26_ist2.csv")

        for fn in fns:
            gs = GDALSamplingFast(fn)
            data = gs.sampling(df["X"].tolist(), df["Y"].tolist())
            name = getfilenamewithoutext(fn)
            df[name] = data["FEATURE_1"]
        df.to_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\is_soil_test\sh2_spl26_ist3.csv", index=False)

    def func7():
        x_len, y_len = RESOLUTION_ANGLE * 200, RESOLUTION_ANGLE * 200
        gr = GDALRaster(r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\SHH2_CD2_look.tif")
        x0, x1, y0, y1 = gr.raster_range
        n_x = int((x1 - x0) / x_len) + 1
        n_y = int((y1 - y0) / y_len) + 1
        gjw = GEOJsonWriteWGS84("SRT")
        n = 1
        to_dict = {"X": [], "Y": [], "SRT": [], "CATEGORY": []}
        for i in range(n_x):
            for j in range(n_y):
                gjw.addPolygon([[
                    [x0 + i * x_len, y0 + j * y_len, ],
                    [x0 + i * x_len, y0 + (j + 1) * y_len, ],
                    [x0 + (i + 1) * x_len, y0 + (j + 1) * y_len, ],
                    [x0 + (i + 1) * x_len, y0 + j * y_len, ],
                    [x0 + i * x_len, y0 + j * y_len, ],
                ]], SRT=n)
                data = np.array([
                    [x0 + i * x_len, y0 + j * y_len, ],
                    [x0 + i * x_len, y0 + (j + 1) * y_len, ],
                    [x0 + (i + 1) * x_len, y0 + (j + 1) * y_len, ],
                    [x0 + (i + 1) * x_len, y0 + j * y_len, ],
                ]).mean(axis=0)
                to_dict["X"].append(float(data[0]))
                to_dict["Y"].append(float(data[1]))
                to_dict["SRT"].append(int(n))
                to_dict["CATEGORY"].append(1)
                n += 1

        gjw.save(r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\is_soil_test\sh2_spl26_ist_grids1.geojson")
        print(pd.DataFrame(to_dict))
        pd.DataFrame(to_dict).to_csv(
            r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\is_soil_test\sh2_spl26_ist_grids1.csv")

    func1()


if __name__ == "__main__":
    sampling()
