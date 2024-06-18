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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal_array

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.GDALUtils import GDALSampling, GDALSamplingFast
from SRTCodes.OGRUtils import sampleSpaceUniform
from SRTCodes.SRTSample import GEOJsonWriteWGS84
from SRTCodes.Utils import getRandom, getfilenamewithoutext
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
    QD_GS_NPY().csvfile(
        csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\2\sh2_spl252_4.csv",
        to_csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\2\sh2_spl252_4_spl.csv",
    )
    # gs = GDALSampling(r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\SHH2_BJ2_envi.dat")
    # gs.csvfile(
    #     csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\sh2_spl27_1.csv",
    #     to_csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\sh2_spl27_1_spl.csv",
    # )


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


def main():
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

        fns = ['OPT', 'OPT+AS', 'OPT+DE', 'OPT+AS+DE', 'OPT+GLCM', 'OPT+BS', 'OPT+C2', 'OPT+HA', 'OPT+SARGLCM']
        images = {fn: os.path.join(dirname, fn + "_imdc.tif") for fn in fns}
        gss = {fn: GDALSampling(os.path.join(dirname, fn + "_imdc.tif")) for fn in fns}
        for fn in images:
            gs = gss[fn]
            categorys = gs.sampling(df["X"].tolist(), df["Y"].tolist())
            for name in categorys:
                df[fn] = categorys[name]
                break
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
    return


if __name__ == "__main__":
    sampling()
