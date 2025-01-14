# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2Funcs.py
@Time    : 2024/6/8 21:36
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2Funcs
-----------------------------------------------------------------------------"""
import os
from shutil import copyfile

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from osgeo import gdal

from SRTCodes.GDALDraw import GDALDrawImages
from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.GDALUtils import RasterToVRTS, GDALSamplingFast
from SRTCodes.OGRUtils import sampleSpaceUniform
from SRTCodes.Utils import FRW, DirFileName, FN, filterFileEndWith, getfilenamewithoutext
from Shadow.Hierarchical import SHH2Config


def main():
    plt.rcParams.update({"font.size":16})

    def func1():
        gdi = GDALDrawImages(win_size=(201, 201))

        qd_name = gdi.addGeoRange(SHH2Config.QD_ENVI_FN, SHH2Config.QD_RANGE_FN)
        bj_name = gdi.addGeoRange(SHH2Config.BJ_ENVI_FN, SHH2Config.BJ_RANGE_FN)
        cd_name = gdi.addGeoRange(SHH2Config.CD_ENVI_FN, SHH2Config.CD_RANGE_FN)
        gdi.addCategoryColor("color", {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255)})

        # gdi.addRCC("RGB", bj_name, cd_name, qd_name, channel_list=["Red", "Green", "Blue"])
        gdi.addRCC("NRG", bj_name, cd_name, qd_name, channel_list=["NIR", "Red", "Green"])
        gdi.addRCC("AS_VV", bj_name, cd_name, qd_name, channel_list=["AS_VV"])
        # gdi.addRCC("DE_VV", bj_name, cd_name, qd_name, channel_list=["DE_VV"])

        gdi.addRCC("Opt",
                   r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240813H104039\2\qd-o_imdc.tif",
                   channel_list=[0], is_01=False, is_min_max=False, )

        gdi.addRCC("OAD",
                   r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240813H104039\2\qd-oad_imdc.tif",
                   channel_list=[0], is_01=False, is_min_max=False, )

        gdi.addRCC("DLoss",
                   r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240813H104039\6\qd-ReduceLogits_imdc.tif",
                   channel_list=[0], is_01=False, is_min_max=False, )

        column_names = ["S2", "SAR", "Opt", "Opt AS DE", "DLoss"]
        row_names = []

        def add_row(name, x, y):
            n_row = len(row_names)
            gdi.addAxisDataXY(n_row, 0, "NRG", x, y, min_list=[200, 200, 200, ], max_list=[3000, 2000, 2000, ])
            gdi.addAxisDataXY(n_row, 1, "AS_VV", x, y, min_list=[-14], max_list=[6])
            gdi.addAxisDataXY(n_row, 2, "Opt", x, y, color_name="color")
            gdi.addAxisDataXY(n_row, 3, "OAD", x, y, color_name="color")
            gdi.addAxisDataXY(n_row, 4, "DLoss", x, y, color_name="color")
            row_names.append(name)

        add_row("(1)   ", 120.14923, 36.30117)
        add_row("(2)   ", 120.27785, 36.38684)

        gdi.draw(n_rows_ex=3.0, n_columns_ex=3.0, row_names=row_names, column_names=column_names)

        plt.show()

    def func2():
        def sigmoid(_x):
            return 1 / (1 + np.exp(-_x))
        x0 = np.linspace(-10, 0, 100)
        x1 = np.linspace(0, 10, 100)
        plt.figure(figsize=(6,6))
        plt.plot(x0, sigmoid(x0), "y")
        plt.plot(x1, sigmoid(x1), "r")
        plt.xlim([-10, 10])
        plt.ylim([-0.05, 1.05])
        plt.show()

    return func1()


def method_name6():
    def dirname_imdc_fn(_dirname):
        for fn in os.listdir(_dirname):
            if fn.endswith("_imdc.tif"):
                return os.path.join(_dirname, fn)

    raster_fns = {
        "QingDao_ML_VHL3": dirname_imdc_fn(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240805H105502"),
        "BeiJing_ML_VHL3": dirname_imdc_fn(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240806H094617"),
        "ChengDu_ML_VHL3": dirname_imdc_fn(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240806H101049"),
        "QingDao_ML_Category4": dirname_imdc_fn(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240806H101844"),
        "BeiJing_ML_Category4": dirname_imdc_fn(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240806H102411"),
        "ChengDu_ML_Category4": dirname_imdc_fn(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240806H102804"),
    }
    gr = GDALRaster(raster_fns["ChengDu_ML_Category4"])
    data = gr.readGDALBand(1)
    categorys, n = np.unique(data, return_counts=True)
    print(n / np.sum(n))


def method_name5():
    def func1():
        dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Mods\Temp")
        city_name = "qd"
        dirname = dfn.fn(city_name)
        to_dirname = r"F:\ProjectSet\Shadow\Hierarchical\Mods\Temp\imdcs"
        for f in os.listdir(dirname):
            if f.endswith("_imdc.tif"):
                fn = os.path.join(dirname, f)
                to_fn = os.path.join(to_dirname, "{}_{}".format(city_name, f))
                copyfile(fn, to_fn)

    def func2():
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\1\shh2_qd1.csv"
        to_csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\1\shh2_qd12.csv"
        df = pd.read_csv(csv_fn)
        fns = filterFileEndWith(r"F:\ProjectSet\Shadow\Hierarchical\Mods\Temp\imdcs", "_imdc.tif")
        names = []
        for fn in fns:
            name = getfilenamewithoutext(fn)
            names.append(name)
            if not os.path.isfile(to_csv_fn):
                gsf = GDALSamplingFast(fn)
                print(name)
                df[name] = gsf.samplingToData(df["X"], df["Y"], ).ravel()
        if not os.path.isfile(to_csv_fn):
            df.to_csv(to_csv_fn, index=False)
        df = pd.read_csv(to_csv_fn)
        data = df[names].values
        category = np.unique(data)
        to_list = []
        for line in df.to_dict("records"):
            n_dict = {cate: 0 for cate in category}
            for name in names:
                n = line[name]
                n_dict[n] += 1
            to_list.append(n_dict)
        df_data = pd.DataFrame(to_list)
        for k in df_data:
            df[k] = df_data[k]
        n_names = [1, 2, 3, 4]
        df["CATEGORY"] = np.argmax(df[n_names].values, axis=1) + 1
        df = df.sort_values(["CATEGORY", 1, 2, 3, 4], ascending=[True, False, False, False, False, ])
        df.to_csv(to_csv_fn, index=False)

    return func2()


def method_name4():
    print('F:\\ProjectSet\\Shadow\\Hierarchical\\Images\\ChengDu\\SH22\\SHH2_CD2_range2.json')
    dfn = DirFileName(r"E:\ImageData\GLCM")
    is_save = True

    def save(raster_fn, json_fn, city_name):
        gr = GDALRaster(raster_fn)
        range_dict = FRW(json_fn).readJson()
        print(gr, range_dict)

        def getdata(name):
            _data = gr.readGDALBand(name)
            x_min, x_max = range_dict[name]["min"], range_dict[name]["max"]
            _data = np.clip(_data, x_min, x_max)
            print(name, np.min(_data), np.max(_data))
            _data = (_data - x_min) / (x_max - x_min)
            if is_save:
                to_fn = dfn.fn(city_name, "{}_{}".format(city_name, name))
                print(to_fn)
                gr.save(_data, to_fn, fmt="ENVI", dtype=gdal.GDT_Float32, descriptions=name)
            return _data

        as_vv_data = getdata("AS_VV")
        as_vh_data = getdata("AS_VH")
        de_vv_data = getdata("DE_VV")
        de_vh_data = getdata("DE_VH")

    save(SHH2Config.QD_ENVI_FN, SHH2Config.QD_RANGE_FN, "QD")
    save(SHH2Config.BJ_ENVI_FN, SHH2Config.BJ_RANGE_FN, "BJ")
    save(SHH2Config.CD_ENVI_FN, SHH2Config.CD_RANGE_FN, "CD")


def method_name3():
    rtv = RasterToVRTS(SHH2Config.BJ_ENVI_FN)
    rtv.save(r"F:\ProjectSet\Shadow\Hierarchical\Images\temp\1\BJ")


def method_name2():
    def func1(csv_fn):
        df = pd.read_csv(csv_fn)
        coors2, out_index_list = sampleSpaceUniform(df[["X", "Y"]].values.tolist(), x_len=500, y_len=500,
                                                    is_trans_jiaodu=True, ret_index=True)
        is_save = np.zeros(len(df))
        is_save[out_index_list] = 1
        df["IS_SAVE"] = is_save
        df = df[df["IS_SAVE"] == 1]
        print(len(out_index_list))
        df.to_csv(FN(csv_fn).changext("_ssu2.csv"), index=False)

        return

    func1(r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\2\sh2_spl27_2_cd_train1.csv")


def method_name1():
    # data = {}
    #
    # def tongjishuliang(name, csv_fn, c_field_name="CNAME"):
    #     df = pd.read_csv(csv_fn)
    #     if "TEST" in df:
    #         df = df[df["TEST"] == 1]
    #     data[name] = pd.value_counts(df[c_field_name]).to_dict()
    #
    # tongjishuliang("bj_test", r"F:\Week\20240609\Data\samples\bj_test.csv")
    # tongjishuliang("bj_train", r"F:\Week\20240609\Data\samples\bj_train.csv")
    # tongjishuliang("cd_test", r"F:\Week\20240609\Data\samples\cd_test.csv")
    # tongjishuliang("cd_train", r"F:\Week\20240609\Data\samples\cd_train.csv")
    # tongjishuliang("qd_test", r"F:\Week\20240609\Data\samples\qd_test.csv")
    # tongjishuliang("qd_train", r"F:\Week\20240609\Data\samples\qd_train.csv")
    #
    # df = pd.DataFrame(data).T
    # print(df)
    # df.to_csv(r"F:\Week\20240609\Data\samples\spl_counts.csv")
    dfn = DirFileName(r"F:\Week\20240609\Data\samples")
    to_dict = FRW(r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\1\20240610H112357_bjtest.json").readJson()
    df = pd.DataFrame(to_dict["acc"])
    df.to_csv(dfn.fn("cd.csv"))
    print(df)


def method_name():
    gr = GDALRaster(r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240607H100838\OPTASDE_epoch36_imdc1.tif")
    data = gr.readAsArray()
    d, n = np.unique(data, return_counts=True)
    print(n * 1.0 / np.sum(n))


if __name__ == "__main__":
    main()
