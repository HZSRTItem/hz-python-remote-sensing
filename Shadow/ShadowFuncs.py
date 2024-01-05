# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowFuncs.py
@Time    : 2023/7/22 3:14
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : GEOCodes of ShadowTest
-----------------------------------------------------------------------------"""
import csv
import os
import os.path
import random
import xml.etree.ElementTree as ElementTree

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axisartist import AxesZero
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
from osgeo import gdal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

import scienceplots
from SRTCodes.GDALRasterClassification import GDALRasterClassificationAccuracy
from SRTCodes.GDALRasterIO import GDALRaster, GDALRasterCollection, readGEORaster, saveGEORaster
from SRTCodes.GDALRasterIO import GDALRasterFeatures
from SRTCodes.GDALUtils import gdalStratifiedRandomSampling, samplingToCSV, RasterToVRTS
from SRTCodes.GEEUtils import GEEImageProperty, geeCSVSelectPropertys
from SRTCodes.GeoRasterRW import GeoRasterWrite
from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.NumpyUtils import neighborhood, calPCA
from SRTCodes.OGRUtils import SRTESRIShapeFileRead, sampleSpaceUniform
from SRTCodes.SRTDraw import SRTDrawHist
from SRTCodes.Utils import savecsv, readcsv, DirFileName, Jdt
from Shadow.ShadowDraw import ShadowDrawDirectLength, cal_10log10
from Shadow.ShadowGeoDraw import _10log10, DrawShadowImage_0
from Shadow.ShadowMainQingDao import ShadowMainQD
from Shadow.ShadowUtils import ShadowSampleAdjustNumber, ShadowFindErrorSamples, ShadowTestAll

imdcDirname = DirFileName(r"F:\ProjectSet\Shadow\QingDao\Mods\20230707H200910")

scienceplots.init()


def main():
    imdc_fn1 = r"F:\ProjectSet\Shadow\QingDao\Mods\20231226H093225\SPL_SH-SVM-TAG-OPTICS-AS_imdc.dat"
    imdc_fn2 = r"F:\ProjectSet\Shadow\QingDao\Mods\20231226H093225\SPL_SH-SVM-TAG-OPTICS-DE_imdc.dat"
    gr1 = GDALRaster(imdc_fn1)
    gr2 = GDALRaster(imdc_fn2)
    imdc1 = gr1.readAsArray()
    imdc2 = gr2.readAsArray()
    imdc_diff = (imdc1 != imdc2) * 1
    gr1.save(imdc_diff, r"F:\ProjectSet\Shadow\Analysis\9\qd_as_de_diff1.tif", dtype=gdal.GDT_Byte, fmt="GTiff")

    haha = 0


def method_name55():
    # DEM
    gr = GDALRaster(r"F:\ProjectSet\Shadow\MkTu\DEM\DEM_china_1.tif")
    d = gr.readAsArray()
    d[d < 0] = 1
    gr.save(d, save_geo_raster_fn=r"F:\ProjectSet\Shadow\MkTu\DEM\DEM_china_3.tif", fmt="GTiff")
    print(d.shape)
    # func3()


def method_name54():
    # 调整样本 Analysis 8
    # csv_fn = r"F:\ProjectSet\Shadow\BeiJing\Mods\20231225H110303\train_data.csv"
    # mod_dirname = r"F:\ProjectSet\Shadow\BeiJing\Mods\20231225H110303"
    def func1(mod_dirname, csv_fn=None):
        method_name47(mod_dirname, True)
        if csv_fn is None:
            csv_fn = os.path.join(mod_dirname, "train_data.csv")
        print(csv_fn)
        sfes = ShadowFindErrorSamples()
        sfes.addCategoryCode(IS=1, VEG=2, SOIL=3, WAT=4, IS_SH=5, VEG_SH=6, SOIL_SH=7, WAT_SH=8)
        df = pd.read_csv(csv_fn)
        # df_test = df[df["TEST"] == 0]
        sfes.initDataFrame(df)
        sfes.addDataFrame()
        to_csv_fn = sfes.fitImdcCSVS(mod_dirname, filter_list=["SPL_SH", "SVM", "OPTICS"])
        print(to_csv_fn)

    # func1(r"F:\ProjectSet\Shadow\BeiJing\Mods\20231225H110303")
    # func1(r"F:\ProjectSet\Shadow\ChengDu\Mods\20231225H110314")
    # func1(r"F:\ProjectSet\Shadow\QingDao\Mods\20231225H110238")
    # updateShadowSamplesCategory(
    #     chang_csv_fn=r"F:\ProjectSet\Shadow\Analysis\8\train\soil_spl_1.csv",
    #     o_csv_fn=r"F:\ProjectSet\Shadow\Analysis\8\train\bj_train_data_t.csv",
    #     is_change_fields=True
    # )
    def func2():
        n = 10
        df = pd.read_excel(r"F:\ProjectSet\Shadow\Analysis\8\train\train1.xlsx", sheet_name="SOIL1")
        plt.scatter(df.loc[:n, "Red"], df.loc[:n, "NIR"])
        x1, x2, y1, y2 = 2899, 948, 3546, 1281
        k = (y2 - y1) / (x2 - x1)

        def y_soil(x):
            return k * (x - x1) + y1

        x = df["Red"].values
        y = y_soil(x)

        plt.plot([2899, 948], [3546, 1281])
        plt.scatter(x[:n], y[:n], c="r")
        plt.show()

        df["SOIL_Y"] = y
        df.to_csv(r"F:\ProjectSet\Shadow\Analysis\8\train\t1.csv")

    def func3():
        csv_fn = r"F:\ProjectSet\Shadow\Analysis\8\chengdu\sh_cd_sample_spl.csv"
        df = pd.read_csv(csv_fn)
        cnames = [
            'IS', 'IS_SH',
            # 'SOIL', 'SOIL_SH',
            'VEG', 'VEG_SH',
            # 'WAT', 'WAT_SH'
        ]
        for cname in cnames:
            df_tmp = df[df["CNAME"] == cname]
            plt.scatter(df_tmp["Blue"], df_tmp["Green"], label=cname)

        plt.plot([0, 5000], [0, 5000])
        plt.legend()
        plt.show()
        print(df)
        print(list(pd.unique(df["CNAME"])))
        # plt.scatter()


def method_name53():
    def func1(dirname):
        for fn in os.listdir(dirname):
            if ("_imdc.dat" in fn) and ("SVM" in fn) and ("SPL_SH" in fn) and ("OPTICS" in fn):
                print(os.path.join(dirname, fn))

    def func2():
        output_vrt = r"F:\ProjectSet\Shadow\QingDao\Mods\20230707H200910\output.vrt"
        ds = gdal.Open(output_vrt)
        d = ds.ReadAsArray()
        out_d = np.zeros([5, d.shape[1], d.shape[2]])
        for i in range(d.shape[1]):
            for j in range(d.shape[2]):
                out_d[:, i, j] = np.bincount(d[:, i, j], minlength=5)
            print(i)
        grw = GeoRasterWrite(output_vrt)
        grw.save(out_d, r"F:\ProjectSet\Shadow\QingDao\Mods\20230707H200910\imdc2.dat", dtype=gdal.GDT_Int16)

    # func1(r"F:\ProjectSet\Shadow\BeiJing\Mods\20231225H110303")
    method_name47()


def method_name52():
    # 测试网格
    class RUNT(ShadowMainQD):

        def __init__(self):
            super().__init__()

        def t_getSample(self):
            spls = self.sct.spl_types
            feats = self.sct.feat_types
            tags = self.sct.tag_types
            print(spls, feats, tags, sep="\n")
            return self.sct.getSample(
                spls['SPL_SH'], feats["OPTICS"] + feats["AS"] + feats["DE"], tags['TAG'])

    rt = RUNT()
    rt.shadowTraining()
    x_train, y_train, x_test, y_test = rt.t_getSample()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    def t_svm():
        find_grid = {"gamma": np.logspace(-1, 1, 20), "C": np.linspace(0.01, 10, 20)}
        print(find_grid)
        n = 1
        for k in find_grid:
            n *= len(find_grid[k])
        n_iter = int(0.2 * n)
        print("Number of Iter: %d", n_iter)

        for i in range(1, 60, 5):
            print("-" * 60)
            print("Number of Iter: %d", i)
            svm_rs_cv = RandomizedSearchCV(
                estimator=SVC(kernel="rbf", cache_size=5000),
                param_distributions=find_grid,
                n_iter=i
            )
            svm_rs_cv.fit(x_train, y_train)
            print(svm_rs_cv.best_params_)
            print(svm_rs_cv.best_estimator_)
            print("Train Accuracy:", svm_rs_cv.score(x_train, y_train))
            print("Test Accuracy:", svm_rs_cv.score(x_test, y_test))

    def t_rf():
        find_grid = {
            "n_estimators": list(range(1, 150, 10)),
            "max_depth": list(range(1, 20)),
            "min_samples_leaf": list(range(1, 5)),
            "min_samples_split": list(range(2, 10))
        }
        print(find_grid)
        n = 1
        for k in find_grid:
            n *= len(find_grid[k])
        n_iter = int(0.2 * n)
        print("Number of Iter: %d", n_iter)

        for i in range(1, 60, 5):
            print("-" * 60)
            print("Number of Iter: %d", i)
            svm_rs_cv = RandomizedSearchCV(
                estimator=RandomForestClassifier(n_jobs=-1),
                param_distributions=find_grid,
                n_iter=i
            )
            svm_rs_cv.fit(x_train, y_train)
            print(svm_rs_cv.best_params_)
            print(svm_rs_cv.best_estimator_)
            print("Train Accuracy:", svm_rs_cv.score(x_train, y_train))
            print("Test Accuracy:", svm_rs_cv.score(x_test, y_test))

    t_rf()


def method_name51():
    # 调整精度
    def func1():
        gr_as_de = GDALRaster(r"F:\ProjectSet\Shadow\BeiJing\Mods\20231221H224253\SPL_SH-SVM-TAG-OPTICS-AS-DE_imdc.dat")
        gr_de = GDALRaster(r"F:\ProjectSet\Shadow\BeiJing\Mods\20231221H224253\SPL_SH-SVM-TAG-OPTICS-AS_imdc.dat")
        gr_as = GDALRaster(r"F:\ProjectSet\Shadow\BeiJing\Mods\20231221H224253\SPL_SH-SVM-TAG-OPTICS-DE_imdc.dat")
        gr = GDALRaster(r"F:\ProjectSet\Shadow\BeiJing\Mods\20231221H224253\SPL_SH-SVM-TAG-OPTICS_imdc.dat")

        d_as_de = gr_as_de.readAsArray()
        d_de = gr_de.readAsArray()
        d_as = gr_as.readAsArray()
        d = gr.readAsArray()
        d1 = (d != d_as_de) * 1
        print(np.sum(d1 * 1))
        print(d.shape[0] * d.shape[1])
        print((np.sum(d1 * 1)) / (d.shape[0] * d.shape[1]))
        gr.save(d1, r"F:\ProjectSet\Shadow\Analysis\6\bj\bj_diff.dat")

        df = {"X": [], "Y": [], "AS_DE": [], "OPT": []}
        c_dict = {1: "IS", 2: "VEG", 3: "SOIL", 4: "WAT"}
        for i in range(d_as_de.shape[0]):
            for j in range(d_as_de.shape[1]):
                if d_as_de[i, j] != d[i, j]:
                    x, y = gr.coorRaster2Geo(i + 0.5, j + 0.5)
                    df["X"].append(x)
                    df["Y"].append(y)
                    df["AS_DE"].append(c_dict[int(d_as_de[i, j])])
                    df["OPT"].append(c_dict[int(d[i, j])])

        df = pd.DataFrame(df)
        df.to_csv(r"F:\ProjectSet\Shadow\Analysis\6\bj\bj_sade_opt_diff.csv")
        print(df)

    method_name47()


def method_name50():
    # 阴影中心点画图
    def func1():
        dsi = DrawShadowImage_0(30, 30, 116.45538071, 39.95543882,
                                raster_fn=r"F:\ProjectSet\Shadow\Release\BeiJingImages\SH_BJ_envi.dat",
                                to_dirname=r"F:\ProjectSet\Shadow\Analysis\6\bj\toIm",
                                width=6, height=6, is_expand=True)
        categorys = [1, (255, 0, 0), 2, (0, 255, 0), 3, (255, 255, 0), 4, (0, 0, 255)]

        dsi.drawOptical("RGB", channel_list=[2, 1, 0])
        dsi.drawOptical("NRB", channel_list=[3, 2, 1])
        dsi.drawIndex("NDVI", d_min=-0.6, d_max=0.9)
        dsi.drawIndex("NDWI", d_min=-0.7, d_max=0.8)
        dsi.drawSAR("AS_VV", d_min=-24.609674, d_max=5.9092603)
        dsi.drawSAR("AS_VH", d_min=-31.865038, d_max=-5.2615275)
        dsi.drawSAR("AS_C11", d_min=-22.61998, d_max=5.8634768)
        dsi.drawSAR("AS_C22", d_min=-28.579813, d_max=-5.2111626)
        dsi.drawSAR("AS_Lambda1", d_min=-21.955856, d_max=6.124724)
        dsi.drawSAR("AS_Lambda2", d_min=-29.869734, d_max=-8.284683)
        dsi.drawSAR("DE_VV", d_min=-27.851603, d_max=5.094706)
        dsi.drawSAR("DE_VH", d_min=-35.427082, d_max=-5.4092093)
        dsi.drawSAR("DE_C11", d_min=-26.245598, d_max=4.9907513)
        dsi.drawSAR("DE_C22", d_min=-32.04232, d_max=-5.322515)
        dsi.drawSAR("DE_Lambda1", d_min=-25.503738, d_max=5.2980003)
        dsi.drawSAR("DE_Lambda2", d_min=-33.442368, d_max=-8.68537)
        dsi.drawImdc("IMDC_AS_DE",
                     raster_fn=r"F:\ProjectSet\Shadow\BeiJing\Mods\20231221H224253\SPL_SH-SVM-TAG-OPTICS-AS-DE_imdc.dat",
                     categorys=categorys)
        dsi.drawImdc("IMDC_AS",
                     raster_fn=r"F:\ProjectSet\Shadow\BeiJing\Mods\20231221H224253\SPL_SH-SVM-TAG-OPTICS-DE_imdc.dat",
                     categorys=categorys)
        dsi.drawImdc("IMDC_DE",
                     raster_fn=r"F:\ProjectSet\Shadow\BeiJing\Mods\20231221H224253\SPL_SH-SVM-TAG-OPTICS-AS_imdc.dat",
                     categorys=categorys)
        dsi.drawImdc("IMDC_OPT",
                     raster_fn=r"F:\ProjectSet\Shadow\BeiJing\Mods\20231221H224253\SPL_SH-SVM-TAG-OPTICS_imdc.dat",
                     categorys=categorys)

    method_name48()


def method_name49():
    # 准备2023年12月23日的组会，上周汇报的分层实验的结果效果不好，找到分类错误的样本，看看是怎么回事，也有可能是样本的问题
    def func1():
        fn = r"F:\ProjectSet\Shadow\Hierarchical\Mods\2023121401\QingDao\QingDao20231214H093733_imdc.dat"
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Analysis\1\QingDao20231214H093733_imdc.dat_test.csv"

        sfes = ShadowFindErrorSamples()
        sfes.addCategoryCode(IS=1, VEG=2, SOIL=3, WAT=4, IS_SH=5, VEG_SH=6, SOIL_SH=7, WAT_SH=8)
        df = pd.read_csv(csv_fn)
        df_test = df[df["TEST"] == 0]
        sfes.imdcFN(fn)
        sfes.initDataFrame(df_test)
        sfes.addDataFrame()
        cm = sfes.calCMImdc()
        print(cm.fmtCM())
        sfes.fitImdc()
        sfes.toCSV(
            keys=['Blue', 'Green', 'Red', 'NIR', "SRT", "CATEGORY", "NDVI", "NDWI",
                  'AS_VV', 'AS_VH', 'DE_VV', 'DE_VH'],
            sort_column=sfes.t_f_name)
        print(fn)

    method_name48()


def method_name48():
    # 对每个结果都调一下，调一遍测试一遍结果
    c_code_dict = {"NOT_KNOW": 0,
                   "IS": 11,
                   "IS_SHADOW": 12,
                   "VEG": 21,
                   "VEG_SHADOW": 22,
                   "SOIL": 31,
                   "SOIL_SHADOW": 32,
                   "WATER": 41,
                   "WATER_SHADOW": 42}
    c_dict = {"NOT_KNOW": "NOT_KNOW",
              "IS": "IS",
              "IS_SHADOW": "IS_SH",
              "VEG": "VEG",
              "VEG_SHADOW": "VEG_SH",
              "SOIL": "SOIL",
              "SOIL_SHADOW": "SOIL_SH",
              "WATER": "WAT",
              "WATER_SHADOW": "WAT_SH"}

    # c_dict = {"NOT_KNOW": "NOT_KNOW",
    #           "IS": "IS",
    #           "IS_SH": "IS_SH",
    #           "VEG": "VEG",
    #           "VEG_SH": "VEG_SH",
    #           "SOIL": "SOIL",
    #           "SOIL_SH": "SOIL_SH",
    #           "WAT": "WAT",
    #           "WAT_SH": "WAT_SH"}
    # c_code_dict = {"NOT_KNOW": 0,
    #                "IS": 11,
    #                "IS_SH": 12,
    #                "VEG": 21,
    #                "VEG_SH": 22,
    #                "SOIL": 31,
    #                "SOIL_SH": 32,
    #                "WAT": 41,
    #                "WAT_SH": 42}

    def func1(csv_fn, dirname):
        sta = ShadowTestAll()
        sta.addCategoryCode(NOT_KNOW=0, IS=1, VEG=2, SOIL=3, WAT=4, IS_SH=5, VEG_SH=6, SOIL_SH=7, WAT_SH=8)
        df = pd.read_csv(csv_fn)
        sta.initDataFrame(df)
        sta.addDataFrame(c_column_name="O_CNAME")
        sta.fitDirName(dirname)
        print(len(df))

    def func2(csv_fn1, csv_fn2):
        df1 = pd.read_csv(csv_fn1).set_index("SRT")
        df2 = pd.read_csv(csv_fn2).set_index("SRT", drop=False)
        df_add = {"X": [], "Y": [], "O_CNAME": [], "IS_CHCANGE": [], "CATEGORY": []}
        for i, line in df1.iterrows():
            category_name = str(line["CATEGORY_NAME"])
            # print(category_name, c_dict[category_name], c_code_dict[category_name])

            if pd.isna(i):
                df_add["X"].append(float(line["X"]))
                df_add["Y"].append(float(line["Y"]))
                df_add["O_CNAME"].append(c_dict[category_name])
                df_add["CATEGORY"].append(c_code_dict[category_name])
                df_add["IS_CHCANGE"].append(1)
            else:
                if df2.loc[i, "IS_CHCANGE"] == 0:
                    if df2.loc[i, "O_CNAME"] != c_dict[category_name]:
                        df2.loc[i, "O_CNAME"] = c_dict[category_name]
                        df2.loc[i, "CATEGORY"] = c_code_dict[category_name]
                        df2.loc[i, "IS_CHCANGE"] = 1
        df2 = pd.concat([df2, pd.DataFrame(df_add)])
        df2.to_csv(csv_fn2, index=False)

    # QingDao
    # func1(r"F:\ProjectSet\Shadow\Analysis\6\qd_test.csv", r"F:\ProjectSet\Shadow\QingDao\Mods\20231221H224548")
    # func2(r"F:\ProjectSet\Shadow\Analysis\6\change3.csv", r"F:\ProjectSet\Shadow\Analysis\6\qd_test.csv")

    # BeiJing
    func1(r"F:\ProjectSet\Shadow\Analysis\6\bj\bj_test.csv", r"F:\ProjectSet\Shadow\BeiJing\Mods\20231221H224253")
    # func2(r"F:\ProjectSet\Shadow\Analysis\6\bj\bj_chang1.csv", r"F:\ProjectSet\Shadow\Analysis\6\bj\bj_test.csv")

    # 青岛分层
    # func1(r"F:\ProjectSet\Shadow\Hierarchical\Analysis\1\qd_test.csv",
    #       r"F:\ProjectSet\Shadow\Hierarchical\Mods\2023121401\QingDao")
    # func2(r"F:\ProjectSet\Shadow\Hierarchical\Analysis\1\qd_change2.csv", r"F:\ProjectSet\Shadow\Hierarchical\Analysis\1\qd_test.csv")


def method_name47(mod_dirname=None, is_test=True):
    # 重新使用样本测试精度
    def func1(dirname, csv_fn=None):
        if csv_fn is None:
            for fn in os.listdir(dirname):
                if "train_data.csv" == fn:
                    csv_fn = os.path.join(dirname, fn)
        print("-" * 60)
        print(dirname)
        for fn in os.listdir(dirname):
            if os.path.splitext(fn)[1] == ".dat":
                fn = os.path.join(dirname, fn)
                sfes = ShadowFindErrorSamples()
                sfes.addCategoryCode(IS=1, VEG=2, SOIL=3, WAT=4, IS_SH=5, VEG_SH=6, SOIL_SH=7, WAT_SH=8)
                df = pd.read_csv(csv_fn)
                df_test = df
                if is_test:
                    df_test = df[df["TEST"] == 0]
                sfes.imdcFN(fn)
                sfes.initDataFrame(df_test)
                sfes.addDataFrame()
                cm = sfes.calCMImdc()
                print(cm.fmtCM())
                sfes.fitImdc()
                sfes.toCSV(
                    keys=['Blue', 'Green', 'Red', 'NIR', "SRT", "CATEGORY", "NDVI", "NDWI",
                          'AS_VV', 'AS_VH', 'DE_VV', 'DE_VH'],
                    sort_column=sfes.t_f_name)
                print(fn)

    def func2(dirname):
        n_list = None
        ks = {}
        n = 0
        for fn in os.listdir(dirname):
            if ("_test.csv" in fn) and ("NOSH" not in fn):
                df = pd.read_csv(os.path.join(dirname, fn))
                # df = df.sort_values(by="SRT")
                df = df.set_index("SRT").sort_index()
                if n_list is None:
                    n_list = np.zeros(len(df))
                    for k in df:
                        ks[k] = df[k].tolist()
                n_list += (df["T_F"].values == "FALSE_C") * 1

                # name1 = fn.split("_")[0] + fn.split("_")[1]
                # name1 = name1.split("-")
                # for name2 in name1:
                #     if name2 not in ks:
                #         ks[name2] = [0 for i in range(len(n_list))]
                #     if name2 in ks:
                #         ks[name2][n] = 1

                print(fn)

        ks["n_list"] = n_list
        savecsv(os.path.join(dirname, "n_list.csv"), ks)

    if mod_dirname is not None:
        func1(mod_dirname)
    # func1(r"F:\ProjectSet\Shadow\QingDao\Mods\20231221H224548", r"F:\ProjectSet\Shadow\Release\QingDaoSamples\sh_qd_sample_spl.csv")
    # func1(r"F:\ProjectSet\Shadow\BeiJing\Mods\20231221H224253", r"F:\ProjectSet\Shadow\Release\BeiJingSamples\sh_bj_sample_spl.csv")
    # func1(r"F:\ProjectSet\Shadow\ChengDu\Mods\20231221H224735", r"F:\ProjectSet\Shadow\Release\ChengDuSamples\sh_cd_sample_spl.csv")
    # func2(r"F:\ProjectSet\Shadow\QingDao\Mods\20231221H224548")
    # func2(r"F:\ProjectSet\Shadow\Release\ChengDuMods\20231117H112558")
    func1(r"F:\ProjectSet\Shadow\BeiJing\Mods\20231225H110303")


def method_name46():
    # 减少一些阴影下样本，让结果变的坏一点
    def func(fn, to_fn, re_dict):
        df = pd.read_csv(fn)
        df_select = df.loc[df["TEST"] == 1]
        df_end = df.loc[df["TEST"] != 1]
        ssan = ShadowSampleAdjustNumber()
        ssan.initDataFrame(df_select)
        ssan.initCKName()
        print(ssan.numbers())
        ssan.printNumber()
        ssan.adjustNumber(re_dict)
        print()
        ssan.printNumber()
        ssan.saveToCSV(to_fn, df_end)

    fn1 = r"F:\ProjectSet\Shadow\Release\BeiJingSamples\sh_bj_sample_spl.csv"
    fn2 = r"F:\ProjectSet\Shadow\Release\QingDaoSamples\sh_qd_sample_spl.csv"
    fn3 = r"F:\ProjectSet\Shadow\Release\ChengDuSamples\sh_cd_sample_spl.csv"
    to_fn1 = r"F:\ProjectSet\Shadow\Analysis\5\sh_bj_sample_spl2.csv"
    to_fn2 = r"F:\ProjectSet\Shadow\Analysis\5\sh_qd_sample_spl2.csv"
    to_fn3 = r"F:\ProjectSet\Shadow\Analysis\5\sh_cd_sample_spl2.csv"
    # func(fn1, to_fn1, {'IS': 1149, 'IS_SH': 100, 'SOIL': 151, 'VEG': 609, 'VEG_SH': 100, 'WAT': 472, 'WAT_SH': 3})
    # func(fn2, to_fn2, {'IS': 794, 'IS_SH': 100, 'SOIL': 261, 'SOIL_SH': 10, 'VEG': 880, 'VEG_SH': 100, 'WAT': 379, 'WAT_SH': 10})
    func(fn3, to_fn3,
         {'IS': 1151, 'IS_SH': 100, 'SOIL': 238, 'SOIL_SH': 1, 'VEG': 896, 'VEG_SH': 100, 'WAT': 245, 'WAT_SH': 6})


def method_name45():
    # 看一下在哪个区域分类不好，重点是精细的位置，精细的样本
    beijing_imdc_fn = r"F:\ProjectSet\Shadow\Hierarchical\Mods\2023121401\BeiJing\BeiJing20231213H194955_imdc.dat"
    qingdao_imdc_fn = r"F:\ProjectSet\Shadow\Hierarchical\Mods\2023121401\QingDao\QingDao20231214H093733_imdc.dat"
    chengdu_imdc_fn = r"F:\ProjectSet\Shadow\Hierarchical\Mods\2023121401\ChengDu\ChengDu20231214H093928_imdc.dat"


def method_name44():
    # 分层的模型的时候，在阴影的分类效果不好，可能是数据范围的问题，现在再看看hist
    sdh = SRTDrawHist()
    # sdh.addCSVFile(r"F:\ProjectSet\Shadow\MkTu\4.1Details\Samples\three_spl_spl.csv")
    sdh.addCSVFile(r"F:\Week\20231217\Data\three_spl_spl.csv")
    # "SRT", "X", "Y", "CNAME", "CATEGORY", "TAG", "TEST", "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
    # "OPT_asm", "OPT_con", "OPT_cor", "OPT_dis", "OPT_ent", "OPT_hom", "OPT_mean", "OPT_var",
    # "AS_VV", "AS_VH", "AS_VHDVV", "AS_C11", "AS_C12_imag", "AS_C12_real", "AS_C22", "AS_Lambda1", "AS_Lambda2",
    # "AS_SPAN", "AS_Epsilon", "AS_Mu", "AS_RVI", "AS_m", "AS_Beta",
    # "AS_VH_asm", "AS_VH_con", "AS_VH_cor", "AS_VH_dis", "AS_VH_ent", "AS_VH_hom", "AS_VH_mean", "AS_VH_var",
    # "AS_VV_asm", "AS_VV_con", "AS_VV_cor", "AS_VV_dis", "AS_VV_ent", "AS_VV_hom", "AS_VV_mean", "AS_VV_var",
    # "DE_VV", "DE_VH", "DE_VHDVV", "DE_C11", "DE_C12_imag", "DE_C12_real", "DE_C22","DE_Lambda1", "DE_Lambda2",
    # "DE_SPAN",  "DE_Epsilon", "DE_Mu", "DE_RVI", "DE_m", "DE_Beta",
    # "DE_VH_asm", "DE_VH_con", "DE_VH_cor", "DE_VH_dis", "DE_VH_ent", "DE_VH_hom", "DE_VH_mean", "DE_VH_var",
    # "DE_VV_asm", "DE_VV_con", "DE_VV_cor", "DE_VV_dis", "DE_VV_ent", "DE_VV_hom", "DE_VV_mean", "DE_VV_var"
    plt.style.use('science')
    sdh.category('SHADOW')

    def draw2(draw_name):
        if ("AS" in draw_name) or ("DE" in draw_name):
            sdh[draw_name].funcCal(cal_10log10)
        sdh.plot(draw_name, c=(0, 0, 0), category="NOSH")
        sdh.plot(draw_name, c=(100 / 255.0, 100 / 255.0, 100 / 255.0), category="SH")
        svg_fn = r"F:\Week\20231217\Data\{0}.svg".format(draw_name)
        plt.legend()
        plt.savefig(svg_fn, dpi=300, format="svg")
        print(draw_name, svg_fn)
        plt.show()

    def draw1(*draw_names):
        for draw_name in draw_names:
            draw2(draw_name)

    draw1("NDVI", "NDWI")
    # plt.legend()
    # plt.savefig(r"F:\Week\20231217\Data\test.svg", dpi=300, format="svg")
    # plt.show()


def method_name43():
    grf = GDALRasterFeatures(r"F:\ProjectSet\Shadow\Release\BeiJingImages\SH_BJ_envi.dat")
    d_as = grf["AS_VV"]
    d_de = grf["DE_VV"]
    d_as_de_min = np.maximum(d_as, d_de)
    d_as_de_min = _10log10(d_as_de_min)
    grf.addFeature("AS_DE_MAX_VV", d_as_de_min)
    grf.saveFeatureToGDALRaster("AS_DE_MAX_VV", save_geo_raster_dir=r"F:\ProjectSet\Shadow\Analysis\3")


def method_name42():
    # 文献整理
    def func1():
        fn = r"F:\Week\20231210\Data\sciencedirect_SAR_title.txt"
        title_list = []
        title_dict = {"year": 0, "title": "", }
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("T1"):
                    title_dict["title"] = line[5:]
                elif line.startswith("PY"):
                    title_dict["year"] = int(line[5:])
                elif line.startswith("ER"):
                    title_list.append(title_dict)
                    title_dict = {"title": "", "year": 0}
        df = pd.DataFrame(title_list)
        print(df)
        df.to_csv(r"F:\Week\20231210\Data\sciencedirect_SAR_title_ty.csv")

    def func2(fn=None, to_fn=None):
        if fn is None:
            fn = r"F:\Week\20231210\Data\sciencedirect_SAR_title.txt"
        title_list = []
        title_dict = {}
        with open(fn, "r", encoding="utf-8") as f:
            i = 1
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                if line.startswith("ER"):
                    title_list.append(title_dict)
                    title_dict = {}
                else:
                    try:
                        lines = line.split("-", 1)
                        k = lines[0].strip()
                        d = lines[1].strip()
                        if k not in title_dict:
                            title_dict[k] = d
                        else:
                            title_dict[k] += " | " + d
                    except:
                        print(i, line)
                i += 1
        df = pd.DataFrame(title_list)
        print()
        print(df)
        if to_fn is None:
            to_fn = r"F:\Week\20231210\Data\sciencedirect_SAR_title_all.csv"
        print()
        print(df.keys())
        df.to_csv(to_fn)

    def func3():
        filelist = [
            # r"F:\Week\20231210\Data\rse ScienceDirect_citations_1701844831876.ris",
            # r"F:\Week\20231210\Data\rseScienceDirect_citations_1701844767156.ris",
            # r"F:\Week\20231210\Data\rseScienceDirect_citations_1701844806745.ris",
            # r"F:\Week\20231210\Data\rseScienceDirect_citations_1701844853580.ris",

            # r"F:\Week\20231210\Data\tgrs IEEE Xplore Citation RIS Download 2023.12.6.15.9.32.ris",
            # r"F:\Week\20231210\Data\tgrs IEEE Xplore Citation RIS Download 2023.12.6.15.10.10.ris",
            # r"F:\Week\20231210\Data\tgrs IEEE Xplore Citation RIS Download 2023.12.6.15.10.41.ris",
            # r"F:\Week\20231210\Data\tgrsIEEE Xplore Citation RIS Download 2023.12.6.15.11.30.ris",

            r"F:\Week\20231210\Data\IS\isprsScienceDirect_citations_1701949697131.ris",
            r"F:\Week\20231210\Data\IS\jag ScienceDirect_citations_1701949786768.ris",
            r"F:\Week\20231210\Data\IS\ScienceDirect_citations_1701949611553.ris",
            r"F:\Week\20231210\Data\IS\tgrs IEEE Xplore Citation RIS Download 2023.12.7.19.51.52.ris",
        ]

        def read_text(filename):
            with open(filename, "r", encoding="utf-8") as fr:
                return fr.read()

        with open(r"F:\Week\20231210\Data\is_s.txt", "w", encoding="utf-8") as f:
            for fn in filelist:
                text = read_text(fn)
                f.write(text)

    def fun4():
        sheet_names = ["RSE原始数据", "ISPRS原始数据", "JAG原始数据", "TGRS原始数据", ]
        dfs = []
        for sheet_name in sheet_names:
            df = pd.read_excel(r"F:\ProjectSet\Shadow\QingDao\Dissertation\文献整理\1\ScienceDirectSAR.xlsx",
                               sheet_name=sheet_name)
            dfs.append(df)
        df_cat = pd.concat(dfs, axis=0)
        print(df_cat)
        df_cat.to_excel(r"F:\ProjectSet\Shadow\QingDao\Dissertation\文献整理\1\ScienceDirectSAR_cat.xlsx")

    def func5(excel_fn, field_name, find_str, to_fn, sheet_name=0):
        df = pd.read_excel(excel_fn, sheet_name=sheet_name)
        to_df = []
        for i in range(len(df)):
            line = df.loc[i].to_dict()
            if find_str in line[field_name]:
                to_df.append(line)
        to_df = pd.DataFrame(to_df)
        to_df.to_excel(to_fn)

    def func6(find_str, csv_fn, to_fn):
        with open(csv_fn, "r", encoding="utf-8") as f:
            fw = open(to_fn, "w", encoding="utf-8")
            fw.write(f.readline())
            for line in f:
                if find_str in line:
                    fw.write(line)
            fw.close()

    # func2(r"F:\Week\20231210\Data\tgrs_sar.txt", r"F:\Week\20231210\Data\tgrs_sar_all.csv")
    # func3()
    # func2(r"F:\Week\20231210\Data\IS\is_2.txt", r"F:\Week\20231210\Data\is_sciencedirect_mdpi.csv")
    # fun4()
    # func5(excel_fn=r"F:\ProjectSet\Shadow\QingDao\Dissertation\文献整理\2\题目中的不透水面的文章.xlsx",
    #       field_name="TI", find_str="SAR",
    #       sheet_name="Sheet6",
    #       to_fn=r"F:\ProjectSet\Shadow\QingDao\Dissertation\文献整理\2\题目中的不透水面的文章_SAR.xlsx")
    func6("SAR", r"F:\ProjectSet\Shadow\QingDao\Dissertation\文献整理\2\is_lines.csv",
          r"F:\ProjectSet\Shadow\QingDao\Dissertation\文献整理\2\is_lines_SAR.csv")


def method_name41():
    # 样本抽出来，整理样本的时候用的
    spl_fn = r"F:\ProjectSet\Shadow\ChengDu\Samples\2\ChengDuSamples2.xlsx"
    cname = "WAT_SH"
    n = 10
    tag = "Random3000"
    df = pd.read_excel(spl_fn, sheet_name="Summary")
    df = df[df["CNAME"] == cname]
    # df = df[df["TAG"] == tag]
    print(len(df))
    df_select = df.sample(n=n, random_state=2222)
    print(df.sample(n=n, random_state=2222))
    df_select.to_csv(r"F:\ProjectSet\Shadow\ChengDu\Samples\2\ChengDuSamples2_tmp.csv")


def method_name40():
    gr1 = GDALRaster(r"F:\ProjectSet\Shadow\Release\CD_SH.vrt")
    gr2 = GDALRaster(r"F:\ProjectSet\Shadow\ChengDu\Image\2\CD_SH2.vrt")
    d = np.array(gr1.geo_transform) - np.array(gr2.geo_transform)
    df = pd.read_csv(r"F:\ProjectSet\Shadow\ChengDu\Samples\1\sh_cd_spl_qjy4_2.csv")
    for i in range(len(df)):
        df["X"][i] += d[0]
        df["Y"][i] += d[3]
    df.to_csv(r"F:\ProjectSet\Shadow\ChengDu\Samples\1\sh_cd_spl_qjy4_3.csv")


def method_name39():
    # 每个波段都2%的线性拉伸
    raster_fn = r"F:\ProjectSet\Shadow\Release\QD_SH.vrt"
    ratio = 0.015
    gr = GDALRaster(raster_fn)
    for name in gr:
        d = gr.readGDALBand(name)
        h, bin_edges = np.histogram(d, bins=612)
        h = h / np.size(d)
        h_sum = 0
        h_min, h_max = np.min(d), np.max(d)
        for i in range(len(h)):
            h_sum += h[i]
            if h_sum >= ratio:
                h_min = bin_edges[i]
                break
        h_sum = 0
        for i in range(len(h)):
            h_sum += h[i]
            if h_sum >= 1 - ratio:
                h_max = bin_edges[i]
                break
        print("sct.featureScaleMinMax(\"{0}\"".format(name), h_min, h_max, sep=",", end=")\n")


def method_name38():
    gr = GDALRaster(r"F:\ProjectSet\Shadow\BeiJing\Image\6\bj_im6_pca1")
    d = gr.readGDALBand(1)
    print(d.shape)
    d = np.clip(d, a_min=-2245.46997, a_max=2289.09883)
    gr.save(d, r"F:\ProjectSet\Shadow\BeiJing\Image\6\bj_im6_pca")
    # gr = GDALRaster(r"F:\ProjectSet\Shadow\QingDao\Image\Temp\tmp16.dat")
    # d = gr.readAsArray()
    # print(d[0, :6, :6])


def method_name37():
    # 看一下SAR的纹理特征是什么样的
    gr = GDALRaster(r"F:\ProjectSet\Shadow\QingDao\Image\GLCM\QD2_s1glcm.tif")
    d = gr.readAsArray()
    for i in range(d.shape[0]):
        d_min = np.min(d[i])
        d_max = np.max(d[i])
        d_mean = np.mean(d[i])
        d_std = np.std(d[i])
        print("{0:20}".format(gr.names[i]), end=" ")
        print("{0:20.6f}".format(d_min), end=" ")
        print("{0:20.6f}".format(d_max), end=" ")
        print("{0:20.6f}".format(d_mean), end=" ")
        print("{0:20.6f}".format(d_std), end=" ")
        print()


def method_name36():
    # 样本空间均匀
    d = SRTESRIShapeFileRead(r"F:\ProjectSet\Shadow\BeiJing\Samples\8\bj_spl_wat2.shp")
    coors = d.getCoorList()
    coors = sampleSpaceUniform(coors, x_len=400, y_len=400, is_trans_jiaodu=True)
    coors = np.array(coors)
    savecsv(r"F:\ProjectSet\Shadow\BeiJing\Samples\8\bj_spl_wat3.csv",
            {"X": coors[:, 0].tolist(), "Y": coors[:, 1].tolist()})
    print(len(coors))


def method_name35():
    # 影像全都读进去然后采样
    # csv_fn = r"F:\ProjectSet\Shadow\BeiJing\Samples\8\bj_spl_wat.csv"
    # raster_fn = r"F:\ProjectSet\Shadow\BeiJing\Image\4\BJ_SH4.vrt"
    # to_csv_fn = r"F:\ProjectSet\Shadow\BeiJing\Samples\8\bj_spl_wat2.csv"
    csv_fn = r"F:\ProjectSet\Shadow\BeiJing\Samples\8\bj_spl8_veg1.csv"
    raster_fn = r"F:\ProjectSet\Shadow\BeiJing\Image\4\BJ_SH4.vrt"
    to_csv_fn = r"F:\ProjectSet\Shadow\BeiJing\Samples\8\bj_spl8_veg2.csv"

    gr = GDALRaster(raster_fn)
    d = gr.readAsArray()
    coors = readcsv(csv_fn)
    jdt = Jdt(len(coors["X"]))
    jdt.start()
    if "" in coors:
        coors.pop("")
    with open(to_csv_fn, "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f)
        line = [k for k in coors]
        for name in gr.names:
            line.append(name)
        cw.writerow(line)
        for i in range(len(coors["X"])):
            x, y = float(coors["X"][i]), float(coors["Y"][i])
            r, c = gr.coorGeo2Raster(x, y, is_int=True)
            d0 = d[:, r, c]
            line = [coors[k][i] for k in coors]
            for j in range(d.shape[0]):
                line.append(d0[j])
            cw.writerow(line)
            jdt.add()
        jdt.end()


def method_name33():
    d = readGEORaster(r"F:\ProjectSet\Shadow\BeiJing\Image\Temp\tmp36")
    print(d.shape)
    as_lamd1, as_lamd2 = calEIG(d[0], d[1], d[2], d[3])
    de_lamd1, de_lamd2 = calEIG(d[4], d[5], d[6], d[7])
    lamd = _10log10(as_lamd1)
    saveGEORaster(lamd, r"F:\ProjectSet\Shadow\BeiJing\Image\Temp\tmp40")


def method_name32():
    # 将样本放到图像的中间
    qjy_fn = r"F:\ProjectSet\Shadow\ChengDu\Samples\1\sh_cd_spl_qjy.txt"
    raster_fn = r"F:\ProjectSet\Shadow\ChengDu\Image\cd_QJY.dat"
    gr = GDALRaster(raster_fn)
    lines = []
    is_lines = False
    n_columns = 0
    with open(qjy_fn, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not is_lines:
                if line == "# Please do not change":
                    is_lines = True
            else:
                if line == "":
                    continue
                lines.append(line.split(","))
                if len(lines[-1]) > n_columns:
                    n_columns = len(lines[-1])
    with open(qjy_fn + ".csv", "w", encoding="utf-8", newline="") as f:
        cw = csv.writer(f)
        for line in lines:
            x = float(line[3])
            y = float(line[4])
            row, column = gr.coorGeo2Raster(x, y, is_int=True)
            x, y = gr.coorRaster2Geo(row + 0.5, column + 0.5)
            line[3] = str(x)
            line[4] = str(y)
            cw.writerow(line)


def method_name31():
    rtvrts = RasterToVRTS(r"F:\ProjectSet\Shadow\ChengDu\Image\cd_no_shadow_1.tif")
    rtvrts.save()


def method_name30():
    # 样本空间均匀
    d = SRTESRIShapeFileRead(r"F:\ProjectSet\Shadow\ChengDu\Samples\1\sh_cd_spl1.shp")
    coors = d.getCoorList()
    coors = sampleSpaceUniform(coors, x_len=400, y_len=400, is_trans_jiaodu=True)
    coors = np.array(coors)
    savecsv(r"F:\ProjectSet\Shadow\ChengDu\Samples\1\sh_cd_spl2.csv",
            {"X": coors[:, 0].tolist(), "Y": coors[:, 1].tolist()})
    print(len(coors))


def method_name29():
    # PCA
    gr = GDALRasterFeatures(r"F:\ProjectSet\Shadow\QingDao\Image\Image1\shadow_qd_im1.dat")
    d = gr.getFeaturesData(["Blue", "Green", "Red", "NIR"])
    d_shape = d.shape
    d = d.reshape((4, -1))
    eig, vec, d1 = calPCA(d, num_components=2)
    print(eig)
    print('\n'.join('{}'.format(' '.join("{:.6f}".format(n) for n in row)) for row in vec))
    # d = d.reshape(d_shape)
    # gr.save(d, r"F:\ProjectSet\Shadow\QingDao\Image\Image1\shadow_qd_im1_pca.dat")
    # calPCA()


def method_name28():
    # samplingToCSV(
    #     csv_fn=r"F:\ProjectSet\Shadow\BeiJing\Samples\7\sh_bj_spl7_1_rand.csv",
    #     gr=GDALRaster(r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_envi.dat"),
    #     to_csv_fn=r"F:\ProjectSet\Shadow\BeiJing\Samples\7\sh_bj_spl7_1_rand_spl2.csv"
    # )
    gr = GDALRaster(r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3.vrt")
    for name in gr.names:
        print(name, end="\t")
    print()


def method_name27():
    # 分割训练样本和测试样本
    # 类别映射
    #  两种分割方法：随机抽样和分层抽样
    #    随机抽样：随机抽样的类别映射和
    # "F:\ProjectSet\Shadow\BeiJing\Samples\7\sh_bj_spl7_1.csv"
    def splitRandomSampleTrainTest(df: pd.DataFrame, save_fn, c_name="CNAME", tag_name="TAG", cate_dict=None,
                                   categorys=None, n=10, test_name="TEST",
                                   select_tags=None):
        if select_tags is None:
            select_tags = []
        select_tags_list = [False for i in range(len(df))]
        for i in range(len(df)):
            tag = df[tag_name][i]
            if tag in select_tags:
                select_tags_list[i] = True
        if categorys is None:
            categorys = []
        if cate_dict is None:
            cate_dict = {}
        c_names = ["NOT_KNOW" for i in range(len(df))]
        for i in range(len(df)):
            if select_tags_list[i]:
                cate = df[c_name][i]
                for k, d in cate_dict.items():
                    if cate in d:
                        c_names[i] = k
                        break
        c_list = []
        for i in range(len(c_names)):
            if c_names[i] in categorys:
                c_list.append(i)
        random.shuffle(c_list)
        df2 = []
        c_list = c_list[:n]
        for i in range(len(df)):
            d0 = df.loc[i].to_dict()
            if i in c_list:
                d0[test_name] = 0
            else:
                d0[test_name] = 1
            df2.append(d0)
        df2 = pd.DataFrame(df2)
        df2.to_csv(save_fn, index=False)
        print(df2)

    splitRandomSampleTrainTest(
        pd.read_csv(r"F:\ProjectSet\Shadow\BeiJing\Samples\7\sh_bj_spl7_1.csv"),
        r"F:\ProjectSet\Shadow\BeiJing\Samples\7\sh_bj_spl7_1_rand.csv",
        c_name="CATEGORY_NAME",
        cate_dict={
            "IS": ["IS", "IS_SH"],
            "VEG": ["VEG", "VEG_SH"],
            "SOIL": ["SOIL", "SOIL_SH"],
            "WAT": ["WAT", "WAT_SH"],
        },
        categorys=["IS", "VEG", "SOIL", "WAT"],
        n=600,
        select_tags=["Select600ToTest", "Select1200ToTest"]
    )

    def splitStratifiedSampleTrainTest(
            df: pd.DataFrame, save_fn, c_name="CNAME", cate_dict=None,
            categorys=None,
            test_name="TEST", tag_name="TAG", select_tags=None):
        if categorys is None:
            categorys = {}
        if select_tags is None:
            select_tags = []
        select_tags_list = [False for i in range(len(df))]
        c_names = ["NOT_KNOW" for i in range(len(df))]
        for i in range(len(df)):
            if select_tags_list[i]:
                cate = df[c_name][i]
                for k, d in cate_dict.items():
                    if cate in d:
                        c_names[i] = k
                        break


def method_name26():
    # 绘制的长度和方向的图
    geojson_fn = r"F:\ProjectSet\Shadow\BeiJing\Temp\1\bj_sh_imcoll_20-23.geojson"
    sddl = ShadowDrawDirectLength(geojson_fn)
    # sddl.plotS("2020-01-01", "2021-01-01")
    # sddl.plotAlpha("2020-01-01", "2021-01-01")
    sddl.plotXY("2020-01-01", "2020-03-01")
    sddl.plotXY("2020-03-01", "2020-06-01")
    sddl.plotXY("2020-06-01", "2020-09-01")
    sddl.plotXY("2020-09-01", "2021-01-01")
    plt.legend()
    plt.show()


def method_name25():
    # 采样到CSV文件
    samplingToCSV(
        csv_fn=r"F:\ProjectSet\Shadow\BeiJing\Samples\2\sh_bj_spl_summary2_1200_qjy2.csv",
        gr=GDALRaster(r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3.vrt"),
        to_csv_fn=r"F:\ProjectSet\Shadow\BeiJing\Samples\2\sh_bj_spl_summary2_1200_qjy2_spl.csv"
    )


def method_name24():
    # 看一看北京研究区的所有vrt中的影像是不是大小一样
    # filelist = [r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_AS_Beta.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_AS_C11.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_AS_C12_imag.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_AS_C12_real.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_AS_C22.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_AS_Epsilon.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_AS_Lambda1.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_AS_Lambda2.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_AS_m.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_AS_Mu.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_AS_RVI.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_AS_SPAN.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_AS_VH.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_AS_VV.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_Blue.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_DE_Beta.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_DE_C11.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_DE_C12_imag.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_DE_C12_real.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_DE_C22.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_DE_Epsilon.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_DE_Lambda1.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_DE_Lambda2.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_DE_m.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_DE_Mu.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_DE_RVI.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_DE_SPAN.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_DE_VH.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_DE_VV.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_Green.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_NDVI.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_NDWI.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_NIR.dat"
    #     , r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_Red.dat"]
    #
    # for f in filelist:
    #     gr = GDALRaster(f)
    #     print(gr.n_rows, gr.n_columns, gr.n_channels, f)
    #
    gr = GDALRaster(r"F:\ProjectSet\Shadow\BeiJing\Image\Temp\t3")
    d = gr.readAsArray()
    gr2 = GDALRaster(r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_Red.dat")
    gr2.save(d, r"F:\ProjectSet\Shadow\BeiJing\Image\Temp\t4")


def method_name23():
    # 在一个样本集中随机抽取样本
    csv_fn = r"F:\ProjectSet\Shadow\BeiJing\Samples\Summary\sh_bj_spl_summary2.csv"
    to_csv_fn = r"F:\ProjectSet\Shadow\BeiJing\Samples\Summary\sh_bj_spl_summary2_1200.csv"
    select_name = "Select1200ToTest"
    select_number = 1200
    category_column_name = None
    is_select_column_name = "__IS_SELECT__"
    select_category = ""

    d = readcsv(csv_fn)
    ks = list(d.keys())
    n_rows = len(d[ks[0]])
    print("n_rows", n_rows)
    if select_number > n_rows:
        select_number = n_rows
    if is_select_column_name not in d:
        d[is_select_column_name] = ["" for _ in range(n_rows)]
    print("select name", select_name)
    for i in range(len(d[is_select_column_name])):
        if d[is_select_column_name][i] == select_name:
            d[is_select_column_name][i] = ""
    if category_column_name in d:
        categorys = d[category_column_name]
    else:
        categorys = ["" for _ in range(n_rows)]
    with open(to_csv_fn, "w", encoding="utf-8", newline="") as fw:
        n_find = 0
        cw = csv.writer(fw)
        cw.writerow(list(d.keys()))
        jdt = Jdt(select_number, desc="Sample select random")
        jdt.start()
        for i in range(n_rows * 2):
            row_select = random.randint(0, n_rows - 1)
            if (d[is_select_column_name][row_select] == "") and (categorys[row_select] == select_category):
                d[is_select_column_name][row_select] = select_name
                cw.writerow([d[k][row_select] for k in d])
                n_find += 1
                jdt.add()
                if select_number == n_find:
                    break
    savecsv(csv_fn, d)


def method_name22():
    # 去掉北京研究区中间的那条缝
    bj_im_dfn = DirFileName(r"F:\ProjectSet\Shadow\BeiJing\Image\1")
    bj_to_im_dfn = DirFileName(r"F:\ProjectSet\Shadow\BeiJing\Image\3")
    bj_to_im_dfn.mkdir()
    c01, c02 = 3, 3
    line = [[116.345321647646, 39.7745751534696], [116.29537259765, 39.9999611850009]]  # AS
    # line = [[116.46133775132, 39.7746032694107], [116.514861851309, 40.0000476955764]]  # DE
    filelist = [
        # r"BJ_SH1_AS_Beta.dat",
        # r"BJ_SH1_AS_C11.dat",
        # r"BJ_SH1_AS_C12_imag.dat",
        # r"BJ_SH1_AS_C12_real.dat",
        # r"BJ_SH1_AS_C22.dat",
        # r"BJ_SH1_AS_Epsilon.dat",
        # r"BJ_SH1_AS_Lambda1.dat",
        # r"BJ_SH1_AS_Lambda2.dat",
        # r"BJ_SH1_AS_m.dat",
        # r"BJ_SH1_AS_Mu.dat",
        # r"BJ_SH1_AS_RVI.dat",
        # r"BJ_SH1_AS_SPAN.dat",
        r"BJ_SH1_AS_VH.dat",
        r"BJ_SH1_AS_VV.dat",
        r"BJ_SH1_Blue.dat",
        # r"BJ_SH1_DE_Beta.dat",
        # r"BJ_SH1_DE_C11.dat",
        # r"BJ_SH1_DE_C12_imag.dat",
        # r"BJ_SH1_DE_C12_real.dat",
        # r"BJ_SH1_DE_C22.dat",
        # r"BJ_SH1_DE_Epsilon.dat",
        # r"BJ_SH1_DE_Lambda1.dat",
        # r"BJ_SH1_DE_Lambda2.dat",
        # r"BJ_SH1_DE_m.dat",
        # r"BJ_SH1_DE_Mu.dat",
        # r"BJ_SH1_DE_RVI.dat",
        # r"BJ_SH1_DE_SPAN.dat",
        r"BJ_SH1_DE_VH.dat",
        r"BJ_SH1_DE_VV.dat",
        r"BJ_SH1_Green.dat",
        r"BJ_SH1_NDVI.dat",
        r"BJ_SH1_NDWI.dat",
        r"BJ_SH1_NIR.dat",
        r"BJ_SH1_Red.dat"
    ]
    # for filename in filelist:
    #     print(bj_im_dfn.fn(filename))
    #     gr = GDALRaster(bj_im_dfn.fn(filename))
    #     d = gr.readAsArray()
    #     print(gr.d.shape)
    #
    #     def two_point_line(r1, c1, r2, c2, r):
    #         return (r - r1) / (r2 - r1) * (c2 - c1) + c1
    #
    #     for i in range(len(line) - 1):
    #         x1, y1, x2, y2 = line[i][0], line[i][1], line[i + 1][0], line[i + 1][1]
    #         r1, c1 = gr.coorGeo2Raster(x1, y1, is_int=True)
    #         r2, c2 = gr.coorGeo2Raster(x2, y2, is_int=True)
    #         print(r1, c1, r2, c2)
    #         for r in range(r2, r1):
    #             c = int(two_point_line(r1, c1, r2, c2, r))
    #             d1 = d[r, c - c01: c + c02]
    #             k = np.argmin(d1)
    #             # d1[k] = (np.sum(d1) - d1[k]) / (d1.shape[0] - 1)
    #             k = np.argmin(d1)
    #             d1[k] = np.median(d1)
    #             k = np.argmin(d1)
    #             d1[k] = np.median(d1)
    #             d[r, c - c01: c + c02] = d1
    #     gr.save(d, bj_to_im_dfn.fn(filename))
    for filename in filelist:
        print(bj_im_dfn.fn(filename))
        gr = GDALRaster(bj_im_dfn.fn(filename))
        d = gr.readAsArray()
        gr.save(d, bj_to_im_dfn.fn(filename))


def method_name21():
    # 使用新的测试样本测试分类影像的精度
    excel_fn = r"F:\ProjectSet\Shadow\QingDao\Sample\Test\3\青岛测试样本3.xlsx"  # sheet_name="选出600个样本作为总体精度"
    excel_fn = r"F:\Week\20230917\Data\看了青岛的样本.xlsx"  # sheet_name="Sheet1" NOSH
    raster_fn = r"F:\ProjectSet\Shadow\QingDao\Mods\20230707H200910\SPL_NOSH-RF-TAG-OPTICS-AS_C2-AS_LAMD-DE_C2_imdc.dat"
    grca = GDALRasterClassificationAccuracy()
    grca.addCategoryCode(IS_SH=1, VEG_SH=2, SOIL_SH=3, WAT_SH=4)
    grca.addSampleExcel(excel_fn, c_column_name="CNAME", sheet_name="NOSH")
    # grca.fit(raster_fn)
    grca.openSaveCSVFileName(r"F:\Week\20230917\Data\test1.csv")
    grca.openSaveCMFileName(r"F:\Week\20230917\Data\test1_cm.txt")
    grca.fitModelDirectory(os.path.dirname(raster_fn))
    grca.closeSaveCSVFileName()
    grca.closeSaveCMFileName()


def method_name20():
    # 结果中包含最大数量类别以及分出是不是最大数量
    csv_fn = r"F:\ProjectSet\Shadow\QingDao\Sample\Test\3\青岛测试样本3.xlsx"
    df = pd.read_excel(csv_fn, sheet_name="Sheet2")
    ks = ['IS', 'VEG', 'SOIL', 'WAT']
    is_max = []
    cates = []
    print(df.keys())
    for i in range(len(df)):
        line = df.loc[i][ks]
        cates.append(line.idxmax())
        is_max.append(int(np.max(line.values) == np.sum(line.values)))
    df["N_IS_CATE"] = cates
    df["N_CATE"] = is_max
    print(df)
    df.to_csv(r"F:\ProjectSet\Shadow\QingDao\Sample\Test\3\qd_sh_testspl3_2_600_2.csv")


def method_name19():
    # 取出在GEE下载的Sentinel-1ImageCollection的某个属性，保存为CSV
    geeCSVSelectPropertys(r"F:\ProjectSet\Shadow\QingDao\temp\s1_ch_prop2.csv",
                          r"F:\ProjectSet\Shadow\QingDao\temp\s1_ch_prop2_1.csv",
                          ["system:index", "orbitProperties_pass", ".geo"])


def plotAZ(csv_fn, label_name, start_time=None, end_time=None):
    df_im_coll = GEEImageProperty(csv_fn) \
        .extractTime() \
        .filterDate(start_time, end_time) \
        .orderbyTime() \
        .intervalSampling(select_step=2)

    df_im_coll.plotZenith(label_name + " Zenith")
    # df_im_coll.plotAzimuth(label_name + " Azimuth")


def plotShadowMax(csv_fn, label_name):
    df_im_coll = GEEImageProperty(csv_fn) \
        .extractTime() \
        .filterDate("2021-01-01", "2022-01-01") \
        .orderbyTime() \
        .intervalSampling(select_step=2)

    fig = plt.figure()
    ax = fig.add_subplot(axes_class=AxesZero)
    for direction in ["xzero", "yzero"]:
        # adds arrows at the ends of each axis
        ax.axis[direction].set_axisline_style("->")
        # adds X and Y-axis from the origin
        ax.axis[direction].set_visible(True)
    for direction in ["left", "right", "bottom", "top"]:
        # hides borders
        ax.axis[direction].set_visible(False)
    ax = plt.gca()
    ax.spines['right'].set_color('none')  # 设置上边和右边无边框
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')  # 设置x坐标刻度数字或名称的位置
    ax.spines['bottom'].set_position(('data', 0))  # 设置边框位置
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))

    df_im_coll.plotShadow(time_split_list=[
        np.datetime64("2021-01-01"), np.datetime64("2021-04-01"),
        np.datetime64("2021-07-01"), np.datetime64("2021-10-01"),
        np.datetime64("2022-01-01")]
        , legend_label=label_name
    )

    # plt.title("Changes in the farthest point of shadow length in four Chinese cities within a year")
    plt.title(label_name)
    plt.legend()
    plt.xlim(-8, 1)
    plt.ylim(-2, 30)
    plt.show()
    # df_im_coll.plt_show(label_name)


def method_name18():
    # 青岛在海水中分层随机抽样
    d = gdalStratifiedRandomSampling(
        r"F:\ProjectSet\Shadow\QingDao\Image\SEA.dat",
        numbers=[0], n_max=200000)
    np.savetxt(r"F:\ProjectSet\Shadow\QingDao\Sample\Test\3\qd_sh_testspl3_2.csv",
               np.array(d), delimiter=",", fmt="%.9f")
    print(np.array(d))


def method_name17():
    # 使用四联通区域，提取非海水的区域
    gr = GDALRaster(imdcDirname.fn("imdc2.dat"))
    d = gr.readAsArray()
    d = d[4, :]
    d = d == 256
    d1 = np.zeros_like(d)
    xys = [(120.309068, 36.179054), (120.286693, 36.175938), (120.299935, 36.143836),
           (120.2959253, 36.1688113), (120.297447, 36.170584), (120.300380, 36.171189),
           (120.298725, 36.171828), (120.3025010, 36.1698416)]
    for x, y in xys:
        x, y = gr.coorGeo2Raster(x, y)
        print(int(x), int(y))
        d2 = seedFill4(d, int(x), int(y))
        d1 = d1 + d2
    d1 = d1 != 0
    gr.save(d1.astype("int8"), r"F:\Week\20230903\Temp\test3.dat", dtype=gdal.GDT_Byte)


def method_name16():
    # 四联通区域获取
    gr = GDALRaster(imdcDirname.fn("SPL_NOSH-RF-TAG-OPTICS_imdc.dat"))
    d = gr.readAsArray()
    print(d.shape)
    d = (d == 4) * 1
    plt.imshow(d)
    plt.colorbar()
    plt.show()
    x, y = 372, 372
    d1 = seedFill4(d, x, y)
    plt.imshow(d1)
    plt.colorbar()
    plt.show()
    gr.save(d1.astype("int8"), r"F:\Week\20230903\Temp\test2.dat", dtype=gdal.GDT_Byte)


def seedFill4(d, x, y):
    # 四联通区域获取
    d1 = np.zeros_like(d)
    xys = [[x, y]]  # 中心点
    while xys:
        xy = xys.pop()
        if 0 <= xy[0] < d.shape[0] and 0 <= xy[1] < d.shape[1]:
            if d[xy[0], xy[1]] == 1 and d1[xy[0], xy[1]] != 1:
                d1[xy[0], xy[1]] = 1
                xys.append([xy[0] + 1, xy[1]])
                xys.append([xy[0] - 1, xy[1]])
                xys.append([xy[0], xy[1] + 1])
                xys.append([xy[0], xy[1] - 1])
    return d1


def method_name15():
    # 测试一下精度评定，使用分类精度最高的影像作为原始影像，分类最低的为分类影像
    # 看一下分层随机抽样和随机撒点的不同
    # 去除海水区域在看看混淆矩阵
    # "F:\ProjectSet\Shadow\QingDao\Image\SEA.dat"

    gr1 = GDALRaster(imdcDirname.fn("SPL_NOSH-RF-TAG-OPTICS_imdc.dat"))
    d1 = gr1.readAsArray().ravel()
    gr2 = GDALRaster(imdcDirname.fn("SPL_SH-RF-TAG-OPTICS-AS_C2-DE_SIGMA_imdc.dat"))
    d2 = gr2.readAsArray().ravel()

    # 去除海水
    gr_sea = GDALRaster(r"F:\ProjectSet\Shadow\QingDao\Image\SEA.dat")
    d3 = gr_sea.readAsArray().ravel()
    d1 = d1[d3 == 0]
    d2 = d2[d3 == 0]

    class_names = ["IS", "VEG", "SOIL", "WAT"]
    cm = ConfusionMatrix(4, class_names=class_names)
    cm.addData(d2, d1)

    oa0 = cm.OA()
    ua0 = cm.UA()
    pa0 = cm.PA()
    print("True")
    print(cm.fmtCM())
    for i in range(4):
        print("{0:6}: {1:6.2f} {2:6.2f}".format(
            class_names[i],
            (d1[d1 == i + 1].shape[0] / d1.shape[0]) * 100,
            (d2[d2 == i + 1].shape[0] / d2.shape[0]) * 100))

    n0 = 10000
    ns = [i for i in range(n0, 21000, n0)]
    d = [[], [], []]
    for n in ns:
        cm.clear()
        # 随机撒点
        select = np.random.randint(0, d2.shape[0], size=n)
        # 分层随机抽样
        # dtmp = d1
        # select0 = np.arange(dtmp.shape[0])
        # select = []
        # for i in range(1, 5):
        #     select1 = select0[dtmp == i]
        #     select2 = np.random.randint(0, select1.shape[0], size=n)
        #     select += select1[select2].tolist()
        # select = np.array(select)
        cm.addData(d2[select], d1[select])
        oa1 = cm.OA()
        ua1 = cm.UA()
        pa1 = cm.PA()
        d[0].append(oa1 - oa0)
        d[1].append(ua1 - ua0)
        d[2].append(pa1 - pa0)
        print("random", n)
        print("OA:", oa1 - oa0)
        print("UA:", ua1 - ua0)
        print("PA:", pa1 - pa0)
        print(cm.fmtCM())
    plt.plot(ns, d[0])
    plt.title("OA")
    plt.show()
    plt.plot(ns, d[1], label=class_names)
    plt.legend()
    plt.title("UA")
    plt.show()
    plt.plot(ns, d[2], label=class_names)
    plt.legend()
    plt.title("PA")
    plt.show()


def categoryCal(x):
    x = float(x) / 10
    return int(x)


def method_name12():
    grf = GDALRasterFeatures(r"G:\ImageData\QingDao\20211023\qd20211023\Temp\qd_im.dat")
    as_channels = ["AS_C11", "AS_C12_real", "AS_C12_imag", "AS_C22"]
    de_channels = ["DE_C11", "DE_C12_real", "DE_C12_imag", "DE_C22"]
    method_name10(as_channels, grf, "AS_ROU_real", "AS_ROU_imag")
    method_name10(de_channels, grf, "DE_ROU_real", "DE_ROU_imag")


def rouConv(x):
    x1_1, x1_2 = np.sum(x[1]), np.sum(x[2])
    x2 = np.sqrt(np.sum(x[0]) * np.sum(x[3]))
    return x1_1 / x2, x1_2 / x2


def method_name11():
    grf = GDALRasterFeatures(r"F:\ProjectSet\Shadow\QingDao\Image\stack2.vrt")
    grf.addFFEDivide("AS_VV_DIVIDE_VH", "AS_VV", "AS_VH")
    grf.addFFEDivide("DE_VV_DIVIDE_VH", "DE_VV", "DE_VH")
    grf.addFFEDivide("AS_C1_DIVIDE_C2", "AS_C1", "AS_C2")
    grf.addFFEDivide("DE_C1_DIVIDE_C2", "DE_C1", "DE_C2")


def method_name10(de_channels, grf, t1, t2):
    d_c2 = grf[de_channels]
    print(d_c2.shape)
    d_rou = neighborhood(d_c2, rouConv, 5, 5, dim=2)
    grf.save(d=d_rou[0], save_geo_raster_fn=r"F:\ProjectSet\Shadow\QingDao\Image\Temp\temp_{0}".format(t1),
             descriptions=[t1])
    grf.save(d=d_rou[1], save_geo_raster_fn=r"F:\ProjectSet\Shadow\QingDao\Image\Temp\temp_{0}".format(t2),
             descriptions=[t2])
    print(d_rou)


def method_name9():
    csv_fn = r"F:\ProjectSet\Shadow\QingDao\Sample\Test\2\qd_testspl2_3.csv"
    csv_d = readcsv(csv_fn)
    to_dict = {}
    for k in csv_d:
        csv_d[k] = np.array(list(map(float, csv_d[k])))
        csv_d[k] = csv_d[k] / np.sum(csv_d[k])
        d1 = np.fft.fft(csv_d[k])
        to_dict["real" + k] = np.real(d1)
        to_dict["imag" + k] = np.imag(d1)
    savecsv(r"F:\ProjectSet\Shadow\QingDao\Sample\Test\2\qd_testspl2_3_2.csv", to_dict)


def method_name8():
    imdc2_fn = r"F:\ProjectSet\Shadow\QingDao\Mods\20230707H200910\imdc2.dat"
    gr = GDALRaster(imdc2_fn)
    d = gr.readAsArray()
    out_dict = {}
    for i in range(d.shape[0]):
        out_dict[i] = [0 for i in range(300)]
        print("out_dict[i]:")
        for j in range(d.shape[1]):
            for k in range(d.shape[2]):
                out_dict[i][int(d[i, j, k])] += 1
        print(out_dict[i])
    savecsv(r"F:\ProjectSet\Shadow\QingDao\Sample\Test\2\qd_testspl2_3.csv", out_dict)


def method_name6():
    tree = ElementTree.parse(
        r"H:\S1B_IW_SLC__1SDV_20211022T215641_20211022T215712_029254_037DC2_3DFC\runtemp1\Stack\2\stack2.xml")
    root = tree.getroot()
    # for node in list(root):
    #     print(node.text, node.tag, node.get('Description'))
    names = [
        "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
        "AS_VV", "AS_VH", "AS_C11", "AS_C12_imag", "AS_C12_real", "AS_C22", "AS_Lambda1", "AS_Lambda2",
        "DE_VV", "DE_VH", "DE_C11", "DE_C12_imag", "DE_C12_real", "DE_C22", "DE_Lambda1", "DE_Lambda2",
        "angle_AS", "angle_DE"
    ]
    i = 0
    for node in root.findall("VRTRasterBand"):
        element = ElementTree.Element("Description")
        element.text = names[i]
        node.append(element)
        i += 1
    for node in root.findall("VRTRasterBand"):
        print(node.find("Description").text)
    tree.write(r"H:\S1B_IW_SLC__1SDV_20211022T215641_20211022T215712_029254_037DC2_3DFC\runtemp1\Stack\2\stack2_2.xml",
               encoding='utf-8', xml_declaration=True)


def method_name5():
    to_dir = r"H:\S1B_IW_SLC__1SDV_20211022T215641_20211022T215712_029254_037DC2_3DFC\runtemp1\Stack\2"
    file_list_fn = os.path.join(to_dir, "filelist.txt")
    names = [
        "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
        "AS_VV", "AS_VH", "AS_C11", "AS_C12_imag", "AS_C12_real", "AS_C22", "AS_Lambda1", "AS_Lambda2",
        "DE_VV", "DE_VH", "DE_C11", "DE_C12_imag", "DE_C12_real", "DE_C22", "DE_Lambda1", "DE_Lambda2",
        "angle_AS", "angle_DE"
    ]
    print("gdalbuildvrt -separate -input_file_list filelist.txt stack2.xml")
    f = open(file_list_fn, "w", encoding="utf-8")
    for i, name in enumerate(names):
        print(name + ".dat", file=f)
        print(i + 1, name)
    f.close()


def method_name3():
    to_dir = r"H:\S1B_IW_SLC__1SDV_20211022T215641_20211022T215712_029254_037DC2_3DFC\runtemp1\Stack\2"
    names = [
        "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
        "AS_VV", "AS_VH", "AS_C11", "AS_C12_imag", "AS_C12_real", "AS_C22",
        "AS_Lambda1", "AS_Lambda2",
        "DE_VV", "DE_VH", "DE_C11", "DE_C12_imag", "DE_C12_real", "DE_C22",
        "DE_Lambda1", "DE_Lambda2",
        "angle_AS", "angle_DE"
    ]
    to_fns = [
        "",
        r"H:\S1B_IW_SLC__1SDV_20211022T215641_20211022T215712_029254_037DC2_3DFC\runtemp1\Stack\2\AS_Lambda1.dat",
        r"H:\S1B_IW_SLC__1SDV_20211022T215641_20211022T215712_029254_037DC2_3DFC\runtemp1\Stack\2\AS_Lambda2.dat",
        r"H:\S1B_IW_SLC__1SDV_20211022T215641_20211022T215712_029254_037DC2_3DFC\runtemp1\Stack\2\DE_Lambda1.dat",
        r"H:\S1B_IW_SLC__1SDV_20211022T215641_20211022T215712_029254_037DC2_3DFC\runtemp1\Stack\2\DE_Lambda2.dat",
    ]
    gr_coll = GDALRasterCollection(names, to_dir, ".dat")
    for i, name in enumerate(["AS_C11", "AS_C12_imag", "AS_C12_real", "AS_C22"]):
        gr_coll[name].readAsArray()
        print(name, gr_coll[name].d.shape)
    as_lamd1, as_lamd2 = calEIG(
        gr_coll["AS_C11"].d,
        gr_coll["AS_C22"].d,
        gr_coll["AS_C12_real"].d,
        gr_coll["AS_C12_imag"].d,
    )
    gr_coll.save(as_lamd1, to_fns[1])
    gr_coll.save(as_lamd2, to_fns[2])
    for i, name in enumerate(["DE_C11", "DE_C12_imag", "DE_C12_real", "DE_C22"]):
        gr_coll[name].readAsArray()
        print(name, gr_coll[name].d.shape)
    de_lamd1, de_lamd2 = calEIG(
        gr_coll["DE_C11"].readAsArray(),
        gr_coll["DE_C22"].readAsArray(),
        gr_coll["DE_C12_real"].readAsArray(),
        gr_coll["DE_C12_imag"].readAsArray(),
    )
    gr_coll.save(de_lamd1, to_fns[3])
    gr_coll.save(de_lamd2, to_fns[4])


def calArrayEIG(x1arr, x2arr, y1arr, y2arr):
    lamd1 = np.zeros(x1arr.shape)
    lamd2 = np.zeros(x1arr.shape)
    for i in range(x1arr.shape[0]):
        for j in range(x1arr.shape[1]):
            d1, d2 = calEIG(x1arr[i, j], x2arr[i, j], y1arr[i, j], y2arr[i, j])
            lamd1[i, j] = d1
            lamd2[i, j] = d2
    return lamd1, lamd2


def method_name2():
    raster_fn = r"H:\S1B_IW_SLC__1SDV_20211022T215641_20211022T215712_029254_037DC2_3DFC\runtemp1\Stack\1\stack2.dat"
    names = ["Blue", "Green", "Red", "NIR", "NDVI", "NDWI", "AS_VV", "AS_VH", "AS_C11", "AS_C12_imag", "AS_C12_real",
             "AS_C22", "DE_VV", "DE_VH", "DE_C11", "DE_C12_imag", "DE_C12_real", "DE_C22", "angle_AS", "angle_DE"]
    to_dir = r"H:\S1B_IW_SLC__1SDV_20211022T215641_20211022T215712_029254_037DC2_3DFC\runtemp1\Stack\2"
    splitImage1(names, raster_fn, to_dir)


def splitImage1(names, raster_fn, to_dir):
    gr = GDALRaster(raster_fn)
    d = gr.readAsArray()
    for i, name in enumerate(names):
        to_fn = os.path.join(to_dir, name + ".dat")
        print(to_fn)
        gr.save(d[i], to_fn)


def method_name():
    gr = GDALRaster(r"H:\S1B_IW_SLC__1SDV_20211022T215641_20211022T215712_029254_037DC2_3DFC"
                    r"\runtemp1\Stack\1\stack1.dat")
    d = gr.readAsArray()
    d = d.astype("float32")
    xigema_channels = [4, 5, 10, 11]
    for i in xigema_channels:
        d0 = np.power(10, d[i] / 10)
        d[i] = d0
    ndvi = (d[3] - d[2]) / (d[3] + d[2])
    ndwi = (d[1] - d[3]) / (d[1] + d[3])
    d1 = np.zeros([gr.n_channels + 2, gr.n_rows, gr.n_columns], dtype="float32")
    print(d.shape, d1.shape)
    d1[:4] = d[:4]
    d1[4] = ndvi
    d1[5] = ndwi
    d1[6:] = d[4:]
    gr.save(d1, r"H:\S1B_IW_SLC__1SDV_20211022T215641_20211022T215712_029254_037DC2_3DFC"
                r"\runtemp1\Stack\1\stack2.dat")


def calEIG(x1, x2, y1, y2):
    """ C11 = x1, C12 = y1 + y2i, C22 = x2"""
    c2 = x1 * x2 - (y1 * y1 + y2 * y2)
    t = (x1 + x2) / 2
    lamd1 = t + np.sqrt(t * t - c2)
    lamd2 = t - np.sqrt(t * t - c2)
    return lamd1, lamd2


def method_name7():
    gr = GDALRaster(r"H:\S1B_IW_SLC__1SDV_20211022T215641_20211022T215712_029254_037DC2_3DFC"
                    r"\runtemp1\Stack\1\stack1.dat")
    d = gr.readAsArray()
    d = d.astype("float32")
    gr.save(d[3], r"F:\ProjectSet\Shadow\QingDao\Image\NIR.dat")


def method_name4():
    fig = plt.figure()  # 定义figure，（1）中的1是什么
    ax_cof = HostAxes(fig, [0.1, 0.1, 0.5, 0.7])  # 用[left, bottom, weight, height]的方式定义axes，0 <= l,b,w,h <= 1
    # parasite addtional axes, share x
    ax_temp = ParasiteAxes(ax_cof, sharex=ax_cof)
    ax_load = ParasiteAxes(ax_cof, sharex=ax_cof)
    ax_cp = ParasiteAxes(ax_cof, sharex=ax_cof)
    ax_wear = ParasiteAxes(ax_cof, sharex=ax_cof)
    # append axes
    ax_cof.parasites.append(ax_temp)
    ax_cof.parasites.append(ax_load)
    ax_cof.parasites.append(ax_cp)
    ax_cof.parasites.append(ax_wear)
    # invisible right axis of ax_cof
    ax_cof.axis['top'].set_visible(False)
    ax_cof.axis['right'].set_visible(False)
    ax_temp.axis['right'].set_visible(True)
    ax_temp.axis['right'].major_ticklabels.set_visible(True)
    ax_temp.axis['right'].label.set_visible(True)
    # set label for axis
    ax_cof.set_ylabel('cof')
    ax_cof.set_xlabel('Distance (m)')
    ax_temp.set_ylabel('Temperature')
    ax_load.set_ylabel('load')
    ax_cp.set_ylabel('CP')
    ax_wear.set_ylabel('Wear')
    load_axisline = ax_load.get_grid_helper().new_fixed_axis
    cp_axisline = ax_cp.get_grid_helper().new_fixed_axis
    wear_axisline = ax_wear.get_grid_helper().new_fixed_axis
    ax_load.axis['right2'] = load_axisline(loc='right', axes=ax_load, offset=(40, 0))
    ax_cp.axis['right3'] = cp_axisline(loc='right', axes=ax_cp, offset=(80, 0))
    ax_wear.axis['right4'] = wear_axisline(loc='right', axes=ax_wear, offset=(120, 0))
    fig.add_axes(ax_cof)
    curve_cof, = ax_cof.plot([0, 1, 2], [0, 1, 2], label="CoF", color='black')
    curve_temp, = ax_temp.plot([0, 1, 2], [0, 3, 2], label="Temp", color='red')
    curve_load, = ax_load.plot([0, 1, 2], [1, 2, 3], label="Load", color='green')
    curve_cp, = ax_cp.plot([0, 1, 2], [0, 40, 25], label="CP", color='pink')
    curve_wear, = ax_wear.plot([0, 1, 2], [25, 18, 9], label="Wear", color='blue')
    ax_temp.set_ylim(0, 4)
    ax_load.set_ylim(0, 4)
    ax_cp.set_ylim(0, 50)
    ax_wear.set_ylim(0, 30)
    ax_cof.legend()
    ax_temp.axis['right'].label.set_color('red')
    ax_load.axis['right2'].label.set_color('green')
    ax_cp.axis['right3'].label.set_color('pink')
    ax_wear.axis['right4'].label.set_color('blue')
    ax_temp.axis['right'].major_ticks.set_color('red')
    ax_load.axis['right2'].major_ticks.set_color('green')
    ax_cp.axis['right3'].major_ticks.set_color('pink')
    ax_wear.axis['right4'].major_ticks.set_color('blue')
    ax_temp.axis['right'].major_ticklabels.set_color('red')
    ax_load.axis['right2'].major_ticklabels.set_color('green')
    ax_cp.axis['right3'].major_ticklabels.set_color('pink')
    ax_wear.axis['right4'].major_ticklabels.set_color('blue')
    ax_temp.axis['right'].line.set_color('red')
    ax_load.axis['right2'].line.set_color('green')
    ax_cp.axis['right3'].line.set_color('pink')
    ax_wear.axis['right4'].line.set_color('blue')
    plt.show()


def method_name14():
    fn0 = r"F:\ProjectSet\Shadow\QingDao\Mods\20230707H200910"
    print(fn0)
    with open(r"F:\ProjectSet\Shadow\QingDao\Mods\Temp\temp1.txt", "w", encoding="utf-8") as fw:
        for fn in os.listdir(fn0):
            if os.path.splitext(fn)[1] == ".dat":
                ff = os.path.join(fn0, fn)
                print(ff, file=fw)


def method_name13():
    csv_fn = r"F:\ProjectSet\Shadow\QingDao\Mods\20230707H200910\train_save_20230707H200910.csv"
    to_csv_fn = r"F:\ProjectSet\Shadow\QingDao\Mods\20230707H200910\train_save_20230707H200910_2.csv"
    to_cm_fn = r"F:\ProjectSet\Shadow\QingDao\Mods\20230707H200910\train_save_20230707H200910_cm_2.txt"
    spl_csv_fn = r"F:\ProjectSet\Shadow\QingDao\Sample\Test\qd_test_t2.csv"
    mod_dir = r"F:\ProjectSet\Shadow\QingDao\Mods\20230707H200910"
    cm_names = ["OATest 2", "KappaTest 2",
                "IS UATest 2", "IS PATest 2", "VEG UATest 2", "VEG PATest 2",
                "SOIL UATest 2", "SOIL PATest 2", "WAT UATest 2", "WAT PATest 2"]
    class_names = ["IS", "VEG", "SOIL", "WAT"]
    cm = ConfusionMatrix(4, class_names=class_names)
    df = readcsv(csv_fn)
    df_spl = readcsv(spl_csv_fn)
    df_spl["X"] = list(map(float, df_spl["X"]))
    df_spl["Y"] = list(map(float, df_spl["Y"]))
    y0 = list(map(categoryCal, df_spl["CATEGORY"]))
    n_rows = len(df["ModelName"])
    for name in cm_names:
        df[name] = [0 for i in range(n_rows)]
    f_cm = open(to_cm_fn, "w", encoding="utf-8")
    for f in os.listdir(mod_dir):
        ff = os.path.join(mod_dir, f)
        if os.path.splitext(f)[1] == ".dat":
            fn = f.split("_")
            row_name = "_".join(fn[:-1])
            print(f, row_name)
            if row_name not in df["ModelName"]:
                print("Warning: not find file name", ff)
                continue
            row = df["ModelName"].index(row_name)
            gr = GDALRaster(ff)
            y1 = gr.sampleCenter(df_spl["X"], df_spl["Y"], is_geo=True)

            cm.clear()
            cm.addData(y0, y1)
            df["KappaTest 2"][row] = cm.getKappa()
            df["OATest 2"][row] = cm.OA()
            for name in class_names:
                df["{0} UATest 2".format(name)][row] = cm.PA(name)
            for name in class_names:
                df["{0} PATest 2".format(name)][row] = cm.PA(name)
            f_cm.write("\n> {0}\n".format(row_name))
            f_cm.write(cm.fmtCM())
    f_cm.close()
    savecsv(to_csv_fn, df)


def method_name1():
    df = pd.read_csv(r"F:\ProjectSet\Shadow\QingDao\Sample\Test\2\qd_testspl2_3_2.csv")
    ks = df.keys()
    for i in range(0, len(df.keys()), 2):
        plt.plot(df[ks[i]], df[ks[i + 1]], label=ks[i])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
