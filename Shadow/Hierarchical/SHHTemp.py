# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHHTemp.py
@Time    : 2024/3/3 20:11
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHHTemp
-----------------------------------------------------------------------------"""
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

import SHHConfig
from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.GDALUtils import GDALRastersSampling
from SRTCodes.NumpyUtils import dataCenter
from SRTCodes.SRTSample import SRTSample
from SRTCodes.Utils import Jdt, DirFileName, changext, SRTDFColumnCal, \
    numberfilename, SRTDataFrame, saveJson
from Shadow.Hierarchical.SHHClasses import SHHT_RandomCoor, SHHT_GDALRandomCoor, SHHModelImdc, SHHGDALSampling
from Shadow.Hierarchical.SHHMLFengCeng import trainMLFC, ext_feat, df_get_data
from Shadow.Hierarchical.ShadowHSample import ShadowHierarchicalSampleCollection
from Shadow.ShadowRaster import ShadowRasterGLCM


def trainMLData(df, nofc_x_keys=None, cate_name=None):
    ext_feat(df, "ndvi", "ndwi", "mndwi")
    if nofc_x_keys is None:
        nofc_x_keys = [
            'B2', 'B3', 'B4', 'B8', 'B11', 'B12',
            'ndvi', 'ndwi', 'mndwi',
            'AS_VV', 'AS_VH', 'DE_VV', 'DE_VH',
            'OPT_mean', 'OPT_var', 'OPT_hom', 'OPT_con', 'OPT_dis', 'OPT_ent', 'OPT_asm'
        ]
    print(nofc_x_keys)
    x_train, y_train = df_get_data(df, nofc_x_keys, 1, cate_name=cate_name)
    x_test, y_test = None, None
    if "TEST" in df:
        x_test, y_test = df_get_data(df, nofc_x_keys, 0, cate_name=cate_name)
    return nofc_x_keys, x_train, y_train, x_test, y_test


def randomSamples(x, y=None):
    random_list = [i for i in range(len(x))]
    random.shuffle(random_list)
    x = x[random_list]
    if y is not None:
        y = y[random_list]
    return x, y


def main():
    # print(method_name17())
    # method_name10()
    # method_name16()

    # 筛选出阴影下样本
    csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\9\shh2_spl9_bj12.csv"

    def addSort(_df: pd.DataFrame, field, is_forward=True):
        _df = _df.sort_values(by=field, ascending=is_forward)
        to_field_name = "{0}_SORT{1}".format(field, 0 if is_forward else 1)
        _df[to_field_name] = [i + 1 for i in range(len(_df))]
        return _df

    df = pd.read_csv(csv_fn)
    df = addSort(df, "B2")
    df = addSort(df, "B3")
    df = addSort(df, "B4")
    df = addSort(df, "B8")
    df = df.sort_values(by="id")

    print(df.keys())

    def map_df(_df, map_func):
        line = []
        jdt = Jdt(len(_df), "map_df").start()
        for i in range(len(df)):
            line.append(map_func(df.loc[i]))
            jdt.add()
        jdt.end()
        return line

    def map_func1(df_line: pd.Series):
        j_dict = {'B2_SORT0': 200, 'B3_SORT0': 200, 'B4_SORT0': 200, 'B8_SORT0': 200}
        if all(df_line[name] < j_dict[name] for name in j_dict):
            return 1
        else:
            return 0

    def map_func2(df_line: pd.Series):
        if int(df_line["SHADOW"]) == 1:
            return int(df_line["CATEGORY"] / 10) * 10 + 2
        else:
            return df_line["CATEGORY"]

    df["SHADOW"] = map_df(df, map_func1)
    df["SHADOW_SUM"] = df[['B2_SORT0', 'B3_SORT0', 'B4_SORT0', 'B8_SORT0']].sum(axis=1)
    df["CATEGORY"] = map_df(df, map_func2)

    print("SHADOW number:", df["SHADOW"].sum())
    df = df.sort_values(by=["F2", "CATEGORY", "NDVI"],ascending=[False, True, False])
    df.to_csv(numberfilename(csv_fn), index=False)
    return


def method_name17():
    # 样本采样
    csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\9\shh2_spl9_bj1.csv"
    to_csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\9\shh2_spl9_bj12.csv"
    SHHGDALSampling().samplingSHH13_CSV(csv_fn, to_csv_fn, )
    df = pd.read_csv(to_csv_fn)
    df["NDVI"] = (df["B8"] - df["B4"]) / (df["B8"] + df["B4"] + 0.0000001)
    df["NDWI"] = (df["B3"] - df["B8"]) / (df["B3"] + df["B8"] + 0.0000001)
    df["MNDWI"] = (df["B3"] - df["B12"]) / (df["B3"] + df["B12"] + 0.0000001)
    df.to_csv(to_csv_fn)
    return to_csv_fn


def method_name16():
    # 精确样本
    imdc_keys = [
        'AS_VV', 'AS_VH', 'DE_VV', 'DE_VH',
        'B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDWI', 'NDVI',
        'OPT_mean', 'OPT_var', 'OPT_hom', 'OPT_con', 'OPT_dis', 'OPT_ent', 'OPT_asm', 'OPT_cor',
        'Map', 'ESA_F', 'NDVI_F', 'NDWI_F', 'F1_SUM', 'NDVI_GT', 'NDVI_WD', 'F2',
    ]

    def error_c_dict_get(is_ret_cn=False):
        cname_true_list = SHHConfig.SHH_CNAMES8
        cname_error_list = SHHConfig.SHH_CNAMES4
        category = 1
        category_dict = {}
        to_dict = {}
        for cname_true in cname_true_list:
            if cname_true not in category_dict:
                category_dict[cname_true] = {}
            for cname_error in cname_error_list:
                category_dict[cname_true][cname_error] = category
                to_dict[category] = "{0} | {1}".format(cname_error, cname_true)
                category += 1

        if is_ret_cn:
            return to_dict
        return category_dict

    code_name_dict = error_c_dict_get(is_ret_cn=True)
    for n in code_name_dict:
        print("{0:<3d}: {1}".format(n, code_name_dict[n]))

    def train_data():
        df_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\6\sh2_spl6_1_2_spl1_1.csv"

        # df = pd.read_csv(df_fn)

        def error_c():
            error_c_dict = error_c_dict_get()
            print(pd.DataFrame(error_c_dict))

            c_true = df["CATEGORY_NAME"].values.tolist()
            c_error = df["CNAME"].values.tolist()
            error_c_list = []
            error_cname_list = []
            for i in range(len(c_true)):
                error_c_list.append(error_c_dict[c_true[i]][c_error[i]])
                error_cname_list.append("{0} | {1}".format(c_error[i], c_true[i]))

            df["ERROR_CODE"] = error_c_list
            df["ERROR_CNAME"] = error_cname_list

        df_fn = changext(df_fn, "_error_c.csv")
        # error_c()
        # df.to_csv(df_fn)

        df = pd.read_csv(df_fn)
        print(df.keys())
        print("Train Sample Number:", len(df))
        _x = df[imdc_keys].values
        _y = df["ERROR_CODE"].values
        return _x, _y

    df_test_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\6\shh2_spl6_qd_random2_11.csv"

    def test_data(is_y=False):
        df_fn = df_test_fn
        df = pd.read_csv(df_fn)
        print(df.keys())
        print("Test Sample Number:", len(df))
        _x = df[imdc_keys].values
        _y = None
        if is_y:
            _y = df["ERROR_CODE"].values
        return _x, _y

    x_train, y_train = train_data()
    x_test, y_test = test_data()

    clf = RandomForestClassifier(60)
    clf.fit(x_train, y_train)
    print("train acc:", clf.score(x_train, y_train))

    df_test = pd.read_csv(df_test_fn)
    df_test["ERROR_CODE"] = clf.predict(x_test)
    df_test["ERROR_CNAME"] = SHHConfig.categoryMap(df_test["ERROR_CODE"], code_name_dict)
    df_test.to_csv(numberfilename(df_test_fn), index=False)

    # nofc_x_keys, x_train, y_train, x_test, y_test = trainMLData(df, cate_name="CATEGORY_CODE")
    # y_train = np.array(SHHConfig.categoryMap(y_train, SHHConfig.CATE_MAP_SH841))
    # x_train, y_train = randomSamples(x_train, y_train)
    # # clf = tree.DecisionTreeClassifier()
    # clf = RandomForestClassifier(60)
    # clf.fit(x_train, y_train)
    # print("train acc:", clf.score(x_train, y_train))
    # shh_mi = SHHModelImdc()
    # shh_mi.initSHH1GLCM("qd", nofc_x_keys)
    # print("IMDC")
    # to_fns = shh_mi.imdc(clf, to_dirname=r"F:\ProjectSet\Shadow\Hierarchical\Images\temp",
    #                      name="2", color_table=SHHConfig.SHH_COLOR8)

    return


def method_name15():
    # train one
    df_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\6\sh2_spl6_1_212_spl.csv"
    df_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\6\sh2_spl6_1_2_spl1_1.csv"
    df = pd.read_csv(df_fn)
    print("Sample Number:", len(df))
    nofc_x_keys, x_train, y_train, x_test, y_test = trainMLData(df, cate_name="CATEGORY_CODE")
    y_train = np.array(SHHConfig.categoryMap(y_train, SHHConfig.CATE_MAP_SH841))
    x_train, y_train = randomSamples(x_train, y_train)
    # clf = tree.DecisionTreeClassifier()
    clf = RandomForestClassifier(60)
    clf.fit(x_train, y_train)
    print("train acc:", clf.score(x_train, y_train))
    shh_mi = SHHModelImdc()
    shh_mi.initSHH1GLCM("qd", nofc_x_keys)
    print("IMDC")
    to_fns = shh_mi.imdc(clf, to_dirname=r"F:\ProjectSet\Shadow\Hierarchical\Images\temp",
                         name="2", color_table=SHHConfig.SHH_COLOR8)
    print(to_fns[0])

    # tiffAddColorTable(r"F:\ProjectSet\Shadow\Hierarchical\Images\temp\qd_1_imdc.tif", code_colors=SHHConfig.SHH_COLOR8)


def method_name14():
    sdf = SRTDataFrame().read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\7\shh2_spl7_qd1.csv", is_auto_type=True)
    gr = GDALRaster(SHHConfig.SHHFNImages.images1().qd)
    to_list = []
    to_dir = r"F:\ProjectSet\Shadow\Hierarchical\Samples\7\shh2_spl7_qd1_spsl1.json"
    with open(to_dir, "w", encoding="utf-8") as f:
        f.write("[")
        jdt = Jdt(len(sdf), "Sampling").start()
        for i in range(len(sdf)):
            line = sdf.rowToDict(i)
            x, y = line["X"], line["Y"]
            line["NPY"] = gr.readAsArrayCenter(x, y, 21, 21, is_geo=True).tolist()
            to_list.append(line)
            jdt.add()
        jdt.end()
    saveJson(to_list, )


def method_name13():
    # 决策树画
    df_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\5\FenCengSamples_glcm.xlsx"
    df = pd.read_excel(df_fn, sheet_name="GLCM")
    ext_feat(df, "ndvi", "ndwi", "mndwi")
    nofc_x_keys = [
        'B2', 'B3', 'B4', 'B8', 'B11', 'B12',
        'ndvi', 'ndwi', 'mndwi',
        'AS_VV', 'AS_VH', 'DE_VV', 'DE_VH',
        'OPT_mean', 'OPT_var', 'OPT_hom', 'OPT_con', 'OPT_dis', 'OPT_ent', 'OPT_asm'
    ]
    print(nofc_x_keys)
    x_train, y_train = df_get_data(df, nofc_x_keys, 1)
    x_test, y_test = df_get_data(df, nofc_x_keys, 0)

    clf = tree.DecisionTreeClassifier(min_samples_split=6, min_samples_leaf=6)
    clf.fit(x_train, y_train)
    print("train acc:", clf.score(x_train, y_train))
    print("test acc:", clf.score(x_test, y_test))
    # plt.figure(figsize=(12, 12))
    # tree.plot_tree(clf, feature_names=nofc_x_keys, class_names=SHHConfig.SHH_CNAMES[1:], fontsize=6)
    # plt.show()
    tree.export_graphviz(clf, feature_names=nofc_x_keys, class_names=SHHConfig.SHH_CNAMES[1:],
                         out_file=r"F:\Week\20240310\Data\tree2.dot", filled=True)
    shh_mi = SHHModelImdc()
    shh_mi.initSHH1GLCM(nofc_x_keys)
    print("IMDC")
    shh_mi.imdc(clf, to_dirname=r"F:\Week\20240310\Data", name="1", color_table=SHHConfig.SHH_COLOR8)


def method_name12():
    # 做一个试试的试一下，像元近距离测试
    shfi = SHHConfig.SHHFNImages.images1()
    gr = GDALRaster(shfi.qd)
    d = gr.readAsArray()
    row, column = gr.coorGeo2Raster(120.4006256, 36.0869219, True)
    d0 = d[:, row, column]
    d_range = np.array([100, 100, 100, 100, 100, 10000, 20000, 200000, 200, 200, 10000, 10000])
    d_min = d0 - d_range
    d_max = d0 + d_range
    np.set_printoptions(linewidth=500)
    print("d0     :", d0)
    print("d_range:", d_range)
    print("d_min  :", d_min)
    print("d_max  :", d_max)
    d_min = np.reshape(d_min, (len(d_min), 1, 1))
    d_max = np.reshape(d_max, (len(d_max), 1, 1))
    d_cal = (d_min < d) & (d < d_max)
    d_cal = np.all(d_cal, axis=0) * 1
    gr.save(d_cal.astype("int8"), numberfilename(r"F:\Week\20240310\Data\tmp.tif"), fmt="GTiff", dtype=gdal.GDT_Byte,
            options=["COMPRESS=PACKBITS"])


def method_name11():
    # 提取纹理
    sh_glcm = ShadowRasterGLCM()
    sh_glcm.meanFourDirection("qd_sh2_1_gray_envi", "OPT_", r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\glcm")
    sh_glcm.meanFourDirection("bj_sh2_1_gray_envi", "OPT_", r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\glcm")
    sh_glcm.meanFourDirection("cd_sh2_1_gray_envi", "OPT_", r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\glcm")


def method_name10():
    # 勾选测试样本
    r"""

        "F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\qd_sh2_1.tif"
        "F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\bj_sh2_1.tif"
        "F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\cd_sh2_1.tif"

        "F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\qd_sh2_esa.tif"
        "F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\bj_sh2_esa.tif"
        "F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\cd_sh2_esa.tif"

        """

    o_rasters_sampling = GDALRastersSampling(
        r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\qd_sh2_1.tif",
        r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\bj_sh2_1.tif",
        r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\cd_sh2_1.tif",
    )
    esa_rasters_sampling = GDALRastersSampling(
        r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\qd_sh2_esa.tif",
        r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\bj_sh2_esa.tif",
        r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\cd_sh2_esa.tif",
    )
    sdfc = SRTDFColumnCal()

    # dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Samples\6")
    # sdfc.read_csv(dfn.fn("shh2_spl6_qd_random2_1.csv"), is_auto_type=True)
    # csv_fn = numberfilename(dfn.fn("shh2_spl6_qd_random2_1.csv"))

    # r"F:\ProjectSet\Shadow\Hierarchical\Samples\9\shh2_spl9_bj12.csv"
    dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Samples\9")
    sdfc.read_csv(dfn.fn("shh2_spl9_bj12.csv"), is_auto_type=True)
    csv_fn = numberfilename(dfn.fn("shh2_spl9_bj12.csv"))

    def sampling_column():
        x, y = sdfc["X"], sdfc["Y"]
        spl_d1 = o_rasters_sampling.samplingIter(x, y)
        spl_d2 = esa_rasters_sampling.samplingIter(x, y)
        return

    def esa_func(line: dict):
        c_name = "ESA21"
        if c_name not in line:
            c_name = "Map"
        x, y = line["X"], line["Y"]
        d = esa_rasters_sampling.sampling(x, y, win_row_size=7, win_column_size=7)
        if np.all(d == line[c_name]):
            return line[c_name]
        else:
            return -1

    def ndvi_func(line: dict):
        x, y = line["X"], line["Y"]
        d = o_rasters_sampling.sampling(x, y, win_row_size=7, win_column_size=7)
        ndvi = (d[9] - d[8]) / (d[9] + d[8])
        if np.all(ndvi > 0.45):
            return 1
        else:
            return -1

    def ndwi_func(line: dict):
        x, y = line["X"], line["Y"]
        d = o_rasters_sampling.sampling(x, y, win_row_size=7, win_column_size=7)
        ndwi = (d[7] - d[9]) / (d[7] + d[9])
        if np.all(ndwi > 0.2):
            return 1
        else:
            return -1

    esa_is_map = {10: 21, 20: 21, 30: 21, 40: 21, 50: 11, 60: 31, 70: 0, 80: 41, 90: 21, 95: 21, 100: 21}

    def category_func(line: dict):
        c_name = "ESA21"
        if c_name not in line:
            c_name = "Map"
        cate = esa_is_map[line[c_name]]
        if line[c_name] == 40:
            if line["NDVI"] < 0.25:
                cate = 31
        return cate

    # sdfc.fit("CATEGORY", category_func)
    def ndvi_gt(line: dict):
        if line["F1_SUM"] == 0:
            return -1
        if line["NDVI"] > 0.5:
            if line["CATEGORY"] == 21:
                return 1
            else:
                return -1
        elif line["NDVI"] < 0.3:
            if line["CATEGORY"] == 21:
                return -1
            else:
                return 1
        else:
            return 1

    def ndvi_wd(line: dict):
        if line["F1_SUM"] != 0:
            return -1
        if line["NDVI"] > 0.6:
            if line["CATEGORY"] == 21:
                return 1
        return -1

    def filter1_sum(line: dict):
        return (line["ESA_F"] != -1) + (line["NDVI_F"] != -1) + (line["NDWI_F"] != -1)

    def filter1_func(line: dict):
        return (line["F1_SUM"] != 0) * 1

    def filter2_sum(line: dict):
        return (line["ESA_F"] != -1) + (line["NDVI_F"] != -1) + (line["NDWI_F"] != -1) + (line["NDVI_GT"] != -1)

    def filter2_func(line: dict):
        return ((line["NDVI_GT"] == 1) or (line["NDVI_WD"] == 1)) * 1

    def filter1():
        sdfc.fit("CATEGORY", category_func)
        sdfc.fit("ESA_F", esa_func)
        sdfc.fit("NDVI_F", ndvi_func)
        sdfc.fit("NDWI_F", ndwi_func)
        sdfc.fit("F1_SUM", filter1_sum)
        sdfc.fit("NDVI_GT", ndvi_gt)
        sdfc.fit("NDVI_WD", ndvi_wd)
        sdfc.fit("F2", filter2_func)

    filter1()
    sdfc.toCSV(csv_fn)
    print(csv_fn)


def method_name9():
    # 计算灰度
    geo_fns = [r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\qd_sh2_1.tif",
               r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\bj_sh2_1.tif",
               r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\cd_sh2_1.tif", ]

    for geo_fn in geo_fns:
        to_fn = changext(geo_fn, "_gray_envi.dat")
        print(to_fn)

        gr = GDALRaster(geo_fn)
        d = gr.readAsArray()
        d_gray = d[8] * 0.299 + d[7] * 0.587 + d[6] * 0.114
        gr.save(d_gray, to_fn, descriptions=["Gray"])


def method_name8():
    # 去除影像中的nan
    geo_fn = r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\qd_sh2_1.tif"
    gr = GDALRaster(geo_fn)
    d = gr.readAsArray()
    d_nan = d[np.isnan(d)]
    print(d_nan)
    if len(d_nan) != 0:
        to_d_nan = np.random.random(np.size(d_nan))
        print(to_d_nan)
        d[np.isnan(d)] = to_d_nan
        to_fn = changext(geo_fn, "_envi.dat")
        gr.save(d, to_fn, descriptions=gr.names)


def method_name6():
    # 图像阴影区域提取
    def image_cal(cal_func, im):
        to_im = np.zeros((im.shape[1], im.shape[2]))
        jdt = Jdt(im.shape[1], "image_cal")
        jdt.start()
        for i in range(im.shape[1]):
            for j in range(im.shape[2]):
                to_im[i, j] = cal_func(im[:, i, j])
        jdt.add()
        jdt.end()
        return to_im

    def cal_func1(data):
        mean_rgb = np.mean(data[6:9], axis=0)
        d1 = (data[6] < 500) * 1 \
             + (data[7] < 500) * 1 \
             + (data[8] < 500) * 1 \
             + (data[9] < 1000) * 1 \
             + (np.abs(mean_rgb - data[9]) < 20) * 1 \
             + (np.abs(mean_rgb - data[9]) < 20) * 1
        # d1 = np.std(data[6:9]qq, axis=0)
        return d1

    gr = GDALRaster(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\qd_sh2_1.tif")
    d = gr.readAsArray()
    to_d = cal_func1(d)
    # to_d = image_cal(cal_func1, d)
    gr.save(to_d.astype("int8"), r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\Temp\tmp11.tif", fmt="GTiff",
            options=["COMPRESS=PACKBITS"], dtype=gdal.GDT_Byte)


def method_name7():
    trainMLFC()


def method_name5():
    # 画散点图
    df = pd.read_excel(r"F:\ProjectSet\Shadow\Hierarchical\Samples\4\sh2_spl4_1_spl1.xlsx", sheet_name="Sheet1")

    def scatter(ax, field_name, x_name, y_name, show_names=None):
        if show_names is not None:
            eq_list = show_names
        else:
            eq_list = pd.unique(df[field_name])
        print(eq_list)
        for eq_data in eq_list:
            df_show = df[df[field_name] == eq_data]
            ax.scatter(df_show[x_name], df_show[y_name], label=eq_data)
        ax.legend()
        ax.set_xlim([0, 8000])
        ax.set_ylim([0, 8000])
        ax.plot([0, 8000], [0, 8000], color="red")

    # ['VEG' 'HIGH' 'LOW']
    fig = plt.figure(figsize=(3 * 6, 1 * 6))
    axes = fig.subplots(1, 3)
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, hspace=0.04, wspace=0.1)
    scatter(axes[0], "CNAME", "B4", "B3", show_names=['VEG'])
    scatter(axes[1], "CNAME", "B4", "B3", show_names=['HIGH'])
    scatter(axes[2], "CNAME", "B4", "B3", show_names=['LOW'])
    plt.legend()
    plt.show()


def method_name4():
    # 样本采样
    shh_sc = ShadowHierarchicalSampleCollection()
    shh_sc.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\2\sh2_spl2_road1.csv")
    shh_sc.shhSampling1()
    shh_sc.toCSV(r"F:\ProjectSet\Shadow\Hierarchical\Samples\2\sh2_spl2_road1_spl.csv")


def method_name3():
    # 使用稳定的样本获取
    def qd_wd(n_spl, esa_lc_code, win_rows, win_columns, to_fn):
        shht_grc = SHHT_GDALRandomCoor()
        shht_grc.initRandomCoor(city_type="qd1")
        shht_grc.initGRSampling(grs_type="qd_esa21")

        def wending(d, *args, **kwargs):
            return np.all(d == esa_lc_code)

        shht_grc.sampling(n_spl, wending, win_rows, win_columns, )
        shh_sc = shht_grc.toSHHSC()
        shh_sc.shhSampling1()
        shh_sc.toCSV(to_fn)

    def qd_wd2(wd_func, n_spl, grs_type, win_rows, win_columns, to_fn):
        """ grs_type qd_sh1 qd_esa21 """
        shht_grc = SHHT_GDALRandomCoor()
        shht_grc.initRandomCoor(city_type="qd1")
        shht_grc.initGRSampling(grs_type=grs_type)
        shht_grc.sampling(n_spl, wd_func, win_rows, win_columns, )
        shh_sc = shht_grc.toSHHSC()
        shh_sc.shhSampling1()
        shh_sc.toCSV(to_fn)

    def qd1():
        qd_wd(1000, 10, 3, 3, r"F:\ProjectSet\Shadow\Hierarchical\Samples\2\sh2_spl2_tree1.csv")  # 林地
        qd_wd(1500, 30, 3, 3, r"F:\ProjectSet\Shadow\Hierarchical\Samples\2\sh2_spl2_grass1.csv")  # 草地
        qd_wd(2000, 40, 5, 5, r"F:\ProjectSet\Shadow\Hierarchical\Samples\2\sh2_spl2_farm1.csv")  # 耕地

        qd_wd(3000, 50, 5, 5, r"F:\ProjectSet\Shadow\Hierarchical\Samples\2\sh2_spl2_building1.csv")  # 建筑
        qd_wd(1500, 60, 3, 3, r"F:\ProjectSet\Shadow\Hierarchical\Samples\2\sh2_spl2_soil1.csv")  # 裸土

        qd_wd(3000, 80, 3, 3, r"F:\ProjectSet\Shadow\Hierarchical\Samples\2\sh2_spl2_water1.csv")  # 水体

    def qd_cat():
        shh_sc = ShadowHierarchicalSampleCollection()
        # shh_sc.initSHHCategory()
        shh_sc.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\2\sh2_spl2_building1.csv", {"VHL": 2})
        shh_sc.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\2\sh2_spl2_farm1.csv", {"VHL": 1})
        shh_sc.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\2\sh2_spl2_grass1.csv", {"VHL": 1})
        shh_sc.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\2\sh2_spl2_soil1.csv", {"VHL": 2})
        shh_sc.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\2\sh2_spl2_tree1.csv", {"VHL": 1})
        shh_sc.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\2\sh2_spl2_water1.csv", {"VHL": 3})
        shh_sc.toCSV(r"F:\ProjectSet\Shadow\Hierarchical\Samples\2\sh2_spl2_1.csv")

    def low_filter1():
        shh_sc = ShadowHierarchicalSampleCollection()
        shh_sc.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\1\qd_spl2_low2_filter2.csv")
        shh_sc.initGDALRastersSampling(grs_type="qd_esa21")

        def low_filter1_func1(spl: SRTSample, d):
            if spl["ESA21"] == 50:
                return True
            elif spl["ESA21"] == 10:
                if np.all(d == 10):
                    return False
                else:
                    return True
            elif spl["ESA21"] == 30:
                if np.all(d == 30):
                    return False
                else:
                    return True

            elif spl["ESA21"] == 40:
                return False
            elif spl["ESA21"] == 60:
                return False
            else:
                return False

        def low_filter1_func2(spl: SRTSample, d):
            if (spl["SUMRGB"] < 1600) and (spl["B8"] < 750):
                return True
            else:
                return False

        shh_sc_2 = shh_sc.filterFuncGRS(low_filter1_func2, 3, 3, )
        print(len(shh_sc_2))
        shh_sc_2.toCSV(r"F:\ProjectSet\Shadow\Hierarchical\Samples\1\qd_spl2_low2_filter2_1.csv")

    def func1():
        is_show = False

        def not_wending(d, *args, **kwargs):
            ndwi = (d[7, :, :] - d[9, :, :]) / (d[7, :, :] + d[9, :, :])
            if np.all(ndwi > 0):
                return False
            ndvi = (d[9, :, :] - d[8, :, :]) / (d[9, :, :] + d[8, :, :])
            d = dataCenter(ndvi)
            if d < 0.42:
                return False
            if is_show:
                plt.imshow(ndvi)
                plt.show()
            return not np.all(ndvi > 0.35)

        # qd_wd2(wd_func, n_spl, grs_type, win_rows, win_columns, to_fn)
        qd_wd2(not_wending, 20000, "qd_sh1", 7, 7, r"F:\ProjectSet\Shadow\Hierarchical\Samples\3\sh2_spl3_ndvi2.csv")

    func1()


def method_name2():
    # 青岛有个地方有NAN
    # (array([646, 646, 646, 647, 647, 647], dtype=int64),
    #  array([2986, 2987, 2988, 2986, 2987, 2988], dtype=int64))
    gr = GDALRaster(r"F:\ProjectSet\Shadow\Hierarchical\Images\qd_sh2_1.tif")
    d = gr.readAsArray()
    d[6:, 646:646 + 2, 2986:2986 + 3] = d[6:, 645:646, 2985:2986]
    gr.save(
        d.astype("float32"), save_geo_raster_fn=r"F:\ProjectSet\Shadow\Hierarchical\Images\qd_sh2_2.tif",
        fmt="GTIFF",
        dtype=GDALRaster.NpType2GDALType["float32"],
        options=[
            # "COMPRESS=JPEG", "PHOTOMETRIC=YCBCR",
            # "COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=9",
            "COMPRESS=PACKBITS"
        ]
    )
    # d = rastersHist(
    #     # r"F:\ProjectSet\Shadow\Hierarchical\Images\bj_sh2_1.tif",
    #     # r"F:\ProjectSet\Shadow\Hierarchical\Images\cd_sh2_1.tif",
    #     r"F:\ProjectSet\Shadow\Hierarchical\Images\qd_sh2_1.tif",
    #     bins=1000,
    # )
    # saveJson(d, r"F:\ProjectSet\Shadow\Hierarchical\Images\qd_sh2_hist.json")


def method_name1():
    gr = GDALRaster(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\qd_sh2_1.tif")
    random_coor = SHHT_RandomCoor(gr.x_min, gr.x_max, gr.y_min, gr.y_max)
    random_coor.generate(200000)
    shh_sc = random_coor.toSHHSC()
    # shh_sc = ShadowHierarchicalSampleCollection()
    # shh_sc.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\2\sh2_spl2_road1.csv")
    shh_sc.shhSampling1()
    shh_sc.toCSV(r"F:\ProjectSet\Shadow\Hierarchical\Samples\1\qd_spl2.csv")


if __name__ == "__main__":
    main()
