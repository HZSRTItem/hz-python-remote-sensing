# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SunChao.py
@Time    : 2024/4/18 9:20
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SunChao
-----------------------------------------------------------------------------"""
import datetime
import json
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from osgeo import gdal


class GDALRaster:

    def __init__(self, geo_fn):
        self.geo_fn = geo_fn

    def readAsArray(self):
        return gdal.Open(self.geo_fn).ReasAsArray()


def readJson(json_fn):
    with open(json_fn, "r", encoding="utf-8") as f:
        return json.load(f)


class Jdt:
    """
    进度条
    """

    def __init__(self, total=100, desc=None, iterable=None, n_cols=20):
        """ 初始化一个进度条对象

        :param iterable: 可迭代的对象, 在手动更新时不需要进行设置
        :param desc: 字符串, 左边进度条描述文字
        :param total: 总的项目数
        :param n_cols: 调整进度条宽度, 默认是根据环境自动调节长度, 如果设置为0, 就没有进度条, 只有输出的信息
        """
        self.total = total
        self.iterable = iterable
        self.n_cols = n_cols
        self.desc = desc if desc is not None else ""

        self.n_split = float(total) / float(n_cols)
        self.n_current = 0
        self.n_print = 0
        self.is_run = False

    def start(self, is_jdt=True):
        """ 开始进度条 """
        if not is_jdt:
            return self
        self.is_run = True
        self._print()
        return self

    def add(self, n=1, is_jdt=True):
        """ 添加n个进度

        :param n: 进度的个数
        :return:
        """
        if not is_jdt:
            return
        if self.is_run:
            self.n_current += n
            if self.n_current > self.n_print * self.n_split:
                self.n_print += 1
                if self.n_print > self.n_cols:
                    self.n_print = self.n_cols
            self._print()

    def setDesc(self, desc):
        """ 添加打印信息 """
        self.desc = desc

    def _print(self):
        des_info = "\r{0}: {1:>3d}% |".format(self.desc, int(self.n_current / self.total * 100))
        des_info += "*" * self.n_print + "-" * (self.n_cols - self.n_print)
        des_info += "| {0}/{1}".format(self.n_current, self.total)
        print(des_info, end="")

    def end(self, is_jdt=True):
        if not is_jdt:
            return
        """ 结束进度条 """
        self.n_split = float(self.total) / float(self.n_split)
        self.n_current = 0
        self.n_print = 0
        self.is_run = False
        print()


def filterFileExt(dirname=".", ext=""):
    filelist = []
    for f in os.listdir(dirname):
        if os.path.splitext(f)[1] == ext:
            filelist.append(os.path.join(dirname, f))
    return filelist


def changefiledirname(filename, dirname):
    filename = os.path.split(filename)[1]
    return os.path.join(dirname, filename)


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():

    def func3():
        gr = GDALRaster(
            r".\drive-download-20240530T103852Z-001\1\sunchao_cjd_image_0_this_tif.tif")
        data = gr.readAsArray()
        print("1", np.sum(data == 1) * 100 / 1000000)
        print("2", np.sum(data == 2) * 100 / 1000000)
        print("3", np.sum(data == 3) * 100 / 1000000)

    def func5():
        gr = GDALRaster(
            r".\drive-download-20240530T103852Z-001\1\sunchao_cjd_image_0_this_tif.tif")
        data = gr.readAsArray()

        n = 60

        class chaojiandaimianji:

            def __init__(self):
                self.gct = []
                self.dct = []
                self.gt = []
                self.zmj = []

            def add(self, _data):
                n_data, counts = np.unique(_data, return_counts=True)
                to_dict = {}
                for i in range(len(n_data)):
                    to_dict[int(n_data[i])] = counts[i] * 100 / 1000000
                d_sum = 0
                if 1 in to_dict:
                    self.gct.append(to_dict[1])
                    d_sum += to_dict[1]
                else:
                    self.gct.append(0)
                if 2 in to_dict:
                    self.dct.append(to_dict[2])
                    d_sum += to_dict[2]
                else:
                    self.dct.append(0)
                if 3 in to_dict:
                    self.gt.append(to_dict[3])
                    d_sum += to_dict[3]
                else:
                    self.gt.append(0)
                self.zmj.append(d_sum)
                return

            def plot(self, x1, x2):
                x = x1 + np.array(list(range(len(self.gct)))) / len(self.gct) * (x2 - x1)
                plt.plot(x, self.gct[::-1], "-", color="lightgreen", label="高潮滩植被")
                plt.plot(x, self.dct[::-1], "-", color="green", label="低潮滩植被")
                plt.plot(x, self.gt[::-1], "-", color="grey", label="光滩")
                plt.plot(x, self.zmj[::-1], "-", color="red", label="总面积")

        cjdmj = chaojiandaimianji()

        for i in range(int(data.shape[1] / n) - 1):
            data_tmp = data[:, i * n:(i + 1) * n]
            cjdmj.add(data_tmp)

        # cjdmj.plot(117.701, 123.611)
        cjdmj.plot(34.594, 38.561)
        plt.xlabel(r"纬度", fontdict={"size": 10})
        plt.ylabel(r"面积\平方千米", fontdict={"size": 10})
        plt.legend()
        plt.show()

        print(data.shape)

    def func6():
        gr = GDALRaster(r".\drive-download-20240530T103852Z-001\sunchao_cjd_image.vrt")
        data = gr.readAsArray()
        print("1", np.sum(data == 1) * 100 / 1000000)


def method_name6():
    def func1(json_fn):
        d = readJson(json_fn)
        d_list = {id_dict[feat["id"]]: feat["properties"] for feat in d["features"]}
        for k in d_list:
            d_list[k]["id"] = k
        return d_list

    def func_id_dict(json_fn):
        d = readJson(json_fn)
        d_list = [feat["id"] for feat in d["features"]]
        for k in d_list:
            if k not in id_dict:
                id_dict[k] = len(id_dict)
        return len(d_list)

    fc_fields = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11',
                 'NDWI', 'mNDWI', 'AWEIsh', 'AWEInsh', 'LSWI', 'WI2015', 'NDVI', 'EVI']
    dirname = r".\drive-download-20240421T083153Z-001"
    id_dict = {}
    [func_id_dict(dirname + r"\samples_spl3_{}.geojson".format(i)) for i in range(36)]
    data_list = [func1(dirname + r"\samples_spl3_{}.geojson".format(i)) for i in range(36)]

    def get_df():
        ks = list(list(data_list[0].values())[0].keys())
        ks.append("FV_N1")
        ks.append("FV_N2")
        print(ks)
        to_dict = {i: {k: 0 for k in ks} for i in range(2000)}

        jdt = Jdt(len(data_list), "data_list").start()
        for data in data_list:
            for i in to_dict:
                if i in data:
                    to_dict[i]["id"] = data[i]["id"]
                    to_dict[i]["FV_N1"] += int(data[i]["NDVI"] > -0.1)
                    to_dict[i]["FV_N2"] += 1
                    to_dict[i]["Category"] = data[i]["Category"]
            jdt.add()
        jdt.end()
        df = pd.DataFrame(to_dict).T
        df["FV"] = df["FV_N1"] / df["FV_N2"]
        df.mean()

        # to_fn = numberfilename(r".\sample\1\sunchao_spl.csv")
        # df.to_csv(to_fn)
        df = df[['Category', 'id', "FV", "FV_N1", "FV_N2"]]
        print(df)
        return df

    df = get_df()

    # df = pd.read_csv(r".\drive-download-20240421T083153Z-001\samples_spl3_fw.csv")

    def text(x, y):
        for i in range(len(x)):
            plt.text(x[i], y[i], '{:.2f}'.format(y[i]), fontsize=9, ha='right')

    fv_y, fv_x = np.histogram(df[(df["Category"] == 1)]["FV"].values, range=[0, 1], bins=10)
    fv_y = fv_y / np.sum(fv_y)
    fv_x = fv_x[1:] - 0.07
    fv_y_sum = np.array([np.sum(fv_y[:i + 1]) for i in range(len(fv_y))])
    plt.bar(fv_x, fv_y, width=0.03, label="高潮滩植被频率")
    plt.plot(fv_x + 0.07, fv_y_sum, "ro--", label="高潮滩植被累计频率")
    text(fv_x + 0.07, fv_y_sum)

    fw_y, fw_x = np.histogram(df[df["Category"] == 2]["FV"].values, range=[0, 1], bins=10)
    fw_y = fw_y / np.sum(fw_y)
    fw_x = fw_x[1:] - 0.03
    fw_y_sum = np.array([np.sum(fw_y[:i + 1]) for i in range(len(fw_y))])
    plt.bar(fw_x, fw_y, width=0.03, label="低潮滩植被频率")
    plt.plot(fw_x + 0.03, fw_y_sum, "go--", label="低潮滩植被累计频率")
    text(fw_x + 0.03, fw_y_sum, )

    plt.xticks(np.linspace(0, 1, 11))
    plt.xlim([0, 1])
    plt.xlabel("植被频率", fontdict={"size": 10})
    plt.ylabel("频率", fontdict={"size": 10})
    plt.legend()
    plt.savefig(r".\tu\潮间带植被的频率分布特征3.jpg", dpi=300)
    plt.show()


def method_name5():
    df = pd.read_csv(r".\drive-download-20240421T083153Z-001\samples_spl3_fw.csv")

    fv_y, fv_x = np.histogram(df[(df["Category"] == 1) | (df["Category"] == 2)]["FW"].values, range=[0, 1], bins=10)
    fv_y = fv_y / np.sum(fv_y)
    plt.bar(fv_x[1:] - 0.07, fv_y, width=0.03, label="潮间带")
    fw_y, fw_x = np.histogram(df[df["Category"] == 4]["FW"].values, range=[0, 1], bins=10)
    fw_y = fw_y / np.sum(fw_y)
    plt.bar(fw_x[1:] - 0.03, fw_y, width=0.03, label="永久性水体")
    plt.xticks(fv_x)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.xlabel("水体频率", fontdict={"size": 10})
    plt.ylabel("频率", fontdict={"size": 10})
    plt.savefig(r".\tu\潮间带和永久性水体的频率分布特征3.jpg", dpi=300)
    plt.show()


def method_name4():
    # saveJson(to_dict, r".\sample\t1.txt")
    # d = readJson(r".\sample\t1.txt")
    # print([d0["id"] for d0 in d["bands"]])
    def func1(json_fn):
        d = readJson(json_fn)
        d_list = {id_dict[feat["id"]]: feat["properties"] for feat in d["features"]}
        for k in d_list:
            d_list[k]["id"] = k
        return d_list

    def func_id_dict(json_fn):
        d = readJson(json_fn)
        d_list = [feat["id"] for feat in d["features"]]
        for k in d_list:
            if k not in id_dict:
                id_dict[k] = len(id_dict)
        return len(d_list)

    fc_fields = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11',
                 'NDWI', 'mNDWI', 'AWEIsh', 'AWEInsh', 'LSWI', 'WI2015', 'NDVI', 'EVI']
    dirname = r".\drive-download-20240421T083153Z-001"
    id_dict = {}
    [func_id_dict(dirname + r"\samples_spl3_{}.geojson".format(i)) for i in range(36)]
    data_list = [func1(dirname + r"\samples_spl3_{}.geojson".format(i)) for i in range(36)]

    def get_df():
        ks = list(list(data_list[0].values())[0].keys())
        print("ks", ks)
        to_dict = {i: {k: 0 for k in ks} for i in range(len(id_dict))}
        jdt = Jdt(len(data_list), "data_list").start()
        for data in data_list:
            for i in to_dict:
                if i in data:
                    for k in fc_fields:
                        to_dict[i][k] += data[i][k]
                    to_dict[i]["id"] = data[i]["id"]
                    to_dict[i]["Category"] = data[i]["Category"]
            jdt.add()
        jdt.end()
        df = pd.DataFrame(to_dict).T
        df[fc_fields] = df[fc_fields] / 36

        print(df)
        # to_fn = numberfilename(r".\sample\1\sunchao_spl.csv")
        # df.to_csv(to_fn)

        return df

    df = get_df()

    def plot1():
        df1 = df[(df["Category"] == 1) | (df["Category"] == 2)]
        filed_names = ['LSWI', 'NDVI', 'EVI']
        to_dict1 = {}
        for filed_name in filed_names:
            y_list, x_list = np.histogram(df1[filed_name].values, bins=20)
            y_list = y_list / len(df1) * 100
            plt.plot(x_list[:-1], y_list, "o--", label=filed_name, )

        plt.legend()
        plt.xlabel("植被光谱指数", fontdict={"size": 11})
        plt.ylabel("频率(%)", fontdict={"size": 11})
        plt.savefig(r".\tu\植被光谱指数分布特征2.jpg", dpi=300)
        plt.show()

    # plot1()

    def plot2(fc_fields_k):
        # fc_fields_2 = ['NDWI', 'mNDWI', 'AWEIsh', 'AWEInsh', 'LSWI', 'WI2015', 'NDVI', 'EVI']
        # fc_fields_k = fc_fields_2[0]
        fig, ax = plt.subplots()

        time_list = []
        time_list_n = []

        def plot2_func1(is_0=True):
            data_list_tmp = data_list[:36]
            jdt = Jdt(len(data_list_tmp), "data_list").start()
            to_dict = {}
            for i, data in enumerate(data_list_tmp):
                # to_dict[i] = {j: data[j][fc_fields_k] for j in data}
                if is_0:
                    to_dict[i] = np.array([data[j][fc_fields_k] for j in data if data[j]["Category"] == 4])
                else:
                    to_dict[i] = np.array([data[j][fc_fields_k] for j in data if data[j]["Category"] != 4])
                jdt.add()
            jdt.end()

            # df = pd.DataFrame(to_dict)
            print(to_dict.keys())

            if is_0:
                color = "orange"
            else:
                color = "blue"
            time0 = datetime.datetime.strptime('2017-01-01', '%Y-%m-%d')

            for i in to_dict:
                if is_0:
                    if i % 5 == 0:
                        time_list.append((time0 + relativedelta(months=i)).strftime('%Y-%m'))
                        time_list_n.append(i)
                plot_1 = ax.boxplot(
                    to_dict[i],
                    # labels=self.names,
                    positions=[i],
                    patch_artist=True,
                    showmeans=True,
                    showfliers=False,
                    widths=0.7,
                    meanprops={"color": "white"},
                    medianprops={"color": "black", "linewidth": 1},
                    boxprops={"facecolor": color, "edgecolor": "black", "linewidth": 1},
                    whiskerprops={"color": "black", "linewidth": 1},
                    capprops={"color": "black", "linewidth": 1},
                )

        plot2_func1(True)
        plot2_func1(False)
        plt.xlabel("时间/月")
        plt.ylabel(fc_fields_k)
        plt.xticks(time_list_n, time_list)
        to_fn = r".\tu\潮间带和永久性水体的水体指数特征3_{0}.jpg".format(fc_fields_k)
        print(to_fn)
        plt.savefig(to_fn, dpi=300)

        plt.show()

    fc_fields_2 = [
        'NDWI', 'mNDWI', 'AWEIsh', 'AWEInsh', 'LSWI', 'NDVI', 'EVI',
        "WI2015"
    ]
    for k in fc_fields_2:
        plot2(k)

    def plot3():
        data_list_tmp = data_list[:12]
        jdt = Jdt(len(data_list_tmp), "data_list").start()
        to_dict = {}
        for i, data in enumerate(data_list_tmp):
            # to_dict[i] = {j: data[j][fc_fields_k] for j in data}
            if is_0:
                to_dict[i] = np.array([data[j][fc_fields_k] for j in data if data[j]["SAMPLE_1"] == 0])
            else:
                to_dict[i] = np.array([data[j][fc_fields_k] for j in data if data[j]["SAMPLE_1"] != 0])
            jdt.add()
        jdt.end()


def method_name3():
    def func1(json_fn):
        d = readJson(json_fn)
        d_list = {feat["properties"]["id"]: feat["properties"] for feat in d["features"]}
        return d_list

    fc_fields = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11',
                 'NDWI', 'mNDWI', 'AWEIsh', 'AWEInsh', 'LSWI', 'WI2015', 'NDVI', 'EVI']
    dirname = r".\sample\1\drive-download-20240418T014323Z-001"
    data_list = [func1(dirname + r"\samples_spl2_{}.geojson".format(i)) for i in range(36)]

    def get_df():
        ks = list(list(data_list[0].values())[0].keys())
        print(ks)
        to_dict = {i: {k: 0 for k in ks} for i in range(2000)}

        jdt = Jdt(len(data_list), "data_list").start()
        for data in data_list:
            for i in to_dict:
                if i in data:
                    to_dict[i]["id"] = data[i]["id"]
                    to_dict[i]["NDWI"] += int(data[i]["NDWI"] > -0.1)
                    to_dict[i]["SAMPLE_1"] = data[i]["SAMPLE_1"]
            jdt.add()
        jdt.end()
        df = pd.DataFrame(to_dict).T
        df[fc_fields] = df[fc_fields] / 36
        df.mean()
        # to_fn = numberfilename(r".\sample\1\sunchao_spl.csv")
        # df.to_csv(to_fn)
        df = df[['SAMPLE_1', 'id', "NDWI"]]
        print(df)
        return df

    df = get_df()

    def getxy(category):
        y, x = np.histogram(df[df["SAMPLE_1"] == category]["NDWI"].values, bins=20, range=[0, 1])
        y = y / np.sum(y)
        return x[:-1], y

    def getxy_noeq(category):
        y, x = np.histogram(df[df["SAMPLE_1"] != category]["NDWI"].values, bins=20, range=[0, 1])
        y = y / np.sum(y)
        return x[:-1], y

    x1, y1 = getxy(0)
    x1 = x1 - 0.01
    plt.bar(x1, y1, width=0.016, label="水体")
    x1, y1 = getxy_noeq(0)
    x1 = x1 + 0.01
    plt.bar(x1, y1, width=0.016, label="植被")
    plt.legend()
    plt.xticks(np.linspace(0, 1, 11))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("水体频率", fontdict={"size": 14})
    plt.ylabel("潮间带", fontdict={"size": 14})
    plt.savefig(r".\tu\潮间带和永久性水体的频率分布特征.jpg", dpi=300)
    plt.show()


def method_name2():
    def func1(json_fn):
        d = readJson(json_fn)
        d_list = [feat["properties"] for feat in d["features"]]
        return d_list

    d_list = func1(r".\sample\1\sunchao_coll_counts.geojson")
    df = pd.DataFrame(d_list)
    df["B2"] = df["B2"] / df["B2"].max()
    print(df)

    def getxy(category):
        y, x = np.histogram(df[df["SAMPLE_1"] == category]["B2"].values, bins=20, range=[0, 1])
        y = y / np.sum(df[df["SAMPLE_1"] == category]["B2"].values)
        return x[:-1], y

    def getxy_noeq(category):
        y, x = np.histogram(df[df["SAMPLE_1"] != category]["B2"].values, bins=20, range=[0, 1])
        y = y / np.sum(df[df["SAMPLE_1"] != category]["B2"].values)
        return x[:-1], y

    x1, y1 = getxy(0)
    x1 = x1 - 0.01
    plt.bar(x1, y1, width=0.016)
    x1, y1 = getxy_noeq(0)
    x1 = x1 + 0.01
    plt.bar(x1, y1, width=0.016)
    plt.xticks(np.linspace(0, 1, 11))
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.show()


def method_name():

    def func1(json_fn):
        d = readJson(json_fn)
        d_list = {feat["properties"]["id"]: feat["properties"] for feat in d["features"]}
        return d_list

    fc_fields = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11',
                 'NDWI', 'mNDWI', 'AWEIsh', 'AWEInsh', 'LSWI', 'WI2015', 'NDVI', 'EVI']
    dirname = r".\sample\1\drive-download-20240418T014323Z-001"
    data_list = [func1(dirname + r"\samples_spl2_{}.geojson".format(i)) for i in range(36)]

    def get_df():
        ks = list(list(data_list[0].values())[0].keys())
        to_dict = {i: {k: 0 for k in ks} for i in range(2000)}
        jdt = Jdt(len(data_list), "data_list").start()
        for data in data_list:
            for i in to_dict:
                if i in data:
                    for k in fc_fields:
                        to_dict[i][k] += data[i][k]
                    to_dict[i]["id"] = data[i]["id"]
                    to_dict[i]["SAMPLE_1"] = data[i]["SAMPLE_1"]
            jdt.add()
        jdt.end()
        df = pd.DataFrame(to_dict).T
        df[fc_fields] = df[fc_fields] / 36
        df.mean()
        print(df)
        # to_fn = numberfilename(r".\sample\1\sunchao_spl.csv")
        # df.to_csv(to_fn)

        return df

    df = get_df()

    def plot1():
        df1 = df[(df["SAMPLE_1"] == 1) | (df["SAMPLE_1"] == 2)]
        filed_names = ['LSWI', 'NDVI', 'EVI']
        to_dict1 = {}
        for filed_name in filed_names:
            y_list, x_list = np.histogram(df1[filed_name].values, bins=20)
            y_list = y_list / len(df1) * 100
            plt.plot(x_list[:-1], y_list, "o--", label=filed_name, )

        plt.legend()
        plt.xlabel("植被光谱指数", fontdict={"size": 11})
        plt.ylabel("频率(%)", fontdict={"size": 11})
        plt.savefig(r".\tu\植被光谱指数分布特征.jpg", dpi=300)
        plt.show()

    def plot2(fc_fields_k):
        # fc_fields_2 = ['NDWI', 'mNDWI', 'AWEIsh', 'AWEInsh', 'LSWI', 'WI2015', 'NDVI', 'EVI']
        # fc_fields_k = fc_fields_2[0]
        fig, ax = plt.subplots()

        def plot2_func1(is_0=True):
            data_list_tmp = data_list[:12]
            jdt = Jdt(len(data_list_tmp), "data_list").start()
            to_dict = {}
            for i, data in enumerate(data_list_tmp):
                # to_dict[i] = {j: data[j][fc_fields_k] for j in data}
                if is_0:
                    to_dict[i] = np.array([data[j][fc_fields_k] for j in data if data[j]["SAMPLE_1"] == 0])
                else:
                    to_dict[i] = np.array([data[j][fc_fields_k] for j in data if data[j]["SAMPLE_1"] != 0])
                jdt.add()
            jdt.end()

            # df = pd.DataFrame(to_dict)
            print(to_dict.keys())

            if is_0:
                color = "blue"
            else:
                color = "orange"
            for i in to_dict:
                plot_1 = ax.boxplot(
                    to_dict[i],
                    # labels=self.names,
                    positions=[i],
                    patch_artist=True,
                    showmeans=True,
                    showfliers=False,
                    widths=0.7,
                    meanprops={"color": "white"},
                    medianprops={"color": "black", "linewidth": 1},
                    boxprops={"facecolor": color, "edgecolor": "black", "linewidth": 1},
                    whiskerprops={"color": "black", "linewidth": 1},
                    capprops={"color": "black", "linewidth": 1},
                )

        plot2_func1(True)
        plot2_func1(False)
        plt.xlabel("时间/月")
        plt.ylabel(fc_fields_k)
        plt.savefig(r".\tu\潮间带和永久性水体的水体指数特征_{0}.jpg".format(fc_fields_k),
                    dpi=300)

        plt.show()

    fc_fields_2 = ['NDWI', 'mNDWI', 'AWEIsh', 'AWEInsh', 'LSWI', 'NDVI', 'EVI']
    for k in fc_fields_2:
        plot2(k)

    def plot3():
        data_list_tmp = data_list[:12]
        jdt = Jdt(len(data_list_tmp), "data_list").start()
        to_dict = {}
        for i, data in enumerate(data_list_tmp):
            # to_dict[i] = {j: data[j][fc_fields_k] for j in data}
            if is_0:
                to_dict[i] = np.array([data[j][fc_fields_k] for j in data if data[j]["SAMPLE_1"] == 0])
            else:
                to_dict[i] = np.array([data[j][fc_fields_k] for j in data if data[j]["SAMPLE_1"] != 0])
            jdt.add()
        jdt.end()

    plot2()


if __name__ == "__main__":
    main()
