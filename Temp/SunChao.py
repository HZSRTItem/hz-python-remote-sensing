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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from SRTCodes.Utils import readJson, Jdt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
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
    dirname = r"F:\ProjectSet\Huo\SunChao\drive-download-20240421T083153Z-001"
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

        # to_fn = numberfilename(r"F:\ProjectSet\Huo\SunChao\sample\1\sunchao_spl.csv")
        # df.to_csv(to_fn)
        df = df[['Category', 'id', "FV", "FV_N1", "FV_N2"]]
        print(df)
        return df

    df = get_df()
    # df = pd.read_csv(r"F:\ProjectSet\Huo\SunChao\drive-download-20240421T083153Z-001\samples_spl3_fw.csv")

    fv_y, fv_x = np.histogram(df[(df["Category"] == 1)]["FV"].values, range=[0, 1], bins=10)
    fv_y = fv_y / len(df)
    fv_x = fv_x[1:] - 0.07
    fv_y_sum = np.array([np.sum(fv_y[:i+1]) for  i in range(len(fv_y))])
    plt.bar(fv_x, fv_y, width=0.03, label="高潮滩植被频率")
    plt.plot(fv_x+0.07, fv_y_sum, "ro--", label="高潮滩植被累计频率")

    fw_y, fw_x = np.histogram(df[df["Category"] == 2]["FV"].values, range=[0, 1], bins=10)
    fw_y = fw_y / len(df)
    fw_x = fw_x[1:] - 0.03
    fw_y_sum = np.array([np.sum(fw_y[:i+1]) for i in range(len(fw_y))])
    plt.bar(fw_x, fw_y, width=0.03, label="低潮滩植被累计频率")
    plt.plot(fw_x+0.03, fw_y_sum,"go--", label="低潮滩植被频率")

    plt.xticks(np.linspace(0, 1, 11))
    plt.xlim([0, 1])
    plt.xlabel("植被频率", fontdict={"size": 10})
    plt.ylabel("频率", fontdict={"size": 10})
    plt.legend()
    plt.savefig(r"F:\ProjectSet\Huo\SunChao\tu\潮间带植被的频率分布特征2.jpg", dpi=300)
    plt.show()


def method_name5():
    df = pd.read_csv(r"F:\ProjectSet\Huo\SunChao\drive-download-20240421T083153Z-001\samples_spl3_fw.csv")

    fv_y, fv_x = np.histogram(df[(df["Category"] == 1) | (df["Category"] == 2)]["FW"].values, range=[0, 1], bins=10)
    fv_y = fv_y / len(df)
    plt.bar(fv_x[1:] - 0.07, fv_y, width=0.03, label="潮间带")
    fw_y, fw_x = np.histogram(df[df["Category"] == 4]["FW"].values, range=[0, 1], bins=10)
    fw_y = fw_y / len(df)
    plt.bar(fw_x[1:] - 0.03, fw_y, width=0.03, label="永久性水体")
    plt.xticks(fv_x)
    plt.xlim([0, 1])
    plt.ylim([0, 0.3])
    plt.xlabel("水体频率", fontdict={"size": 10})
    plt.ylabel("频率", fontdict={"size": 10})
    plt.savefig(r"F:\ProjectSet\Huo\SunChao\tu\潮间带和永久性水体的频率分布特征2.jpg", dpi=300)
    plt.show()


def method_name4():
    # saveJson(to_dict, r"F:\ProjectSet\Huo\SunChao\sample\t1.txt")
    # d = readJson(r"F:\ProjectSet\Huo\SunChao\sample\t1.txt")
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
    dirname = r"F:\ProjectSet\Huo\SunChao\drive-download-20240421T083153Z-001"
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
        # to_fn = numberfilename(r"F:\ProjectSet\Huo\SunChao\sample\1\sunchao_spl.csv")
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
        plt.savefig(r"F:\ProjectSet\Huo\SunChao\tu\植被光谱指数分布特征2.jpg", dpi=300)
        plt.show()

    plot1()

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
                color = "blue"
            else:
                color = "orange"
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
        plt.savefig(r"F:\ProjectSet\Huo\SunChao\tu\潮间带和永久性水体的水体指数特征2_{0}.jpg".format(fc_fields_k),
                    dpi=300)

        plt.show()

    # fc_fields_2 = ['NDWI', 'mNDWI', 'AWEIsh', 'AWEInsh', 'LSWI', 'NDVI', 'EVI']
    # for k in fc_fields_2:
    #     plot2(k)
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
    dirname = r"F:\ProjectSet\Huo\SunChao\sample\1\drive-download-20240418T014323Z-001"
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
        # to_fn = numberfilename(r"F:\ProjectSet\Huo\SunChao\sample\1\sunchao_spl.csv")
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
    plt.savefig(r"F:\ProjectSet\Huo\SunChao\tu\潮间带和永久性水体的频率分布特征.jpg", dpi=300)
    plt.show()


def method_name2():
    def func1(json_fn):
        d = readJson(json_fn)
        d_list = [feat["properties"] for feat in d["features"]]
        return d_list

    d_list = func1(r"F:\ProjectSet\Huo\SunChao\sample\1\sunchao_coll_counts.geojson")
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
    # saveJson(to_dict, r"F:\ProjectSet\Huo\SunChao\sample\t1.txt")
    # d = readJson(r"F:\ProjectSet\Huo\SunChao\sample\t1.txt")
    # print([d0["id"] for d0 in d["bands"]])
    def func1(json_fn):
        d = readJson(json_fn)
        d_list = {feat["properties"]["id"]: feat["properties"] for feat in d["features"]}
        return d_list

    fc_fields = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11',
                 'NDWI', 'mNDWI', 'AWEIsh', 'AWEInsh', 'LSWI', 'WI2015', 'NDVI', 'EVI']
    dirname = r"F:\ProjectSet\Huo\SunChao\sample\1\drive-download-20240418T014323Z-001"
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
        # to_fn = numberfilename(r"F:\ProjectSet\Huo\SunChao\sample\1\sunchao_spl.csv")
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
        plt.savefig(r"F:\ProjectSet\Huo\SunChao\tu\植被光谱指数分布特征.jpg", dpi=300)
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
        plt.savefig(r"F:\ProjectSet\Huo\SunChao\tu\潮间带和永久性水体的水体指数特征_{0}.jpg".format(fc_fields_k),
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
