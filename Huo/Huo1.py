# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : Huo1.py
@Time    : 2023/9/19 20:25
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of Huo1

分类：读取 ->
-----------------------------------------------------------------------------"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from SRTCodes.Utils import readJson, saveJson

plt.rcParams['font.family'] = 'Times New Roman'


def main():
    df = pd.read_excel(r"F:\ProjectSet\Huo\shufen\Sentinel-1\新建 Microsoft Excel 工作表.xlsx", sheet_name="Sheet1")
    have_types = df["HAVE_TYPE"].values
    vv_t = df["vv_t"].values
    sub = df["sub"].values
    cates = np.unique(have_types)
    print(cates)
    cates = ['have_to_have', 'nohave_to_nohave', 'nohave_to_have']
    lables = ["Trees", "Others", "Others to Trees"]
    colors = ["green", "blue", "red"]

    cates.reverse()
    lables.reverse()
    colors.reverse()

    i = 0
    for cate in cates:
        # if cate == "nohave_to_have":
        #     tmp1 = lables[1]
        # else:
        #     tmp1 = lables[0]
        x1, x2 = vv_t[have_types == cate], sub[have_types == cate]
        plt.scatter(x1, x2, label=lables[i], c=colors[i])
        i += 1
        # x1_1, x1_2 = np.histogram(x1, bins=30)
        # plt.plot(x1_2[:-1], x1_1, label=cate)
        # x2_1, x2_2 = np.histogram(x2, bins=30)
        # plt.plot(x2_2[:-1], x2_1, label=cate)
    plt.xlabel("Standard deviation of time series VV images")
    plt.ylabel("Difference between two years of normalized VV images")
    plt.legend()
    plt.show()

    pass


def method_name():
    d = readJson(r"F:\ProjectSet\Huo\shufen\Sentinel-1\stats_polygon_1.geojson")
    df = []
    for feat in d["features"]:
        df.append(feat["properties"])
    df = pd.DataFrame(df)
    print(df)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
    kmeans.fit(df[["vv_t", "sub"]])
    # Get the cluster centers and labels
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    # plt.scatter(df["vv_t"], df["std_18"], c=labels, cmap='viridis')
    # plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=300, c='red')
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    # plt.title('KMeans Clustering')
    # plt.show()
    df['label'] = labels
    print(cluster_centers)
    print(labels)
    for i in range(len(d["features"])):
        d["features"][i]["properties"]["label"] = float(labels[i])
    df.to_excel(r"F:\ProjectSet\Huo\shufen\test1.xlsx")
    saveJson(d, r"F:\ProjectSet\Huo\shufen\test1.geojson")


if __name__ == "__main__":
    main()
