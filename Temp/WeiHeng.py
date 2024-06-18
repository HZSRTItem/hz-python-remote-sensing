# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : WeiHeng.py
@Time    : 2024/5/24 12:40
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of WeiHeng
-----------------------------------------------------------------------------"""
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from osgeo import gdal

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题


def main():
    data = gdal.Open(r"F:\ProjectSet\Huo\weiheng\weiheng_ld\weiheng_ld_tif.tif").ReadAsArray()
    print( np.unique(data, return_counts=True)[1]*100/1000000)
    pass


def method_name1():
    # spl_dict = readJson(r"F:\ProjectSet\Huo\weiheng\weiheng_samples_spl2.geojson")
    df = pd.read_csv(r"F:\ProjectSet\Huo\weiheng\weiheng_samples_spl4.csv")
    df_g = df.groupby("leibie").mean(numeric_only=True).T
    date_list = [int(line.split("_")[0]) for line in df_g.index]
    df_g["date_n"] = date_list
    df_g = df_g.sort_values("date_n")
    df_g["con"] = 1.0
    df_g["t"] = (df_g["date_n"] + 1) / 36 * 6 * np.pi
    df_g["sin"] = np.sin(df_g["t"])
    df_g["cos"] = np.cos(df_g["t"])
    x = df_g[["con", "t", "sin", "cos"]].values
    k = (np.linalg.inv(x.T @ x) @ x.T).T

    def linear(y_name):
        y = np.array([df_g[y_name].values])
        a = y @ k
        x_tmp = a @ x.T
        df_g["{}_xb".format(y_name)] = x_tmp[0]

    linear(1)
    linear(2)
    linear(3)
    linear(4)
    print(df_g)

    def caodi():
        plt.plot(df_g["date_n"], df_g[1], "ro-", label="草地")
        plt.plot(df_g["date_n"], df_g["1_xb"], "ro--", label="cao di xb")

    def guanmu():
        plt.plot(df_g["date_n"], df_g[2], "go-", label="guan mu")
        plt.plot(df_g["date_n"], df_g["2_xb"], "go--", label="guan mu xb")

    def luoye():
        plt.plot(df_g["date_n"], df_g[3], "bo-", label="luo ye")
        plt.plot(df_g["date_n"], df_g["3_xb"], "bo--", label="luo ye xb")

    def changlv():
        plt.plot(df_g["date_n"], df_g[4], "yo-", label="chang lv")
        plt.plot(df_g["date_n"], df_g["4_xb"], "yo--", label="chang lv xb")

    # caodi()
    # guanmu()
    # luoye()
    changlv()
    time0 = datetime.strptime('2021-01-01', '%Y-%m-%d')
    n = 3
    time_list = [(time0 + relativedelta(months=i)).strftime('%Y-%m') for i in range(0, 37, n)]
    time_list_n = [i for i in range(0, 37, n)]
    plt.xticks(time_list_n, time_list, )
    # plt.ylim([0,0.8])
    plt.legend()
    df_g.to_excel(r"F:\ProjectSet\Huo\weiheng\wh_wuhou.xlsx", index=False)
    plt.savefig(r"F:\ProjectSet\Huo\weiheng\wh_wuhou_caodi.jpg", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
