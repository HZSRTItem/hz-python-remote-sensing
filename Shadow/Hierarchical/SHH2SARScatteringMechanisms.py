# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2SARScatteringMechanisms.py
@Time    : 2024/6/16 11:41
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2SARScatteringMechanisms
-----------------------------------------------------------------------------"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SRTCodes.GDALRasterIO import GDALRasterChannel
from Shadow.Hierarchical.SHHDraw import SHHGDALDrawImagesColumn

plt.rc('font', family='Times New Roman')
FONT_SIZE = 12
plt.rcParams.update({'font.size': 12})


def main():
    def read1(raster_fn):
        grc = GDALRasterChannel()
        grc.addGDALDatas(raster_fn=raster_fn)
        return grc

    df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\2\sh2_spl26_4_spl.csv")
    grc_is = read1(r"F:\ProjectSet\Shadow\Hierarchical\Analysis\2\is1.tif")
    grc_soil = read1(r"F:\ProjectSet\Shadow\Hierarchical\Analysis\2\soil1.tif")
    grc_sh = read1(r"F:\ProjectSet\Shadow\Hierarchical\Analysis\2\sh2.tif")

    fig = plt.figure(figsize=(8, 8))
    # plt.subplots_adjust(top=0.90, bottom=0.1, left=0.1, right=0.9)

    x0, y0 = 0.1, 0.1
    s_len1, s_len2 = 0.65, 0.16
    jiange = 0.02
    ax1 = fig.add_axes([x0, y0, s_len1, s_len1])  # 左下角坐标(0.1, 0.55)，宽度0.35，高度0.35
    ax2 = fig.add_axes([x0, y0 + s_len1 + jiange, s_len1, s_len2])  # 左下角坐标(0.55, 0.55)，宽度0.35，高度0.35
    ax3 = fig.add_axes([x0 + s_len1 + jiange, y0, s_len2, s_len1])  # 左下角坐标(0.55, 0.55)，宽度0.35，高度0.35

    def scatter1(color):
        _df = pd.DataFrame(df[df["CNAME"] == filter_c].to_dict("records"))
        _df[y_k] = np.rad2deg(_df[y_k].values)
        print(_df[[x_k, y_k]].describe())
        plt.scatter(-_df[x_k], _df[y_k], color=color, label=filter_c)

    def scatter2(color, _grc, ax):
        x = _grc[x_k].ravel()
        y = _grc[y_k].ravel()
        # y = np.rad2deg(y)
        ax.scatter(x, y, color=color, label=filter_c, )

    def hist(color, data, ax, name, is_xy):
        n, x = np.histogram(data, bins=256, range=(-60, 60))
        n = n / np.sum(n) * 100
        x = x[:-1]
        x = x + (x[1] - x[1]) / 2
        print(max(x), min(x))
        if is_xy:
            x, n = n, x
        ax.plot(x, n, color=color, label=name)


    grc = grc_sh
    filter_c = "IS AS"
    x_k, y_k = "AS_C11", "AS_C22"
    scatter2("red", grc, ax1)
    hist("red", grc[x_k], ax2, x_k, False)
    hist("red", grc[y_k], ax3, y_k, True)

    filter_c = "IS DE"
    x_k, y_k = "DE_C11", "DE_C22"
    scatter2("green", grc, ax1)
    hist("green", grc[x_k], ax2, x_k, False)
    hist("green", grc[y_k], ax3, y_k, True)

    ax1.set_xlabel("C11", fontdict={"size": FONT_SIZE})
    ax1.set_ylabel("C22", fontdict={"size": FONT_SIZE})

    ax2.set_xticks([])
    ax2.set_yticks([0, 0.5, 1.0, 1.5])
    ax2.set_ylabel("frequency(%)", fontdict={"size": FONT_SIZE})

    ax3.set_xticks([0, 0.5, 1.0, 1.5])
    ax3.set_yticks([])
    ax3.set_xlabel("frequency(%)", fontdict={"size": FONT_SIZE})

    ax1.set_xlim([-32, 32])
    ax2.set_xlim([-32, 32])
    ax3.set_xlim([0, 5])

    ax1.set_ylim([-32, 32])
    ax2.set_ylim([0, 5])
    ax3.set_ylim([-32, 32])

    ax1.set_xticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    ax1.set_yticks([-30, -25, -20, -15, -10, -5, 0])

    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    ax3.legend(frameon=False)

    plt.savefig(r"F:\ProjectSet\Shadow\Hierarchical\Analysis\2\sh_c2.jpg", dpi=300)
    plt.show()
    pass


def method_name2():
    # 组会 2024年6月16日15:36:25
    sgdic = SHHGDALDrawImagesColumn((120, 120))
    sgdic.addRCC_Im1()
    sgdic.fontdict["size"] = 16
    column_names = ["RGB", "NRG"]
    row_names = []

    def add1(name, x, y):
        n_row = len(row_names)
        row_names.append(name)
        sgdic.addAxisDataXY(n_row, 0, "RGB", x, y, min_list=[100, 100, 100], max_list=[2000, 2000, 2000])
        sgdic.addAxisDataXY(n_row, 1, "NRG", x, y, min_list=[100, 100, 100], max_list=[2000, 2000, 2000])

    add1("(1)    ", 120.103112, 36.297384)
    add1("(2)    ", 120.154764, 36.299025)
    add1("(3)    ", 120.373510,36.090501)
    sgdic.draw(n_columns_ex=2.6, n_rows_ex=2.6, row_names=row_names, column_names=column_names)
    plt.savefig(r"F:\ProjectSet\Shadow\Hierarchical\Analysis\2\issoil.jpg", dpi=300)
    plt.show()


def method_name1():
    def read1(raster_fn):
        grc = GDALRasterChannel()
        grc.addGDALDatas(raster_fn=raster_fn)
        return grc

    df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\2\sh2_spl26_4_spl.csv")
    grc_is = read1(r"F:\ProjectSet\Shadow\Hierarchical\Analysis\2\is1.tif")
    grc_soil = read1(r"F:\ProjectSet\Shadow\Hierarchical\Analysis\2\soil1.tif")
    fig = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(top=0.90, bottom=0.1, left=0.1, right=0.9)

    def scatter1(color):
        _df = pd.DataFrame(df[df["CNAME"] == filter_c].to_dict("records"))
        _df[y_k] = np.rad2deg(_df[y_k].values)
        print(_df[[x_k, y_k]].describe())
        plt.scatter(-_df[x_k], _df[y_k], color=color, label=filter_c)

    def scatter2(color, grc):
        x = grc[x_k].ravel()
        y = grc[y_k].ravel()
        # y = np.rad2deg(y)
        plt.scatter(x, y, color=color, label=filter_c, )

    filter_c = "IS AS"
    x_k, y_k = "AS_C11", "AS_C22"
    scatter2("red", grc_is)
    filter_c = "IS DE"
    x_k, y_k = "DE_C11", "DE_C22"
    scatter2("salmon", grc_is)
    filter_c = "SOIL AS"
    x_k, y_k = "AS_C11", "AS_C22"
    scatter2("yellow", grc_soil)
    filter_c = "SOIL DE"
    x_k, y_k = "DE_C11", "DE_C22"
    scatter2("orange", grc_soil)
    # filter_c = "VEG"
    # x_k, y_k = "AS_H", "AS_Alpha"
    # scatter1("green")
    # filter_c = "SOIL"
    # x_k, y_k = "AS_H", "AS_Alpha"
    # scatter2("yellow", grc_soil)
    # filter_c = "WAT"
    # x_k, y_k = "AS_H", "AS_Alpha"
    # scatter1("blue")
    # x_k, y_k = "DE_H", "DE_Alpha"
    # scatter1("green")
    plt.legend()
    plt.xlabel("C11")
    plt.ylabel("C22")
    plt.xlim([-20, 20])
    plt.ylim([-30, 0])
    plt.show()


if __name__ == "__main__":
    main()
