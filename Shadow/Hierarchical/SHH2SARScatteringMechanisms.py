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
from matplotlib import colors

from SRTCodes.GDALRasterIO import GDALRasterChannel
from SRTCodes.GDALUtils import dictRasterToVRT, getImageDataFromGeoJson
from SRTCodes.Utils import DirFileName
from Shadow.Hierarchical.SHHDraw import SHHGDALDrawImagesColumn, SHHGDALDrawImages

plt.rc('font', family='Times New Roman')
FONT_SIZE = 12
plt.rcParams.update({'font.size': 12})


# COLOR https://blog.csdn.net/major_in_data_/article/details/132380084


class SHH2DrawImage_SSM(SHHGDALDrawImages):

    def __init__(self, win_size=None, is_min_max=True, is_01=True):
        super().__init__(win_size, is_min_max, is_01)

    def addRCC_HAlpha(self):
        self.addRasterCenterCollection(
            "AS_H",
            r"F:\ProjectSet\Shadow\Hierarchical\Images\HA\BJ\AS_H.dat",
            r"F:\ProjectSet\Shadow\Hierarchical\Images\HA\CD\AS_H.dat",
            r"F:\ProjectSet\Shadow\Hierarchical\Images\HA\QD\AS_H.dat",

            channel_list=["AS_H"])
        self.addRasterCenterCollection(
            "AS_Alpha",
            r"F:\ProjectSet\Shadow\Hierarchical\Images\HA\BJ\AS_Alpha.dat",
            r"F:\ProjectSet\Shadow\Hierarchical\Images\HA\CD\AS_Alpha.dat",
            r"F:\ProjectSet\Shadow\Hierarchical\Images\HA\QD\AS_Alpha.dat",
            channel_list=["AS_Alpha"])
        self.addRasterCenterCollection(
            "AS_A",
            r"F:\ProjectSet\Shadow\Hierarchical\Images\HA\BJ\AS_A.dat",
            r"F:\ProjectSet\Shadow\Hierarchical\Images\HA\CD\AS_A.dat",
            r"F:\ProjectSet\Shadow\Hierarchical\Images\HA\QD\AS_A.dat",
            channel_list=["AS_A"])
        self.addRasterCenterCollection(
            "DE_H",
            r"F:\ProjectSet\Shadow\Hierarchical\Images\HA\BJ\DE_H.dat",
            r"F:\ProjectSet\Shadow\Hierarchical\Images\HA\CD\DE_H.dat",
            r"F:\ProjectSet\Shadow\Hierarchical\Images\HA\QD\DE_H.dat",

            channel_list=["DE_H"])
        self.addRasterCenterCollection(
            "DE_Alpha",
            r"F:\ProjectSet\Shadow\Hierarchical\Images\HA\BJ\DE_Alpha.dat",
            r"F:\ProjectSet\Shadow\Hierarchical\Images\HA\CD\DE_Alpha.dat",
            r"F:\ProjectSet\Shadow\Hierarchical\Images\HA\QD\DE_Alpha.dat",

            channel_list=["DE_Alpha"])
        self.addRasterCenterCollection(
            "DE_A",
            r"F:\ProjectSet\Shadow\Hierarchical\Images\HA\BJ\DE_A.dat",
            r"F:\ProjectSet\Shadow\Hierarchical\Images\HA\CD\DE_A.dat",
            r"F:\ProjectSet\Shadow\Hierarchical\Images\HA\QD\DE_A.dat",
            channel_list=["DE_A"])


def main():
    method_name3()

    pass


def method_name5():
    # 2024年6月22日10:35:10
    sgdic = SHH2DrawImage_SSM((800, 800))
    sgdic.addRCC_Im2()
    sgdic.addRCC_HAlpha()
    sgdic.fontdict["size"] = 20
    row_names = ["AS     ", "DE    "]
    column_names = ["Opt", "H", "Alpha", "A"]
    x, y = 120.17044, 36.30231
    sgdic.addAxisDataXY(0, 0, "RGB", x, y, min_list=[100, 100, 100], max_list=[2000, 2000, 2000])
    sgdic.addAxisDataXY(0, 1, "AS_H", x, y, min_list=[0], max_list=[1])
    sgdic.addAxisDataXY(0, 2, "AS_Alpha", x, y, min_list=[0], max_list=[90])
    sgdic.addAxisDataXY(0, 3, "AS_A", x, y, min_list=[0], max_list=[1])
    sgdic.addAxisDataXY(1, 0, "NRG", x, y, min_list=[100, 100, 100], max_list=[2000, 2000, 2000])
    sgdic.addAxisDataXY(1, 1, "DE_H", x, y, min_list=[0], max_list=[1])
    sgdic.addAxisDataXY(1, 2, "DE_Alpha", x, y, min_list=[0], max_list=[90])
    sgdic.addAxisDataXY(1, 3, "DE_A", x, y, min_list=[0], max_list=[1])
    sgdic.draw(n_columns_ex=3.6, n_rows_ex=3.6, row_names=row_names, column_names=column_names)
    plt.show()


def method_name4():
    # H Alpha scatter
    to_dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Analysis\3")
    fns = ["AS_A", "AS_Alpha", "AS_H", "DE_A", "DE_Alpha", "DE_H"]

    def func1():
        dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Images\HA\2\QD")
        qd_fns = {fn: dfn.fn("{}.dat".format(fn)) for fn in fns}
        print(dfn.fn("QD_HA2.vrt"))
        dictRasterToVRT(dfn.fn("QD_HA2.vrt"), raster_dict=qd_fns)

    func1()
    # sys.exit()

    to_dict = getImageDataFromGeoJson(
        r"F:\ProjectSet\Shadow\Hierarchical\Analysis\3\range.geojson",
        r"F:\ProjectSet\Shadow\Hierarchical\Images\HA\QD\QD_HA.vrt",
    )
    df = pd.DataFrame(to_dict)
    print(df)
    df.to_csv(to_dfn.fn("qd.csv"), index=False)

    def read1(raster_fn):
        grc = GDALRasterChannel()
        grc.addGDALDatas(raster_fn=raster_fn)
        return grc

    # df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\2\sh2_spl26_4_spl.csv")
    fig = plt.figure(figsize=(6.5, 6.5))
    # plt.subplots_adjust(top=0.90, bottom=0.1, left=0.1, right=0.9)
    x0, y0 = 0.1, 0.1
    s_len1, s_len2 = 0.65, 0.20
    jiange = 0.03
    ax1 = fig.add_axes([x0, y0, s_len1, s_len1])  # 左下角坐标(0.1, 0.55)，宽度0.35，高度0.35
    ax2 = fig.add_axes([x0, y0 + s_len1 + jiange, s_len1, s_len2])  # 左下角坐标(0.55, 0.55)，宽度0.35，高度0.35
    ax3 = fig.add_axes([x0 + s_len1 + jiange, y0, s_len2, s_len1])  # 左下角坐标(0.55, 0.55)，宽度0.35，高度0.35

    def scatter1(color, x_k, y_k, filter_c):
        _df = pd.DataFrame(df[df["CNAME"] == filter_c].to_dict("records"))
        _df[y_k] = np.rad2deg(_df[y_k].values)
        print(_df[[x_k, y_k]].describe())
        plt.scatter(-_df[x_k], _df[y_k], color=color, label=filter_c)

    def scatter2(x_k, y_k, filter_c, color, marker, _df, n, ax):
        _df = _df[_df["CATEGORY"] == n]
        x = _df[x_k]
        y = _df[y_k]
        ax.scatter(x, y, color=color, label=filter_c, **marker)

    def hist(color, _df, n, ax, name, is_xy):
        _df = _df[_df["CATEGORY"] == n]
        data = _df[name].values
        n, x = np.histogram(data, bins=256)
        n = n / np.sum(n) * 100
        x = x[:-1]
        x = x + (x[1] - x[1]) / 2
        print(max(x), min(x))
        if is_xy:
            x, n = n, x
        ax.plot(x, n, color=color, label=name)

    fns = ["AS_A", "AS_Alpha", "AS_H", "DE_A", "DE_Alpha", "DE_H"]

    def show1(filter_n, _filter_c, color1, color2):
        color1 = np.array(color1) / 255
        color1 = (float(color1[0]), float(color1[1]), float(color1[2]), float(color1[3]))
        color1 = colors.to_hex(color1, keep_alpha=True)

        def _as():
            filter_c = "{} AS".format(_filter_c)
            x_k, y_k = "AS_H", "AS_Alpha",
            scatter2(x_k, y_k, filter_c, color1,
                     {"marker": "o", "edgecolors": color1, "facecolors": 'none', "linewidths": 1},
                     df, filter_n, ax1)
            hist(color1, df, filter_n, ax2, x_k, False)
            hist(color1, df, filter_n, ax3, y_k, True)

        color2 = np.array(color2) / 255
        color2 = (float(color2[0]), float(color2[1]), float(color2[2]), float(color2[3]))
        color2 = colors.to_hex(color2, keep_alpha=True)

        def _de():
            filter_c = "{} DE".format(_filter_c)
            x_k, y_k = "DE_H", "DE_Alpha",
            scatter2(x_k, y_k, filter_c, color2,
                     {"marker": "^", "edgecolors": color2, "facecolors": 'none', "linewidths": 1},
                     df, filter_n, ax1)
            hist(color2, df, filter_n, ax2, x_k, False)
            hist(color2, df, filter_n, ax3, y_k, True)

        _as()
        _de()

    def h_alpha_2():
        """
        alpha 11.92: 0,  3.49,  6.51, 11.92 | 0.    , 0.2928, 0.5461, 1.
        alpha 11.92: 0,  5.00,  7.04, 11.92 | 0.    , 0.4195, 0.5906, 1.
        alpha 11.92: 0,  4.23,  7.06, 11.92 | 0.    , 0.3549, 0.5923, 1.
        h     15.2 : 0, 10.45, 14.25, 15.20 | 0.    , 0.6875, 0.9375, 1.
        :return:
        """
        y11, y12 = 0.2928 * 90, 0.5461 * 90
        x1 = 0.6875
        ax1.axvline(x1, c="black")
        ax1.axhline(y11, 0, x1, c="black")
        ax1.axhline(y12, 0, x1, c="black")
        ax1.text(x1 / 2, y11 / 2, "Z1")
        ax1.text(x1 / 2, (y11 + y12) / 2, "Z2")
        ax1.text(x1 / 2, (y12 + 90) / 2, "Z3")

        y11, y12 = 0.4195 * 90, 0.5906 * 90
        x2 = 0.9375
        ax1.axvline(x2, c="black")
        ax1.axhline(y11, x1, x2, c="black")
        ax1.axhline(y12, x1, x2, c="black")
        ax1.text((x1 + x2) / 2, y11 / 2, "Z4")
        ax1.text((x1 + x2) / 2, (y11 + y12) / 2, "Z5")
        ax1.text((x1 + x2) / 2, (y12 + 90) / 2, "Z6")

        y11, y12 = 0.3549 * 90, 0.5923 * 90
        x3 = 1.0
        ax1.axhline(y11, x2, x3, c="black")
        ax1.axhline(y12, x2, x3, c="black")
        ax1.text((x2 + x3) / 2 - 0.02, y11 / 2, "Z7")
        ax1.text((x2 + x3) / 2 - 0.02, (y11 + y12) / 2, "Z8")
        ax1.text((x2 + x3) / 2 - 0.02, (y12 + 90) / 2, "Z9")

    # show1(1, "IS", "red", "green")
    show1(2, "SOIL", (128, 128, 0, 128), (0, 0, 255, 128))
    # show1(3, "Shadow", (255, 0, 0, 128), (0, 255, 0, 128))
    # show1(4, "IS", (255, 0, 0, 128), (0, 255, 0, 128))

    x_lim = [0, 1]
    y_lim = [0, 90]

    h_alpha_2()

    ax1.set_xlabel("Entropy", fontdict={"size": FONT_SIZE})
    ax1.set_ylabel("Alpha", fontdict={"size": FONT_SIZE})
    ax2.set_xticks([])
    ax2.set_yticks([0, 0.5, 1.0, 1.5, 2])
    ax2.set_ylabel("frequency(%)", fontdict={"size": FONT_SIZE})
    ax3.set_xticks([0, 0.5, 1.0, 1.5, 2])
    ax3.set_yticks([])
    ax3.set_xlabel("frequency(%)", fontdict={"size": FONT_SIZE})
    ax1.set_xlim(x_lim)
    ax2.set_xlim(x_lim)
    ax3.set_xlim([0, 2])
    ax1.set_ylim(y_lim)
    ax2.set_ylim([0, 2])
    ax3.set_ylim(y_lim)
    ax1.set_xticks([0, 0.5, 1])
    ax1.set_yticks([0, 45, 90])
    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    ax3.legend(frameon=False)
    plt.savefig(to_dfn.fn(r"HAlpha_ASDESOIL.jpg"), dpi=300)
    plt.show()


def method_name3():
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
    jiange = 0.03
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
        n, x = np.histogram(data, bins=256,
                            # range=(-60, 60)
                            )
        n = n / np.sum(n) * 100
        x = x[:-1]
        x = x + (x[1] - x[1]) / 2
        print(max(x), min(x))
        if is_xy:
            x, n = n, x
        ax.plot(x, n, color=color, label=name)

    # grc = grc_is
    # filter_c = "IS AS"
    # x_k, y_k = "AS_C11", "AS_C22"
    # scatter2("red", grc, ax1)
    # hist("red", grc[x_k], ax2, x_k, False)
    # hist("red", grc[y_k], ax3, y_k, True)
    # filter_c = "IS DE"
    # x_k, y_k = "DE_C11", "DE_C22"
    # scatter2("green", grc, ax1)
    # hist("green", grc[x_k], ax2, x_k, False)
    # hist("green", grc[y_k], ax3, y_k, True)

    # grc = grc_soil
    # filter_c = "SOIL AS"
    # x_k, y_k = "AS_C11", "AS_C22"
    # scatter2("y", grc, ax1)
    # hist("y", grc[x_k], ax2, x_k, False)
    # hist("y", grc[y_k], ax3, y_k, True)
    # filter_c = "SOIL DE"
    # x_k, y_k = "DE_C11", "DE_C22"
    # scatter2("orange", grc, ax1)
    # hist("orange", grc[x_k], ax2, x_k, False)
    # hist("orange", grc[y_k], ax3, y_k, True)

    # ax1.set_xlabel("C11", fontdict={"size": FONT_SIZE})
    # ax1.set_ylabel("C22", fontdict={"size": FONT_SIZE})
    # ax2.set_xticks([])
    # ax2.set_yticks([0, 2.5, 5, 7.5,10])
    # ax2.set_ylabel("frequency(%)", fontdict={"size": FONT_SIZE})
    # ax3.set_xticks([0, 2.5, 5, 7.5,10])
    # ax3.set_yticks([])
    # ax3.set_xlabel("frequency(%)", fontdict={"size": FONT_SIZE})

    # ax1.set_xlim([-20, 20])
    # ax2.set_xlim([-20, 20])
    # ax3.set_xlim([0, 10])

    # ax1.set_ylim([-32, 0])
    # ax2.set_ylim([0, 10])
    # ax3.set_ylim([-32, 0])

    # ax1.set_xticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # ax1.set_yticks([-30, -25, -20, -15, -10, -5, 0])

    def grids():
        """
        alpha 11.92: 0,  3.49,  6.51, 11.92 | 0.    , 0.2928, 0.5461, 1.
        alpha 11.92: 0,  5.00,  7.04, 11.92 | 0.    , 0.4195, 0.5906, 1.
        alpha 11.92: 0,  4.23,  7.06, 11.92 | 0.    , 0.3549, 0.5923, 1.
        h     15.2 : 0, 10.45, 14.25, 15.20 | 0.    , 0.6875, 0.9375, 1.
        :return:
        """

        ax1.axvline(0, linestyle="-", c="black")
        ax1.axhline(0, linestyle="-", c="black")
        ax1.axvline(-10, linestyle="--", c="gray")
        ax1.axvline(10, linestyle="--", c="gray")
        ax1.axvline(20, linestyle="--", c="gray")
        ax1.axhline(-10, linestyle="--", c="gray")
        ax1.axhline(10, linestyle="--", c="gray")

    grc = grc_soil
    data = {"C11": grc["AS_C11"] - grc["DE_C11"], "C22": grc["AS_C22"] - grc["DE_C22"]}
    filter_c = "SOIL"
    x_k, y_k = "C11", "C22"
    scatter2("orange", data, ax1)
    hist("orange", data[x_k], ax2, x_k, False)
    hist("orange", data[y_k], ax3, y_k, True)

    grc = grc_is
    data = {"C11": grc["AS_C11"] - grc["DE_C11"], "C22": grc["AS_C22"] - grc["DE_C22"]}
    filter_c = "IS"
    x_k, y_k = "C11", "C22"
    scatter2("red", data, ax1)
    hist("red", data[x_k], ax2, x_k, False)
    hist("red", data[y_k], ax3, y_k, True)

    grids()

    ax1.set_xlabel("C11", fontdict={"size": FONT_SIZE})
    ax1.set_ylabel("C22", fontdict={"size": FONT_SIZE})
    ax2.set_xticks([])
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_ylabel("frequency(%)", fontdict={"size": FONT_SIZE})
    ax3.set_xticks([0, 1, 2, 3])
    ax3.set_yticks([])
    ax3.set_xlabel("frequency(%)", fontdict={"size": FONT_SIZE})

    ax1.set_xlim([-12, 30])
    ax2.set_xlim([-12, 30])
    ax3.set_xlim([0, 3])

    ax1.set_ylim([-15, 15])
    ax2.set_ylim([0, 3])
    ax3.set_ylim([-15, 15])

    ax1.set_xticks([-10, -5, 0, 5, 10, 15, 20, 25, 30])
    ax1.set_yticks([-15, -10, -5, 0, 5, 10, 15])

    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    ax3.legend(frameon=False)
    plt.savefig(r"F:\ProjectSet\Shadow\Hierarchical\Analysis\2\sh_c2.jpg", dpi=300)
    plt.show()


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
    add1("(3)    ", 120.373510, 36.090501)
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
