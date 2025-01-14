# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowWrite.py
@Time    : 2024/7/27 16:43
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of ShadowWrite
-----------------------------------------------------------------------------"""
import pandas as pd
from matplotlib import pyplot as plt

from SRTCodes.GDALDraw import GDALDrawImage
from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.NumpyUtils import update10Log10
from SRTCodes.SRTDraw import MplPatchesEllipse
from SRTCodes.Utils import DirFileName


def main():
    dfn = DirFileName(r"F:\ProjectSet\Shadow\Release")
    sar_mean_dfn = DirFileName(r"F:\ASDEWrite\Images\SARMean")
    images_dfn = DirFileName(r"F:\ASDEWrite\Images")
    result_dfn = DirFileName(r"F:\ASDEWrite\Result")

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    plt.rc('text', usetex=True)

    def func1():
        fn = dfn.fn("ChengDuImages", "SH_CD_envi.dat")
        gr = GDALRaster(fn)

        def read_data(_name):
            _data = gr.readGDALBand(_name)
            print("{:>10} {:>10.3f} {:>10.3f} {:>10.3f}".format(_name, _data.min(), _data.max(), _data.mean()))
            return _data

        def mean(_data1, _data2):
            # _data = (_data1 + _data2) / 2
            # _data = update10Log10(_data)
            _data1 = update10Log10(_data1)
            _data2 = update10Log10(_data2)
            _data = (_data1 + _data2) / 2
            print("{:>10} {:>10.3f} {:>10.3f} {:>10.3f}".format(" ", _data.min(), _data.max(), _data.mean()))
            return _data

        as_vv = read_data("AS_VV")
        as_vh = read_data("AS_VH")
        de_vv = read_data("DE_VV")
        de_vh = read_data("DE_VH")
        as_mean = mean(as_vv, as_vh)
        de_mean = mean(de_vv, de_vh)

        gr.save(as_mean, sar_mean_dfn.fn("cd_as_mean2.tif"), fmt="GTiff")
        gr.save(de_mean, sar_mean_dfn.fn("cd_de_mean2.tif"), fmt="GTiff")

    def func2():
        FONT_SIZE = 16

        win_size = (31, 31)
        gi_win_size = int(win_size[0] * 2640.0 / 121.0), int(win_size[1] * 2026.0 / 121.0)
        print("win_size", win_size)
        print("gi_win_size", gi_win_size)
        gdi = GDALDrawImage(win_size)
        qd = gdi.addGR(dfn.fn("BeiJingImages", "SH_BJ_envi.dat"), dfn.fn("BeiJingImages", "SH_BJ_look_envi.range"))
        bj = gdi.addGR(dfn.fn("QingDaoImages", "SH_QD_envi.dat"), dfn.fn("QingDaoImages", "SH_QD_look_envi.range"))
        cd = gdi.addGR(dfn.fn("ChengDuImages", "SH_CD_envi.dat"), dfn.fn("ChengDuImages", "SH_CD_look_envi.range"))
        gdi.addRCC("RGB", qd, bj, cd, channel_list=["Red", "Green", "Blue"])
        gdi.addRCC("NRG", qd, bj, cd, channel_list=["NIR", "Red", "Green"])
        gdi.addRCC(
            "AS_SAR",
            sar_mean_dfn.fn("bj_as_mean2.tif"),
            sar_mean_dfn.fn("qd_as_mean2.tif"),
            sar_mean_dfn.fn("cd_as_mean2.tif"),
            channel_list=[0])
        gdi.addRCC(
            "DE_SAR",
            sar_mean_dfn.fn("bj_de_mean2.tif"),
            sar_mean_dfn.fn("qd_de_mean2.tif"),
            sar_mean_dfn.fn("cd_de_mean2.tif"),
            channel_list=[0])
        gdi.addRCC(
            "GI",
            r"F:\ProjectSet\Shadow\MkTu\4.1Details\BingImages\bj_googleimages.mbtiles",
            r"F:\ProjectSet\Shadow\MkTu\4.1Details\BingImages\cd_googleimages.mbtiles",
            r"F:\ProjectSet\Shadow\MkTu\4.1Details\BingImages\qd_googleimages.mbtiles",
            channel_list=[0, 1, 2], win_size=gi_win_size
        )

        fig = plt.figure(figsize=(10, 8), )
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, hspace=0.03, wspace=0.03)
        n_rows, n_columns = 4, 5
        column_names = ["Google Earth", "S2", "S1 AS", "S1 DE"]

        from matplotlib import colors
        color_dict = {
            1: colors.to_hex((1, 0, 0)),
            2: colors.to_hex((0, 1, 0)),
            3: colors.to_hex((1, 1, 0)),
            4: colors.to_hex((0, 0, 1)),
        }
        cname_dict = {1: "IS", 2: "VEG", 3: "SO", 4: "WAT"}
        sh_type_dict = {
            1: "Optical Shadow",
            2: "AS SAR Shadow",
            3: "DE SAR Shadow",
        }

        class draw_column:

            def __init__(self):
                self.row = 1
                self.column = 1
                self.name = ""
                self.opt_list = None
                self.as_list = None
                self.de_list = None

            def fit(self, name, x, y, opt_list=None, as_list=None, de_list=None):
                self.column = 1
                self.name = name
                self.opt_list = opt_list
                self.as_list = as_list
                self.de_list = de_list

                plt.subplot(n_rows, n_columns, (self.row - 1) * n_columns + 1)
                gdi.readDraw("GI", x, y, is_trans=True, min_list=[0, 0, 0], max_list=[255, 255, 255])
                self.column_draw()

                plt.subplot(n_rows, n_columns, (self.row - 1) * n_columns + 2)
                gdi.readDraw("NRG", x, y, min_list=[300, 300, 300], max_list=[2900, 1500, 1500])
                self.column_draw()

                plt.subplot(n_rows, n_columns, (self.row - 1) * n_columns + 3)
                gdi.readDraw("AS_SAR", x, y, min_list=[-16], max_list=[3.5])
                self.column_draw()

                plt.subplot(n_rows, n_columns, (self.row - 1) * n_columns + 4)
                gdi.readDraw("DE_SAR", x, y, min_list=[-16], max_list=[3.5])
                self.column_draw()

                self.row += 1

            def column_draw(self):
                ax = plt.gca()
                if self.row == 1:
                    ax.set_title(column_names[self.column - 1], fontdict={"size": FONT_SIZE})
                if self.column == 1:
                    ax.set_ylabel(self.name, rotation=0, labelpad=10, fontdict={"size": FONT_SIZE})

                def scatter_xy_list(_list, sh_type, *args, **kwargs):
                    _x_list, _y_list = [], []
                    _color_list = []
                    for i, (x, y, category) in enumerate(_list):
                        if self.column == 1:
                            x, y = x * 2026.0 / 121.0, y * 2640.0 / 121.0
                        _x_list.append(x)
                        _y_list.append(y)
                        _color_list.append(category)
                        cname = "{} {}".format(cname_dict[category], sh_type_dict[sh_type])
                        plt.scatter([x], [y], color=color_dict[category], label=cname, *args, **kwargs)
                    return _x_list, _y_list

                s = 86
                edgecolors = "black"

                if self.opt_list is not None:
                    x_list, y_list = scatter_xy_list(self.opt_list, 1, marker=",", s=s, edgecolors=edgecolors)

                if self.as_list is not None:
                    x_list, y_list = scatter_xy_list(self.as_list, 2, marker="^", s=s, edgecolors=edgecolors)

                if self.de_list is not None:
                    x_list, y_list = scatter_xy_list(self.de_list, 3, marker="o", s=s, edgecolors=edgecolors)

                if self.column == 4:
                    plt.legend(
                        loc='upper left', bbox_to_anchor=(1.0, 0.75),
                        prop={"size": FONT_SIZE - 4}, frameon=False, ncol=1,
                        handletextpad=0, borderaxespad=0,
                    )

                self.column += 1

        column = draw_column()
        column.fit(r"(a)", 104.0629481, 30.6528887,
                   opt_list=[(12, 13, 1)], as_list=[(20.5, 11, 1)], de_list=[(11, 12, 1)])
        column.fit(r"(b)", 116.4121497, 39.8492792,
                   opt_list=[(15, 12, 2)], as_list=[(17, 11, 2)], de_list=[(15, 19, 2)])
        column.fit(r"(c)", 116.3555112, 39.7940205,
                   opt_list=[(12, 15, 3)], as_list=[(19, 15, 3)], de_list=[(7, 16, 3)])
        column.fit(r"(d)", 104.061904, 30.649237,
                   opt_list=[(13, 14, 4)], as_list=[(18, 14.5, 4)], de_list=[(9, 13.5, 4)])
        # column.fit("(5)    ", 116.4340498, 39.9156977,
        #            opt_list=[(9.5, 24, 1)], as_list=[(18, 12, 1)], de_list=[(10, 14, 1)])

        plt.savefig(images_dfn.fn("fig4112.jpg"), dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.show()

    def func3():
        FONT_SIZE = 16

        win_size = (41, 41)
        gi_win_size = int(win_size[0] * 2640.0 / 121.0), int(win_size[1] * 2026.0 / 121.0)
        print("win_size", win_size)
        print("gi_win_size", gi_win_size)
        gdi = GDALDrawImage(win_size)
        qd = gdi.addGR(dfn.fn("BeiJingImages", "SH_BJ_envi.dat"), dfn.fn("BeiJingImages", "SH_BJ_look_envi.range"))
        bj = gdi.addGR(dfn.fn("QingDaoImages", "SH_QD_envi.dat"), dfn.fn("QingDaoImages", "SH_QD_look_envi.range"))
        cd = gdi.addGR(dfn.fn("ChengDuImages", "SH_CD_envi.dat"), dfn.fn("ChengDuImages", "SH_CD_look_envi.range"))
        gdi.addRCC("RGB", qd, bj, cd, channel_list=["Red", "Green", "Blue"])
        gdi.addRCC("NRG", qd, bj, cd, channel_list=["NIR", "Red", "Green"])
        gdi.addRCC(
            "AS_SAR",
            sar_mean_dfn.fn("bj_as_mean2.tif"),
            sar_mean_dfn.fn("qd_as_mean2.tif"),
            sar_mean_dfn.fn("cd_as_mean2.tif"),
            channel_list=[0])
        gdi.addRCC(
            "DE_SAR",
            sar_mean_dfn.fn("bj_de_mean2.tif"),
            sar_mean_dfn.fn("qd_de_mean2.tif"),
            sar_mean_dfn.fn("cd_de_mean2.tif"),
            channel_list=[0])
        gdi.addRCC(
            "GI",
            r"F:\ProjectSet\Shadow\MkTu\4.1Details\BingImages\bj_googleimages.mbtiles",
            r"F:\ProjectSet\Shadow\MkTu\4.1Details\BingImages\cd_googleimages.mbtiles",
            r"F:\ProjectSet\Shadow\MkTu\4.1Details\BingImages\qd_googleimages.mbtiles",
            channel_list=[0, 1, 2], win_size=gi_win_size
        )

        imdc_color_dict = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255)}

        imdcs = [
            ("SAR-Opt-AS-DE_imdc.tif", "HS-OAD"),
            ("Opt-Opt-AS-DE_imdc.tif", "OS-OAD"),
            ("FREE-Opt-AS-DE_imdc.tif", "NS-OAD"),
        ]
        imdc_column_names = []

        for fn, name in imdcs:
            gdi.addRCC(
                name,
                result_dfn.fn("QingDao", "qd_{}".format(fn)),
                result_dfn.fn("Beijing", "bj_{}".format(fn)),
                result_dfn.fn("Chengdu", "cd_{}".format(fn)),
                channel_list=[0], is_01=False, is_min_max=False,
            )
            imdc_column_names.append(name)

        fig = plt.figure(figsize=(14 * 0.75, 12 * 0.75), )
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, hspace=0.03, wspace=0.03)
        n_rows, n_columns = 6, 7
        column_names = ["Google Earth", "S2", "S1 AS", "S1 DE"] + imdc_column_names

        from matplotlib import colors

        color_dict = {
            1: colors.to_hex((1, 0, 0)),
            2: colors.to_hex((0, 1, 0)),
            3: colors.to_hex((1, 1, 0)),
            4: colors.to_hex((0, 0, 1)),
        }

        cname_dict = {1: "IS", 2: "VEG", 3: "SO", 4: "WAT"}

        sh_type_dict = {
            1: "Optical Shadow",
            2: "AS SAR Shadow",
            3: "DE SAR Shadow",
        }

        class draw_column:

            def __init__(self):
                self.row = 1
                self.column = 1
                self.name = ""

                self.ells = []

            def fit(self, name, x, y, ells=None):
                if ells is None:
                    ells = []
                self.column = 1
                self.name = name
                self.ells = ells

                plt.subplot(n_rows, n_columns, (self.row - 1) * n_columns + 1)
                gdi.readDraw("GI", x, y, is_trans=True, min_list=[0, 0, 0], max_list=[255, 255, 255])
                self.column_draw()

                plt.subplot(n_rows, n_columns, (self.row - 1) * n_columns + 2)
                gdi.readDraw("NRG", x, y, min_list=[300, 300, 300], max_list=[2900, 1500, 1500])
                self.column_draw()

                plt.subplot(n_rows, n_columns, (self.row - 1) * n_columns + 3)
                gdi.readDraw("AS_SAR", x, y, min_list=[-16], max_list=[3.5])
                self.column_draw()

                plt.subplot(n_rows, n_columns, (self.row - 1) * n_columns + 4)
                gdi.readDraw("DE_SAR", x, y, min_list=[-16], max_list=[3.5])
                self.column_draw()

                plt.subplot(n_rows, n_columns, (self.row - 1) * n_columns + 5)
                gdi.readDraw("HS-OAD", x, y, color_dict=imdc_color_dict)
                self.column_draw()

                plt.subplot(n_rows, n_columns, (self.row - 1) * n_columns + 6)
                gdi.readDraw("OS-OAD", x, y, color_dict=imdc_color_dict)
                self.column_draw()

                plt.subplot(n_rows, n_columns, (self.row - 1) * n_columns + 7)
                gdi.readDraw("NS-OAD", x, y, color_dict=imdc_color_dict)
                self.column_draw()

                self.row += 1

            def column_draw(self):
                ax = plt.gca()

                if self.row == 1:
                    ax.set_title(column_names[self.column - 1], fontdict={"size": FONT_SIZE})
                if self.column == 1:
                    ax.set_ylabel(self.name, rotation=0, labelpad=10, fontdict={"size": FONT_SIZE})

                if self.ells is not None:
                    ell: MplPatchesEllipse
                    for ell in self.ells:
                        if self.column == 1:
                            xy = ell.xy.copy()
                            width, height = ell.width, ell.height
                            ell.xy[0] *= 2026.0 / 121.0
                            ell.xy[1] *= 2640.0 / 121.0
                            ell.width*= 2026.0 / 121.0
                            ell.height*= 2640.0 / 121.0
                            ax.add_patch(ell.fit())
                            ell.xy = xy
                            ell.width, ell.height = width, height
                        else:
                            ax.add_patch(ell.fit())

                if (self.row == n_rows) and (self.column == n_columns):
                    plt.scatter([1], [1], marker=",", color=color_dict[1], edgecolors="black", s=100, label="IS")
                    plt.scatter([1], [1], marker=",", color=color_dict[2], edgecolors="black", s=100, label="VEG")
                    plt.scatter([1], [1], marker=",", color=color_dict[3], edgecolors="black", s=100, label="SO")
                    plt.scatter([1], [1], marker=",", color=color_dict[4], edgecolors="black", s=100, label="WAT")
                    plt.legend(
                        loc='lower left', bbox_to_anchor=(-4.6, -0.4),
                        prop={"size": FONT_SIZE}, frameon=False, ncol=4,
                        handletextpad=0, borderaxespad=0,
                    )

                self.column += 1

        def _ell(xy, width=18, height=16, angle=0, linewidth=2, fill=False, zorder=2, edgecolor="black"):
            ell = MplPatchesEllipse(xy=xy, width=width, height=height, angle=angle, linewidth=linewidth,
                                    fill=fill, zorder=zorder, edgecolor=edgecolor)
            return ell

        column = draw_column()
        column.fit(r"(a)", 120.429395, 36.114999, ells=[_ell((16, 18), 20, 20)])
        column.fit(r"(d)", 120.414563, 36.160460, ells=[_ell((17, 17))])
        # column.fit(r"(b)", 120.393296,36.134271 )
        # column.fit(r"(b)", 120.439103,36.114294)
        # column.fit(r"(c)", 116.483022,39.886159 )
        column.fit(r"(c)", 116.393431, 39.858198, ells=[_ell((23, 23), 16, 32)])
        # column.fit(r"(c)", 116.363135,39.948879)
        column.fit(r"(d)", 116.491673, 39.890778, ells=[_ell((22, 12), 28, 16)])
        column.fit(r"(e)", 104.076017, 30.645997, ells=[_ell((13, 10), 16, 16), _ell((8, 30), 10, 16, 20)])
        column.fit(r"(f)", 104.004844, 30.703297, ells=[_ell((19, 15), 10, 36, 50)])

        fn = images_dfn.fn("fig4131.jpg")
        plt.savefig(fn, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print(fn)
        plt.show()

    def func4():
        FONT_SIZE = 16

        win_size = (61, 61)
        gi_win_size = int(win_size[0] * 2640.0 / 121.0), int(win_size[1] * 2026.0 / 121.0)
        print("win_size", win_size)
        print("gi_win_size", gi_win_size)
        gdi = GDALDrawImage(win_size)
        qd = gdi.addGR(dfn.fn("BeiJingImages", "SH_BJ_envi.dat"), dfn.fn("BeiJingImages", "SH_BJ_look_envi.range"))
        bj = gdi.addGR(dfn.fn("QingDaoImages", "SH_QD_envi.dat"), dfn.fn("QingDaoImages", "SH_QD_look_envi.range"))
        cd = gdi.addGR(dfn.fn("ChengDuImages", "SH_CD_envi.dat"), dfn.fn("ChengDuImages", "SH_CD_look_envi.range"))
        gdi.addRCC("RGB", qd, bj, cd, channel_list=["Red", "Green", "Blue"])
        gdi.addRCC("NRG", qd, bj, cd, channel_list=["NIR", "Red", "Green"])
        gdi.addRCC(
            "AS_SAR",
            sar_mean_dfn.fn("bj_as_mean2.tif"),
            sar_mean_dfn.fn("qd_as_mean2.tif"),
            sar_mean_dfn.fn("cd_as_mean2.tif"),
            channel_list=[0])
        gdi.addRCC(
            "DE_SAR",
            sar_mean_dfn.fn("bj_de_mean2.tif"),
            sar_mean_dfn.fn("qd_de_mean2.tif"),
            sar_mean_dfn.fn("cd_de_mean2.tif"),
            channel_list=[0])
        gdi.addRCC(
            "GI",
            r"F:\ProjectSet\Shadow\MkTu\4.1Details\BingImages\bj_googleimages.mbtiles",
            r"F:\ProjectSet\Shadow\MkTu\4.1Details\BingImages\cd_googleimages.mbtiles",
            r"F:\ProjectSet\Shadow\MkTu\4.1Details\BingImages\qd_googleimages.mbtiles",
            channel_list=[0, 1, 2], win_size=gi_win_size
        )

        imdc_color_dict = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255)}

        imdcs = [
            ("FREE-Opt-AS-DE_imdc.tif", "NS-OAD"),
            ("FREE-Opt-AS_imdc.tif", "NS-OA"),
            ("FREE-Opt-DE_imdc.tif", "NS-OD"),
            ("FREE-Opt_imdc.tif", "NS-O"),
            ("Opt-Opt-AS-DE_imdc.tif", "OS-OAD"),
            ("SAR-Opt-AS-DE_imdc.tif", "HS-OAD"),
        ]

        imdc_column_names = []

        for fn, name in imdcs:
            gdi.addRCC(
                name,
                result_dfn.fn("QingDao", "qd_{}".format(fn)),
                result_dfn.fn("Beijing", "bj_{}".format(fn)),
                result_dfn.fn("Chengdu", "cd_{}".format(fn)),
                channel_list=[0], is_01=False, is_min_max=False,
            )
            imdc_column_names.append(name)

        fig = plt.figure(figsize=(9, 9), )
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, hspace=0.03, wspace=0.03)
        n_rows, n_columns = 6, 6
        column_names = ["Google Earth", "S2", ] + imdc_column_names

        from matplotlib import colors

        color_dict = {
            1: colors.to_hex((1, 0, 0)),
            2: colors.to_hex((0, 1, 0)),
            3: colors.to_hex((1, 1, 0)),
            4: colors.to_hex((0, 0, 1)),
        }

        cname_dict = {1: "IS", 2: "VEG", 3: "SO", 4: "WAT"}

        sh_type_dict = {
            1: "Optical Shadow",
            2: "AS SAR Shadow",
            3: "DE SAR Shadow",
        }

        class draw_column:

            def __init__(self):
                self.row = 1
                self.column = 1
                self.name = ""

            def fit(self, name, x, y):
                self.column = 1
                self.name = name

                plt.subplot(n_rows, n_columns, (self.row - 1) * n_columns + 1)
                gdi.readDraw("GI", x, y, is_trans=True, min_list=[0, 0, 0], max_list=[255, 255, 255])
                self.column_draw()

                plt.subplot(n_rows, n_columns, (self.row - 1) * n_columns + 2)
                gdi.readDraw("NRG", x, y, min_list=[300, 300, 300], max_list=[2900, 1500, 1500])
                self.column_draw()

                plt.subplot(n_rows, n_columns, (self.row - 1) * n_columns + 3)
                gdi.readDraw("NS-OAD", x, y, color_dict=imdc_color_dict)
                self.column_draw()

                plt.subplot(n_rows, n_columns, (self.row - 1) * n_columns + 4)
                gdi.readDraw("NS-OA", x, y, color_dict=imdc_color_dict)
                self.column_draw()

                plt.subplot(n_rows, n_columns, (self.row - 1) * n_columns + 5)
                gdi.readDraw("NS-OD", x, y, color_dict=imdc_color_dict)
                self.column_draw()

                plt.subplot(n_rows, n_columns, (self.row - 1) * n_columns + 6)
                gdi.readDraw("NS-O", x, y, color_dict=imdc_color_dict)
                self.column_draw()

                self.row += 1

            def column_draw(self):
                ax = plt.gca()

                if self.row == 1:
                    ax.set_title(column_names[self.column - 1], fontdict={"size": FONT_SIZE})
                if self.column == 1:
                    ax.set_ylabel(self.name, rotation=0, labelpad=10, fontdict={"size": FONT_SIZE})

                if (self.row == n_rows) and (self.column == n_columns):
                    plt.scatter([1], [1], marker=",", color=color_dict[1], edgecolors="black", s=100, label="IS")
                    plt.scatter([1], [1], marker=",", color=color_dict[2], edgecolors="black", s=100, label="VEG")
                    plt.scatter([1], [1], marker=",", color=color_dict[3], edgecolors="black", s=100, label="SO")
                    plt.scatter([1], [1], marker=",", color=color_dict[4], edgecolors="black", s=100, label="WAT")
                    plt.legend(
                        loc='lower left', bbox_to_anchor=(-4.0, -0.4),
                        prop={"size": FONT_SIZE}, frameon=False, ncol=4,
                        handletextpad=0, borderaxespad=0,
                    )

                self.column += 1

        column = draw_column()
        column.fit(r"(a)", 120.346020, 36.120435)
        column.fit(r"(b)", 120.387358, 36.125549)
        column.fit(r"(c)", 116.441548, 39.906636)
        column.fit(r"(d)", 116.516795, 39.920875)
        column.fit(r"(e)", 104.066101, 30.656686)
        column.fit(r"(f)", 104.110842, 30.647660)

        fn = images_dfn.fn("fig4132.jpg")
        plt.savefig(fn, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print(fn)
        plt.show()

    return func2()


def method_name1():
    csv_fn = r"F:\ASDEWrite\Result\QingDao\qd_data_spl.csv"
    # csv_fn = r"F:\ASDEWrite\Result\BeiJing\HSPL_BJ_select.csv"
    # csv_fn = r"F:\ASDEWrite\Result\ChengDu\HSPL_CD_select.csv"
    csv_fns = [
        r"F:\ASDEWrite\Result\QingDao\qd_data_spl.csv",
        r"F:\ASDEWrite\Result\BeiJing\HSPL_BJ_select.csv",
        r"F:\ASDEWrite\Result\ChengDu\HSPL_CD_select.csv",
    ]

    def func1():
        df = pd.read_csv(csv_fn)
        df = df[df["TEST"] == 1]
        df = df[df["OS"] == 1]
        counts_dict = df["CNAME"].value_counts().to_dict()
        names = [
            'IS', 'IS_SH', 'IS_AS_SH', 'IS_DE_SH',
            'VEG', 'VEG_SH', 'VEG_AS_SH', 'VEG_DE_SH',
            'SOIL', 'SOIL_SH', 'SOIL_AS_SH', 'SOIL_DE_SH',
            'WAT', 'WAT_SH', 'WAT_AS_SH', 'WAT_DE_SH'
        ]

        def print_line(*_names):
            print(*[counts_dict[name] for name in _names])

        print_line('IS', 'IS_SH', 'IS_AS_SH', 'IS_DE_SH', )
        print_line('VEG', 'VEG_SH', 'VEG_AS_SH', 'VEG_DE_SH')
        print_line('SOIL', 'SOIL_SH', 'SOIL_AS_SH', 'SOIL_DE_SH', )
        print_line('WAT', 'WAT_SH', 'WAT_AS_SH', 'WAT_DE_SH')

    def func2():
        for _csv_fn in csv_fns:
            df = pd.read_csv(_csv_fn)
            df = df[df["TEST"] == 1]
            for _spl_type in ["HS", "OS", "NS"]:
                _df = df[df[_spl_type] == 1]
                counts_dict = _df["CNAME"].value_counts().to_dict()

                def _sum(*_names):
                    return sum(counts_dict[name] for name in _names if name in counts_dict)

                print(
                    _spl_type,
                    _sum("IS", "VEG", "SOIL", "WAT"),
                    _sum("IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH"),
                    _sum("IS_AS_SH", "VEG_AS_SH", "SOIL_AS_SH", "WAT_AS_SH"),
                    _sum("IS_DE_SH", "VEG_DE_SH", "SOIL_DE_SH", "WAT_DE_SH"),
                )

    def func3():
        for _csv_fn in csv_fns:
            df = pd.read_csv(_csv_fn)
            df = df[df["TEST"] == 0]
            for _spl_type in ["TEST_IS", "TEST_SH"]:
                _df = df[df[_spl_type] == 1]
                counts_dict = _df["CNAME"].value_counts().to_dict()

                def _sum(*_names):
                    return sum(counts_dict[name] for name in _names if name in counts_dict)

                print(
                    _spl_type,
                    _sum("IS", "IS_SH"),
                    _sum("VEG", "VEG_SH"),
                    _sum("SOIL", "SOIL_SH"),
                    _sum("WAT", "WAT_SH"),
                )

    func3()


if __name__ == "__main__":
    main()
