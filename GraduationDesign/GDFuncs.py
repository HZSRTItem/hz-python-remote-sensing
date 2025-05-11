# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : GDFuncs.py
@Time    : 2024/12/12 21:58
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of ZH
-----------------------------------------------------------------------------"""
import os
import random

import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from osgeo import gdal
from tabulate import tabulate

from SRTCodes.GDALRasterIO import GDALRaster, tiffAddColorTable
from SRTCodes.GDALUtils import GDALSamplingFast
from SRTCodes.GeoMap import GMRaster
from SRTCodes.NumpyUtils import scaleMinMax, update10EDivide10
from SRTCodes.SampleUtils import SamplesUtil
from SRTCodes.Utils import readJson, printList, DirFileName, TableLinePrint, ChangeInitDirname, getfilenme, changext, \
    getfilenamewithoutext

_cid = ChangeInitDirname().initTrack(None)
_cid_G = ChangeInitDirname().initTrack(None)


def _samplesDescription(df, field_name="TEST", is_print=True):
    df_des = pd.pivot_table(df, index="CNAME", columns=field_name, aggfunc='size', fill_value=0)
    df_des[pd.isna(df_des)] = 0
    df_des["SUM"] = df_des.apply(lambda x: x.sum(), axis=1)
    df_des.loc["SUM"] = df_des.apply(lambda x: x.sum())
    if is_print:
        print(tabulate(df_des, headers="keys"))
    return df_des


class _FUNCS3_SamplesUtil(SamplesUtil):

    def __init__(self):
        super(_FUNCS3_SamplesUtil, self).__init__()
        self.images_fns = readJson(r"G:\SHImages\image.json")

    def samplingName(self, name, _func=None, _filters_and=None, _filters_or=None,
                     x_field_name="X", y_field_name="Y", is_jdt=True):
        return self.sampling1(name, self.images_fns[name], _func=_func,
                              _filters_and=_filters_and, _filters_or=_filters_or,
                              x_field_name=x_field_name, y_field_name=y_field_name, is_jdt=is_jdt)

    def samplingNames(self, *names, _func=None, _filters_and=None, _filters_or=None,
                      x_field_name="X", y_field_name="Y", is_jdt=True):
        for name in names:
            self.samplingName(name, _func=_func,
                              _filters_and=_filters_and, _filters_or=_filters_or,
                              x_field_name=x_field_name, y_field_name=y_field_name, is_jdt=is_jdt)


def main():
    def func1():

        def _sample_update(city_name, csv_fn, n_dict=None):
            def _show_spl1(_samples_cname_dict, _str=None):
                if _str is not None:
                    print(_str)
                _sum = 0
                for _name in _samples_cname_dict:
                    print("{:>10}: {}".format(_name, len(_samples_cname_dict[_name])))
                    _sum += len(_samples_cname_dict[_name])
                print("{:>10}: {}".format("SUM", _sum))
                # print({_name:len(_samples_cname_dict[_name]) for _name in _samples_cname_dict})

            df = pd.read_csv(csv_fn)
            _samplesDescription(df)
            samples = df.to_dict("records")

            def _split_spls(_test):
                _samples_cname_dict = {}
                for spl in samples:
                    if spl["TEST"] == _test:
                        if spl["CNAME"] not in _samples_cname_dict:
                            _samples_cname_dict[spl["CNAME"]] = []
                        _samples_cname_dict[spl["CNAME"]].append(spl)
                return _samples_cname_dict

            samples_cname_dict_0 = _split_spls(0)
            _show_spl1(samples_cname_dict_0, "TEST 0:")
            samples_cname_dict_1 = _split_spls(1)
            _show_spl1(samples_cname_dict_1, "TEST 1:")

            if n_dict is None:
                return

            def _samples_select(_samples_cname_dict, _n_dict=None):
                if _n_dict is None:
                    _n_dict = {}
                to_spls = {}
                for _name in _n_dict:
                    if _n_dict[_name] == -1:
                        to_spls[_name] = _samples_cname_dict[_name]
                    else:
                        to_spls[_name] = random.sample(_samples_cname_dict[_name], _n_dict[_name])
                return to_spls

            samples_cname_dict_1_select = _samples_select(samples_cname_dict_1, n_dict)

            _show_spl1(samples_cname_dict_1_select, "TEST 1 2:")

            def _cat_spl(*_spls):
                to_list = []
                for _spl in _spls:
                    for _name in _spl:
                        to_list.extend(_spl[_name])
                return to_list

            to_df = pd.DataFrame(_cat_spl(samples_cname_dict_0, samples_cname_dict_1_select))
            print(to_df)
            _samplesDescription(to_df)

            to_df.to_csv(r"F:\GraduationDesign\Result\run\Samples\1\{}_spl1.csv".format(city_name))

        # _sample_update(
        #     city_name="qd",
        #     csv_fn=r"F:\GraduationDesign\Result\run\Samples\sh2_spl30_qd6_spl.csv",
        #     n_dict={
        #         'IS': 835, 'SOIL': 436, 'IS_SH': 0, 'VEG_SH': 0,
        #         'SOIL_SH': 0, 'WAT': 378, 'WAT_SH': 0, 'VEG': 839
        #     })

        # _sample_update(
        #     city_name="bj",
        #     csv_fn=r"F:\GraduationDesign\Result\run\Samples\sh2_spl30_bj1_spl.csv",
        #     n_dict={
        #         'IS': -1, 'SOIL': -1, 'IS_SH': 0, 'VEG_SH': 0,
        #         'SOIL_SH': 0, 'WAT': -1, 'WAT_SH': 0, 'VEG': -1
        #     }
        # )

        _sample_update(
            city_name="cd",
            csv_fn=r"F:\GraduationDesign\Result\run\Samples\sh2_spl30_cd6_spl.csv",
            n_dict={
                'IS': -1, 'SOIL': -1, 'IS_SH': 0, 'VEG_SH': 0,
                'SOIL_SH': 0, 'WAT': -1, 'WAT_SH': 0, 'VEG': -1
            }
        )
        return

    def func2():
        title_dict = {}
        is_find = {}

        def _get_title(_line: str):
            _lines = _line.split(".")
            _title = _lines[1].split("[")[0].strip()
            if ";" in _title:
                return _title.split(";")[0].strip()
            else:
                return _title

        with open(r"F:\GraduationDesign\参考文献.txt", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                # if not line.strip().endswith("."):
                #     print(line)
                # if "]." not in line:
                #     print(line)
                title = _get_title(line)
                title = title.lower()
                print("[{}]\t{}".format(i + 1, title))
                title_dict[title] = line.split("\t", 2)[1].strip()
                is_find[title] = False

        print("-" * 60)
        fn = r"F:\GraduationDesign\参考文献\参考文献-1.txt"
        to_fn = changext(fn, "-fmt.txt")
        fw = open(to_fn, "w", encoding="utf-8")
        with open(fn, "r", encoding="utf-8") as f:

            for i, line in enumerate(f):
                title = _get_title(line)
                title = title.lower()

                if title in title_dict:
                    print("[{}]\t{}".format(i + 1, title_dict[title]))
                    is_find[title] = True
                    fw.write("{}\n".format(title_dict[title]))
                else:
                    print("[{}]\t -----------------------({})".format(i + 1, title))
                    fw.write("-----------------------({})\n".format(title))

            fw.write("\n\n")
            print("\n# Not find")
            for name in is_find:
                if not is_find[name]:
                    print(title_dict[name])
                    fw.write("{}\n".format(title_dict[name]))
            fw.write("\n\n")
            fw.close()

    def func3():
        image_data = np.random.random((100, 100))
        image_data = np.concatenate([[image_data], [image_data], [image_data]]).transpose((2, 1, 0))

        plt.imshow(image_data)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(r"F:\Week\20250330\Data\dddd.jpg", dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.show()

    return func3()


class GD_GMRaster:
    COLOR_TABLE_4 = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255)}

    def __init__(
            self,
            raster_type,
            raster_fn=None,
            geo_region=None,
            raster_region=None,
            min_list=None,
            max_list=None,
            color_table=None,
            _func=None,
    ):
        self.raster_type = raster_type
        self.raster_fn = raster_fn
        self.geo_region = geo_region
        self.raster_region = raster_region
        self.min_list = min_list
        self.max_list = max_list
        self.color_table = color_table
        self._func = _func
        self.gmr = None

    def draw(self, channels, to_fn=None,
             x_major_len=(0, 8, 0), y_major_len=(0, 8, 0),
             x_minor_len=(0, 0, 20), y_minor_len=(0, 0, 20), ):
        gmr = GMRaster(self.raster_fn)

        if isinstance(channels, int) or isinstance(channels, str) or isinstance(channels, list):
            gmr.read(channels, geo_region=self.geo_region, raster_region=self.raster_region,
                     min_list=self.min_list, max_list=self.max_list, color_table=self.color_table,
                     _func=self._func)
        else:
            for raster_fn in channels:
                gmr.read(channels[raster_fn], geo_region=self.geo_region, raster_region=self.raster_region,
                         min_list=self.min_list, max_list=self.max_list, color_table=self.color_table,
                         _func=self._func, raster_fn=raster_fn)

        ax = gmr.draw(
            9,
            x_major_len=x_major_len, y_major_len=y_major_len,
            x_minor_len=x_minor_len, y_minor_len=y_minor_len,
            fontdict={"size": 26, "style": 'italic'}
        )
        gmr.scale(
            loc=(0.40, 0.02), length=0.5269,
            offset=0.06, offset2=0.13, offset3=0.3, high=0.08, high2=0.016,
            fontdict={'family': 'Times New Roman', "size": 21}
        )
        gmr.compass(loc=(0.92, 0.76), size=1.8)

        if to_fn is not None:
            plt.savefig(to_fn, bbox_inches='tight', dpi=300)

        self.gmr = gmr
        return ax

    def drawImages(self, channels, raster_fns, to_dirname=None):
        for raster_fn in raster_fns:
            self.raster_fn = raster_fn
            to_fn = None
            if to_dirname is not None:
                to_fn = os.path.join(to_dirname, getfilenamewithoutext(self.raster_fn) + ".jpg")
            print(to_fn)
            self.draw(channels, to_fn=to_fn)


def funcs3(*args):
    def func1():
        # GDALSamplingFast(r"F:\ProjectSet\Shadow\ASDEHSamples\Images\BeiJing\bj_down1_swir.tif").csvfile(r"F:\GraduationDesign\Result\run\Samples\Funcs3\HSPL_BJ_select.csv", r"F:\GraduationDesign\Result\run\Samples\Funcs3\HSPL_BJ_select_spl.csv")
        # GDALSamplingFast(r"F:\ProjectSet\Shadow\ASDEHSamples\Images\ChengDu\cd_down1_swir.tif").csvfile(r"F:\GraduationDesign\Result\run\Samples\Funcs3\HSPL_CD_select.csv", r"F:\GraduationDesign\Result\run\Samples\Funcs3\HSPL_CD_select_spl.csv")
        # GDALSamplingFast(r"F:\ProjectSet\Shadow\ASDEHSamples\Images\QingDao\qd_down1_swir.tif").csvfile(r"F:\GraduationDesign\Result\run\Samples\Funcs3\HSPL_QD_select.csv", r"F:\GraduationDesign\Result\run\Samples\Funcs3\HSPL_QD_select_spl.csv")

        FONT_SIZE = 16
        plt.rcParams.update({'font.size': FONT_SIZE})
        plt.rcParams['font.family'] = ["Times New Roman", 'SimSun', ] + plt.rcParams['font.family']
        plt.rcParams['mathtext.fontset'] = 'stix'

        gsu = _FUNCS3_SamplesUtil()
        gsu.addCSV(r"F:\GraduationDesign\Result\run\Samples\Funcs3\HSPL_BJ_select_spl.csv")
        gsu.addCSV(r"F:\GraduationDesign\Result\run\Samples\Funcs3\HSPL_CD_select_spl.csv")
        gsu.addCSV(r"F:\GraduationDesign\Result\run\Samples\Funcs3\HSPL_QD_select_spl.csv")

        df = gsu.toDF()
        print(df.value_counts("CNAME"))
        printList("DF List", list(df.keys()))

        def to_01(_names):
            for name in _names:
                df[name] = (df[name] - df[name].mean()) / df[name].std()
                df[name] = (df[name] - df[name].min()) / (df[name].max() - df[name].min())
            return _names

        opt_names = to_01([
            "NDVI", "NDWI", "Blue", "Green", "Red", "NIR", "B11", "B12", "OPT_mean",
            "OPT_cor", "OPT_ent", "OPT_dis", "OPT_hom", "OPT_con",
            "OPT_var", "OPT_asm",
        ])
        opt_show_names = [
            '$ NDVI $', '$ NDWI $', '$ Blue $', '$ Green $', '$ Red $', '$ NIR $', '$ OPT_{mean} $',
            '$ OPT_{cor} $', '$ OPT_{ent} $', '$ OPT_{dis} $', '$ OPT_{hom} $', '$ OPT_{con} $',
            '$ OPT_{var} $', '$ OPT_{asm} $',
        ]
        opt_show_names = [
            "NDVI", "NDWI", "B2", "B3", "B4", "B8", "B11", "B12", "GRAY mean",
            "GRAY cor", "GRAY ent", "GRAY dis", "GRAY hom", "GRAY con",
            "GRAY var", "GRAY asm",
        ]

        sar_names = to_01([
            "AS_VV", "AS_VH", "AS_C11", "AS_C22", "AS_H", "AS_Alpha",
            "DE_VV", "DE_VH", "DE_C11", "DE_C22", "DE_H", "DE_Alpha",
        ])
        sar_names_as = ["AS_VV", "AS_VH", "AS_C11", "AS_C22", "AS_H", "AS_Alpha", ]
        sar_names_de = ["DE_VV", "DE_VH", "DE_C11", "DE_C22", "DE_H", "DE_Alpha", ]

        sar_show_names = [
            "$ \\sigma_{VV} $", "$ \\sigma_{VH} $", "$ C_{11} $", "$ C_{22} $", "$ H $", r"$ \alpha $",
        ]

        def _get_data_1(_cname, _fields=None):
            _df = df[df["CNAME"] == _cname]
            if _fields is not None:
                return _df[_fields]
            else:
                return _df

        def draw1(_cname, _fields, *_args, _x_names=None, **_kwargs):
            if _x_names is None:
                _x_names = ["$ {} $".format(_field) for _field in _fields]
            _df = _get_data_1(_cname, _fields)
            _df = _df.mean()

            plt.plot(_x_names, _df.values, *_args, **_kwargs)

        # plt.rc('font', family='Times New Roman')
        # plt.rcParams.update({'font.size': 16})

        # font_SimHei = FontProperties(family="SimHei")

        dfn = DirFileName(r"F:\GraduationDesign\Images\3")

        def draw1_im1(_fn, _draw1_data, ):

            # plt.figure(figsize=(8, 5))
            # plt.subplots_adjust(top=0.97, bottom=0.21, left=0.081, right=0.981, hspace=0.2, wspace=0.2)

            fontsize = 16

            for _data in _draw1_data:
                draw1(*_data[0], **_data[1])

            plt.xlabel("特征", fontsize=fontsize)
            plt.ylabel("特征均值", fontsize=fontsize)

            plt.ylim([None, 1.0])
            plt.legend(fontsize=fontsize, loc='upper left')
            if _fn.startswith("sar"):
                _rotation=0
            else:
                _rotation=45
            plt.xticks(rotation=_rotation, fontsize=fontsize - 2)
            if _fn is not None:
                to_fn = dfn.fn("{}.jpg".format(_fn))
                print(to_fn)
                plt.savefig(to_fn, dpi=300, bbox_inches='tight', pad_inches=0.03)

        def _func_draw1_data(*_args, **_kwargs):
            return [_args, _kwargs]

        def draw1_im2(_fn, _draw2_data, ):

            n = len(_draw2_data)

            for i, _data2 in enumerate(_draw2_data):
                plt.subplot(n * 100 + 10 + (i + 1))
                for _data in _data2:
                    draw1(*_data[0], **_data[1])
                    plt.xlabel("特征", )
                    plt.ylabel("特征均值", )
                    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0, frameon=False)
                    plt.xticks(rotation=45)
                # plt.text(0.02, 0.85, "({})".format(i + 1), fontsize=14, transform=plt.gca().transAxes)

            if _fn is not None:
                plt.savefig(dfn.fn("{}.jpg".format(_fn)), dpi=300)

        names = opt_names
        show_names = opt_show_names

        MARKER_SIZE = 10

        is_draw1_data = _func_draw1_data(
            "IS", names, "*-", _x_names=show_names, label="不透水面", color="red", markersize=MARKER_SIZE)
        is_sh_draw1_data = _func_draw1_data(
            "IS_SH", names, "*-", _x_names=show_names, label="光学阴影不透水面", color="darkred",
            markersize=MARKER_SIZE)

        is_draw1_data_as = _func_draw1_data(
            "IS", sar_names_as, "*-", _x_names=sar_show_names, label="升轨SAR—不透水面", color="red",
            markersize=MARKER_SIZE)
        is_sh_draw1_data_as = _func_draw1_data(
            "IS_SH", sar_names_as, "*-", _x_names=sar_show_names, label="升轨SAR—光学阴影不透水面", color="darkred",
            markersize=MARKER_SIZE)
        is_draw1_data_de = _func_draw1_data(
            "IS", sar_names_de, "*-", _x_names=sar_show_names, label="降轨SAR—不透水面", color="#FFC1C1",
            markersize=MARKER_SIZE)
        is_sh_draw1_data_de = _func_draw1_data(
            "IS_SH", sar_names_de, "*-", _x_names=sar_show_names, label="降轨SAR—光学阴影不透水面", color="#8B6969",
            markersize=MARKER_SIZE)

        is_as_sh_draw1_data = _func_draw1_data(
            "IS_AS_SH", names, "*-", label="升轨SAR阴影不透水面", color="#FFC1C1", markersize=MARKER_SIZE)
        is_de_sh_draw1_data = _func_draw1_data(
            "IS_DE_SH", names, "*-", label="降轨SAR阴影不透水面", color="#8B6969", markersize=MARKER_SIZE)

        veg_draw1_data = _func_draw1_data(
            "VEG", names, "^-", _x_names=show_names, label="植被", color="lime", markersize=MARKER_SIZE)
        veg_sh_draw1_data = _func_draw1_data(
            "VEG_SH", names, "^-", _x_names=show_names, label="光学阴影植被", color="darkgreen", markersize=MARKER_SIZE)

        veg_draw1_data_as = _func_draw1_data(
            "VEG", sar_names_as, "^-", _x_names=sar_show_names, label="升轨SAR—植被", color="lime",
            markersize=MARKER_SIZE)
        veg_sh_draw1_data_as = _func_draw1_data(
            "VEG_SH", sar_names_as, "^-", _x_names=sar_show_names, label="升轨SAR—光学阴影植被", color="darkgreen",
            markersize=MARKER_SIZE)
        veg_draw1_data_de = _func_draw1_data(
            "VEG", sar_names_de, "^-", _x_names=sar_show_names, label="降轨SAR—植被", color="#008B45",
            markersize=MARKER_SIZE)
        veg_sh_draw1_data_de = _func_draw1_data(
            "VEG_SH", sar_names_de, "^-", _x_names=sar_show_names, label="降轨SAR—光学阴影植被", color="#C0FF3E",
            markersize=MARKER_SIZE)

        veg_as_sh_draw1_data = _func_draw1_data(
            "VEG_AS_SH", names, "^-", _x_names=show_names, label="升轨SAR阴影植被", color="#008B45",
            markersize=MARKER_SIZE)
        veg_de_sh_draw1_data = _func_draw1_data(
            "VEG_DE_SH", names, "^-", _x_names=show_names, label="降轨SAR阴影植被", color="#C0FF3E",
            markersize=MARKER_SIZE)

        soil_draw1_data = _func_draw1_data(
            "SOIL", names, "s-", _x_names=show_names, label="裸土", color="#FF8247", markersize=MARKER_SIZE)
        soil_sh_draw1_data = _func_draw1_data(
            "SOIL_SH", names, "s-", _x_names=show_names, label="光学阴影裸土", color="#8B7355", markersize=MARKER_SIZE)

        soil_draw1_data_as = _func_draw1_data(
            "SOIL", sar_names_as, "s-", _x_names=sar_show_names, label="升轨SAR—裸土", color="#FF8247",
            markersize=MARKER_SIZE)
        soil_sh_draw1_data_as = _func_draw1_data(
            "SOIL_SH", sar_names_as, "s-", _x_names=sar_show_names, label="升轨SAR—光学阴影裸土", color="#8B7355",
            markersize=MARKER_SIZE)
        soil_draw1_data_de = _func_draw1_data(
            "SOIL", sar_names_de, "s-", _x_names=sar_show_names, label="降轨SAR—裸土", color="#FFEC8B",
            markersize=MARKER_SIZE)
        soil_sh_draw1_data_de = _func_draw1_data(
            "SOIL_SH", sar_names_de, "s-", _x_names=sar_show_names, label="降轨SAR—光学阴影裸土", color="#8B8B00",
            markersize=MARKER_SIZE)

        soil_as_sh_draw1_data = _func_draw1_data(
            "SOIL_AS_SH", names, "s-", _x_names=show_names, label="升轨SAR阴影裸土", color="#FFEC8B",
            markersize=MARKER_SIZE)
        soil_de_sh_draw1_data = _func_draw1_data(
            "SOIL_DE_SH", names, "s-", _x_names=show_names, label="降轨SAR阴影裸土", color="#8B8B00",
            markersize=MARKER_SIZE)

        wat_draw1_data = _func_draw1_data(
            "WAT", names, "o-", _x_names=show_names, label="水体", color="#4876FF", markersize=MARKER_SIZE)
        wat_sh_draw1_data = _func_draw1_data(
            "WAT_SH", names, "o-", _x_names=show_names, label="光学阴影水体", color="#27408B", markersize=MARKER_SIZE)

        wat_draw1_data_as = _func_draw1_data(
            "WAT", sar_names_as, "o-", _x_names=sar_show_names, label="升轨SAR—水体", color="#4876FF",
            markersize=MARKER_SIZE)
        wat_sh_draw1_data_as = _func_draw1_data(
            "WAT_SH", sar_names_as, "o-", _x_names=sar_show_names, label="升轨SAR—光学阴影水体", color="#27408B",
            markersize=MARKER_SIZE)
        wat_draw1_data_de = _func_draw1_data(
            "WAT", sar_names_de, "o-", _x_names=sar_show_names, label="降轨SAR—水体", color="#00BFFF",
            markersize=MARKER_SIZE)
        wat_sh_draw1_data_de = _func_draw1_data(
            "WAT_SH", sar_names_de, "o-", _x_names=sar_show_names, label="降轨SAR—光学阴影水体", color="#5F9EA0",
            markersize=MARKER_SIZE)

        wat_as_sh_draw1_data = _func_draw1_data(
            "WAT_AS_SH", names, "o-", _x_names=show_names, label="升轨SAR阴影水体", color="#00BFFF",
            markersize=MARKER_SIZE)
        wat_de_sh_draw1_data = _func_draw1_data(
            "WAT_DE_SH", names, "o-", _x_names=show_names, label="降轨SAR阴影水体", color="#5F9EA0",
            markersize=MARKER_SIZE)

        # draw_im1("im41", [is_draw1_data, is_sh_draw1_data])
        # draw_im1("im42", [is_draw1_data, veg_draw1_data, veg_sh_draw1_data])
        # draw_im1("im43", [is_draw1_data, wat_draw1_data, wat_sh_draw1_data])
        # draw_im1("im44", [is_draw1_data, soil_draw1_data, soil_sh_draw1_data])
        # draw_im1("im45", [is_sh_draw1_data, veg_draw1_data, veg_sh_draw1_data])
        # draw_im1("im46", [is_sh_draw1_data, wat_draw1_data, wat_sh_draw1_data])
        # draw_im1("im47", [is_sh_draw1_data, soil_draw1_data, soil_sh_draw1_data])

        # plt.figure(figsize=(10, 9.5))
        # plt.subplots_adjust(top=0.975, bottom=0.137, left=0.099, right=0.704, hspace=0.732, wspace=0.2)
        # draw1_im2(
        #     "im31",
        #     [[veg_draw1_data, veg_sh_draw1_data, is_draw1_data, is_sh_draw1_data, ],
        #      [wat_draw1_data, wat_sh_draw1_data, is_draw1_data, is_sh_draw1_data, ],
        #      [soil_draw1_data, soil_sh_draw1_data, is_draw1_data, is_sh_draw1_data, ]]
        # )
        # plt.show()

        plt.figure(figsize=(9, 6))
        plt.subplots_adjust(top=0.96, bottom=0.217, left=0.098, right=0.965, hspace=0.732, wspace=0.2)
        draw1_im1("opt_is", [is_draw1_data, is_sh_draw1_data, ])
        plt.clf()
        draw1_im1("opt_veg", [veg_draw1_data, veg_sh_draw1_data, is_draw1_data, is_sh_draw1_data, ])
        plt.clf()
        draw1_im1("opt_soil", [soil_draw1_data, soil_sh_draw1_data, is_draw1_data, is_sh_draw1_data, ])
        plt.clf()
        draw1_im1("opt_wat", [wat_draw1_data, wat_sh_draw1_data, is_draw1_data, is_sh_draw1_data, ])
        plt.clf()

        draw1_im1("sar_is1", [is_draw1_data_as, is_sh_draw1_data_as, is_draw1_data_de, is_sh_draw1_data_de])
        plt.clf()
        draw1_im1("sar_veg1", [
            is_draw1_data_as, is_sh_draw1_data_as, is_draw1_data_de, is_sh_draw1_data_de,
            veg_draw1_data_as, veg_sh_draw1_data_as, veg_draw1_data_de, veg_sh_draw1_data_de])
        plt.clf()
        draw1_im1("sar_soil1", [
            is_draw1_data_as, is_sh_draw1_data_as, is_draw1_data_de, is_sh_draw1_data_de,
            soil_draw1_data_as, soil_sh_draw1_data_as, soil_draw1_data_de, soil_sh_draw1_data_de])
        plt.clf()
        draw1_im1("sar_wat1", [
            is_draw1_data_as, is_sh_draw1_data_as, is_draw1_data_de, is_sh_draw1_data_de,
            wat_draw1_data_as, wat_sh_draw1_data_as, wat_draw1_data_de, wat_sh_draw1_data_de])
        plt.clf()

        # plt.figure(figsize=(8, 9.5))
        # plt.subplots_adjust(top=0.984, bottom=0.111, left=0.081, right=0.781, hspace=0.51, wspace=0.2)
        # draw1_im2(
        #     "im32",
        #     [[is_draw1_data, is_sh_draw1_data, is_as_sh_draw1_data, is_de_sh_draw1_data],
        #      [veg_draw1_data, veg_sh_draw1_data, veg_as_sh_draw1_data, veg_de_sh_draw1_data],
        #      [soil_draw1_data, soil_sh_draw1_data, soil_as_sh_draw1_data, soil_de_sh_draw1_data],
        #      [wat_draw1_data, wat_sh_draw1_data, wat_as_sh_draw1_data, wat_de_sh_draw1_data],]
        # )
        # plt.show()

        # plt.figure(figsize=(8, 9.5))
        # plt.subplots_adjust(top=0.984, bottom=0.105, left=0.081, right=0.74, hspace=0.759, wspace=0.2)
        # draw1_im2(
        #     "im33",
        #     [[is_draw1_data, veg_draw1_data, soil_draw1_data, wat_draw1_data, ],
        #      [is_sh_draw1_data, veg_sh_draw1_data, soil_sh_draw1_data, wat_sh_draw1_data],
        #      [is_as_sh_draw1_data, veg_as_sh_draw1_data, soil_as_sh_draw1_data, wat_as_sh_draw1_data],
        #      [is_de_sh_draw1_data, veg_de_sh_draw1_data, soil_de_sh_draw1_data, wat_de_sh_draw1_data], ]
        # )
        # plt.show()

        # draw1_im1(None, [is_draw1_data, is_as_sh_draw1_data, is_de_sh_draw1_data,veg_draw1_data ])

        # plt.figure(figsize=(8, 9.5))
        # plt.subplots_adjust(top=0.984, bottom=0.111, left=0.081, right=0.781, hspace=0.51, wspace=0.2)
        # draw1_im2(
        #     "im33",
        #     [[is_draw1_data, veg_draw1_data, veg_sh_draw1_data, veg_as_sh_draw1_data, veg_de_sh_draw1_data],
        #      [is_draw1_data, wat_draw1_data, wat_sh_draw1_data, wat_as_sh_draw1_data, wat_de_sh_draw1_data],
        #      [is_draw1_data, soil_draw1_data, soil_sh_draw1_data, soil_as_sh_draw1_data, soil_de_sh_draw1_data]]
        # )
        # plt.show()

        # plt.figure(figsize=(8, 9.5))
        # plt.subplots_adjust(top=0.984, bottom=0.111, left=0.081, right=0.781, hspace=0.51, wspace=0.2)
        # draw1_im2(
        #     "im34",
        #     [[is_sh_draw1_data, veg_draw1_data, veg_sh_draw1_data, veg_as_sh_draw1_data, veg_de_sh_draw1_data],
        #      [is_sh_draw1_data, wat_draw1_data, wat_sh_draw1_data, wat_as_sh_draw1_data, wat_de_sh_draw1_data],
        #      [is_sh_draw1_data, soil_draw1_data, soil_sh_draw1_data, soil_as_sh_draw1_data, soil_de_sh_draw1_data]]
        # )
        # plt.show()

        return

    def func2():
        tlp = TableLinePrint().firstLine("Number", "Name", "Min", "Max", "Mean")

        def func21(_im_fn):
            gr = GDALRaster(_im_fn)
            for i, name in enumerate(gr.names):
                data = gr.readGDALBand(name)
                tlp.print(i + 1, name, float(data.min()), float(data.max()), float(data.mean()), )

        def func22(_im_fn, _im_to_fn):
            gr = GDALRaster(_im_fn)
            data = np.zeros((3, gr.n_rows, gr.n_columns), dtype="int8")

            def _get_data(_name, _min, _max):
                _data = gr.readGDALBand(_name)
                return (scaleMinMax(_data, _min, _max) * 255).astype("int8")

            data[0] = _get_data("Red", 0, 2500)
            data[1] = _get_data("Green", 0, 2500)
            data[2] = _get_data("Blue", 0, 2500)

            gr.save(data, _im_to_fn.format("RGB"), fmt="GTiff", dtype=gdal.GDT_Byte, options=["COMPRESS=PACKBITS"])

            data[0] = _get_data("NIR", 0, 3000)
            data[1] = _get_data("Red", 0, 2500)
            data[2] = _get_data("Green", 0, 2500)

            gr.save(data, _im_to_fn.format("NRG"), fmt="GTiff", dtype=gdal.GDT_Byte, options=["COMPRESS=PACKBITS"])

        # func22(
        #     r"F:\ProjectSet\Shadow\Release\BeiJingImages\SH_BJ_look_envi.dat",
        #     r"F:\GraduationDesign\MkTu\SH_BJ_{}.tif"
        # )
        #
        # func22(
        #     r"F:\ProjectSet\Shadow\Release\QingDaoImages\SH_QD_look_envi.dat",
        #     r"F:\GraduationDesign\MkTu\SH_QD_{}.tif"
        # )

        func22(
            r"F:\ProjectSet\Shadow\Release\ChengDuImages\SH_CD_look_envi.dat",
            r"F:\GraduationDesign\MkTu\SH_CD_{}.tif"
        )

    def func3():
        json_data = readJson(r"G:\SHImages\image.json")

        def _show_1(_name):
            plt.figure(figsize=(12, 4))
            for i, _fn in enumerate(json_data[_name]):
                _fn = _cid_G.change(_fn)
                print(_fn)
                gr = GDALRaster(_fn)
                plt.subplot(1, 3, i + 1)
                data = gr.readGDALBand(1)
                data = scaleMinMax(update10EDivide10(data), 0, 0.3)
                plt.imshow(data, cmap=plt.get_cmap("gray"))

            plt.show()

        to_dfn = DirFileName(r"F:\GraduationDesign\MkTu\Images")

        def _save_1(_name, _min, _max, _to_dfn: DirFileName):
            for i, _fn in enumerate(json_data[_name]):
                _fn = _cid_G.change(_fn)
                print(_fn)
                gr = GDALRaster(_fn)
                data = gr.readGDALBand(1)
                data = scaleMinMax(update10EDivide10(data), _min, _max) * 255
                data = data.astype("int8")
                _to_fn = _to_dfn.fn(changext(getfilenme(_fn), "_look.tif"))
                print(_to_fn)
                if os.path.isfile(_to_fn):
                    os.remove(_to_fn)
                gr.saveGTiff(data, _to_fn, options=["COMPRESS=PACKBITS"])

        # _save_1("AS_C11", 0, 1, to_dfn)
        _save_1("AS_C22", 0, 0.3, to_dfn)
        # _save_1("DE_C11", 0, 1, to_dfn)
        _save_1("DE_C22", 0, 0.3, to_dfn)

    def func4(_run="sigma"):
        # GD_GMRaster("qd", ).drawImages(
        #     [1, 2, 3],
        #     [r"F:\GraduationDesign\MkTu\SH_BJ_NRG.tif",
        #      r"F:\GraduationDesign\MkTu\SH_BJ_RGB.tif",
        #      r"F:\GraduationDesign\MkTu\SH_CD_NRG.tif",
        #      r"F:\GraduationDesign\MkTu\SH_CD_RGB.tif",
        #      r"F:\GraduationDesign\MkTu\SH_QD_NRG.tif",
        #      r"F:\GraduationDesign\MkTu\SH_QD_RGB.tif",
        #      r"F:\GraduationDesign\MkTu\SH22_BJ_NRG.tif",
        #      r"F:\GraduationDesign\MkTu\SH22_BJ_RGB.tif",
        #      r"F:\GraduationDesign\MkTu\SH22_CD_NRG.tif",
        #      r"F:\GraduationDesign\MkTu\SH22_CD_RGB.tif",
        #      r"F:\GraduationDesign\MkTu\SH22_QD_NRG.tif",
        #      r"F:\GraduationDesign\MkTu\SH22_QD_RGB.tif", ],
        #     r"F:\GraduationDesign\MkTu\Images\images",
        # )

        json_data = readJson(r"G:\SHImages\image.json")

        ells = {"QD": [], "BJ": [], "CD": []}

        class _ell:

            def __init__(self, xy, width, height, edgecolor,
                         facecolor='none', linewidth=5):
                self.xy = xy
                self.data = dict(width=width, height=height, edgecolor=edgecolor,
                                 facecolor=facecolor, linewidth=linewidth)

            def fit(self, _gr: GDALRaster):
                _row, _column = _gr.coorGeo2Raster(self.xy[0], self.xy[1], True)
                self.data["xy"] = (_column, _row,)
                return Ellipse(**self.data)

        def _add_ell(city_name, xy, width, height, edgecolor):
            ells[city_name].append(
                _ell(xy=xy, width=width, height=height, edgecolor=edgecolor,
                     facecolor='none', linewidth=5)
            )

        def _draw_1(_name, _min, _max, _func, _to_dirname, ):
            for _fn in json_data[_name]:
                # if "CD" not in getfilenamewithoutext(_fn):
                #     continue
                to_fn = os.path.join(_to_dirname, getfilenamewithoutext(_fn) + ".jpg")
                min_list = [_min] if _min is not None else None
                max_list = [_max] if _max is not None else None
                _gmr = GD_GMRaster(
                    "qd", raster_fn=_cid_G.change(_fn), min_list=min_list, max_list=max_list, _func=_func
                )
                ax = _gmr.draw(1, to_fn=to_fn, )
                for city_name in ells:
                    if city_name in getfilenamewithoutext(_fn):
                        for d in ells[city_name]:
                            ax.add_patch(d.fit(_gmr.gmr.gr))
                print(to_fn)
                plt.savefig(to_fn, bbox_inches='tight', dpi=300)
                # plt.show()
                plt.clf()

        to_dirname = r"F:\GraduationDesign\MkTu\Images\images"

        _add_ell("QD", (120.42991, 36.38329), 1000, 600, "red")
        _add_ell("QD", (120.19427, 36.24705), 500, 600, "red")
        _add_ell("QD", (120.14196, 36.29585), 600, 600, "yellow")
        _add_ell("QD", (120.04549, 36.20751), 500, 300, "yellow")
        _add_ell("BJ", (116.54959, 39.67683), 1000, 600, "red")
        _add_ell("BJ", (116.60903, 39.81002), 500, 300, "yellow")
        _add_ell("CD", (104.01196, 30.67262), 500, 300, "red")
        _add_ell("CD", (103.83861, 30.75427), 500, 400, "yellow")

        if _run == "sigma":
            _draw_1("AS_VV", -15.0, 1.0, None, to_dirname)
            _draw_1("AS_VH", -20, -6, None, to_dirname)
            _draw_1("DE_VV", -15.0, 1.0, None, to_dirname)
            _draw_1("DE_VH", -20, -6, None, to_dirname)

        if _run == "C2":
            update10EDivide10 = None
            _draw_1("AS_C11", 0, 0.8, update10EDivide10, to_dirname)
            _draw_1("AS_C22", 0, 0.2, update10EDivide10, to_dirname)
            _draw_1("DE_C11", 0, 0.8, update10EDivide10, to_dirname)
            _draw_1("DE_C22", 0, 0.2, update10EDivide10, to_dirname)

        if _run == "HA":
            _draw_1("AS_H", 0.2, 0.98, None, to_dirname)
            _draw_1("DE_H", 0.2, 0.98, None, to_dirname)
            _draw_1("AS_Alpha", 26.0, 78.0, None, to_dirname)
            _draw_1("DE_Alpha", 26.0, 78.0, None, to_dirname)

        return

    def func5(_run="sigma"):
        plt.rcParams.update({'font.size': 12})
        plt.rcParams['font.family'] = ['SimSun', "Times New Roman", ] + plt.rcParams['font.family']
        plt.rcParams['mathtext.fontset'] = 'stix'

        dfn = DirFileName(r"F:\GraduationDesign\MkTu\Images\images")
        c2_cnames1 = ["$C_{11}$", "$C_{22}$"]
        sigma_names1 = [r"$\sigma_{VV}$", r"$\sigma_{VH}$"]
        ha_names1 = ["$H$", r"$\alpha$"]

        c2_names2 = ["AS_C11", "DE_C11", "AS_C22", "DE_C22"]
        sigma_names2 = ["AS_VV", "DE_VV", "AS_VH", "DE_VH"]
        ha_names2 = ["AS_H", "DE_H", "AS_Alpha", "DE_Alpha"]

        _n = sigma_names1
        image_names = sigma_names2

        if _run == "sigma":
            _n = sigma_names1
            image_names = sigma_names2

        if _run == "C2":
            _n = c2_cnames1
            image_names = c2_names2

        if _run == "HA":
            _n = ha_names1
            image_names = ha_names2

        _names = [
            "（$%s$）青岛升轨 {}".format(_n[0]), "（$%s$）北京升轨 {}".format(_n[0]), "（$%s$）成都升轨 {}".format(_n[0]),
            "（$%s$）青岛降轨 {}".format(_n[0]), "（$%s$）北京降轨 {}".format(_n[0]), "（$%s$）成都降轨 {}".format(_n[0]),
            "（$%s$）青岛升轨 {}".format(_n[1]), "（$%s$）北京升轨 {}".format(_n[1]), "（$%s$）成都升轨 {}".format(_n[1]),
            "（$%s$）青岛降轨 {}".format(_n[1]), "（$%s$）北京降轨 {}".format(_n[1]), "（$%s$）成都降轨 {}".format(_n[1]),
        ]
        numbers = "abcdefghijklmnopqrstuvwxyz"
        n = 1
        plt.figure(figsize=(8, 9))
        plt.subplots_adjust(top=0.983, bottom=0.017, left=0.019, right=0.981, hspace=0.0, wspace=0.061)
        for image_name in image_names:
            for city_name in ["QD", "BJ", "CD"]:
                plt.subplot(4, 3, n)
                plt.imshow(matplotlib.image.imread(dfn.fn("{}_{}.jpg".format(city_name, image_name))))
                plt.xlabel(_names[n - 1] % numbers[n - 1])
                n += 1

                plt.gca().spines['right'].set_visible(False)
                plt.gca().spines['left'].set_visible(False)
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['bottom'].set_visible(False)

                plt.xticks([])
                plt.yticks([])

        plt.savefig(dfn.fn("{}.jpg".format(_run)), bbox_inches='tight', dpi=300)

    def func6(_run, n):
        if n == 1:
            func4(_run)
        if n == 2:
            func5(_run)

    def func7():

        def _func_1(_name, _fn):
            _gmr = GD_GMRaster(
                "qd", raster_fn=_fn,
                color_table={1: (200, 0, 0), 2: (255, 255, 255), 3: (255, 255, 255), 4: (255, 255, 255)}
            )
            _gmr.draw(
                [1],
                x_major_len=(0, 10, 0), y_major_len=(0, 10, 0),
                x_minor_len=(0, 0, 20), y_minor_len=(0, 0, 20),
            )
            to_fn = r"F:\GraduationDesign\MkTu\Images\images\{}_imdc_adesi.jpg".format(_name)
            print(to_fn)
            plt.savefig(to_fn, bbox_inches='tight', dpi=300)
            plt.show()

        # _func_1("qd", r"F:\ASDEWrite\Result\QingDao\qd_SAR-Opt-AS-DE_imdc.tif")
        # _func_1("bj", r"F:\ASDEWrite\Result\BeiJing\bj_SAR-Opt-AS-DE_imdc.tif")
        # _func_1("cd", r"F:\ASDEWrite\Result\ChengDu\cd_SAR-Opt-AS-DE_imdc.tif")

        _func_1("qd", r"F:\GraduationDesign\Result\QingDao\20250120H183522\qd-RF_9ADESI_GLCM_OPT-RF_imdc.tif")
        _func_1("bj", r"F:\GraduationDesign\Result\BeiJing\20250120H190546\bj-RF_9ADESI_GLCM_OPT-RF_imdc.tif")
        _func_1("cd", r"F:\GraduationDesign\Result\ChengDu\20250120H193443\cd-RF_9ADESI_GLCM_OPT-RF_imdc.tif")

    def func8():
        json_data = readJson(r"G:\SHImages\image.json")
        to_dirname = r"F:\GraduationDesign\MkTu\Images\images"
        dfn = DirFileName(to_dirname)
        _n = []
        image_names = []

        def _draw_1(_name, _min, _max, _func, _to_dirname, ):
            _n.append(_name)
            image_names.append(_name)
            for _fn in json_data[_name]:
                to_fn = os.path.join(_to_dirname, getfilenamewithoutext(_fn) + ".jpg")
                if not os.path.isfile(to_fn):
                    min_list = [_min] if _min is not None else None
                    max_list = [_max] if _max is not None else None
                    _gmr = GD_GMRaster(
                        "qd", raster_fn=_cid_G.change(_fn), min_list=min_list, max_list=max_list, _func=_func
                    )
                    ax = _gmr.draw(1, to_fn=to_fn, )
                    plt.savefig(to_fn, bbox_inches='tight', dpi=300)
                    # plt.show()
                    plt.clf()
                print(to_fn)

        def _draw_3(_name, _raster_names, _min, _max, _func, _to_dirname, ):
            _n.append(_name)
            image_names.append(_name)
            _images = {"QD": [], "BJ": [], "CD": []}

            for _raster_name in _raster_names:
                for _fn in json_data[_raster_name]:
                    for _image_name in _images:
                        if _image_name in _fn:
                            _images[_image_name].append(_fn)

            for _image_name in _images:
                to_fn = os.path.join(_to_dirname, "{}_{}.jpg".format(_image_name, _name))
                if not os.path.isfile(to_fn):
                    _gmr = GD_GMRaster(
                        "qd", raster_fn=_cid_G.change(_images[_image_name][0]),
                        min_list=_min, max_list=_max, _func=_func
                    )
                    _gmr.draw({raster_fn: [1] for raster_fn in _images[_image_name]}, to_fn=to_fn, )
                    plt.savefig(to_fn, bbox_inches='tight', dpi=300)
                    # plt.show()
                    plt.clf()
                print(to_fn)

        # _draw_1("Red", 300, 2500, None, to_dirname)
        # _draw_1("Green", 300, 2500, None, to_dirname)
        # _draw_1("Blue", 300, 2500, None, to_dirname)
        # _draw_1("NIR", 300, 2500, None, to_dirname)
        # _draw_1("SWIR1", 300, 2500, None, to_dirname)
        # _draw_1("SWIR2", 300, 2500, None, to_dirname)

        _draw_3("RGB", ["Red", "Green", "Blue"], [300, 300, 300], [2500, 2500, 2500], None, to_dirname)
        _draw_3("NRG", ["NIR", "Red", "Green"], [300, 300, 300], [2500, 2500, 2500], None, to_dirname)
        _draw_1("NDVI", -0.3, 0.5, None, to_dirname)
        _draw_1("NDWI", -0.6, 0.1, None, to_dirname)

        plt.rcParams.update({'font.size': 12})
        plt.rcParams['font.family'] = ['SimSun', "Times New Roman", ] + plt.rcParams['font.family']
        plt.rcParams['mathtext.fontset'] = 'stix'

        _n = ["真彩色", "假彩色（近红外 红 绿）", "$NDVI$", "$NDWI$"]
        _names = [
            "（$%s$）青岛 {}".format(_n[0]), "（$%s$）北京 {}".format(_n[0]), "（$%s$）成都 {}".format(_n[0]),
            "（$%s$）青岛 {}".format(_n[1]), "（$%s$）北京 {}".format(_n[1]), "（$%s$）成都 {}".format(_n[1]),
            "（$%s$）青岛 {}".format(_n[2]), "（$%s$）北京 {}".format(_n[2]), "（$%s$）成都 {}".format(_n[2]),
            "（$%s$）青岛 {}".format(_n[3]), "（$%s$）北京 {}".format(_n[3]), "（$%s$）成都 {}".format(_n[3]),
        ]

        numbers = "abcdefghijklmnopqrstuvwxyz"
        n = 1
        plt.figure(figsize=(8, 9))
        plt.subplots_adjust(top=0.983, bottom=0.017, left=0.019, right=0.981, hspace=0.0, wspace=0.061)
        for image_name in image_names:
            for city_name in ["QD", "BJ", "CD"]:
                plt.subplot(len(_n), 3, n)
                plt.imshow(matplotlib.image.imread(dfn.fn("{}_{}.jpg".format(city_name, image_name))))
                plt.xlabel(_names[n - 1] % numbers[n - 1])
                n += 1

                plt.gca().spines['right'].set_visible(False)
                plt.gca().spines['left'].set_visible(False)
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['bottom'].set_visible(False)

                plt.xticks([])
                plt.yticks([])

        plt.savefig(dfn.fn("{}.jpg".format("optical_original")), bbox_inches='tight', dpi=300)

    if len(args) != 0:
        return func6(*args)
    else:
        return func1()


def adsiHS():
    def func1():
        gr = GDALRaster(r"F:\ProjectSet\Shadow\ASDEHSamples\Images\QingDao\HSPL_QD_envi.dat")
        data_as = gr.readGDALBand("AS_VV")
        data_de = gr.readGDALBand("DE_VV")
        data = np.abs((data_as - data_de) / (data_as + data_de + 0.0000001))
        # data = update10Log10(data)
        gr.save(data.astype("float32"), r"F:\ASDEWrite\Run\Images\adsi_2.dat", fmt="ENVI", dtype=gdal.GDT_Float32)

    def showColor(data, color_dict):
        to_data = np.zeros((data.shape[0], data.shape[1], 3))
        for n in color_dict:
            to_data[data == n, :] = np.array(color_dict[n]) / 255
        return to_data

    def func2():
        gr = GDALRaster(r"F:\ProjectSet\Shadow\ASDEHSamples\Threshold\1\cd_sei_adsi.vrt")

        data_csi = gr.readGDALBand(1)
        data_adsi = gr.readGDALBand(2)

        data = np.zeros(data_adsi.shape)
        data[data_adsi > 0.507] = 2
        data[data_csi > -0.068] = 1

        to_fn = r"F:\ProjectSet\Shadow\ASDEHSamples\Threshold\1\cd_hs.tif"
        color_table = {0: (242,242,242), 1: (105,105,105), 2: (47,85,150)}

        if os.path.isfile(to_fn):
            os.remove(to_fn)
        gr.save(data.astype("int8"), to_fn, fmt="GTiff", dtype=gdal.GDT_Byte)
        tiffAddColorTable(to_fn, code_colors=color_table)

        plt.figure(figsize=(16, 8))
        plt.imshow(showColor(data[1000:1200, 1000:1200], color_table))
        plt.show()

    return func2()


def write():
    def func1():
        images_fn = r"F:\GraduationDesign\Images\update\ReadMe.md"
        images = []
        dfn = DirFileName(r"F:\GraduationDesign\Images\update")
        is_copy = True

        with open(images_fn, "r", encoding="utf-8") as f:
            section_names = [None, None, None]
            image_lines = []
            for line in f:
                line = line.strip()
                if line.startswith("# "):
                    section_names[0] = line[2:]
                    section_names[1], section_names[2] = None, None
                elif line.startswith("## "):
                    section_names[1] = line[3:]
                    section_names[2] = None
                elif line.startswith("## "):
                    section_names[2] = line[4:]
                else:
                    if line != "":
                        image_lines.append(line)
                    else:
                        if image_lines:
                            fn = image_lines[0].strip("\"")
                            images.append({
                                "FN": fn,
                                "NAME_CH": image_lines[1],
                                "NAME_EN": image_lines[2],
                                "SEC_1": section_names[0],
                                "SEC_2": section_names[1],
                                "SEC_3": section_names[2],
                                "OTHER": image_lines[3:] if len(image_lines) > 3 else None,
                            })
                            if is_copy:
                                if os.path.isfile(fn):
                                    dfn.copyfile(fn)
                                else:
                                    print("Not find file:", fn)
                            image_lines = []
        print(tabulate(pd.DataFrame(images), headers="keys", tablefmt="simple"))

        return

    def func2():
        title_dict = {}
        is_find = {}

        def _get_title(_line: str):
            _lines = _line.split(".")
            _title = _lines[1].split("[")[0].strip()
            if ";" in _title:
                return _title.split(";")[0].strip()
            else:
                return _title

        with open(r"F:\GraduationDesign\参考文献\参考文献.txt", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                # if not line.strip().endswith("."):
                #     print(line)
                # if "]." not in line:
                #     print(line)
                title = _get_title(line)
                title = title.lower()
                print("[{}]\t{}".format(i + 1, title))
                title_dict[title] = line.split("\t", 2)[1].strip()
                is_find[title] = False

        print("-" * 60)
        fn = r"F:\GraduationDesign\参考文献\参考文献-3.txt"
        to_fn = changext(fn, "-fmt.txt")
        fw = open(to_fn, "w", encoding="utf-8")
        with open(fn, "r", encoding="utf-8") as f:

            for i, line in enumerate(f):
                title = _get_title(line)
                title = title.lower()

                if title in title_dict:
                    print("[{}]\t{}".format(i + 1, title_dict[title]))
                    is_find[title] = True
                    fw.write("{}\n".format(title_dict[title]))
                else:
                    print("[{}]\t -----------------------({}) {}".format(i + 1, title, line.strip()))
                    fw.write("-----------------------({}) {}\n".format(title, line.strip()))

            fw.write("\n\n")
            print("\n# Not find")
            for name in is_find:
                if not is_find[name]:
                    print(title_dict[name])
                    fw.write("{}\n".format(title_dict[name]))
            fw.write("\n\n")
            fw.close()

    return func2()


if __name__ == "__main__":
    funcs3()

r"""
E:\Anaconda3\python -c "import sys; sys.path.append(r'F:\PyCodes'); from GraduationDesign.GDFuncs import funcs3; funcs3('sigma', 1)"
E:\Anaconda3\python -c "import sys; sys.path.append(r'F:\PyCodes'); from GraduationDesign.GDFuncs import funcs3; funcs3('sigma', 2)"

E:\Anaconda3\python -c "import sys; sys.path.append(r'F:\PyCodes'); from GraduationDesign.GDFuncs import funcs3; funcs3('C2', 1)"
E:\Anaconda3\python -c "import sys; sys.path.append(r'F:\PyCodes'); from GraduationDesign.GDFuncs import funcs3; funcs3('C2', 2)"


E:\Anaconda3\python -c "import sys; sys.path.append(r'F:\PyCodes'); from GraduationDesign.GDFuncs import funcs3; funcs3('HA', 1)"
E:\Anaconda3\python -c "import sys; sys.path.append(r'F:\PyCodes'); from GraduationDesign.GDFuncs import funcs3; funcs3('HA', 2)"


"""
