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

from SRTCodes.GDALRasterIO import GDALRaster
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

    return func1()


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
        gmr.read(channels, geo_region=self.geo_region, raster_region=self.raster_region,
                 min_list=self.min_list, max_list=self.max_list, color_table=self.color_table,
                 _func=self._func)

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
        FONT_SIZE = 16
        plt.rcParams.update({'font.size': FONT_SIZE})
        plt.rcParams['font.family'] = ["Times New Roman", 'SimSun', ] + plt.rcParams['font.family']
        plt.rcParams['mathtext.fontset'] = 'stix'

        gsu = _FUNCS3_SamplesUtil()
        gsu.addCSV(r"F:\GraduationDesign\Result\run\Samples\Funcs3\HSPL_BJ_select.csv")
        gsu.addCSV(r"F:\GraduationDesign\Result\run\Samples\Funcs3\HSPL_CD_select.csv")
        gsu.addCSV(r"F:\GraduationDesign\Result\run\Samples\Funcs3\HSPL_QD_select.csv")

        df = gsu.toDF()
        print(df.value_counts("CNAME"))
        printList("DF List", list(df.keys()))

        def to_01(_names):
            for name in _names:
                df[name] = (df[name] - df[name].mean()) / df[name].std()
                df[name] = (df[name] - df[name].min()) / (df[name].max() - df[name].min())
            return _names

        opt_names = to_01([
            "NDVI", "NDWI", "Blue", "Green", "Red", "NIR",
            "OPT_mean", "OPT_cor", "OPT_ent", "OPT_dis", "OPT_hom", "OPT_con", "OPT_var", "OPT_asm",
        ])
        opt_show_names = [
            '$ NDVI $', '$ NDWI $', '$ Blue $', '$ Green $', '$ Red $', '$ NIR $', '$ OPT_{mean} $',
            '$ OPT_{cor} $', '$ OPT_{ent} $', '$ OPT_{dis} $', '$ OPT_{hom} $', '$ OPT_{con} $',
            '$ OPT_{var} $', '$ OPT_{asm} $',
        ]

        sar_names = to_01([
            "AS_VV", "AS_VH", "AS_C11", "AS_C22", "AS_H", "AS_Alpha",
            "DE_VV", "DE_VH", "DE_C11", "DE_C22", "DE_H", "DE_Alpha",
        ])
        sar_names_as = ["AS_VV", "AS_VH", "AS_C11", "AS_C22", "AS_H", "AS_Alpha", ]
        sar_names_de = ["DE_VV", "DE_VH", "DE_C11", "DE_C22", "DE_H", "DE_Alpha", ]

        sar_show_names = [
            "$ VV $", "$ VH $", "$ C_{11} $", "$ C_{22} $", "$ H $", r"$ \alpha $",
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

            for _data in _draw1_data:
                draw1(*_data[0], **_data[1])

            plt.xlabel("特征", )
            plt.ylabel("特征均值", )

            plt.ylim([None, 1])
            plt.legend()

            plt.xticks(rotation=0)
            if _fn is not None:
                to_fn = dfn.fn("{}.jpg".format(_fn))
                print(to_fn)
                plt.savefig(to_fn, dpi=300, bbox_inches='tight', )

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

        names = sar_names
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
        # draw1_im1("opt_is", [is_draw1_data, is_sh_draw1_data, ])
        # plt.clf()
        # draw1_im1("opt_veg", [veg_draw1_data, veg_sh_draw1_data, is_draw1_data, is_sh_draw1_data, ])
        # plt.clf()
        # draw1_im1("opt_soil", [soil_draw1_data, soil_sh_draw1_data, is_draw1_data, is_sh_draw1_data, ])
        # plt.clf()
        # draw1_im1("opt_wat", [wat_draw1_data, wat_sh_draw1_data, is_draw1_data, is_sh_draw1_data, ])
        # plt.clf()

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
        sigma_names1 = ["$VV$", "$VH$"]
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

    def func8():

        def _func_1(_name, _fn):
            _gmr = GD_GMRaster(
                "qd", raster_fn=_fn,
                color_table={1: (200, 0, 0), 2: (255, 255, 255), 3: (255, 255, 255), 4: (255, 255, 255)}
            )
            _gmr.draw(
                [1],
                x_major_len=(0, 6, 0), y_major_len=(0, 6, 0),
                x_minor_len=(0, 0, 20), y_minor_len=(0, 0, 20),
            )
            to_fn = r"F:\GraduationDesign\MkTu\Images\images\{}_imdc1.jpg".format(_name)
            print(to_fn)
            plt.savefig(to_fn, bbox_inches='tight', dpi=300)
            plt.show()

        _func_1("qd", r"F:\ASDEWrite\Result\QingDao\qd_SAR-Opt-AS-DE_imdc.tif")
        _func_1("bj", r"F:\ASDEWrite\Result\BeiJing\bj_SAR-Opt-AS-DE_imdc.tif")
        _func_1("cd", r"F:\ASDEWrite\Result\ChengDu\cd_SAR-Opt-AS-DE_imdc.tif")

    if len(args) != 0:
        return func6(*args)
    else:
        return func8()


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
