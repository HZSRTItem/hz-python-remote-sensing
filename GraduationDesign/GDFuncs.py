# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : GDFuncs.py
@Time    : 2024/12/12 21:58
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of ZH
-----------------------------------------------------------------------------"""
import random

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from tabulate import tabulate

from SRTCodes.SampleUtils import SamplesUtil
from SRTCodes.Utils import readJson, printList, DirFileName


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


def funcs3():
    def func1():
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
            "OPT_mean", "OPT_var", "OPT_asm", "OPT_dis", "OPT_hom", "OPT_con", "OPT_cor", "OPT_ent",
        ])

        sar_names = to_01([
            "AS_VV", "AS_VH", "AS_C11", "AS_C22", "AS_H", "AS_Alpha",
            "DE_VV", "DE_VH", "DE_C11", "DE_C22", "DE_H", "DE_Alpha",
        ])

        def _get_data_1(_cname, _fields=None):
            _df = df[df["CNAME"] == _cname]
            if _fields is not None:
                return _df[_fields]
            else:
                return _df

        def draw1(_cname, _fields, *_args, _x_names=None, **_kwargs):
            if _x_names is None:
                _x_names = _fields
            _df = _get_data_1(_cname, _fields)
            _df = _df.mean()

            plt.plot(_x_names, _df.values, *_args, **_kwargs)

        # plt.rc('font', family='Times New Roman')
        # plt.rcParams.update({'font.size': 16})

        font_SimHei = FontProperties(family="SimHei")

        dfn = DirFileName(r"F:\GraduationDesign\Images\3")

        def draw1_im1(_fn, _draw1_data, ):

            plt.figure(figsize=(8, 5))
            plt.subplots_adjust(top=0.97, bottom=0.21, left=0.081, right=0.981, hspace=0.2, wspace=0.2)

            for _data in _draw1_data:
                draw1(*_data[0], **_data[1])

            plt.xlabel("特征", fontproperties=font_SimHei)
            plt.ylabel("特征均值", fontproperties=font_SimHei)
            plt.legend(prop=font_SimHei)

            plt.xticks(rotation=45)
            if _fn is not None:
                plt.savefig(dfn.fn("{}.jpg".format(_fn)), dpi=300)
            plt.show()

        def _func_draw1_data(*_args, **_kwargs):
            return [_args, _kwargs]

        def draw1_im2(_fn, _draw2_data, ):

            n = len(_draw2_data)

            for i, _data2 in enumerate(_draw2_data):
                plt.subplot(n * 100 + 10 + (i + 1))
                for _data in _data2:
                    draw1(*_data[0], **_data[1])
                    plt.xlabel("特征", fontproperties=font_SimHei)
                    plt.ylabel("特征均值", fontproperties=font_SimHei)
                    plt.legend(prop=font_SimHei, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0, frameon=False)
                    plt.xticks(rotation=45)
                plt.text(0.02, 0.85, "({})".format(i+1),fontsize=14, transform=plt.gca().transAxes)

            if _fn is not None:
                plt.savefig(dfn.fn("{}.jpg".format(_fn)), dpi=300)

        names = sar_names

        is_draw1_data = _func_draw1_data("IS", names, "*-", label="不透水面", color="red", markersize=6)
        is_sh_draw1_data = _func_draw1_data("IS_SH", names, "*-", label="光学阴影不透水面", color="darkred",
                                            markersize=6)
        is_as_sh_draw1_data = _func_draw1_data("IS_AS_SH", names, "*-", label="升轨SAR阴影不透水面", color="#FFC1C1",
                                               markersize=6)
        is_de_sh_draw1_data = _func_draw1_data("IS_DE_SH", names, "*-", label="降轨SAR阴影不透水面", color="#8B6969",
                                               markersize=6)

        veg_draw1_data = _func_draw1_data("VEG", names, "^-", label="植被", color="lime")
        veg_sh_draw1_data = _func_draw1_data("VEG_SH", names, "^-", label="光学阴影植被", color="darkgreen")
        veg_as_sh_draw1_data = _func_draw1_data("VEG_AS_SH", names, "^-", label="升轨SAR阴影植被", color="#008B45")
        veg_de_sh_draw1_data = _func_draw1_data("VEG_DE_SH", names, "^-", label="降轨SAR阴影植被", color="#C0FF3E")

        soil_draw1_data = _func_draw1_data("SOIL", names, "s-", label="裸土", color="#FFD39B")
        soil_sh_draw1_data = _func_draw1_data("SOIL_SH", names, "s-", label="光学阴影裸土", color="peru")
        soil_as_sh_draw1_data = _func_draw1_data("SOIL_AS_SH", names, "s-", label="升轨SAR阴影裸土", color="#FFEC8B")
        soil_de_sh_draw1_data = _func_draw1_data("SOIL_DE_SH", names, "s-", label="降轨SAR阴影裸土", color="#8B8B00")

        wat_draw1_data = _func_draw1_data("WAT", names, "o-", label="水体", color="lightblue")
        wat_sh_draw1_data = _func_draw1_data("WAT_SH", names, "o-", label="光学阴影水体", color="blue")
        wat_as_sh_draw1_data = _func_draw1_data("WAT_AS_SH", names, "o-", label="升轨SAR阴影水体", color="#00BFFF")
        wat_de_sh_draw1_data = _func_draw1_data("WAT_DE_SH", names, "o-", label="降轨SAR阴影水体", color="#5F9EA0")

        # draw_im1("im41", [is_draw1_data, is_sh_draw1_data])
        # draw_im1("im42", [is_draw1_data, veg_draw1_data, veg_sh_draw1_data])
        # draw_im1("im43", [is_draw1_data, wat_draw1_data, wat_sh_draw1_data])
        # draw_im1("im44", [is_draw1_data, soil_draw1_data, soil_sh_draw1_data])
        # draw_im1("im45", [is_sh_draw1_data, veg_draw1_data, veg_sh_draw1_data])
        # draw_im1("im46", [is_sh_draw1_data, wat_draw1_data, wat_sh_draw1_data])
        # draw_im1("im47", [is_sh_draw1_data, soil_draw1_data, soil_sh_draw1_data])

        plt.figure(figsize=(8, 9.5))
        plt.subplots_adjust(top=0.984, bottom=0.111, left=0.081, right=0.781, hspace=0.51, wspace=0.2)
        draw1_im2(
            "im31",
            [[is_draw1_data, is_sh_draw1_data, veg_draw1_data, veg_sh_draw1_data],
             [is_draw1_data, is_sh_draw1_data, wat_draw1_data, wat_sh_draw1_data],
             [is_draw1_data, is_sh_draw1_data, soil_draw1_data, soil_sh_draw1_data]]
        )
        plt.show()

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

    return func1()


if __name__ == "__main__":
    funcs3()
