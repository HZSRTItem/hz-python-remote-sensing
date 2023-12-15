# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowDraw.py
@Time    : 2023/6/28 15:30
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : BaseCodes of Draw
-----------------------------------------------------------------------------"""
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from SRTCodes.GDALRasterIO import samplingGDALRastersToCSV, GDALRasterChannel, getGDALRasterNames
from SRTCodes.NumpyUtils import filterEq
from SRTCodes.SRTDraw import SRTDrawHist
from SRTCodes.SRTSample import CSVSamples
from SRTCodes.Utils import printList, printDict, readJson, angleToRadian, radianToAngle, changext
from Shadow.DeepLearning.SHDLData import SHDLDataSampleCollection, SHDLDataSample

science_style = r"F:\ProjectSet\PycharmEnvs\BaseCodes\scienceplots\styles\science.mplstyle"


class SampleDraw:

    def __init__(self):
        self.draw_categorys = []
        self.draw_features = []
        self.draw_tags = []
        self._c_colors = {}

        self.sample = CSVSamples()
        self.c_names = []

        plt.rcParams['ytick.direction'] = 'in'
        plt.rc('font', family='Times New Roman')

    def setSample(self, sample: CSVSamples):
        self.sample = sample
        self.c_names = sample.getCategoryNames()

    def addCategorys(self, *categorys):
        self.draw_categorys.extend(list(categorys))

    def addFeatures(self, *features):
        self.draw_features.extend(list(features))

    def addTags(self, *tags):
        self.draw_tags.extend(list(tags))

    def colors(self, *args):
        self._c_colors[self.c_names[0]] = "black"
        for i, color_0 in enumerate(args):
            self._c_colors[self.c_names[i + 1]] = color_0

    def print(self):
        print("-" * 70)
        self.sample.print()
        print("-" * 70)
        printDict("Colors", self._c_colors)
        printList("DRAW Categorys:", self.draw_categorys)
        printList("DRAW Features:", self.draw_features)
        printList("DRAW Tags:", self.draw_tags)
        print("-" * 70)

    def fit(self, *args, **kwargs):
        return None

    def end(self, *args, **kwargs):
        return None


class SampleDrawBox(SampleDraw):

    def __init__(self):
        super(SampleDrawBox, self).__init__()
        self.csv_fn = None

    def fit(self, patch_artist=True, axvline_ratio=0.0):
        n_cate = len(self.draw_categorys)
        n_feat = len(self.draw_features)

        fig, ax = plt.subplots()
        fig.subplots_adjust(top=0.88, bottom=0.11, left=0.085, right=0.835, hspace=0.2, wspace=0.2)

        boxplots_ = []

        for i in range(n_cate):
            draw_category = self.draw_categorys[i]
            d, y = self.sample.get(c_names=[draw_category], feat_names=self.draw_features, tags=self.draw_tags)
            positions = [j + i for j in range(1, n_feat * n_cate, n_cate)]
            plot_1 = ax.boxplot(
                d.values,
                # labels=self.names,
                positions=positions,
                patch_artist=patch_artist,
                showmeans=True,
                showfliers=False,
                meanprops={"color": "white"},
                medianprops={"color": "black", "linewidth": 1},
                boxprops={"facecolor": self._c_colors[draw_category], "edgecolor": "black", "linewidth": 1},
                whiskerprops={"color": "black", "linewidth": 1},
                capprops={"color": "black", "linewidth": 1},
            )
            boxplots_.append(plot_1)

        plt.legend([test1["boxes"][0] for test1 in boxplots_], self.draw_categorys,
                   bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0)

        plt.xticks([j + n_cate / 2 for j in range(1, n_feat * n_cate, n_cate)], self.draw_features)
        plt.xlim([0.5, n_feat * n_cate + 0.5])
        for i in range(n_cate, n_feat * n_cate, n_cate):
            plt.axvline(i + 0.5, axvline_ratio, 1 - axvline_ratio, color="black",
                        linestyle="--", linewidth=1)
        plt.grid(True, axis="y", color="lightgrey", linestyle="-")

        title = "{0} of box".format(" ".join(self.draw_categorys))
        print(title)
        plt.title(title)


class SampleDrawScatter(SampleDraw):

    def __init__(self):
        super(SampleDrawScatter, self).__init__()

    def fit(self, *args, **kwargs):
        # plt.figure(figsize=(3,3))
        x_feat_name, y_feat_name = self.draw_features[0], self.draw_features[1]
        for i, category in enumerate(self.draw_categorys):
            spl, y0 = self.sample.get(c_names=[category], feat_names=[x_feat_name, y_feat_name], tags=self.draw_tags)
            x, y = spl[x_feat_name], spl[y_feat_name]
            plt.scatter(x, y, c=self._c_colors[category], label=category, s=3)

        plt.legend()
        plt.xlabel(x_feat_name)
        plt.ylabel(y_feat_name)


def cal_10log10(x):
    return 10 * np.log10(x + 0.0001)


class ShadowSampleNumberDraw:

    def __init__(self, csv_fn):
        self.data: pd.DataFrame = pd.read_csv(csv_fn)

        # MOD_TYPE	SPL_TYPE	FEAT_TYPE	SPL_NUMBER
        self.mod_types = self._getColumnUnique("MOD_TYPE")
        self.spl_types = self._getColumnUnique("SPL_TYPE")
        self.feat_types = self._getColumnUnique("FEAT_TYPE")
        self.spl_number = self._getColumnUnique("SPL_NUMBER")
        self.col_names = ['OATrain', 'KappaTrain', 'IS UATrain', 'IS PATrain',
                          'VEG UATrain', 'VEG PATrain', 'SOIL UATrain', 'SOIL PATrain',
                          'WAT UATrain', 'WAT PATrain', 'OATest', 'KappaTest', 'IS UATest',
                          'IS PATest', 'VEG UATest', 'VEG PATest', 'SOIL UATest', 'SOIL PATest',
                          'WAT UATest', 'WAT PATest', ]

        self.show_mod_type = None
        self.show_spl_type = None
        self.show_feat_type = None
        self.show_column_name = None

    def _getColumnUnique(self, column_name) -> list:
        d = self.data[column_name].values
        return np.unique(d).tolist()

    def modelType(self, model_type=None):
        self.show_mod_type = self._showType(self.mod_types, model_type)

    def sampleType(self, spl_type=None):
        self.show_spl_type = self._showType(self.spl_types, spl_type)

    def featureType(self, feat_type=None):
        self.show_feat_type = self._showType(self.feat_types, feat_type)

    def columnName(self, column_name):
        self.show_column_name = self._showType(self.col_names, column_name)

    def plot(self, mod_type=None, spl_type=None, feat_type=None, column_name=None, name=None, **kwargs):
        if mod_type is not None:
            self.modelType(mod_type)
        if spl_type is not None:
            self.sampleType(spl_type)
        if feat_type is not None:
            self.featureType(feat_type)
        if column_name is not None:
            self.columnName(column_name)
        if name is None:
            name = " ".join([self.show_spl_type, self.show_feat_type, self.show_mod_type, self.show_column_name])

        df = self.data.copy()
        df = filterEq(df, "MOD_TYPE", self.show_mod_type)
        df = filterEq(df, "SPL_TYPE", self.show_spl_type)
        df = filterEq(df, "FEAT_TYPE", self.show_feat_type)
        plt.plot(df["SPL_NUMBER"], df[self.show_column_name], label=name, **kwargs)

        print()
        print(">>> SHOW ---------------------------------")
        print("  + SHOW MOD_TYPE: ", self.show_mod_type)
        print("  + SHOW SPL_TYPE: ", self.show_spl_type)
        print("  + SHOW FEAT_TYPE: ", self.show_feat_type)
        print("  + SHOW COL_NAMES: ", self.show_column_name)
        print()

    @staticmethod
    def _showType(d, k):
        if len(d) == 0:
            return None
        if k is None:
            return d[0]
        else:
            if isinstance(k, int):
                return d[k]
            elif isinstance(k, str):
                if k in d:
                    return k
                else:
                    return None

    def print(self):
        printList("* MOD_TYPE: ", self.mod_types)
        printList("* SPL_TYPE: ", self.spl_types)
        printList("* FEAT_TYPE: ", self.feat_types)
        printList("* SPL_NUMBER: ", self.spl_number)
        printList("* COL_NAMES: ", self.col_names)


class ShadowDrawDirectLength:

    def __init__(self, geojson_fn, H=10.0):
        # 'MEAN_SOLAR_AZIMUTH_ANGLE' 平均太阳方位角
        # 'MEAN_SOLAR_ZENITH_ANGLE'  平均太阳天顶角
        self.d = readJson(geojson_fn)
        self.H = H

    def _cal(self):
        s, a, x, y, times = [], [], [], [], []
        for feat in self.d["features"]:
            azimuth = feat["properties"]["MEAN_SOLAR_AZIMUTH_ANGLE"]
            zenith = feat["properties"]["MEAN_SOLAR_ZENITH_ANGLE"]
            name = feat["id"]
            s0 = self.H * math.tan(angleToRadian(zenith))
            a0 = angleToRadian(azimuth + 180)
            s.append(s0)
            a.append(radianToAngle(a0))
            x.append(s0 * math.sin(a0))
            y.append(s0 * math.cos(a0))
            times.append(datetime.strptime(name[:8], "%Y%m%d"))
        return np.array(s), np.array(a), np.array(x), np.array(y), np.array(times)

    def plotXY(self, time_min=None, time_max=None):
        s, a, x, y, times = self._cal()
        time_min, time_max, filter_list = self.filterTime(times, time_min, time_max)
        plt.plot(x[filter_list], y[filter_list],
                 label="XY {0} {1}".format(time_min.strftime("%Y-%m-%d"), time_max.strftime("%Y-%m-%d")))

    def plotS(self, time_min=None, time_max=None):
        s, a, x, y, times = self._cal()
        time_min, time_max, filter_list = self.filterTime(times, time_min, time_max)
        plt.plot(times[filter_list], s[filter_list],
                 label="A {0} {1}".format(time_min.strftime("%Y-%m-%d"), time_max.strftime("%Y-%m-%d")))

    def plotAlpha(self, time_min=None, time_max=None):
        s, a, x, y, times = self._cal()
        time_min, time_max, filter_list = self.filterTime(times, time_min, time_max)
        plt.plot(times[filter_list], a[filter_list],
                 label="Alpha {0} {1}".format(time_min.strftime("%Y-%m-%d"), time_max.strftime("%Y-%m-%d")))

    def filterTime(self, times, time_min=None, time_max=None):
        if time_min is None:
            time_min = np.min(times)
        if time_max is None:
            time_max = np.min(times)
        if isinstance(time_min, str):
            time_min = datetime.strptime(time_min, "%Y-%m-%d")
        if isinstance(time_max, str):
            time_max = datetime.strptime(time_max, "%Y-%m-%d")
        filter_list = np.array([True for _ in range(len(times))])
        for i in range(len(times)):
            filter_list[i] = time_min < times[i] < time_max
        return time_min, time_max, filter_list


def drawShadowBox():
    csv_fn = r"F:\ProjectSet\Shadow\QingDao\Sample\qd_shadow_spl3_s1.csv"
    csv_spl = CSVSamples(csv_fn)

    csv_spl.fieldNameCategory("CNAME")
    csv_spl.fieldNameTag("TAG")
    csv_spl.addCategoryNames(["IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH"])
    csv_spl.readData()

    csv_spl.featureCallBack("AS_VV", cal_10log10)
    csv_spl.featureCallBack("AS_VH", cal_10log10)
    csv_spl.featureCallBack("AS_C11", cal_10log10)
    csv_spl.featureCallBack("AS_C12_imag", cal_10log10)
    csv_spl.featureCallBack("AS_C12_real", cal_10log10)
    csv_spl.featureCallBack("AS_C22", cal_10log10)
    csv_spl.featureCallBack("AS_Lambda1", cal_10log10)
    csv_spl.featureCallBack("AS_Lambda2", cal_10log10)
    csv_spl.featureCallBack("DE_VV", cal_10log10)
    csv_spl.featureCallBack("DE_VH", cal_10log10)
    csv_spl.featureCallBack("DE_C11", cal_10log10)
    csv_spl.featureCallBack("DE_C12_imag", cal_10log10)
    csv_spl.featureCallBack("DE_C12_real", cal_10log10)
    csv_spl.featureCallBack("DE_C22", cal_10log10)
    csv_spl.featureCallBack("DE_Lambda1", cal_10log10)
    csv_spl.featureCallBack("DE_Lambda2", cal_10log10)

    spl_draw_box = SampleDrawBox()
    spl_draw_box.setSample(csv_spl)

    spl_draw_box.colors("red", "darkred", "lightgreen", "darkgreen", "yellow", "y", "lightblue", "darkblue")

    # "NOT_KNOW", "IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH",
    # "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH"
    # "IS", "VEG", "SOIL", "WAT",
    spl_draw_box.addCategorys("IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH", )

    # "X", "Y", "CATEGORY", "SRT", "TEST", "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
    # "AS_VV", "AS_VH", "AS_C11","AS_C22", "AS_Lambda1", "AS_Lambda2",
    #  "AS_C12_imag", "AS_C12_real",
    # "DE_VV", "DE_VH", "DE_C11",  "DE_C22", "DE_Lambda1", "DE_Lambda2",
    # "DE_C12_imag", "DE_C12_real",
    # "angle_AS", "angle_DE",
    spl_draw_box.addFeatures("AS_C11", "AS_C22")

    # "STATIFY_NOSH", "SEA", "SELECT", "STATIFY_SH", "GE_MAP_RAND", "NIR_SH",
    spl_draw_box.addTags("STATIFY_NOSH", "SEA", "SELECT", "STATIFY_SH", "GE_MAP_RAND", "NIR_SH")

    spl_draw_box.print()

    spl_draw_box.fit()

    # plt.ylim([-1, 1])
    # plt.savefig()
    plt.show()


class SampleDrawMain:

    def __init__(self):
        self.csv_fn = None
        self.csv_spl = CSVSamples()

    def initCSVSamples(self, csv_fn=None):
        if csv_fn is None:
            csv_fn = self.csv_fn
        self.csv_spl = CSVSamples(csv_fn)
        self.csv_spl.fieldNameCategory("CNAME")
        self.csv_spl.fieldNameTag("TAG")
        self.csv_spl.addCategoryNames(["IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH"])
        self.csv_spl.readData()

    def drawBox(self):
        sdb = SampleDrawBox()
        sdb.setSample(self.csv_spl)

        sdb.colors("red", "darkred", "lightgreen", "darkgreen", "yellow", "y", "lightblue", "darkblue")

        # "NOT_KNOW", "IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH",
        # sdb.addCategorys(
        #     "IS", "IS_SH",
        #     "VEG", "VEG_SH",
        #     "SOIL", "SOIL_SH",
        #     "WAT", "WAT_SH"
        # )
        sdb.addCategorys(
            # "IS", "VEG", "SOIL", "WAT",
            "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH",
        )

        # "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
        #   "OPT_asm", "OPT_con", "OPT_cor", "OPT_dis", "OPT_ent", "OPT_hom", "OPT_mean", "OPT_var",
        # "AS_VV", "AS_VH", "AS_VHDVV", "AS_C11", "AS_C12_imag", "AS_C12_real", "AS_C22", "AS_Lambda1", "AS_Lambda2",
        #   "AS_SPAN", "AS_Epsilon", "AS_Mu", "AS_RVI", "AS_m", "AS_Beta",
        #   "AS_VV_asm", "AS_VV_con", "AS_VV_cor", "AS_VV_dis", "AS_VV_ent", "AS_VV_hom", "AS_VV_mean", "AS_VV_var",
        #   "AS_VH_asm", "AS_VH_con", "AS_VH_cor", "AS_VH_dis", "AS_VH_ent", "AS_VH_hom", "AS_VH_mean", "AS_VH_var",
        # "DE_VV", "DE_VH", "DE_VHDVV", "DE_C11", "DE_C12_imag", "DE_C12_real", "DE_C22", "DE_Lambda1", "DE_Lambda2",
        #   "DE_SPAN", "DE_Epsilon", "DE_Mu", "DE_RVI", "DE_Beta", "DE_m",
        #   "DE_VH_asm", "DE_VH_con", "DE_VH_cor", "DE_VH_dis", "DE_VH_ent", "DE_VH_hom", "DE_VH_mean", "DE_VH_var",
        #   "DE_VV_asm", "DE_VV_con", "DE_VV_cor", "DE_VV_dis", "DE_VV_ent", "DE_VV_hom", "DE_VV_mean", "DE_VV_var",
        sdb.addFeatures(
            # "AS_VV", "AS_VH", "AS_VHDVV",
            # "AS_C11", "AS_C22",
            "AS_Lambda1", "AS_Lambda2",
            # "AS_SPAN",
            # "AS_Epsilon",
            # "AS_Mu",
            # "AS_RVI",
            # "AS_m",
            # "AS_Beta",

            # "DE_VV", "DE_VH", "DE_VHDVV",
            # "DE_C11", "DE_C22",
            # "DE_Lambda1", "DE_Lambda2",
            # "DE_SPAN",
            # "DE_Epsilon",
            # "DE_Mu",
            # "DE_RVI",
            # "DE_m",
            # "DE_Beta"
        )

        # sdb.addTags("SELECT")

        sdb.print()

        sdb.fit()

        # plt.ylim([-1, 1])
        # plt.savefig()
        plt.show()
        # sdb.fit()

    def drawScatter(self):
        self.csv_spl.featureCallBack("AS_VV", cal_10log10)
        self.csv_spl.featureCallBack("AS_VH", cal_10log10)
        self.csv_spl.featureCallBack("AS_C11", cal_10log10)
        self.csv_spl.featureCallBack("AS_C22", cal_10log10)
        # self.csv_spl.featureCallBack("AS_Lambda1", cal_10log10)
        # self.csv_spl.featureCallBack("AS_Lambda2", cal_10log10)
        self.csv_spl.featureCallBack("DE_VV", cal_10log10)
        self.csv_spl.featureCallBack("DE_VH", cal_10log10)
        self.csv_spl.featureCallBack("DE_C11", cal_10log10)
        self.csv_spl.featureCallBack("DE_C22", cal_10log10)
        self.csv_spl.featureCallBack("DE_Lambda1", cal_10log10)
        self.csv_spl.featureCallBack("DE_Lambda2", cal_10log10)

        self.csv_spl.featureScaleMinMax("AS_Lambda1", 0, 0.25, is_01=False)
        self.csv_spl.featureScaleMinMax("AS_Lambda2", 0, 0.05, is_01=False)

        self.csv_spl.featureScaleMinMax("AS_Epsilon", 0, 100, is_01=False)
        self.csv_spl.featureScaleMinMax("DE_Epsilon", 0, 100, is_01=False)

        sdc = SampleDrawScatter()
        sdc.setSample(self.csv_spl)

        sdc.colors("red", "darkred", "lightgreen", "darkgreen", "yellow", "y", "lightblue", "darkblue")

        # "NOT_KNOW", "IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH",
        # sdb.addCategorys(
        #     "IS", "IS_SH",
        #     "VEG", "VEG_SH",
        #     "SOIL", "SOIL_SH",
        #     "WAT", "WAT_SH"
        # )
        sdc.addCategorys(
            "IS",
            "VEG",
            # "SOIL",
            # "WAT",
            # "IS_SH",
            # "VEG_SH",
            # "SOIL_SH",
            # "WAT_SH",
        )

        # "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
        #   "OPT_asm", "OPT_con", "OPT_cor", "OPT_dis", "OPT_ent", "OPT_hom", "OPT_mean", "OPT_var",
        # "AS_VV", "AS_VH", "AS_VHDVV", "AS_C11", "AS_C12_imag", "AS_C12_real", "AS_C22", "AS_Lambda1", "AS_Lambda2",
        #   "AS_SPAN", "AS_Epsilon", "AS_Mu", "AS_RVI", "AS_m", "AS_Beta",
        #   "AS_VV_asm", "AS_VV_con", "AS_VV_cor", "AS_VV_dis", "AS_VV_ent", "AS_VV_hom", "AS_VV_mean", "AS_VV_var",
        #   "AS_VH_asm", "AS_VH_con", "AS_VH_cor", "AS_VH_dis", "AS_VH_ent", "AS_VH_hom", "AS_VH_mean", "AS_VH_var",
        # "DE_VV", "DE_VH", "DE_VHDVV", "DE_C11", "DE_C12_imag", "DE_C12_real", "DE_C22", "DE_Lambda1", "DE_Lambda2",
        #   "DE_SPAN", "DE_Epsilon", "DE_Mu", "DE_RVI", "DE_Beta", "DE_m",
        #   "DE_VH_asm", "DE_VH_con", "DE_VH_cor", "DE_VH_dis", "DE_VH_ent", "DE_VH_hom", "DE_VH_mean", "DE_VH_var",
        #   "DE_VV_asm", "DE_VV_con", "DE_VV_cor", "DE_VV_dis", "DE_VV_ent", "DE_VV_hom", "DE_VV_mean", "DE_VV_var",
        sdc.addFeatures(
            # "AS_VV", "AS_VH", "AS_VHDVV",
            # "AS_C11", "AS_C22",
            # "AS_Lambda1", "AS_Lambda2",
            # "AS_SPAN",
            "AS_Epsilon",
            # "AS_Mu",
            # "AS_RVI",
            # "AS_m",
            # "AS_Beta",

            # "DE_VV", "DE_VH", "DE_VHDVV",
            # "DE_C11", "DE_C22",
            # "DE_Lambda1", "DE_Lambda2",
            # "DE_SPAN",
            "DE_Epsilon",
            # "DE_Mu",
            # "DE_RVI",
            # "DE_m",
            # "DE_Beta"
        )

        # sdb.addTags("SELECT")

        sdc.print()

        sdc.fit()

        # plt.ylim([-1, 1])
        # plt.savefig()
        plt.show()
        # sdb.fit()

    def excelToCSV(self, excel_fns: dict, to_csv_fn):
        self.csv_fn = to_csv_fn
        ssc = SHDLDataSampleCollection()
        for k, fn in excel_fns.items():
            ssc.addExcel(fn, k)
        dfs = []
        for i in range(len(ssc)):
            spl: SHDLDataSample = ssc.samples[i]
            to_dict = spl.toDict()
            dfs.append(to_dict)
        df = pd.DataFrame(dfs)
        df.to_csv(to_csv_fn, index=False)
        return self.csv_fn

    def sampling(self, geo_fns, csv_fn=None, to_csv_fn=None):
        if csv_fn is None:
            csv_fn = self.csv_fn
        if to_csv_fn is None:
            to_csv_fn = changext(csv_fn, "_spl.csv")
        samplingGDALRastersToCSV(geo_fns, csv_fn, to_csv_fn)

    def funcSampling(self):
        self.excelToCSV({
            "QingDao": r"F:\ProjectSet\Shadow\Release\QingDaoSamples\QingDaoSamples.xlsx",
            "BeiJing": r"F:\ProjectSet\Shadow\Release\BeiJingSamples\BeiJingSamples.xlsx",
            "ChengDu": r"F:\ProjectSet\Shadow\Release\ChengDuSamples\ChengDuSamples.xlsx",
        },
            r"F:\ProjectSet\Shadow\MkTu\4.1Details\Samples\three_spl.csv")
        self.sampling(geo_fns=[r"F:\ProjectSet\Shadow\Release\QingDaoImages\SH_QD_envi.dat",
                               r"F:\ProjectSet\Shadow\Release\BeiJingImages\SH_BJ_envi.dat",
                               r"F:\ProjectSet\Shadow\Release\ChengDuImages\SH_CD_envi.dat"])


def tempFuncs():
    def filter_eq(df, filter_field, data):
        return df[df[filter_field] == data]

    def func1():
        csv_fn = r"F:\ProjectSet\Shadow\Analysis\3\sh_cate_1_tp_spl.csv"
        x_draw_feat, y_draw_feat = "AS_VV", "AS_VH"
        names = ["IS", "VEG", "SOIL", "WAT", ]
        colors = ["red", "lightgreen", "yellow", "lightblue"]
        select_list = [2, 3, 4, 1, ]
        is_10log10 = True
        df = pd.read_csv(csv_fn)
        for i in select_list:
            df_tmp = filter_eq(df, "Category", i)
            if is_10log10:
                df_tmp[x_draw_feat] = np.power(10, df_tmp[x_draw_feat] / 10)
                df_tmp[x_draw_feat] = np.clip(df_tmp[x_draw_feat], 0, 0.6)
                df_tmp[y_draw_feat] = np.power(10, df_tmp[y_draw_feat] / 10)
                df_tmp[y_draw_feat] = np.clip(df_tmp[y_draw_feat], 0, 0.1)
            plt.scatter(df_tmp[x_draw_feat], df_tmp[y_draw_feat], label=names[i - 1], c=colors[i - 1])
        plt.legend()
        plt.xlabel(x_draw_feat)
        plt.ylabel(y_draw_feat)
        plt.show()

    func1()


class SRTDrawHistShadow(SRTDrawHist):

    def __init__(self):
        super(SRTDrawHistShadow, self).__init__()
        self.grc = GDALRasterChannel()
        self.names = []

    def addGDALRaster(self, raster_fn, field_name):
        self.grc.addGDALData(raster_fn, field_name)
        self.add(field_name, self.grc[field_name])

    def featureScaleMinMax(self, field_name, x_min, x_max):
        self._collection[field_name].scaleMinMax(x_min, x_max)

    def featureCallBack(self, field_name, func_callback):
        self._collection[field_name].addCallBack(func_callback)


def main():
    sdhs = SRTDrawHistShadow()
    qd_raster_fn = r"F:\ProjectSet\Shadow\Release\QingDaoImages\SH_QD_envi.dat"
    bj_raster_fn = r"F:\ProjectSet\Shadow\Release\BeiJingImages\SH_BJ_envi.dat"
    cd_raster_fn = r"F:\ProjectSet\Shadow\Release\ChengDuImages\SH_CD_envi.dat"
    # 'SRT', 'X', 'Y', 'CNAME', 'CATEGORY', 'TAG', 'TEST', 'Blue', 'Green', 'Red', 'NIR', 'NDVI', 'NDWI',
    # 'OPT_asm', 'OPT_con', 'OPT_cor', 'OPT_dis', 'OPT_ent', 'OPT_hom', 'OPT_mean', 'OPT_var',
    # 'AS_VV', 'AS_VH', 'AS_VHDVV', 'AS_C11', 'AS_C12_imag', 'AS_C12_real', 'AS_C22', 'AS_Lambda1', 'AS_Lambda2',
    # 'AS_SPAN', 'AS_Epsilon', 'AS_Mu', 'AS_RVI', 'AS_m', 'AS_Beta',
    # 'AS_VH_asm', 'AS_VH_con', 'AS_VH_cor', 'AS_VH_dis', 'AS_VH_ent', 'AS_VH_hom', 'AS_VH_mean', 'AS_VH_var',
    # 'AS_VV_asm', 'AS_VV_con', 'AS_VV_cor', 'AS_VV_dis', 'AS_VV_ent', 'AS_VV_hom', 'AS_VV_mean', 'AS_VV_var',
    # 'DE_VV', 'DE_VH', 'DE_VHDVV', 'DE_C11', 'DE_C12_imag', 'DE_C12_real', 'DE_C22','DE_Lambda1', 'DE_Lambda2',
    # 'DE_SPAN',  'DE_Epsilon', 'DE_Mu', 'DE_RVI', 'DE_m', 'DE_Beta',
    # 'DE_VH_asm', 'DE_VH_con', 'DE_VH_cor', 'DE_VH_dis', 'DE_VH_ent', 'DE_VH_hom', 'DE_VH_mean', 'DE_VH_var',
    # 'DE_VV_asm', 'DE_VV_con', 'DE_VV_cor', 'DE_VV_dis', 'DE_VV_ent', 'DE_VV_hom', 'DE_VV_mean', 'DE_VV_var'
    field_names = getGDALRasterNames(qd_raster_fn)
    for field_name in field_names:
        sdhs.addGDALRaster(qd_raster_fn, field_name)

    sdhs.featureScaleMinMax("Blue", 99.76996, 2397.184)
    sdhs.featureScaleMinMax("Green", 45.83414, 2395.735)
    sdhs.featureScaleMinMax("Red", 77.79654, 2726.7026)
    sdhs.featureScaleMinMax("NIR", 0.66086, 3498.4321)
    sdhs.featureScaleMinMax("NDVI", -0.8007727, 0.8354284)
    sdhs.featureScaleMinMax("NDWI", -0.8572631, 0.8623875)
    sdhs.featureScaleMinMax("OPT_asm", 0.02124183, 0.998366)
    sdhs.featureScaleMinMax("OPT_con", 0.0, 169.74791)
    sdhs.featureScaleMinMax("OPT_cor", -0.036879253, 0.99688625)
    sdhs.featureScaleMinMax("OPT_dis", 0.0, 9.799746)
    sdhs.featureScaleMinMax("OPT_ent", 0.0, 3.8249474)
    sdhs.featureScaleMinMax("OPT_hom", 0.12091503, 0.998366)
    sdhs.featureScaleMinMax("OPT_mean", 4.941177, 53.7353)
    sdhs.featureScaleMinMax("OPT_var", 0.0, 236.09961)

    sdhs.featureCallBack("AS_VV", cal_10log10)
    sdhs.featureCallBack("AS_VH", cal_10log10)
    sdhs.featureCallBack("AS_C11", cal_10log10)
    sdhs.featureCallBack("AS_C22", cal_10log10)
    sdhs.featureCallBack("AS_Lambda1", cal_10log10)
    sdhs.featureCallBack("AS_Lambda2", cal_10log10)
    sdhs.featureCallBack("AS_SPAN", cal_10log10)
    sdhs.featureCallBack("AS_Epsilon", cal_10log10)
    sdhs.featureCallBack("DE_VV", cal_10log10)
    sdhs.featureCallBack("DE_VH", cal_10log10)
    sdhs.featureCallBack("DE_C11", cal_10log10)
    sdhs.featureCallBack("DE_C22", cal_10log10)
    sdhs.featureCallBack("DE_Lambda1", cal_10log10)
    sdhs.featureCallBack("DE_Lambda2", cal_10log10)
    sdhs.featureCallBack("DE_SPAN", cal_10log10)
    sdhs.featureCallBack("DE_Epsilon", cal_10log10)

    sdhs.featureScaleMinMax("AS_VV", -30.609674, 11.9092603)
    sdhs.featureScaleMinMax("AS_VH", -35.865038, 1.2615275)
    sdhs.featureScaleMinMax("AS_VHDVV", 0.0, 1.5)
    sdhs.featureScaleMinMax("AS_C11", -28.61998, 11.8634768)
    sdhs.featureScaleMinMax("AS_C22", -30.579813, 1.2111626)
    sdhs.featureScaleMinMax("AS_Lambda1", -26.955856, 11.124724)
    sdhs.featureScaleMinMax("AS_Lambda2", -31.869734, 1.284683)
    sdhs.featureScaleMinMax("AS_SPAN", -22.58362, 6.97997)
    sdhs.featureScaleMinMax("AS_Epsilon", 0.0, 35.12922)
    sdhs.featureScaleMinMax("AS_Mu", -0.7263123, 0.7037629)
    sdhs.featureScaleMinMax("AS_RVI", 0.07459847, 2.076324)
    sdhs.featureScaleMinMax("AS_m", 0.26469338, 0.97544414)
    sdhs.featureScaleMinMax("AS_Beta", 0.632338, 0.9869048)

    sdhs.featureScaleMinMax("AS_VH_asm", 0.02124183, 0.050653595)
    sdhs.featureScaleMinMax("AS_VH_con", 6.572378, 59.151405)
    sdhs.featureScaleMinMax("AS_VH_cor", 0.006340516, 0.86876196)
    sdhs.featureScaleMinMax("AS_VH_dis", 1.9767247, 5.8193297)
    sdhs.featureScaleMinMax("AS_VH_ent", 3.0939856, 3.8060431)
    sdhs.featureScaleMinMax("AS_VH_hom", 0.16666667, 0.40849674)
    sdhs.featureScaleMinMax("AS_VH_mean", 7.514706, 54.04412)
    sdhs.featureScaleMinMax("AS_VH_var", 5.9986033, 108.64137)
    sdhs.featureScaleMinMax("AS_VV_asm", 0.022875817, 0.050653595)
    sdhs.featureScaleMinMax("AS_VV_con", 4.5305123, 48.325462)
    sdhs.featureScaleMinMax("AS_VV_cor", 0.21234758, 0.88228023)
    sdhs.featureScaleMinMax("AS_VV_dis", 1.5990733, 5.22229)
    sdhs.featureScaleMinMax("AS_VV_ent", 3.1254923, 3.7871387)
    sdhs.featureScaleMinMax("AS_VV_hom", 0.18464053, 0.45261437)
    sdhs.featureScaleMinMax("AS_VV_mean", 8.544118, 51.573532)
    sdhs.featureScaleMinMax("AS_VV_var", 3.8744159, 96.8604)

    sdhs.featureScaleMinMax("DE_VV", -27.851603, 5.094706)
    sdhs.featureScaleMinMax("DE_VH", -35.427082, -5.4092093)
    sdhs.featureScaleMinMax("DE_VHDVV", 0.0, 1.0289364)
    sdhs.featureScaleMinMax("DE_C11", -26.245598, 4.9907513)
    sdhs.featureScaleMinMax("DE_C22", -32.042320, -5.322515)
    sdhs.featureScaleMinMax("DE_Lambda1", -25.503738, 5.2980003)
    sdhs.featureScaleMinMax("DE_Lambda2", -33.442368, -8.68537)
    sdhs.featureScaleMinMax("DE_SPAN", -24.81076, 4.82663)
    sdhs.featureScaleMinMax("DE_Epsilon", 0.0, 21.882689)
    sdhs.featureScaleMinMax("DE_Mu", -0.6823329, 0.7723537)
    sdhs.featureScaleMinMax("DE_RVI", 0.0940072, 2.1935015)
    sdhs.featureScaleMinMax("DE_m", 0.24836189, 0.9705721)
    sdhs.featureScaleMinMax("DE_Beta", 0.6241778, 0.9852859)

    sdhs.featureScaleMinMax("DE_VH_asm", 0.022875817, 0.05392157)
    sdhs.featureScaleMinMax("DE_VH_con", 5.6798058, 51.11825)
    sdhs.featureScaleMinMax("DE_VH_cor", 0.12444292, 0.87177193)
    sdhs.featureScaleMinMax("DE_VH_dis", 1.8186697, 5.456009)
    sdhs.featureScaleMinMax("DE_VH_ent", 2.9679575, 3.7997417)
    sdhs.featureScaleMinMax("DE_VH_hom", 0.1748366, 0.42810458)
    sdhs.featureScaleMinMax("DE_VH_mean", 7.6176476, 55.176476)
    sdhs.featureScaleMinMax("DE_VH_var", 5.513511, 95.38374)
    sdhs.featureScaleMinMax("DE_VV_asm", 0.02124183, 0.057189543)
    sdhs.featureScaleMinMax("DE_VV_con", 5.0987973, 57.54357)
    sdhs.featureScaleMinMax("DE_VV_cor", 0.19514601, 0.88254523)
    sdhs.featureScaleMinMax("DE_VV_dis", 1.7117102, 5.6928787)
    sdhs.featureScaleMinMax("DE_VV_ent", 2.993163, 3.7997417)
    sdhs.featureScaleMinMax("DE_VV_hom", 0.17320262, 0.44444445)
    sdhs.featureScaleMinMax("DE_VV_mean", 6.4852943, 54.04412)
    sdhs.featureScaleMinMax("DE_VV_var", 4.44714, 111.17851)

    for i, field_name in enumerate (field_names):
        sdhs.plot(field_name)
        image_fn = os.path.join(r"F:\Week\20231217\Data\qd", "{0:03d}_{1}.jpg".format(i+1, field_name))
        print(image_fn)
        plt.legend()
        plt.savefig(image_fn, format="jpeg")
        plt.close()


def method_name():
    sdm = SampleDrawMain()
    sdm.initCSVSamples(r"F:\ProjectSet\Shadow\MkTu\4.1Details\Samples\three_spl_spl.csv")
    # sdm.drawBox()
    sdm.drawScatter()


if __name__ == '__main__':
    main()
    # tempFuncs()
