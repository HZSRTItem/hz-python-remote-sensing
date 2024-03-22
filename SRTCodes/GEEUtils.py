# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : GEEUtils.py
@Time    : 2023/9/1 11:15
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of GEEUtils
-----------------------------------------------------------------------------"""
import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axisartist import AxesZero

from SRTCodes.Utils import readJson

plt.rc('font', family='Times New Roman')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 #有中文出现的情况，需要u'内容'
config = {
    "font.family": 'serif',
    "font.size": 12,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
plt.rcParams.update(config)


class GEEProperty:
    """
    manage collection features property of loading from gee
    Contact: tourensong@gmail.com
    (C)Copyright 2023, ZhengHan. All rights reserved.
    """

    def __init__(self, csv_fn=None):
        self.csv_fn = None
        self.df = None
        self.initFromCsvFN(csv_fn)
        self.start_time_t = None
        self.end_time_t = None

    def initFromCsvFN(self, csv_fn):
        if csv_fn is not None:
            self.csv_fn = csv_fn
            self.df = pd.read_csv(csv_fn)
        return self

    def extractTime(self):
        if "TIME" not in self.df:
            print("Warning: not extract time. Please override function `extractTime`")
        return self

    def filterDate(self, start_time=None, end_time=None):
        if start_time is None:
            start_time = str(self.df["TIME"].min())
        if end_time is None:
            end_time = str(self.df["TIME"].max())
        start_time_t = pd.to_datetime(start_time)
        end_time_t = pd.to_datetime(end_time)
        self.start_time_t, self.end_time_t = start_time_t, end_time_t
        df_select = (start_time_t < self.df["TIME"]) & (self.df["TIME"] < end_time_t)
        self.df = self.df[df_select]
        return self


def plt_show(title, x_label=None, y_label=None, h_setxticks=None):
    # "Time" "Angle(°)"
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if h_setxticks is not None:
        hSetXticks(h_setxticks)
    # plt.legend(
    #     bbox_to_anchor=(0, 0.1), loc=3, borderaxespad=0
    # )
    # plt.subplots_adjust(right=0.7)
    plt.title(title)
    plt.show()
    pass


def plt_fig(fig_style="double_row"):
    if fig_style == "double_row":
        fig = plt.figure()
        ax = fig.add_subplot(axes_class=AxesZero)
        for direction in ["xzero", "yzero"]:
            # adds arrows at the ends of each axis
            ax.axis[direction].set_axisline_style("->")
            # adds X and Y-axis from the origin
            ax.axis[direction].set_visible(True)
        for direction in ["left", "right", "bottom", "top"]:
            # hides borders
            ax.axis[direction].set_visible(False)
        ax = plt.gca()
        ax.spines['right'].set_color('none')  # 设置上边和右边无边框
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')  # 设置x坐标刻度数字或名称的位置
        ax.spines['bottom'].set_position(('data', 0))  # 设置边框位置
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('data', 0))


class GEEImageProperty(GEEProperty):
    """
    manage gee image collection features property of loading from gee
    Contact: tourensong@gmail.com
    (C)Copyright 2023, ZhengHan. All rights reserved.
    """

    def __init__(self, csv_fn=None):
        super(GEEImageProperty, self).__init__(csv_fn)

    def extractTime(self):
        df_product_id_split = self.df.PRODUCT_ID.str.split("_", expand=True)
        self.df["TIME"] = pd.to_datetime(df_product_id_split[2], format="%Y%m%dT%H%M%S")
        self.start_time_t = str(self.df["TIME"].min())
        self.end_time_t = str(self.df["TIME"].max())
        return self

    def plotAzimuth(self, legend_label: str):
        plt.plot(self.df.index, self.df["MEAN_SOLAR_AZIMUTH_ANGLE"] + 180, label=legend_label)

    def plotZenith(self, legend_label: str):
        plt.plot(self.df.index, self.df["MEAN_SOLAR_ZENITH_ANGLE"], label=legend_label)

    def orderbyTime(self, agg="mean"):
        self.df = self.df.groupby("TIME").agg(agg, numeric_only=True).sort_index()
        return self

    def intervalSampling(self, select_list=None, select_step=None):
        if select_list is None:
            if select_step is not None:
                select_list = [i for i in range(0, len(self.df), select_step)]
        if select_list is not None:
            df_select = self.df.index[select_list]
            self.df = self.df.loc[df_select]
        else:
            return None
        return self

    def plotShadow(self, building_height=10, legend_label: str = "Shadow", time_split_list=None):
        # 计算坐标
        df_azi = self.df["MEAN_SOLAR_AZIMUTH_ANGLE"]
        df_zen = self.df["MEAN_SOLAR_ZENITH_ANGLE"]
        shadow_length = building_height * np.tan(np.deg2rad(df_zen))
        i_max = np.argmax(shadow_length)
        print("{0}:\n"
              "MaxLength: {1:.2f}\n"
              "Time: {2}\n".format(legend_label, shadow_length[i_max], self.df.index[i_max]))
        x = shadow_length * np.sin(np.deg2rad(df_azi + 180))
        y = shadow_length * np.cos(np.deg2rad(df_azi + 180))

        # 分时间
        if time_split_list is not None:
            df_time_index = self.df.index
            for i in range(len(time_split_list) - 1):
                t1 = time_split_list[i]
                t2 = time_split_list[i + 1]
                df_ti = (t1 < df_time_index) & (df_time_index < t2)
                plt.plot(x[df_ti], y[df_ti], label=legend_label + " {0} to {1}".format(t1, t2))
        else:
            plt.plot(x, y, label=legend_label)

        plt.scatter(x[i_max], y[i_max], c="black")
        # lim = max([np.abs(x.min()), np.abs(y.min()), np.abs(y.max()), np.abs(y.max())]) + 1
        # plt.xlim(-10, 3)
        # plt.ylim(-5, 30)


def geeCSVSelectPropertys(gee_csv_fn, to_csv_fn, fields=None):
    if fields is None:
        fields = []
    if not fields:
        return None
    fr = open(gee_csv_fn, "r", encoding="utf-8")
    cr = csv.reader(fr)
    fw = open(to_csv_fn, "w", encoding="utf-8", newline="")
    cw = csv.writer(fw)
    cw.writerow(fields)
    keys = next(cr)
    for line in cr:
        line2 = []
        for i, k in enumerate(line):
            if keys[i] in fields:
                line2.append(k)
        cw.writerow(line2)
    fr.close()
    fw.close()


def hSetXticks(n_split=6):
    x, _ = plt.xticks()
    x_ticks = np.linspace(x.min(), x.max(), n_split + 1)
    plt.xticks(x_ticks)


class SRTGEEProperties:

    def __init__(self):
        self.datas = {}
        self._n_iter = 0
        self._keys = []

    def keys(self):
        return list(self.datas.keys())

    def __len__(self):
        return len(self.datas)

    def __iter__(self):
        return self

    def __next__(self):
        if self._n_iter == 0:
            self._keys = list(self.keys())
        if self._n_iter == len(self._keys):
            self._n_iter = 0
            raise StopIteration()
        self._n_iter += 1
        return self._keys[self._n_iter - 1]

    def __contains__(self, item):
        return item in self.datas

    def __getitem__(self, item):
        return self.datas[item]

    def __setitem__(self, key, value):
        self.datas[key] = value

    def initDict(self, _data: dict):
        self.datas = {}
        self._n_iter = 0
        self._keys = []
        self.addDict(_data)
        return self

    def addDict(self, _data: dict):
        for k in _data:
            self.add(k, _data[k])

    def add(self, key, data):
        if key == "system:time_start":
            data = time.localtime(data / 1000)
        elif key == "system:time_end":
            data = time.localtime(data / 1000)
        self.datas[key] = data


class SRTGEEGeometry:

    def __init__(self):
        self.type = ""
        self.coordinates = []

    def initDict(self, _data: dict):
        self.type = _data["type"]
        self.coordinates = _data["coordinates"]
        return self


class SRTGEEFeature:

    def __init__(self):
        self.id = ""
        self.type = ""
        self.geometry = SRTGEEGeometry()
        self.properties = SRTGEEProperties()

    def initDict(self, _data: dict):
        self.id = _data["id"]
        self.type = _data["type"]
        self.geometry = SRTGEEGeometry().initDict(_data["geometry"])
        self.properties = SRTGEEProperties().initDict(_data["properties"])
        return self

    def keys(self):
        return self.properties.keys()

    def __len__(self):
        return len(self.properties)

    def __iter__(self):
        return self

    def __next__(self):
        return self.properties.__next__()

    def __contains__(self, item):
        return item in self.properties

    def __getitem__(self, item):
        return self.properties[item]

    def __setitem__(self, key, value):
        self.properties[key] = value


class SRTGEEFeatures:

    def __init__(self):
        self.datas = []
        self._n_iter = 0
        self.fields = []

    def addFeatureDict(self, _data: dict):
        self.addFeature(SRTGEEFeature().initDict(_data))

    def addFeature(self, feat: SRTGEEFeature):
        for k in feat:
            if k not in self.fields:
                self.fields.append(k)
        self.datas.append(feat)

    def initDict(self, _data: dict):
        for feat_data in _data:
            self.addFeatureDict(feat_data)
        return self

    def keys(self):
        return self.fields

    def ids(self):
        return [x.id for x in self.datas]

    def idsIter(self):
        return iter(x.id for x in self.datas)

    def __len__(self):
        return len(self.datas)

    def __iter__(self):
        return self

    def __next__(self):
        if self._n_iter == len(self.datas):
            self._n_iter = 0
            raise StopIteration()
        self._n_iter += 1
        return self.datas[self._n_iter - 1]

    def __contains__(self, item):
        return item in self.datas

    def __getitem__(self, item) -> SRTGEEFeature:
        if isinstance(item, str):
            data = next(filter(lambda x: x.id == item, self.datas))
        elif isinstance(item, int):
            data = self.datas[item]
        else:
            data = None
        return data

    def propertiesIter(self, field_name=None):
        if field_name is None:
            return iter(feat.properties for feat in self.datas)
        else:
            return iter(feat[field_name] for feat in self.datas)

    def properties(self, field_name=None):
        return list(self.propertiesIter(field_name))


class SRTGEECollection:

    def __init__(self):
        self.type = ""
        self.features = SRTGEEFeatures()

    def keys(self):
        return self.features.keys()

    def initJson(self, json_fn):
        json_data = readJson(json_fn)
        self.initDict(json_data)
        return self

    def initDict(self, _data):
        self.type = _data["type"]
        self.features = SRTGEEFeatures().initDict(_data["features"])
        return self

    def ids(self):
        return self.features.ids()

    def idsIter(self):
        return self.features.idsIter()

    def propertiesIter(self, field_name=None):
        return self.features.propertiesIter(field_name)

    def properties(self, field_name=None):
        return self.features.properties(field_name)

    def __len__(self):
        return len(self.features)

    def __iter__(self):
        return self

    def __next__(self) -> SRTGEEFeature:
        return self.features.__next__()

    def __contains__(self, item):
        return item in self.features

    def __getitem__(self, item) -> SRTGEEFeature:
        return self.features[item]


def main():
    # df_im_coll = GEEImageProperty(r"D:\RemoteShadow\Analysis\ImageCollection\gz_s2_improps_1.csv") \
    #     .extractTime() \
    #     .filterDate("2021-01-01", "2022-01-01") \
    #     .orderbyTime() \
    #     .intervalSampling(select_step=2)
    #
    # df_im_coll.plotZenith("Qing Dao Zenith")
    # df_im_coll.plotAzimuth("Qing Dao Azimuth")
    #
    # df_im_coll.plt_show("Qing Dao", "Time", "Angle(°)", 4)
    #
    # df_im_coll.plt_fig()
    # df_im_coll.plotShadow(time_split_list=[
    #     np.datetime64("2021-01-01"), np.datetime64("2021-04-01"),
    #     np.datetime64("2021-07-01"), np.datetime64("2021-10-01"),
    #     np.datetime64("2022-01-01")
    # ])
    # df_im_coll.plt_show("Qing Dao")
    # plt.figure(figsize=(1.2, .9))

    # plotAZ(r"D:\RemoteShadow\Analysis\ImageCollection\hrb_s2_improps_1.csv", "Ha Er Bin", "2021-01-01", "2022-01-01")
    # plotAZ(r"D:\RemoteShadow\Analysis\ImageCollection\qd_s2_improps_1.csv", "Qing Dao", "2021-01-01", "2022-01-01")
    # plotAZ(r"D:\RemoteShadow\Analysis\ImageCollection\sh_s2_improps_1.csv", "Shang Hai", "2021-01-01", "2022-01-01")
    # plotAZ(r"D:\RemoteShadow\Analysis\ImageCollection\gz_s2_improps_1.csv", "Guang Zhou", "2021-01-01", "2022-01-01")
    #
    # plt.title("Changes in zenith in different regions within a year")
    # plt.legend()
    # plt.show()

    # plotShadowMax(r"D:\RemoteShadow\Analysis\ImageCollection\hrb_s2_improps_1.csv", "Ha Er Bin")
    # plotShadowMax(r"D:\RemoteShadow\Analysis\ImageCollection\qd_s2_improps_1.csv", "Qing Dao")
    # plotShadowMax(r"D:\RemoteShadow\Analysis\ImageCollection\sh_s2_improps_1.csv", "Shang Hai")
    # plotShadowMax(r"D:\RemoteShadow\Analysis\ImageCollection\gz_s2_improps_1.csv", "Guang Zhou")

    se_coll = SRTGEECollection().initJson(r"M:\ProjectSet\Shadow\Hierarchical\Images\Find\cd_s2_coll.geojson")

    pass


if __name__ == "__main__":
    main()


def main():
    pass
