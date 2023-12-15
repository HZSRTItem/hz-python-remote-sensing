# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SRTDraw.py
@Time    : 2023/12/14 14:39
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of SRTDraw
-----------------------------------------------------------------------------"""
import matplotlib.pyplot as plt
import numpy as np

from SRTCodes.SRTFeature import SRTFeatureDataCollection


class SRTDrawData(SRTFeatureDataCollection):
    MAX_CATEGORY_N = 256

    def __init__(self):
        super().__init__()
        self.category_list = []
        self.category_data = []

    def category(self, name):
        self.category_data = self.get(name)
        d = np.unique(self.category_data).tolist()
        if len(d) > self.MAX_CATEGORY_N:
            raise Exception("number of unique this name of "
                            "\"{0}\" is more than MAX_CATEGORY_N={1}".format(name, self.MAX_CATEGORY_N))
        self.category_list = d

    def filterCategory(self, name, category):
        d = self.get(name)
        select_d = self.category_data == category
        return d[select_d]


class SRTDrawHist(SRTDrawData):

    def __init__(self):
        super().__init__()
        self.name = ""
        self.data_coll = {}

    def plot(self, name, *plot_args, category=None, bins=256, d_range=None, density=True, scalex=True, scaley=True,
             **kwargs):
        if category is None:
            d = self.get(name)
        else:
            d = self.filterCategory(name, category)
            name += " " + category
        hist, bin_edges = np.histogram(d, bins=bins, range=d_range, density=density)
        bin_edges = (bin_edges[1] - bin_edges[0]) / 2 + bin_edges
        bin_edges = bin_edges[:-1]
        return plt.plot(bin_edges, hist, label=name, *plot_args, scalex=scalex, scaley=scaley, **kwargs)


def main():
    pass


if __name__ == "__main__":
    main()
