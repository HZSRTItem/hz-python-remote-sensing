# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SRTDraw.py
@Time    : 2023/12/14 14:39
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of SRTDraw
-----------------------------------------------------------------------------"""

import matplotlib.patches as mpl_patches
import numpy as np
from matplotlib import pyplot as plt

from SRTCodes.SRTFeature import SRTFeatureDataCollection
from SRTCodes.Utils import CheckLoc, NONE_CLASS


class MplPatchesEllipse:

    def __init__(self, xy, width, height, angle=0, is_ratio=False, select_columns=None, **kwargs):
        if select_columns is None:
            select_columns = []
        self.xy = xy
        self.width = width
        self.height = height
        self.angle = angle
        self.is_ratio = is_ratio
        self.select_columns = select_columns
        self.kwargs = kwargs

        self.check_loc = CheckLoc()

    def fit(self, *args, **kwargs):
        xy = self.xy
        width, height = self.width, self.height
        if self.is_ratio:
            row_size, column_size = args[0], args[1]
            xy = self.xy[0] * row_size, self.xy[1] * column_size
        if "up" in kwargs:
            if kwargs["up"] is not None:
                width *= kwargs["up"].width_up
                height *= kwargs["up"].height_up
            kwargs.pop("up")
        return mpl_patches.Ellipse(xy=xy, width=width, height=height, angle=self.angle, **self.kwargs)

    def initLoc(self, n_row=None, n_column=None, not_rows=None, not_columns=None, ):
        self.check_loc = CheckLoc(n_row=n_row, n_column=n_column, not_rows=not_rows, not_columns=not_columns, )

    def isLoc(self, i_row, i_column):
        return self.check_loc.isLoc(i_row, i_column)


class MplPatchesEllipseColl:

    def __init__(self):
        self.ells = []
        self.ups = []

    def add(self, n_row, xy, width, height, angle=0, is_ratio=False, select_columns=None, **kwargs):
        ell = MplPatchesEllipse(xy=xy, width=width, height=height, angle=angle, is_ratio=is_ratio,
                                   select_columns=select_columns, **kwargs)
        ell.initLoc(n_row=n_row, n_column=None)
        self.ells.append(ell)
        return ell

    def add2(self, xy, width, height, n_row=None, n_column=None, angle=0, is_ratio=False, select_columns=None,
             not_rows=None, not_columns=None,
             **kwargs):
        ell = MplPatchesEllipse(xy=xy, width=width, height=height, angle=angle, is_ratio=is_ratio,
                                select_columns=select_columns, **kwargs)
        ell.initLoc(n_row=n_row, n_column=n_column, not_rows=not_rows, not_columns=not_columns)
        self.ells.append(ell)
        return ell

    def fit(self, ax, i_row, i_column, *args, **kwargs):
        for ell in self.ells:
            ell: MplPatchesEllipse
            if ell.isLoc(i_row, i_column):
                up = self.find(self.ups, i_row, i_column, 0)
                ax.add_patch(ell.fit(*args, up=up, **kwargs))

    def addUp(self, width_up=None, height_up=None, n_row=None, n_column=None, not_rows=None, not_columns=None, ):
        if (width_up is None) and (height_up is None):
            return
        if (width_up is not None) and (height_up is None):
            height_up = width_up
        if (height_up is not None) and (width_up is None):
            width_up = height_up
        up = NONE_CLASS()
        up.width_up = width_up
        up.height_up = height_up
        up.loc = CheckLoc(n_row=n_row, n_column=n_column, not_rows=not_rows, not_columns=not_columns)
        self.ups.append(up)

    def find(self, loc_list, i_row, i_column, n=None):
        to_loc_list = []
        for loc in loc_list:
            if loc.loc.isLoc(i_row, i_column):
                to_loc_list.append(loc)
        if n is None:
            return to_loc_list
        else:
            if to_loc_list:
                return to_loc_list[n]
            else:
                return None


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

    def __init__(self, ):
        super().__init__()
        self.name = ""
        self.data_coll = {}
        self.hist = np.array([])
        self.bin_edges = np.array([])

    def plot(self, name, *plot_args, category=None, bins=256, d_range=None, density=True, scalex=True, scaley=True,
             **kwargs):
        if category is None:
            d = self.get(name)
        else:
            d = self.filterCategory(name, category)
            name += " " + category
        hist, bin_edges = np.histogram(d, bins=bins, range=d_range, density=density)
        self.hist, self.bin_edges = hist, bin_edges
        if density:
            hist *= 100
        bin_edges = (bin_edges[1] - bin_edges[0]) / 2 + bin_edges
        bin_edges = bin_edges[:-1]
        return plt.plot(bin_edges, hist, label=name, *plot_args, scalex=scalex, scaley=scaley, **kwargs)


def main():
    pass


if __name__ == "__main__":
    main()
