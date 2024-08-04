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
from PIL import Image
from matplotlib import pyplot as plt

from SRTCodes.SRTFeature import SRTFeatureDataCollection
from SRTCodes.Utils import CheckLoc, NONE_CLASS


class MplPatchesEllipse:

    def __init__(self, xy, width, height, angle=0, is_ratio=False, select_columns=None, **kwargs):
        if select_columns is None:
            select_columns = []
        self.xy = list(xy)
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


def dataToCategory(c_dict, d):
    d = d.astype("int")
    d = d[:, :, 0]
    c_d = np.zeros((d.shape[0], d.shape[1], 3))
    for k in c_dict:
        c_d[d == k, :] = np.array(c_dict[k])
    c_d = c_d / 255
    return c_d


class DrawImage:

    def __init__(self):
        self.category_colors = {}
        self.ell_coll = MplPatchesEllipseColl()
        self.ells = []

        self.data = None

    def addCategoryColor(self, name, *category_colors, **category_color):
        c_dict = {}
        i = 0
        c_name = ""
        for c_color in category_colors:
            if isinstance(c_color, dict):
                for k in c_color:
                    c_dict[k] = c_color[k]
            else:
                if i % 2 == 0:
                    c_name = c_color
                else:
                    c_dict[c_name] = c_color
        for k in category_color:
            c_dict[k] = category_color[k]
        self.category_colors[name] = c_dict
        return name

    def toCategory(self, *args, **kwargs):
        color_name = args[0]
        c_dict = self.category_colors[color_name]
        c_d = dataToCategory(c_dict, self.data)
        return c_d

    def draw(self, *args, **kwargs):
        if len(args) >= 1:
            ax = args[0]
        elif "ax" in kwargs:
            ax = kwargs["ax"]
        else:
            ax = plt.gca()
        ax.imshow(self.data, aspect="auto")

        if "fontdict" in kwargs:
            fontdict = kwargs["fontdict"]
        else:
            fontdict = {}
        for ell in self.ells:
            ax.add_patch(ell.fit())
        ax.set_xticks([])
        ax.set_yticks([])
        if "xlabel" in kwargs:
            ax.set_xlabel(kwargs["xlabel"], rotation=0, fontdict=fontdict)
        if "ylabel" in kwargs:
            ax.set_ylabel(kwargs["ylabel"], rotation=0, fontdict=fontdict)
        if "title" in kwargs:
            ax.set_title(kwargs["title"], fontdict=fontdict)
        return ax

    def addEllipse(self, xy, width, height, angle=0, is_ratio=False, *args, **kwargs):
        ell = MplPatchesEllipse(xy=xy, width=width, height=height, angle=angle, is_ratio=is_ratio, **kwargs)
        self.ells.append(ell)
        return ell


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


class SRTDrawImages:

    def __init__(self, fontdict=None):
        self.name = ""
        self.data = [[None]]
        if fontdict is None:
            fontdict = {
                # 'family': 'Times New Roman',
                'size': 16
            }
        self.fontdict = fontdict
        self.ell_coll = MplPatchesEllipseColl()

    def changeDataList(self, n_row, n_column):
        if n_row >= len(self.data):
            for i in range(n_row - len(self.data) + 1):
                self.data.append([None for _ in range(len(self.data[0]))])
        if n_column >= len(self.data[0]):
            n_column_tmp = len(self.data[0])
            for i in range(len(self.data)):
                for j in range(n_column - n_column_tmp + 1):
                    self.data[i].append(None)
        return n_row, n_column

    def addImage(self, n_row, n_column, image_fn, ):
        n_row, n_column = self.changeDataList(n_row, n_column)
        self.data[n_row][n_column] = image_fn

    def shape(self, dim=None):
        if dim is None:
            return len(self.data), len(self.data[0])
        else:
            if dim == 0:
                return len(self.data)
            elif dim == 1:
                return len(self.data[0])

    def initFig(self, fontdict, n_columns_ex, n_rows_ex, *args, **kwargs):
        if fontdict is None:
            fontdict = self.fontdict
        n_rows, n_columns = self.shape()
        fig = plt.figure(figsize=(n_columns * n_columns_ex, n_rows * n_rows_ex), )
        axes = fig.subplots(n_rows, n_columns)
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, hspace=0.04, wspace=0.03)
        return axes, fontdict, n_columns, n_rows

    def draw(self, n_columns_ex=1.0, n_rows_ex=1.0, row_names=None, column_names=None, fontdict=None):
        axes, fontdict, n_columns, n_rows = self.initFig(fontdict, n_columns_ex, n_rows_ex)

        for i in range(n_rows):
            for j in range(n_columns):
                if n_rows == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]
                if j == 0:
                    if row_names is not None:
                        ax.set_ylabel(row_names[i],rotation=0, fontdict=fontdict)
                if i == 0:
                    if column_names is not None:
                        ax.set_title(column_names[j], fontdict=fontdict)
                d = self.getData(i, j)
                if d is not None:
                    ax.imshow(d)
                    self.ell_coll.fit(ax, i, j)
                else:
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])

    def getData(self, row, column):
        fn = self.data[row][column]
        if fn is None:
            return None
        return Image.open(fn)


def main():
    pass


if __name__ == "__main__":
    main()
