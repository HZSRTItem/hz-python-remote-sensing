# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : GDALDraw.py
@Time    : 2024/1/1 20:11
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of GDALDraw
-----------------------------------------------------------------------------"""
import numpy as np
from matplotlib import pyplot as plt

from SRTCodes.GDALUtils import GDALRasterCenterDatas, GDALRasterCenterData
from SRTCodes.SRTDraw import DrawImage, dataToCategory


class GDALDrawImage(DrawImage):

    def __init__(self, win_size=None, is_min_max=True, is_01=True):
        super().__init__()
        self.grcd = GDALRasterCenterData(win_size=win_size, is_min_max=is_min_max, is_01=is_01)

    def addRCC(self, name, *raster_fns, channel_list=None, fns=None, win_size=None, is_min_max=None,
               is_01=None, no_data=0.0, min_list=None, max_list=None, callback_funcs=None):
        return self.grcd.addRasterCenterCollection(
            name, *raster_fns, channel_list=channel_list, fns=fns, win_size=win_size, is_min_max=is_min_max,
            is_01=is_01, no_data=no_data, min_list=min_list, max_list=max_list, callback_funcs=callback_funcs
        )

    def addGR(self, raster_fn, geo_range=None):
        return self.grcd.addGeoRange(raster_fn=raster_fn, geo_range=geo_range)

    def read(self, grcc_name, x, y, win_size=None, is_trans=False, *args, **kwargs):
        self.data = self.grcd.readAxisDataXY(grcc_name, x, y, win_size=win_size, is_trans=is_trans, *args, **kwargs)
        if self.data.shape[2] == 1:
            data_tmp = np.zeros((*self.data.shape[:2], 3))
            data_tmp[:, :, 0] = self.data[:, :, 0]
            data_tmp[:, :, 1] = self.data[:, :, 0]
            data_tmp[:, :, 2] = self.data[:, :, 0]
        return self.data

    def readDraw(self, grcc_name, x, y, win_size=None, is_trans=False, *args, **kwargs):
        self.read(grcc_name, x, y, win_size=win_size, is_trans=is_trans, *args, **kwargs)
        self.draw(*args, **kwargs)


class GDALDrawImages(GDALRasterCenterDatas, DrawImage):

    def __init__(self, win_size=None, is_min_max=True, is_01=True, fontdict=None):
        DrawImage.__init__(self)
        GDALRasterCenterDatas.__init__(self, win_size, is_min_max, is_01)

        if fontdict is None:
            fontdict = {'family': 'Times New Roman', 'size': 16}

        self.column_names = None
        self.row_names = None
        self.fontdict = fontdict

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
                        ax.set_ylabel(row_names[i], rotation=0, fontdict=fontdict)
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

    def initFig(self, fontdict, n_columns_ex, n_rows_ex, *args, **kwargs):
        if fontdict is None:
            fontdict = self.fontdict
        n_rows, n_columns = self.shape()
        fig = plt.figure(figsize=(n_columns * n_columns_ex, n_rows * n_rows_ex), )
        axes = fig.subplots(n_rows, n_columns)
        # fig.subplots_adjust(top=0.96, bottom=0.04, left=0.04, right=0.96, hspace=0.04, wspace=0.03)
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, hspace=0.04, wspace=0.03)

        return axes, fontdict, n_columns, n_rows

    def toCategory(self, n_row, n_column, color_name, *args, **kwargs):
        d = np.array(self.getData(n_row, n_column))
        c_dict = self.category_colors[color_name]
        c_d = dataToCategory(c_dict, d)
        self.setData(n_row, n_column, c_d)

    def addAxisDataXY(self, n_row, n_column, grcc_name, x, y, win_size=None, color_name=None, is_trans=False, *args,
                      **kwargs):
        d = super(GDALDrawImages, self).addAxisDataXY(n_row=n_row, n_column=n_column, grcc_name=grcc_name, x=x, y=y,
                                                      win_size=win_size, is_trans=is_trans, *args, **kwargs)
        if d.shape[2] == 1:
            to_d = np.zeros((d.shape[0], d.shape[1], 3))
            for i in range(3):
                to_d[:, :, i] = d[:, :, 0]
            d = to_d
        self.setData(n_row, n_column, data=d)
        if color_name is not None:
            self.toCategory(n_row, n_column, color_name)

    def addEllipse(self, xy, width, height, n_row=None, n_column=None, angle=0, is_ratio=False, select_columns=None,
                   not_rows=None, not_columns=None, **kwargs):
        ell = self.ell_coll.add2(xy, width, height, n_row=n_row, n_column=n_column, angle=angle, is_ratio=is_ratio,
                                 select_columns=select_columns, not_rows=not_rows, not_columns=not_columns, **kwargs)
        return ell


class _ColumnKey:

    def __init__(self, name, grcc_name, win_size=None, color_name=None, *args, **kwargs):
        self.name = name
        self.grcc_name = grcc_name
        self.color_name = color_name
        self.args = args
        self.kwargs = kwargs
        self.win_size = win_size


class _RowKey:

    def __init__(self, name, x, y, win_size=None, *args, **kwargs):
        self.name = name
        self.x = x
        self.y = y
        self.args = args
        self.kwargs = kwargs
        self.win_size = win_size


def catColumnRowArgs(*args, **kwargs):
    return args, kwargs


class GDALDrawImagesColumns(GDALDrawImages):

    def __init__(self, win_size=None, is_min_max=True, is_01=True):
        super().__init__(win_size, is_min_max, is_01)
        self.columns = []
        self.rows = []

    def addColumn(self, name, grcc_name, win_size=None, color_name=None, *args, **kwargs):
        self.columns.append(_ColumnKey(
            name, grcc_name, win_size=win_size, color_name=color_name, *args, **kwargs))
        return self.columns[-1]

    def addRow(self, name, x, y, win_size=None, *args, **kwargs):
        self.rows.append(_RowKey(name, x, y, win_size=win_size, *args, **kwargs))
        return self.rows[-1]

    def fitColumn(self, n_columns_ex=1.0, n_rows_ex=1.0, fontdict=None):
        row_names, column_names = [], []
        for column, column_key in enumerate(self.columns):
            column_key: _ColumnKey
            column_names.append(column_key.name)
            for row, row_key in enumerate(self.rows):
                if column == 0:
                    row_names.append(row_key.name)
                row_key: _RowKey
                win_size = column_key.win_size
                if row_key.win_size is not None:
                    win_size = row_key.win_size
                _args, _kwargs = catColumnRowArgs(*column_key.args, *row_key.args, **column_key.kwargs,
                                                  **row_key.kwargs)
                self.addAxisDataXY(n_row=row, n_column=column, grcc_name=column_key.grcc_name, x=row_key.x, y=row_key.y,
                                   color_name=column_key.color_name, win_size=win_size, *_args, **_kwargs)
        self.draw(n_columns_ex=n_columns_ex, n_rows_ex=n_rows_ex, row_names=row_names, column_names=column_names,
                  fontdict=fontdict)


def main():
    gdi = GDALDrawImages((100, 100))
    bj_name = gdi.addGeoRange(r"F:\ProjectSet\Shadow\Release\BeiJingImages\SH_BJ_look_tif.tif",
                              r"F:\ProjectSet\Shadow\MkTu\Draw\SH_BJ_envi.dat.npy.json")
    cd_name = gdi.addGeoRange(r"F:\ProjectSet\Shadow\Release\ChengDuImages\SH_CD_look_tif.tif",
                              r"F:\ProjectSet\Shadow\MkTu\Draw\SH_CD_envi.dat.npy.json")
    qd_name = gdi.addGeoRange(r"F:\ProjectSet\Shadow\Release\QingDaoImages\SH_QD_look_tif.tif",
                              r"F:\ProjectSet\Shadow\MkTu\Draw\SH_QD_envi.dat.npy.json")
    gdi.addCategoryColor("color", {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255)})
    gdi.addRasterCenterCollection("RGB", bj_name, cd_name, qd_name, channel_list=["Red", "Green", "Blue"])
    gdi.addRasterCenterCollection("NRG", bj_name, cd_name, qd_name, channel_list=["NIR", "Red", "Green"])
    gdi.addRasterCenterCollection("AS_VV", bj_name, cd_name, qd_name, channel_list=["AS_VV"])
    gdi.addRasterCenterCollection("AS_C22", bj_name, cd_name, qd_name, channel_list=["AS_C22"])
    gdi.addRasterCenterCollection(
        "IMDC",
        r"F:\ProjectSet\Shadow\Release\BeiJingMods\20231114H094632\SPL_SH-SVM-TAG-OPTICS-AS-DE_imdc.dat",
        r"F:\ProjectSet\Shadow\Release\ChengDuMods\20231117H112558\SPL_SH-SVM-TAG-OPTICS-AS-DE_imdc.dat",
        r"F:\ProjectSet\Shadow\Release\QingDaoMods\20231221H224548\SPL_SH-SVM-TAG-OPTICS-AS-DE_imdc.dat",
        is_min_max=False,
        channel_list=[0])
    xys = [
        120.330806, 36.135239,
        120.397521, 36.144224,
        116.363178, 39.858848,
        116.34989, 39.79670,
        104.07385, 30.65005,
        104.13064, 30.62272,
    ]
    gdi.addAxisDataXY(0, 0, "RGB", 120.330806, 36.135239)
    gdi.addAxisDataXY(0, 1, "NRG", 120.330806, 36.135239)
    gdi.addAxisDataXY(1, 0, "AS_VV", 120.330806, 36.135239)
    gdi.addAxisDataXY(0, 2, "IMDC", 120.330806, 36.135239, color_name="color")
    gdi.addAxisDataXY(1, 1, "IMDC", 104.07385, 30.65005, color_name="color")
    gdi.addAxisDataXY(2, 0, "AS_C22", 120.330806, 36.135239)

    gdi.draw(n_rows_ex=2, n_columns_ex=2)
    plt.show()
    pass


if __name__ == "__main__":
    main()
