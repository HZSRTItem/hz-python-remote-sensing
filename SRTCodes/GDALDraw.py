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

from SRTCodes.GDALUtils import GDALRasterCenterDatas


class GDALDrawImages(GDALRasterCenterDatas):

    def __init__(self, win_size=None, is_min_max=True, is_01=True):
        super().__init__(win_size, is_min_max, is_01)

        self.column_names = None
        self.row_names = None

    def draw(self, n_columns_ex=1.0, n_rows_ex=1.0, row_names=None, column_names=None, fontdict=None):
        if None is None:
            fontdict = {'family': 'Times New Roman', 'size': 16}
        n_rows, n_columns = self.shape()
        fig = plt.figure(figsize=(n_columns * n_columns_ex, n_rows * n_rows_ex), )
        axes = fig.subplots(n_rows, n_columns)
        fig.subplots_adjust(top=0.96, bottom=0.04, left=0.04, right=0.96, hspace=0.04, wspace=0.03)
        for i in range(n_rows):
            for j in range(n_columns):
                if n_rows == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]
                if j == 0:
                    if row_names is not None:
                        ax.set_ylabel(row_names[i], fontdict=fontdict)
                if i == 0:
                    if column_names is not None:
                        ax.set_title(column_names[j], fontdict=fontdict)
                d = self.getData(i, j)
                if d is not None:
                    ax.imshow(d)
                else:
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])

    def toCategory(self, n_row, n_column, color_name):
        d = np.array(self.getData(n_row, n_column))
        d = d.astype("int")
        d = d[:, :, 0]
        c_d = np.zeros((d.shape[0], d.shape[1], 3))
        c_dict = self.category_colors[color_name]
        for k in c_dict:
            c_d[d == k, :] = np.array(c_dict[k])
        c_d = c_d / 255
        self.setData(n_row, n_column, c_d)

    def addAxisDataXY(self, n_row, n_column, grcc_name, x, y, color_name=None, *args, **kwargs):
        d = super(GDALDrawImages, self).addAxisDataXY(n_row=n_row, n_column=n_column, grcc_name=grcc_name, x=x, y=y)
        if d.shape[2] == 1:
            to_d = np.zeros((d.shape[0], d.shape[1], 3))
            for i in range(3):
                to_d[:, :, i] = d[:, :, 0]
            d = to_d
        self.setData(n_row, n_column, data=d)
        if color_name is not None:
            self.toCategory(n_row, n_column, color_name)


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
