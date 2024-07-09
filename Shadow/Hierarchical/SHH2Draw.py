# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2Draw.py
@Time    : 2024/6/28 14:53
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2Draw
-----------------------------------------------------------------------------"""
import matplotlib.pyplot as plt

from SRTCodes.GDALDraw import GDALDrawImages
from SRTCodes.Utils import DirFileName, FRW, filterFileEndWith
from Shadow.Hierarchical import SHH2Config


def columnLoc(row, column, max_column):
    column += 1
    if column == max_column:
        row += 1
        column = 0
    return [row, column]


def getGDI(win_size, is_min_max=True, is_01=True, fontdict=None):
    gdi = GDALDrawImages(win_size=win_size, is_min_max=is_min_max, is_01=is_01, fontdict=fontdict)
    qd_name = gdi.addGeoRange(SHH2Config.QD_ENVI_FN, SHH2Config.QD_RANGE_FN)
    bj_name = gdi.addGeoRange(SHH2Config.BJ_ENVI_FN, SHH2Config.BJ_RANGE_FN)
    cd_name = gdi.addGeoRange(SHH2Config.CD_ENVI_FN, SHH2Config.CD_RANGE_FN)
    gdi.addCategoryColor("color", {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255)})

    gdi.addRasterCenterCollection("RGB", bj_name, cd_name, qd_name, channel_list=["Red", "Green", "Blue"])
    gdi.addRasterCenterCollection("NRG", bj_name, cd_name, qd_name, channel_list=["NIR", "Red", "Green"])
    gdi.addRasterCenterCollection("NDVI", bj_name, cd_name, qd_name, channel_list=["NDVI"])
    gdi.addRasterCenterCollection("NDWI", bj_name, cd_name, qd_name, channel_list=["NDWI"])

    gdi.addRasterCenterCollection("AS_VV", bj_name, cd_name, qd_name, channel_list=["AS_VV"])
    gdi.addRasterCenterCollection("AS_VH", bj_name, cd_name, qd_name, channel_list=["AS_VH"])
    gdi.addRasterCenterCollection("DE_VV", bj_name, cd_name, qd_name, channel_list=["DE_VV"])
    gdi.addRasterCenterCollection("DE_VH", bj_name, cd_name, qd_name, channel_list=["DE_VH"])

    gdi.addRasterCenterCollection("AS_C11", bj_name, cd_name, qd_name, channel_list=["AS_C11"])
    gdi.addRasterCenterCollection("AS_C22", bj_name, cd_name, qd_name, channel_list=["AS_C22"])
    gdi.addRasterCenterCollection("DE_C11", bj_name, cd_name, qd_name, channel_list=["DE_C11"])
    gdi.addRasterCenterCollection("DE_C22", bj_name, cd_name, qd_name, channel_list=["DE_C22"])

    gdi.addRasterCenterCollection("AS_Lambda1", bj_name, cd_name, qd_name, channel_list=["AS_Lambda1"])
    gdi.addRasterCenterCollection("AS_Lambda2", bj_name, cd_name, qd_name, channel_list=["AS_Lambda2"])
    gdi.addRasterCenterCollection("DE_Lambda1", bj_name, cd_name, qd_name, channel_list=["DE_Lambda1"])
    gdi.addRasterCenterCollection("DE_Lambda2", bj_name, cd_name, qd_name, channel_list=["DE_Lambda2"])

    gdi.addRasterCenterCollection("AS_Epsilon", bj_name, cd_name, qd_name, channel_list=["AS_Epsilon"])
    gdi.addRasterCenterCollection("DE_Epsilon", bj_name, cd_name, qd_name, channel_list=["DE_Epsilon"])
    gdi.addRasterCenterCollection("AS_Mu", bj_name, cd_name, qd_name, channel_list=["AS_Mu"])
    gdi.addRasterCenterCollection("DE_Mu", bj_name, cd_name, qd_name, channel_list=["DE_Mu"])
    gdi.addRasterCenterCollection("AS_Beta", bj_name, cd_name, qd_name, channel_list=["AS_Beta"])
    gdi.addRasterCenterCollection("DE_Beta", bj_name, cd_name, qd_name, channel_list=["DE_Beta"])

    return gdi


class SHH2DrawTR:

    def __init__(self, city_name):
        self.city_name = city_name
        self.gdi = self.initGDI((200, 200))
        self.draw9_init = [0, 0]
        self.draw18_init = [0, 0]
        self.imdcs = {}

    def initGDI(self, win_size=None, is_min_max=True, is_01=True, fontdict=None):
        gdi = getGDI(win_size, is_01, is_min_max, fontdict)
        self.gdi = gdi
        return gdi

    def getTRDiranme(self):
        dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Images")
        if self.city_name == "qd":
            return dfn.fn("QingDao", "SH22", "TestRegions")
        elif self.city_name == "bj":
            return dfn.fn("BeiJing", "SH22", "TestRegions")
        elif self.city_name == "cd":
            return dfn.fn("ChengDu", "SH22", "TestRegions")
        else:
            return None

    def _draw9(self, name, x, y, color_name=None):
        self.gdi.addAxisDataXY(self.draw9_init[0], self.draw9_init[1], name, x, y, color_name=color_name)
        self.draw9_init = columnLoc(*self.draw9_init, max_column=3)

    def _draw18(self, name, x, y, color_name=None):
        self.gdi.addAxisDataXY(self.draw18_init[0], self.draw18_init[1], name, x, y, color_name=color_name)
        self.draw18_init = columnLoc(*self.draw18_init, max_column=6)

    def draw9(self, name):
        dfn = DirFileName(self.getTRDiranme())
        json_dict = FRW(dfn.fn("test_regions.json")).readJson()
        color_name = None
        if name in self.imdcs:
            color_name = self.imdcs[name]
        for fn in json_dict:
            self._draw9(name, json_dict[fn]["X"], json_dict[fn]["Y"], color_name=color_name)
        self.gdi.draw(n_rows_ex=3, n_columns_ex=3)
        return self

    def draw18(self, name1, name2, ):
        dfn = DirFileName(self.getTRDiranme())
        json_dict = FRW(dfn.fn("test_regions.json")).readJson()

        color_name1, color_name2 = None, None
        if name1 in self.imdcs:
            color_name1 = self.imdcs[name1]
        if name2 in self.imdcs:
            color_name2 = self.imdcs[name2]

        for fn in json_dict:
            self._draw18(name1, json_dict[fn]["X"], json_dict[fn]["Y"], color_name=color_name1)
            self._draw18(name2, json_dict[fn]["X"], json_dict[fn]["Y"], color_name=color_name2)

        self.gdi.draw(n_rows_ex=3, n_columns_ex=3)
        return self

    def show(self, to_fn=None):
        if to_fn is not None:
            plt.savefig(to_fn, dpi=300)
        plt.show()
        return self

    def addImdc(self, name, *fns, color_name=None):
        self.gdi.addRasterCenterCollection(name, *fns, channel_list=[0], is_min_max=False)
        if color_name is None:
            color_name = "color"
        self.imdcs[name] = color_name
        return self

    def addImdcDirName(self, name, dirname, color_name=None):
        fns = filterFileEndWith(dirname, ".tif")
        self.addImdc(name, *fns, color_name=color_name)
        return self


def main():
    def func1():
        gdi = getGDI((200, 200))
        row_names = []
        column_names = ["NRG", "AS_VV", "AS_VH", "DE_VV", "DE_VH"]

        def add_row(name, x, y):
            gdi.addAxisDataXY(len(row_names), 0, "NRG", x, y, min_list=[0, 0, 0], max_list=[3000, 2000, 2000])
            gdi.addAxisDataXY(len(row_names), 1, "AS_VV", x, y, min_list=[-23], max_list=[4])
            gdi.addAxisDataXY(len(row_names), 2, "AS_VH", x, y, min_list=[-30], max_list=[-9])
            gdi.addAxisDataXY(len(row_names), 3, "DE_VV", x, y, min_list=[-27], max_list=[1])
            gdi.addAxisDataXY(len(row_names), 4, "DE_VH", x, y, min_list=[-35], max_list=[-8])
            row_names.append(name)

        add_row("qd(1)    ", 120.10256, 36.29804)
        add_row("qd(2)    ", 120.28513, 36.33342)
        add_row("qd(3)    ", 120.399169, 36.125422)
        add_row("qd(4)    ", 120.471486, 36.139496)

        add_row("bj(4)    ", 116.39246,39.64814)


        gdi.draw(n_rows_ex=2, n_columns_ex=2, row_names=row_names, column_names=column_names)
        plt.show()

    func1()

    return


if __name__ == "__main__":
    main()
