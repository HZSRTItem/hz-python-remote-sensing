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


class SHH2DrawTR:

    def __init__(self, city_name):
        self.city_name = city_name
        self.gdi = self.initGDI((200, 200))
        self.draw9_init = [0, 0]
        self.draw18_init = [0, 0]
        self.imdcs = {}

    def initGDI(self, win_size=None, is_min_max=True, is_01=True, fontdict=None):
        gdi = GDALDrawImages(win_size=win_size, is_min_max=is_min_max, is_01=is_01, fontdict=fontdict)

        qd_name = gdi.addGeoRange(SHH2Config.QD_ENVI_FN, SHH2Config.QD_RANGE_FN)
        bj_name = gdi.addGeoRange(SHH2Config.BJ_ENVI_FN, SHH2Config.BJ_RANGE_FN)
        cd_name = gdi.addGeoRange(SHH2Config.CD_ENVI_FN, SHH2Config.CD_RANGE_FN)

        gdi.addCategoryColor("color", {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255)})
        gdi.addRasterCenterCollection("RGB", bj_name, cd_name, qd_name, channel_list=["Red", "Green", "Blue"])
        gdi.addRasterCenterCollection("NRG", bj_name, cd_name, qd_name, channel_list=["NIR", "Red", "Green"])
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
    pass


if __name__ == "__main__":
    main()
