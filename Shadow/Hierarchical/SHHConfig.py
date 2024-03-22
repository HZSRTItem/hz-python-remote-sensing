# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHHConfig.py
@Time    : 2024/3/9 10:53
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHHConfig

-----------------------------------------------------------------------------"""
from SRTCodes.Utils import DirFileName, FileName


class SHHDfn:

    @staticmethod
    def Analysis():
        return DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Analysis")

    @staticmethod
    def Images():
        return DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Images")

    @staticmethod
    def MLMods():
        return DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\MLMods")

    @staticmethod
    def Mods():
        return DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Mods")

    @staticmethod
    def Samples():
        return DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Samples")

    @staticmethod
    def Temp():
        return DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Temp")


class ImagesName3:

    def __init__(self, *fns):
        self.qd = None
        self.bj = None
        self.cd = None
        self.qd_fn = FileName()
        self.bj_fn = FileName()
        self.cd_fn = FileName()
        self.initFns(*fns)

    def initFns(self, *fns):
        self.qd = fns[0]
        self.bj = fns[1]
        self.cd = fns[2]
        self.qd_fn = FileName(self.qd)
        self.bj_fn = FileName(self.bj)
        self.cd_fn = FileName(self.cd)

    def iter(self):
        return iter([self.qd, self.bj, self.cd])

    def iterFN(self):
        return iter([self.qd_fn, self.bj_fn, self.cd_fn])


SHH2_IMAGE1_FNS = [
    r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\qd_sh2_1.tif",
    r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\bj_sh2_1.tif",
    r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\cd_sh2_1.tif",
]

SHH2_IMAGE1_ESA21_FNS = [
    r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\qd_sh2_esa.tif",
    r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\bj_sh2_esa.tif",
    r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\cd_sh2_esa.tif",
]

SHH2_IMAGE1_GLCM_FNS = [
    r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\glcm\qd_sh2_1_gray_envi_mean",
    r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\glcm\bj_sh2_1_gray_envi_mean",
    r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\glcm\cd_sh2_1_gray_envi_mean",
]


class SHHFNImages:

    @staticmethod
    def images1():
        return ImagesName3(*SHH2_IMAGE1_FNS)

    @staticmethod
    def esa21():
        return ImagesName3(*SHH2_IMAGE1_ESA21_FNS)

    @staticmethod
    def image1GLCM():
        return ImagesName3(*SHH2_IMAGE1_GLCM_FNS)


SHH_COLOR4 = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), }

SHH_COLOR8 = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), 5: (128, 0, 0),
              6: (0, 128, 0), 7: (128, 128, 0), 8: (0, 0, 128)}

SHH_COLOR82 = {0: (0, 0, 0), 11: (255, 0, 0), 21: (0, 255, 0), 31: (255, 255, 0), 41: (0, 0, 255), 12: (128, 0, 0),
               22: (0, 128, 0), 32: (128, 128, 0), 42: (0, 0, 128)}

SHH_COLOR_VNL_8 = {0: (0, 0, 0), 1: (200, 200, 200), 2: (0, 255, 0), 3: (200, 200, 200), 4: (60, 60, 60),
                   5: (60, 60, 60),
                   6: (60, 60, 60), 7: (60, 60, 60), 8: (60, 60, 60)}

SHH_COLOR_IS_8 = {0: (0, 0, 0), 1: (255, 0, 0), 2: (255, 255, 255), 3: (255, 255, 0), 4: (255, 255, 255),
                  5: (255, 255, 255), 6: (255, 255, 255), 7: (255, 255, 255), 8: (255, 255, 255)}

SHH_CNAMES = ["NOT_KNOW", "IS", "VEG", "SOIL", "WAT", "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH"]
SHH_CNAMES4 = ["IS", "VEG", "SOIL", "WAT"]
SHH_CNAMES_SH4 = ["IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH"]
SHH_CNAMES8 = ["IS", "VEG", "SOIL", "WAT", "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH"]

VHL_CNAMES = ["VEG", "HIGH", "LOW"]
IS_CNAMES = ["IS", "SOIL"]

CATE_MAP_VHL_8 = {1: 2, 2: 1, 3: 2, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3}
CATE_MAP_IS_8 = {1: 1, 2: 0, 3: 2, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}

CATE_MAP_SH881 = {11: 1, 21: 2, 31: 3, 41: 4, 12: 5, 22: 6, 32: 7, 42: 8}
CATE_MAP_SH882 = {1: 11, 2: 21, 3: 31, 4: 41, 5: 12, 6: 22, 7: 32, 8: 42}
CATE_MAP_SH841 = {11: 1, 21: 2, 31: 3, 41: 4, 12: 1, 22: 2, 32: 3, 42: 4}
CATE_MAP_SH842 = {1: 1, 2: 2, 3: 3, 4: 4, 5: 1, 6: 2, 7: 3, 8: 4}


def categoryMap(categorys, map_dict, is_notfind_to0=False):
    to_category = []
    for d in categorys:
        if d in map_dict:
            to_category.append(map_dict[d])
        else:
            if is_notfind_to0:
                to_category.append(0)
            else:
                to_category.append(d)
    return to_category


def main():
    pass


if __name__ == "__main__":
    main()
