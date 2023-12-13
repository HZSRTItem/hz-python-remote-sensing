# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ENVIRasterClassification.py
@Time    : 2023/6/23 15:56
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PytorchGeo of ENVIRasterClassification
-----------------------------------------------------------------------------"""
import os
import random

from SRTCodes.ENVIRasterIO import ENVIRaster
from SRTCodes.RasterClassification import RasterPrediction


class ENVIRasterClassification(ENVIRaster, RasterPrediction):

    def __init__(self, dat_fn=None, hdr_fn=None):
        RasterPrediction.__init__(self)
        ENVIRaster.__init__(self, dat_fn, hdr_fn)

        self.n_category = 1
        self.cate_names = ["Unclassified"]
        self.back_cate_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]
        self.cate_colors = [(0, 0, 0)]

        self.save_hdr = {
            "bands": "1",
            "description": "ENVI Raster Classification",
            "file type": "ENVI Classification",
            "data type": "1",
            "interleave": "bsq",
            "classes": "",
            "class lookup": "{ 0,   0,   0, 255,   0,   0,   0, 255,   0}",
            "class names": "{Unclassified, NOImper, Imper}",
            "band names": ""
        }

    def addCategory(self, name: str, color: tuple = None):
        if name in self.cate_names:
            print("Warning: name \"{0}\" in category names.".format(name))
            return
        self.cate_names.append(name)
        if color is not None:
            self.cate_colors.append(color)
        else:
            self.cate_colors.append(self._getColor())
        self.n_category += 1

    def saveImdc(self, imdc_fn):
        self.save_hdr["classes"] = str(self.n_category)
        class_lookup = "{ "
        for c in self.cate_colors:
            class_lookup += " {0:>3}, {1:>3}, {2:>3},".format(c[0], c[1], c[2])
        class_lookup = class_lookup[:-1]
        class_lookup += " }"
        self.save_hdr["class lookup"] = class_lookup
        class_names = "{ " + self.cate_names[0]
        for name in self.cate_names[1:]:
            class_names += ", " + name
        class_names += " }"
        self.save_hdr["class names"] = class_names
        self.save_hdr["band names"] = "{" + os.path.split(self.dat_fn)[1] + " Category }"
        self.save(self.imdc.astype("int8"), imdc_fn, hdrs=self.save_hdr)

    def _getColor(self):
        if self.n_category - 1 >= len(self.back_cate_colors):
            return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        else:
            return self.back_cate_colors[self.n_category - 1]

    def run(self, imdc_fn, mod=None, spl_size=None, row_start=0, row_end=-1, column_start=0, column_end=-1, *args,
            **kwargs):
        # 读取图像数据
        if self.d is None:
            self.readAsArray(interleave="b,r,c")
        # 添加模型
        if mod is not None:
            self.addModel(mod)

        # 分类
        self.fit(spl_size=spl_size, row_start=row_start, row_end=row_end, column_start=column_start,
                 column_end=column_end, *args, **kwargs)

        # 保存
        self.saveImdc(imdc_fn)


def main():
    pass


if __name__ == "__main__":
    main()
