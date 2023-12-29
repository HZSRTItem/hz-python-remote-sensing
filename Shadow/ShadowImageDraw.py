# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowImageDraw.py
@Time    : 2023/7/15 15:40
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : BaseCodes of ShadowImageDraw
-----------------------------------------------------------------------------"""
import os

import matplotlib.pyplot as plt
import numpy as np

from SRTCodes.SRTFeature import SRTFeatureCallBackCollection
from SRTCodes.Utils import readJson


class ImageDrawOne:

    def __init__(self, name, d):
        self.name = name
        if len(d.shape) == 3:
            if d.shape[2] == 1:
                d = d[:, :, 0]
        self.d = d
        self.fcb = SRTFeatureCallBackCollection(is_trans=True)
        self.c_d = None

    def scaleMinMax(self, d_min, d_max):
        self.d = np.clip(self.d, d_min, d_max)
        self.d = (self.d - d_min) / (d_max - d_min)

    def callBack(self, callback, *args, **kwargs):
        self.d = callback(self.d, args=args, kwargs=kwargs)

    def save(self, fn):
        if len(self.d.shape) == 2:
            plt.imsave(fn, self.d, cmap="gray")
        elif len(self.d.shape) == 3:
            plt.imsave(fn, self.d)

    def draw(self):
        if len(self.d.shape) == 2:
            plt.imshow(self.d, cmap="gray")
        elif len(self.d.shape) == 3:
            plt.imshow(self.d)

    def toCategory(self):
        self.c_d = self.d.astype("int")
        to_shape = (self.c_d.shape[0], self.c_d.shape[1], 3)
        self.d = np.zeros(to_shape, dtype="uint8")

    def categoryColor(self, category, color: tuple = (0, 0, 0)):
        self.d[self.c_d == category, :] = np.array(color, dtype="uint8")


class ShadowImageDraw:

    def __init__(self, json_fn):
        self.json_fn = json_fn
        self.draw_dict = readJson(self.json_fn)
        self.name = self.draw_dict["Name"]
        self.draws = {}
        self.d = np.load(self.draw_dict["Npy File Name"])
        self.save_dir = os.path.dirname(json_fn)
        self.init()

    def init(self):
        for draw in self.draw_dict["Draw"]:
            self.draws[draw["Name"]] = ImageDrawOne(draw["Name"], self.d[:, :, draw["Data"]])

    def draw(self, name):
        self.draws[name].draw()

    def scaleMinMax(self, name, d_min, d_max):
        self.draws[name].scaleMinMax(d_min, d_max)

    def callBack(self, name, callback, *args, **kwargs):
        self.draws[name].callBack(callback=callback, args=args, kwargs=kwargs)

    def save(self, name, fn=None):
        if fn is None:
            fn = self.getFileName(name, ".png")
        self.draws[name].save(fn)

    def getFileName(self, name, ext=""):
        return os.path.join(self.save_dir, self.name + " " + name + ext)

    def print(self):
        print("JSON FileName: ", self.json_fn)
        print("Directory Name: ", self.save_dir)
        for draw in self.draws:
            print("  * {0}: {1}".format(draw, self.draws[draw].d.shape))

    def addCategory(self, name, *args):
        if self.draws[name].c_d is None:
            self.draws[name].toCategory()
        for i in range(0, len(args) - 1, 2):
            self.draws[name].categoryColor(args[i], args[i + 1])


def _10log10(x, *args, **kwargs):
    return 10 * np.log10(x)


def drawShadowImage_Optical(simd: ShadowImageDraw, name, is_save=False):
    simd.scaleMinMax(name, 0, 3500)
    simd.draw(name)
    if is_save:
        simd.save(name)


def drawShadowImage_Imdc(simd: ShadowImageDraw, name, is_save=False, is_show=True):
    simd.addCategory(name, 1, (255, 0, 0), 2, (0, 255, 0), 3, (255, 255, 0), 4, (0, 0, 255))
    simd.draw(name)
    if is_save:
        simd.save(name)
    if is_show:
        plt.show()


def main():
    simd = ShadowImageDraw(r"F:\ProjectSet\Shadow\QingDao\mktu\Draw1.json")
    # drawShadowImage_Optical(simd, "RGB")
    drawShadowImage_Imdc(simd, "IMDC_1", is_save=True)
    plt.show()
    pass


if __name__ == "__main__":
    main()
