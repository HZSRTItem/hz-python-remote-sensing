# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : RasterIO.py
@Time    : 2023/5/9 17:38
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PytorchModel of ZY5MImage
-----------------------------------------------------------------------------"""

from io import BytesIO

import numpy as np
from PIL import Image

eps = 0.000001
np.set_printoptions(suppress=True, precision=3)


class Raster:
    """ Raster base class """

    def __init__(self, raster_fn=None):
        self.n_rows = 0  # number of rows
        self.n_columns = 0  # number of columns
        self.n_channels = 0  # number of channels
        self.d = None  # data of this raster

        self.initRaster()

    def initRaster(self):
        self.n_rows = 0  # number of rows
        self.n_columns = 0  # number of columns
        self.n_channels = 0  # number of channels
        self.d = None  # data of this raster

    def readAsArray(self, *args, **kwargs):
        return None

    def save(self, *args, **kwargs):
        return None


class GEORaster(Raster):
    """ GEORaster class """

    def __init__(self):
        super(GEORaster, self).__init__()

        self.srs = None
        self.y_size = None
        self.x_size = None
        self.y_max = None
        self.x_max = None
        self.y_min = None
        self.x_min = None

        self.initGEORaster()

    def initGEORaster(self):
        self.x_min = 0
        self.y_min = 0
        self.x_max = 0
        self.y_max = 0
        self.x_size = 0
        self.y_size = 0
        self.srs = None

    def coorRaster2Geo(self, row, column):
        x = self.x_min + column * self.x_size
        y = self.y_max - row * self.y_size
        return x, y

    def coorGeo2Raster(self, x, y):
        column = (x - self.x_min) / self.x_size
        row = (self.y_max - y) / self.y_size
        return column, row


