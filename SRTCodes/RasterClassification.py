# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : RasterClassification.py
@Time    : 2023/6/12 10:15
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : BaseCodes of RasterClassification
-----------------------------------------------------------------------------"""

import numpy as np

from SRTCodes.RasterIO import Raster
from SRTCodes.Utils import Jdt

eps = 0.000001
np.set_printoptions(suppress=True, precision=3)


class Prediction:

    def __init__(self):
        self.model = None
        self.models = []
        self.spl = 0

    def addModel(self, model):
        self.model = model
        self.models.append(model)

    def sample(self, *args, **kwargs):
        return None

    def predict(self, x, *args, **kwargs) -> np.ndarray:
        return self.model(x)

    def preDeal(self, *args, **kwargs):
        return self.spl

    def fit(self, *args, **kwargs):
        return self


class RasterPrediction(Raster, Prediction):

    def __init__(self):
        Prediction.__init__(self)

        self.imdc = None
        self.win_spl = [0, 0, 1, 1]

        self.iter_n_row = 0
        self.iter_n_column = 0
        self.iter_is_end = False

        self.row_start = 0
        self.row_end = -1
        self.column_start = 0
        self.column_end = -1

    def _initImdc(self):
        self.imdc = np.zeros([self.n_rows, self.n_columns])

    def sample(self, row=0, column=0, *args, **kwargs):
        d1 = self.d[:, row + self.win_spl[0]: row + self.win_spl[1], column + self.win_spl[2]: column + self.win_spl[3]]
        return d1

    def fit(self, spl_size=None, row_start=0, row_end=-1, column_start=0, column_end=-1, *args, **kwargs):
        if spl_size is None:
            spl_size = [1, 1]
        if self.d is None:
            raise Exception("RasterClassification d error: d is None")
        if self.model is None:
            raise Exception("RasterClassification model error: can not find model.")
        if self.imdc is None:
            self._initImdc()
        if column_end < 0:
            column_end += self.n_columns + 1
        if row_end < 0:
            row_end += self.n_rows + 1

        jdt = Jdt(total=row_end - row_start, desc="RasterClassification")
        jdt.start()
        col_imdc = np.zeros([self.n_columns, self.n_channels, spl_size[0], spl_size[1]])
        select_spl = np.array([False for i in range(self.n_columns)])

        row_size, column_size = spl_size[0], spl_size[1]
        self.win_spl[0] = 0 - int(row_size / 2)
        self.win_spl[1] = 0 + round(row_size / 2 + 0.1)
        self.win_spl[2] = 0 - int(column_size / 2)
        self.win_spl[3] = 0 + round(column_size / 2 + 0.1)

        for i in range(row_start, row_end):

            select_spl[column_start:column_end] = self.preDeal(i, column_start, column_end)
            j_select = 0

            for j in range(column_start, column_end):
                if select_spl[j]:
                    col_imdc[j_select, :, :, :] = self.sample(i, j)
                    j_select += 1

            jdt.add()

            if j_select == 0:
                self.imdc[i, select_spl] = 0
            else:
                self.imdc[i, select_spl] = self.predict(col_imdc[:j_select, :, :, :])

        jdt.end()

    def fit2(self, spl_size=None, row_start=0, row_end=-1, column_start=0, column_end=-1, n_one_t=2000, *args,
             **kwargs):
        if spl_size is None:
            spl_size = [1, 1]
        if self.d is None:
            raise Exception("RasterClassification d error: d is None")
        if self.model is None:
            raise Exception("RasterClassification model error: can not find model.")
        if self.imdc is None:
            self._initImdc()
        if column_end < 0:
            column_end += self.n_columns + 1
        if row_end < 0:
            row_end += self.n_rows + 1
        self.row_start = row_start
        self.row_end = row_end
        self.column_start = column_start
        self.column_end = column_end
        self.iter_n_row = row_start
        self.iter_n_column = column_start
        row_size, column_size = spl_size[0], spl_size[1]
        self.win_spl[0] = 0 - int(row_size / 2)
        self.win_spl[1] = 0 + round(row_size / 2 + 0.1)
        self.win_spl[2] = 0 - int(column_size / 2)
        self.win_spl[3] = 0 + round(column_size / 2 + 0.1)

        jdt = Jdt(total=(row_end - row_start)*(column_end-column_start), desc="RasterClassification")
        jdt.start()
        col_imdc = np.zeros([n_one_t, self.n_channels, spl_size[0], spl_size[1]])
        select_spl = []
        for i in range(n_one_t):
            select_spl.append([0, 0])

        while True:
            k = 0
            while k < n_one_t:
                n_row, n_column, spl = self.getIterSample()
                jdt.add()
                if self.iter_is_end:
                    break
                if self.preDeal2(spl):
                    select_spl[k][0] = n_row
                    select_spl[k][1] = n_column
                    col_imdc[k, :, :, :] = spl
                    k += 1
            cate = self.predict(col_imdc)
            for k0 in range(k):
                n_row = select_spl[k0][0]
                n_column = select_spl[k0][1]
                self.imdc[n_row, n_column] = cate[k0]

            if self.iter_is_end:
                break

    def getIterSample(self):
        if self.iter_is_end:
            return None
        n_row, n_column = self.iter_n_row, self.iter_n_column
        spl = self.sample(n_row, n_column)
        self.iter_n_row += 1
        self.iter_n_column += 1
        if self.iter_n_column == self.column_end:
            self.iter_n_column = self.column_start
        if self.iter_n_row == self.row_end:
            self.iter_is_end = True
        return n_row, n_column, spl

    def preDeal(self, row, column_start, column_end):
        d_row = self.d[:, row, column_start:column_end]
        return np.ones(d_row.shape) == 0

    def preDeal2(self, x):
        return True

    def featureScaleMinMax(self, field_name, x_min, x_max, is_trans=None, is_01=None):
        if is_trans is None:
            is_trans = self.is_trans
        if is_01 is None:
            is_01 = self.is_01
        self.scale_min_max[field_name] = {"min": x_min, "max": x_max, "is_trans": is_trans, "is_01": is_01}

    def featureCallBack(self, feat_name, callback):
        self._feat_callback[feat_name] = callback
