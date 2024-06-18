# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : TrainingUtils.py
@Time    : 2024/4/7 18:55
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of TrainingUtils
-----------------------------------------------------------------------------"""
import numpy as np

from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.NumpyUtils import categoryMap


class SRTAccuracyConfusionMatrix:

    def __init__(self, n_class=0, class_names=None):
        self.cm = ConfusionMatrix(n_class=n_class, class_names=class_names)

    def add(self, y_true, y_pred):
        self.cm.addData(y_true, y_pred)


class SRTAccuracy:

    def __init__(self, y1=None, y2=None, y1_map_dict=None, y2_map_dict=None, cnames=None):
        self.y1 = y1
        self.y2 = y2
        self.y1_map_dict = y1_map_dict
        self.y2_map_dict = y2_map_dict
        self.cnames = cnames

    def cm(self):
        y1 = self.categoryMap(self.y1, self.y1_map_dict)
        y2 = self.categoryMap(self.y2, self.y2_map_dict)
        if self.cnames is None:
            cnames = np.unique(y1).tolist() + np.unique(y2).tolist()
            cnames = np.unique(cnames).tolist()
        else:
            cnames = self.cnames
        cm = ConfusionMatrix(class_names=cnames)
        cm.addData(y1, y2)
        return cm

    def categoryMap(self, y, y_map_dict):
        if y_map_dict is None:
            return y
        else:
            return categoryMap(y, y_map_dict, is_notfind_to0=True)


def main():
    pass


if __name__ == "__main__":
    main()
