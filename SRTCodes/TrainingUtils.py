# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : TrainingUtils.py
@Time    : 2024/4/7 18:55
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of TrainingUtils
-----------------------------------------------------------------------------"""
from SRTCodes.ModelTraining import ConfusionMatrix


class SRTAccuracyConfusionMatrix:

    def __init__(self, n_class=0, class_names=None):
        self.cm = ConfusionMatrix(n_class=n_class, class_names=class_names)

    def add(self, y_true, y_pred):
        self.cm.addData(y_true, y_pred)


def main():
    pass


if __name__ == "__main__":
    main()
