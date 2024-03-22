# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowHModel.py
@Time    : 2024/2/26 20:33
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHHModel
-----------------------------------------------------------------------------"""
from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.SRTModel import SRTModelInit


class ShadowHierarchicalCategoryModel(SRTModelInit):

    def __init__(self, name="", c_names=None, field_names=None, colors=None, ):
        super(ShadowHierarchicalCategoryModel, self).__init__()
        if colors is None:
            colors = []
        if field_names is None:
            field_names = []
        if c_names is None:
            c_names = []

        self.name = name
        self.field_names = field_names
        self.c_names = c_names
        self.colors = colors

        self.cm = ConfusionMatrix()


class ShadowHierarchicalModel(SRTModelInit):

    def __init__(self):
        super(ShadowHierarchicalModel, self).__init__()
        self.models = []

    def train(self, *args, **kwargs):
        for mod in self.models:
            mod.train()


class SHHModel_T1(ShadowHierarchicalModel):

    def __init__(self):
        super(SHHModel_T1, self).__init__()
        self.veg_high_low_mod = ShadowHierarchicalCategoryModel()
        self.high_mod = ShadowHierarchicalCategoryModel()
        self.low_mod = ShadowHierarchicalCategoryModel()

    def train(self, *args, **kwargs):
        self.veg_high_low_mod.train()
        self.high_mod.train()
        self.low_mod.train()


def main():
    pass


if __name__ == "__main__":
    main()
