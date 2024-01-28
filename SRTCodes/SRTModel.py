# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SRTModel.py
@Time    : 2023/12/9 21:53
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of SRTModel
-----------------------------------------------------------------------------"""
import os
from datetime import datetime

from SRTCodes.Utils import saveJson, readJson


class SRTModelInit:

    def __init__(self):
        self.name = "MODEL"
        self.save_dict = {}

    def train(self, *args, **kwargs):
        train_args = {"name": self.name}
        return train_args

    def predict(self, *args, **kwargs):
        return 0

    def score(self, *args, **kwargs):
        return 0

    def load(self, *args, **kwargs):
        return True

    def save(self, *args, **kwargs):
        return True

    def __getitem__(self, item):
        return self.save_dict[item]

    def __setitem__(self, key, value):
        self.save_dict[key] = value

    def saveDict(self, filename, *args, **kwargs):
        for k in kwargs:
            self.save_dict[k] = kwargs[k]
        self.save_dict["__BACK__"] = list(args)
        saveJson(self.save_dict, filename)

    def loadDict(self, filename):
        self.save_dict = readJson(filename)


class SRTHierarchicalModel(SRTModelInit):

    def __init__(self):
        super(SRTHierarchicalModel, self).__init__()

        self.models = {}
        self.train_data = {}
        self.test_data = {}
        self.save_dirname = None
        self.formatted_time = None

    def saveDirName(self, dirname=None):
        self.save_dirname = dirname
        if not os.path.isdir(self.save_dirname):
            os.mkdir(self.save_dirname)

    def timeDirName(self, dirname=None):
        if dirname is not None:
            self.save_dirname = dirname
        current_time = datetime.now()
        self.formatted_time = current_time.strftime("%Y%m%d%H%M%S")
        self.save_dirname = os.path.join(self.save_dirname, self.formatted_time)
        if not os.path.isdir(self.save_dirname):
            os.mkdir(self.save_dirname)

    def model(self, name, mod, *args, **kwargs):
        self.models[name] = mod

    def trainData(self, name, x, y, *args, **kwargs):
        self.train_data[name] = (x, y)

    def testData(self, name, x, y, *args, **kwargs):
        self.test_data[name] = (x, y)

    def train(self, *args, **kwargs):
        for name in self.models:
            x, y = self.train_data[name]
            self.models[name].train(x, y)

    def save(self, *args, **kwargs):
        for name in self.models:
            self.models[name].save(os.path.join(self.save_dirname, name + ".mod"))

    def load(self, dirname, *args, **kwargs):
        for name in self.models:
            self.models[name].load(os.path.join(dirname, name + ".mod"))


def main():
    pass


if __name__ == "__main__":
    main()
