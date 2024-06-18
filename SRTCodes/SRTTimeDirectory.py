# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SRTTimeDirectory.py
@Time    : 2024/6/12 10:53
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SRTTimeDirectory
-----------------------------------------------------------------------------"""
import os
from shutil import copyfile

import joblib

from SRTCodes.Utils import TimeName, SRTLog, SRTWriteText, DirFileName


class TimeDirectory:

    def __init__(self, dirname, fmt="%Y%m%dH%H%M%S"):
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        self.dirname = dirname
        self.time_dfn = None
        self.initTimeName(fmt)
        self._log = SRTLog()
        self.save_dict = {}

    def initTimeName(self, fmt="%Y%m%dH%H%M%S"):
        self.time_dfn = DirFileName(TimeName(fmt, _dirname=self.dirname).dirname(is_mk=True))

    def initLog(self, log_fn="log.txt", mode="w", is_print=True):
        to_fn = self.time_dfn.fn(log_fn)
        self._log = SRTLog(to_fn, mode, is_print)

    def saveDF(self, name, df, *args, **kwargs):
        to_fn = self.time_dfn.fn(name)
        df.to_csv(to_fn, *args, **kwargs)

    def saveSklearnModel(self, name, model):
        to_fn = self.time_dfn.fn(name)
        joblib.dump(model, to_fn)

    def saveTorchModel(self, name, model):
        to_fn = self.time_dfn.fn(name)
        # torch.save(model, to_fn)

    def log(self, *text, sep=" ", end="\n", is_print=None):
        self._log.log(*text, sep=sep, end=end, is_print=is_print)

    def wl(self, line, end="\n", is_print=None):
        return self._log.wl(line, end=end, is_print=is_print)

    def kw(self, key, value, sep=": ", end="\n", is_print=None):
        return self._log.kw(key, value, sep=sep, end=end, is_print=is_print)

    def buildWriteText(self, name, mode="w"):
        to_fn = self.time_dfn.fn(name)
        return SRTWriteText(to_fn, mode=mode)

    def copyfile(self, fn):
        name = os.path.split(fn)[1]
        to_fn = self.time_dfn.fn(name)
        copyfile(fn, to_fn)
        return to_fn

    def fn(self, name):
        return self.time_dfn.fn(name)

    def dn(self, name, is_mkdir=False):
        dirname = self.time_dfn.fn(name)
        if not os.path.isdir(dirname):
            if is_mkdir:
                os.mkdir(dirname)
        return dirname


def main():
    pass


if __name__ == "__main__":
    main()
