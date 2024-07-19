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

from SRTCodes.Utils import TimeName, SRTLog, SRTWriteText, DirFileName, saveJson, FRW


class TimeDirectory:

    def __init__(self, dirname, fmt="%Y%m%dH%H%M%S"):
        self.dirname = dirname
        if self.dirname is None:
            return
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        self.time_dfn = None
        self.initTimeName(fmt)
        self._log = SRTLog()
        self.save_dict = {}

    def initTimeName(self, fmt="%Y%m%dH%H%M%S"):
        if self.dirname is None:
            return
        self.time_dfn = DirFileName(TimeName(fmt, _dirname=self.dirname).dirname(is_mk=True))

    def initLog(self, log_fn="log.txt", mode="w", is_print=True):
        if self.dirname is None:
            return
        to_fn = self.time_dfn.fn(log_fn)
        self._log = SRTLog(to_fn, mode, is_print)
        return self

    def saveDF(self, name, df, *args, **kwargs):
        if self.dirname is None:
            return
        to_fn = self.time_dfn.fn(name)
        df.to_csv(to_fn, *args, **kwargs)
        return to_fn

    def saveSklearnModel(self, name, model):
        if self.dirname is None:
            return
        to_fn = self.time_dfn.fn(name)
        joblib.dump(model, to_fn)

    def saveTorchModel(self, name, model):
        if self.dirname is None:
            return
        to_fn = self.time_dfn.fn(name)
        # torch.save(model, to_fn)

    def log(self, *text, sep=" ", end="\n", is_print=None):
        if self.dirname is None:
            return
        self._log.log(*text, sep=sep, end=end, is_print=is_print)

    def wl(self, line, end="\n", is_print=None):
        if self.dirname is None:
            return
        return self._log.wl(line, end=end, is_print=is_print)

    def kw(self, key, value, sep=": ", end="\n", is_print=None):
        if self.dirname is None:
            return
        _key = key
        n = 0
        while _key in self.save_dict:
            n += 1
            _key = "{}_{}".format(_key, n)
        self.save_dict[_key] = value
        return self._log.kw(key, value, sep=sep, end=end, is_print=is_print)

    def buildWriteText(self, name, mode="w"):
        if self.dirname is None:
            return
        to_fn = self.time_dfn.fn(name)
        return SRTWriteText(to_fn, mode=mode)

    def copyfile(self, fn):
        if self.dirname is None:
            return
        name = os.path.split(fn)[1]
        to_fn = self.time_dfn.fn(name)
        copyfile(fn, to_fn)
        return to_fn

    def fn(self, name):
        if self.dirname is None:
            return
        return self.time_dfn.fn(name)

    def dn(self, name, is_mkdir=False):
        if self.dirname is None:
            return
        dirname = self.time_dfn.fn(name)
        if not os.path.isdir(dirname):
            if is_mkdir:
                os.mkdir(dirname)
        return dirname

    def saveJson(self, name, to_dict):
        if self.dirname is None:
            return
        saveJson(to_dict, self.fn(name), )

    def saveToJson(self, name):
        if self.dirname is None:
            return
        to_fn = self.time_dfn.fn(name)
        FRW(to_fn).saveJson(self.save_dict)

    def time_dirname(self):
        return self.time_dfn.fn()

def main():
    pass


if __name__ == "__main__":
    main()
