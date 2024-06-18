# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2Config.py
@Time    : 2024/6/8 16:57
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2Config
-----------------------------------------------------------------------------"""
import os.path
from datetime import datetime
from shutil import copyfile

QD_ENVI_FN = r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\SHH2_QD2_envi.dat"
BJ_ENVI_FN = r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\SHH2_BJ2_envi.dat"
CD_ENVI_FN = r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\SHH2_CD2_envi.dat"

QD_NPY_FN = r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\SHH2_QD2_data.npy"
BJ_NPY_FN = r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\SHH2_BJ2_data.npy"
CD_NPY_FN = r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\SHH2_CD2_data.npy"

QD_LOOK_FN = r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\SHH2_QD2_look.tif"
BJ_LOOK_FN = r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\SHH2_BJ2_look.tif"
CD_LOOK_FN = r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\SHH2_CD2_look.tif"


class _SHH2RS:

    def __init__(self, name, init_dirname):
        self.init_dirname = init_dirname
        self.name = name
        self.this_filename = os.path.join(self.init_dirname, name)
        self.this_init_dirname = os.path.join(self.init_dirname, self.name)
        if not os.path.isdir(self.this_init_dirname):
            os.mkdir(self.this_init_dirname)
        self.current_time_str = ""
        self.filelist = []

    def add(self, fn, description="", is_update=False):
        fn = os.path.abspath(fn)
        current_time = datetime.now()
        time_str = current_time.strftime("%Y%m%d%H%M%S")
        self.filelist.append({"filename": fn, "time_str": time_str, "description": description,})
        to_fn = self._tofn(fn, time_str)
        copyfile(fn, to_fn)

    def _tofn(self, fn, time_str):
        to_fn = os.path.split(fn)[1]
        to_fn = "{}_{}".format(time_str, to_fn)
        to_fn = os.path.join(self.this_init_dirname, to_fn)
        return to_fn

    def update(self, time_str):
        file_tors = self.find(time_str)
        to_fn = self._tofn(file_tors["filename"], file_tors["time_str"])

    def find(self, time_str):
        for line in self.filelist:
            if line["time_str"] == time_str:
                return line
        return None


class SHH2ReleaseSamples:

    def __init__(self):
        self.init_dirname = r"F:\ProjectSet\Shadow\Hierarchical\Samples\SaveSamples"
        self.current_fn = ""
        self.filelist


def main():
    pass


if __name__ == "__main__":
    main()
