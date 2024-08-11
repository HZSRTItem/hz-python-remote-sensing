# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : Utils.py
@Time    : 2023/6/12 10:16
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : BaseCodes of Utils
-----------------------------------------------------------------------------"""
import csv
import inspect
import json
import math
import os
import random
import shutil
import time
import warnings
from datetime import datetime
from typing import Iterable

import numpy as np

from SRTCodes.DEFINE import StrOrBytesPath
from SRTCodes.SRTLinux import W2LF


def stoprun(fn, n_line):
    input("{0} {1}>".format(fn, n_line))


def getRandom(x0, x1):
    return x0 + random.random() * (x1 - x0)


def changext(fn, ext=""):
    fn1, ext1 = os.path.splitext(fn)
    return fn1 + ext


def changefilename(filename, fn):
    dirname = os.path.split(filename)[0]
    return os.path.join(dirname, fn)


def filterStringAnd(string_list, *filters):
    out_list = []
    for str_ in string_list:
        is_in = True
        for f in filters:
            if f not in str_:
                is_in = False
                break
        if is_in:
            out_list.append(str_)
    return out_list


def findfile(dirname, filename):
    filename = os.path.split(filename)[1]
    for root, dirs, files in os.walk(dirname):
        for fn in files:
            if fn == filename:
                return os.path.join(root, fn)
    return ""


def readcsv(csv_fn):
    d = {}
    with open(csv_fn, "r", encoding="utf-8-sig") as fr:
        cr = csv.reader(fr)
        ks = next(cr)
        for k in ks:
            d[k] = []
        for line in cr:
            for i, k in enumerate(line):
                d[ks[i]].append(k)
    return d


def readcsvlines(csv_fn):
    with open(csv_fn, "r", encoding="utf-8-sig") as fr:
        cr = csv.reader(fr)
        lines = [line for line in cr]
    return lines


def savecsv(csv_fn, d: dict):
    with open(csv_fn, "w", encoding="utf-8", newline="") as fr:
        cw = csv.writer(fr)
        ks = list(d.keys())
        cw.writerow(ks)
        for i in range(len(d[ks[0]])):
            cw.writerow([d[k][i] for k in ks])


def getfilext(fn):
    return os.path.splitext(fn)[1]


def getfilenamewithoutext(fn):
    fn = os.path.splitext(fn)[0]
    return os.path.split(fn)[1]


def getfilenme(fn):
    to_fn = os.path.split(fn)[1]
    return to_fn


def changefiledirname(filename, dirname):
    filename = os.path.split(filename)[1]
    return os.path.join(dirname, filename)


def numberfilename(init_fn, sep="", start_number=1):
    dirname = os.path.dirname(init_fn)
    if not os.path.isdir(dirname):
        raise Exception("Can not find dirname: {0}".format(dirname))
    fn, ext = os.path.splitext(init_fn)
    while True:
        to_fn = "{0}{1}{2}{3}".format(fn, sep, start_number, ext)
        if not os.path.isfile(to_fn):
            return to_fn
        start_number += 1


def readJson(json_fn):
    with open(json_fn, "r", encoding="utf-8") as f:
        return json.load(f)


def funcCodeString(func):
    if func is None:
        return ""
    try:
        return inspect.getsource(func)
    except Exception as ex:
        return str(ex)


def saveJson(d, json_fn):
    with open(json_fn, "w", encoding="utf-8") as f:
        json.dump(d, f)
    return d


def readLines(filename, strip_str="\n"):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if strip_str is not None:
            for i in range(len(lines)):
                lines[i] = lines[i].strip(strip_str)
        return lines


def readLinesList(filename, sep=" ", strip_str="\n", ):
    with open(filename, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            if strip_str is not None:
                line = line.strip(strip_str)
            lines.append(line.split(sep))
        return lines


def readText(filename, encoding="utf-8", **kwargs):
    with open(filename, mode="r", encoding=encoding, **kwargs) as f:
        return f.read()


def listfilename(fn_list):
    return [os.path.split(fn)[1] for fn in fn_list]


def filterFileExt(dirname=".", ext="", is_full=True):
    filelist = []
    for f in os.listdir(dirname):
        if os.path.splitext(f)[1] == ext:
            filelist.append(os.path.join(dirname, f))

    if not is_full:
        return listfilename(filelist)
    return filelist


def filterFileEndWith(dirname=".", end="", is_full=True):
    filelist = []
    for f in os.listdir(dirname):
        if f.endswith(end):
            filelist.append(os.path.join(dirname, f))
    if not is_full:
        return listfilename(filelist)
    return filelist


def filterFileContain(dirname=".", *filters, is_full=True):
    out_fns = []
    filter_list = []
    for filter_ele in filters:
        if isinstance(filter_ele, list) or isinstance(filter_ele, tuple):
            filter_list += list(filter_ele)
        else:
            filter_list.append(str(filter_ele))
    for fn in os.listdir(dirname):
        n = 0
        for fiter_str in filter_list:
            if fiter_str in fn:
                n += 1
        if n == len(filter_list):
            out_fns.append(os.path.join(dirname, fn))
    if not is_full:
        return listfilename(out_fns)
    return out_fns


def writeTexts(filename, *texts, mode="w", end=""):
    with open(filename, mode, encoding="utf-8") as f:
        for text in texts:
            f.write(str(text))
        f.write(end)
    return filename


def writeTextLine(filename, *texts, mode="a", end="\n"):
    writeTexts(filename, *texts, mode=mode, end=end)


def writeLines(filename, *lines, mode="w", sep="\n"):
    lines = datasCaiFen(lines)
    with open(filename, mode=mode, encoding="utf-8") as f:
        for line in lines:
            f.write(str(line))
            f.write(sep)
    return filename


def copyFile(filename, to_filename):
    shutil.copyfile(filename, to_filename)


def listMap(_list, _func):
    return list(map(_func, _list))


def writeCSVLine(filename, line: list):
    with open(filename, "a", encoding="utf-8", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(line)


class DirFileName:
    """ Directory file name """

    def __init__(self, dirname: StrOrBytesPath = None, init_week_dir=False):
        self.dirname = dirname
        if dirname is None:
            self.dirname = os.getcwd()
        self.dirname = W2LF(self.dirname)
        self._week_dir = W2LF(r"F:\Week")
        if init_week_dir:
            self.dirname = self._week_dir
        self._data_dir = self._getDataDir()

    def _getDataDir(self):
        dir_list = []
        for f in os.listdir(self._week_dir):
            ff = os.path.join(self._week_dir, f)
            if os.path.isdir(ff):
                if f.isdigit() and len(f) == 8:
                    dir_list.append(ff)
        return os.path.join(max(dir_list), "Data")

    def mkdir(self):
        if not os.path.isdir(self.dirname):
            os.mkdir(self.dirname)

    def fn(self, *names):
        """ add directory or file name in the end of cwd
        /
        :param names:
        :return:
        """
        if len(names) == 0:
            return self.dirname
        return os.path.join(self.dirname, *names)

    def listdir(self, ext=None):
        print("FILE:")
        for f in os.listdir(self.dirname):
            ff = os.path.join(self.dirname, f)
            if os.path.isfile(ff):
                if ext is not None:
                    if ext == os.path.splitext(f)[1]:
                        print("    " + f)
                else:
                    print("    " + f)
        print("DIRECTORY:")
        for f in os.listdir(self.dirname):
            ff = os.path.join(self.dirname, f)
            if os.path.isdir(ff):
                print("    " + f)

    def ddir(self, filename=""):
        return os.path.join(self._data_dir, filename)


def mkdir(dirname):
    dirname = os.path.abspath(dirname)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    return dirname


ofn = DirFileName()


def timeStringNow(str_fmt="%Y-%m-%d %H:%M:%S"):
    current_time = datetime.now()
    save_dirname = current_time.strftime(str_fmt)
    return save_dirname


class Jdt:
    """
    进度条
    """

    def __init__(self, total=100, desc=None, iterable=None, n_cols=20):
        """ 初始化一个进度条对象

        :param iterable: 可迭代的对象, 在手动更新时不需要进行设置
        :param desc: 字符串, 左边进度条描述文字
        :param total: 总的项目数
        :param n_cols: 调整进度条宽度, 默认是根据环境自动调节长度, 如果设置为0, 就没有进度条, 只有输出的信息
        """
        self.total = total
        self.iterable = iterable
        self.n_cols = n_cols
        self.desc = desc if desc is not None else ""

        self.n_split = float(total) / float(n_cols)
        self.n_current = 0
        self.n_print = 0
        self.is_run = False

        self.current_time = time.time()
        self.init_time = time.time()

    def start(self, is_jdt=True):
        """ 开始进度条 """
        if not is_jdt:
            return self
        self.is_run = True
        self._print()
        self.current_time = time.time()
        self.init_time = time.time()
        return self

    def add(self, n=1, is_jdt=True):
        """ 添加n个进度

        :param n: 进度的个数
        :return:
        """
        if not is_jdt:
            return
        if self.is_run:
            self.n_current += n
            self.current_time = time.time()
            if self.n_current > self.n_print * self.n_split:
                self.n_print += 1
                if self.n_print > self.n_cols:
                    self.n_print = self.n_cols
            self._print()

    def setDesc(self, desc):
        """ 添加打印信息 """
        self.desc = desc

    def fmttime(self):
        d_time = (self.current_time - self.init_time) / (self.n_current + 0.000001)

        def timestr(_time):

            tian = int(_time // (60 * 60 * 24))
            _time = _time % (60 * 60 * 24)
            shi = int(_time // (60 * 60))
            _time = _time % (60 * 60)
            fen = int(_time // 60)
            miao = int(_time % 60)

            if tian >= 1:
                return "{}D {:02d}:{:02d}:{:02d}".format(tian, shi, fen, miao)
            if shi >= 1:
                return "{:02d}:{:02d}:{:02d}".format(shi, fen, miao)
            return "{:02d}:{:02d}".format(fen, miao)

        n = 1 / (d_time + 0.000001)
        n_str = "{:.1f}".format(n)
        if len(n_str) >= 6:
            n_str = ">1000".format(n)

        fmt = "[{}<{}, {}it/s]            ".format(timestr(
            self.current_time - self.init_time), timestr(d_time * self.total), n_str)
        return fmt

    def _print(self):
        des_info = "\r{0}: {1:>3d}% |".format(self.desc, int(self.n_current / self.total * 100))
        des_info += "*" * self.n_print + "-" * (self.n_cols - self.n_print)
        des_info += "| {0}/{1}".format(self.n_current, self.total)
        des_info += " {}".format(self.fmttime())
        print(des_info, end="")

    def end(self, is_jdt=True):
        if not is_jdt:
            return
        """ 结束进度条 """
        self.n_split = float(self.total) / float(self.n_split)
        self.n_current = 0
        self.n_print = 0
        self.is_run = False
        print()


class RumTime:

    def __init__(self, n_all=0, func_print=print):
        self.n_all = n_all
        self.n_current = 0
        self.strat_time = time.time()
        self.current_time = time.time()
        self.func_print = func_print

    def strat(self):
        self.n_current = 0
        self.strat_time = time.time()
        self.current_time = time.time()
        return self

    def add(self, n=1):
        self.n_current += 1
        self.current_time = time.time()
        return self

    def printInfo(self):
        out_s = f"+ {self.n_current}/{self.n_all} "
        out_s += RumTime.fmtTime((self.current_time - self.strat_time) / (self.n_current + + 0.0000001)) + "/it "
        out_s += " RUN:"
        out_s += RumTime.fmtTime(self.current_time - self.strat_time)
        if self.n_all != 0:
            out_s += " ALL:"
            t1 = (self.current_time - self.strat_time) / (self.n_current + 0.0000001) * self.n_all
            out_s += RumTime.fmtTime(t1)
        self.func_print(out_s)

    def end(self):
        self.func_print("end")

    @classmethod
    def fmtTime(cls, t):
        hours = t // 3600
        minutes = (t - 3600 * hours) // 60
        seconds = t - 3600 * hours - minutes * 60
        return f"({int(hours)}:{int(minutes)}:{seconds:.2f})"


def is_in_poly(p, poly):
    px, py = p
    is_in = False
    for i, corner in enumerate(poly):
        next_i = i + 1 if i + 1 < len(poly) else 0
        x1, y1 = corner
        x2, y2 = poly[next_i]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in


class CoorInPoly:

    def __init__(self, coors=None):
        if coors is None:
            coors = []
        self.coors = coors

    def readCoors(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                lines = line.split(',')
                self.coors.append([float(lines[0]), float(lines[1])])

    def addCoor(self, x, y):
        self.coors.append([x, y])

    def t(self, x, y):
        poi = [x, y]
        sinsc = 0  # 交点个数
        for i in range(len(self.coors) - 1):  # [0,len-1]
            s_poi = self.coors[i]
            e_poi = self.coors[i + 1]
            # 输入：判断点，边起点，边终点，都是[lng,lat]格式数组
            if s_poi[1] == e_poi[1]:  # 排除与射线平行、重合，线段首尾端点重合的情况
                continue
            elif s_poi[1] > poi[1] and e_poi[1] > poi[1]:  # 线段在射线上边
                continue
            elif s_poi[1] < poi[1] and e_poi[1] < poi[1]:  # 线段在射线下边
                continue
            elif s_poi[1] == poi[1] and e_poi[1] > poi[1]:  # 交点为下端点，对应spoint
                continue
            elif e_poi[1] == poi[1] and s_poi[1] > poi[1]:  # 交点为下端点，对应epoint
                continue
            elif s_poi[0] < poi[0] and e_poi[1] < poi[1]:  # 线段在射线左边
                continue
            elif (e_poi[0] - (e_poi[0] - s_poi[0]) * (e_poi[1] - poi[1]) / (e_poi[1] - s_poi[1])) < poi[0]:
                # 交点在射线起点的左侧
                continue
            else:
                # print(s_poi[0], s_poi[1], sep=",", end="\n")
                # print(e_poi[0], e_poi[1], sep=",")
                sinsc += 1
        # print(sinsc)
        return sinsc % 2 == 1

    def t2(self, x, y):
        return is_in_poly([x, y], self.coors)


def printList(front_str, to_list, max_line_width=60):
    print(front_str)
    line_width = 2
    print("  ", end="")
    for name in to_list:
        str0 = "\"{0}\", ".format(name)
        line_width += len(str0)
        if line_width >= max_line_width:
            print("\n  ", end="")
            line_width = 2
        print(str0, end="")
    print()


def printLines(lines, front_str="", is_line_number=False, line_end="\n"):
    n = len(str(len(lines)))
    fmt_line_number = "{:>" + str(n) + "d} "
    for i, line in enumerate(lines):
        if is_line_number:
            print(fmt_line_number.format(i), end="")
        print(front_str, end="")
        print(line, end=line_end)


def catIterToStr(_iterable: Iterable[str], sep=" ", ) -> str:
    return sep.join(str(data) for data in _iterable)


def printDict(front_str, d, print_func=print, end=""):
    print_func(front_str)
    for i, k in enumerate(d):
        print_func("{:>2}. \"{}\": {}".format(i + 1, k, d[k]))
    print_func(end=end)


def printKeyValue(key, value, fs=""):
    print("{0}{1}: {2}".format(key, value, fs))


def angleToRadian(angle=1.0):
    return angle * math.pi / 180.0


def radianToAngle(radian=1.0):
    return radian * 180 / math.pi


def listUnique(t_list):
    new_li = list(set(t_list))
    new_li.sort(key=t_list.index)
    return t_list


class SRTMultipleOpening:
    """  Multiple Opening """

    def __init__(self, fn):
        self.filename = fn
        self.oids = {}
        self.initOIDS()

    def is_run(self, oid):
        """
        1. not in this == not run -> return 0                \
        2. in this and oids[this] is false == running -> 1   \
        3. in this and oids[this] is true == run end -> -1   \

        :param oid:
        :return: is run
        """
        self.initOIDS()
        if oid in self.oids:
            if self.oids[oid]:
                return -1
            else:
                return 1
        else:
            return 0

    def initOIDS(self):
        if self.oids:
            del self.oids
            self.oids = {}
        while True:
            try:
                if not os.path.isfile(self.filename):
                    break
                self.oids = readJson(self.filename)
                break
            except:
                time.sleep(0.1)

    def saveOIDS(self):
        saveJson(self.oids, self.filename)

    def add(self, oid):
        if self.is_run(oid) != 0:
            return False
        self.oids[oid] = False
        self.saveOIDS()
        return True

    def end(self, oid):
        is_r = self.is_run(oid)
        if is_r == 0:
            print("Warning: Can not find this oid:[{0}] in oids.".format(oid))
        elif is_r == 1:
            self.oids[oid] = True
        self.saveOIDS()


class SRTTablePrint:

    def __init__(self, column_names=None):
        self.c_names = {}


class _SRTPrintListTableFmt:
    """ _SRT Print List Table Fmt """

    class SRTPrintListTableFmt_Int:

        def __init__(self, n=0, rcl="^"):
            self.n = n
            self.fmt = ""
            self.rcl = rcl

        def getFmt(self):
            self.fmt = "{" + ":" + self.rcl + str(self.n) + "d}"
            return self.fmt

    class SRTPrintListTableFmt_Float:

        def __init__(self, n=0, precision=2, rcl="^"):
            self.fmt = ""
            self.n = n
            self.precision = precision
            self.rcl = rcl

        def getFmt(self):
            self.fmt = "{" + ":" + self.rcl + str(self.n) + "." + str(self.precision) + "f}"
            return self.fmt

    class SRTPrintListTableFmt_Str:

        def __init__(self, n=0, rcl="^"):
            self.n = n
            self.rcl = rcl
            self.fmt = ""

        def getFmt(self):
            self.fmt = "{" + ":" + self.rcl + str(self.n) + "}"
            return self.fmt

    class SRTPrintListTableFmt_Init:

        def __init__(self, precision=2):
            self.fmt_int = _SRTPrintListTableFmt.SRTPrintListTableFmt_Int()
            self.fmt_float = _SRTPrintListTableFmt.SRTPrintListTableFmt_Float(precision=precision)
            self.fmt_str = _SRTPrintListTableFmt.SRTPrintListTableFmt_Str()

        def initFmt(self, d):
            if isinstance(d, int):
                n = len(str(d))
                if self.fmt_int.n < n:
                    self.fmt_int.n = n
            elif isinstance(d, float):
                n = len(str(int(d)))
                if self.fmt_float.n < (n + self.fmt_float.precision + 1):
                    self.fmt_float.n = n + self.fmt_float.precision + 1
            else:
                n = len(str(d))
                if self.fmt_str.n < n:
                    self.fmt_str.n = n

        def fmt(self, d):
            if isinstance(d, int):
                return self.fmt_int.fmt.format(d)
            elif isinstance(d, float):
                return self.fmt_float.fmt.format(d)
            else:
                return self.fmt_str.fmt.format(d)

        def getFmt(self, rcl="^"):
            self.fmt_int.rcl = rcl
            self.fmt_int.getFmt()
            self.fmt_float.rcl = rcl
            self.fmt_float.getFmt()
            self.fmt_str.rcl = rcl
            self.fmt_str.getFmt()

    def __init__(self, n_columns, precision=2):
        self.n_columns = n_columns
        self.columns = [self.SRTPrintListTableFmt_Init(precision=precision) for _ in range(n_columns)]

    def __getitem__(self, n_column) -> SRTPrintListTableFmt_Init:
        return self.columns[n_column]

    def __len__(self):
        return self.n_columns


def printListTable(d_list, columns_names=None, precision=2, rcl="^"):
    if columns_names is not None:
        n_columns = len(columns_names)
    else:
        n_columns = len(d_list[0])
    fmt = _SRTPrintListTableFmt(n_columns, precision=precision)
    for line in d_list:
        for i in range(n_columns):
            fmt[i].initFmt(line[i])
    for i in range(fmt.n_columns):
        fmt[i].getFmt(rcl)
    for line in d_list:
        for i in range(n_columns):
            print(fmt[i].fmt(line[i]), end=" ")
        print()


def timeDirName(dirname=None, is_mk=False):
    current_time = datetime.now()
    save_dirname = current_time.strftime("%Y%m%dH%H%M%S")
    if dirname is not None:
        save_dirname = os.path.join(dirname, save_dirname)
    if is_mk:
        if not os.path.isdir(save_dirname):
            os.mkdir(save_dirname)
    return save_dirname


def timeFileName(fmt="{}", dirname=None):
    current_time = datetime.now()
    time_name = current_time.strftime("%Y%m%dH%H%M%S")
    to_fn = fmt.format(time_name)
    if dirname is not None:
        to_fn = os.path.join(dirname, to_fn)
    return to_fn


def isStringInt(_str: str):
    if _str.isdigit():
        return True
    if len(_str) == 0:
        return False
    try:
        int(_str)
        return True
    except ValueError:
        return False


def isStringFloat(_str: str):
    if "." not in _str:
        return False
    if len(_str) <= 1:
        return False
    try:
        float(_str)
        return True
    except ValueError:
        return False


def listAutoType(_list):
    d_type = "int"

    for data in _list:
        if not isinstance(data, str):
            d_type = "none"
            break
        if data == "":
            d_type = "string"
            break

        if isStringFloat(data):
            if d_type == "float":
                continue
            d_type = "float"
        elif isStringInt(data):
            if d_type == "float":
                continue
            d_type = "int"
        else:
            d_type = "string"
            break

    def as_column_type(_type):
        def func(_str):
            if _str == "":
                return None
            else:
                return _type(_str)

        return list(map(func, _list))

    if d_type == "int":
        return as_column_type(int)
    elif d_type == "float":
        return as_column_type(float)

    return _list


def dictAutoType(_dict):
    for k in _dict:
        _dict[k] = listAutoType(_dict[k])
    return _dict


class SRTDataFrame:

    def __init__(self):
        self._data = {}
        self._n_length = 0
        self._n_iter = 0
        self._index_name = None
        self._keys = []
        self._index = {}

        self._to_str_width = 100

    def copyEmpty(self):
        sdf = SRTDataFrame()
        sdf._data = {k: [] for k in self._data}
        return sdf

    def read_csv(self, csv_fn, index_column_name=None, is_auto_type=False):
        self._index_name = index_column_name
        self._data = {}
        self._index = {}
        with open(csv_fn, "r", encoding="utf-8-sig") as fr:
            self._n_length = 0
            cr = csv.reader(fr)
            ks = next(cr)
            for k in ks:
                self._data[self._initColumnName(k)] = []
            ks = list(self._data.keys())
            i = 0
            for line in cr:
                self._n_length += 1
                for i, k in enumerate(line):
                    self._data[ks[i]].append(k)
                self._index[i] = i
                i += 1
        if is_auto_type:
            for k in self._data:
                self.autoColumnType(k)
        for name in list(self._data.keys()):
            if not self._data[name]:
                self._data.pop(name)
        return self

    def indexColumnName(self, column_name):
        self._index_name = column_name
        self.buildIndex(d for d in self._data[column_name])

    def buildIndex(self, _index_iter):
        self._index = {}
        for i in range(self._n_length):
            _index = next(_index_iter)
            self._index[_index] = i

    def _initColumnName(self, name):
        if name == "":
            name = "_UName"
        if name not in self._data:
            return name
        new_name = "{0}_1".format(name)
        while True:
            if new_name not in self._data:
                break
        return new_name

    def autoColumnType(self, column_name):
        self._data[column_name] = listAutoType(self._data[column_name])

    def asColumnType(self, column_name, _type):
        def func(_str):
            if _str == "":
                return None
            else:
                return _type(_str)

        self._data[column_name] = list(map(func, self._data[column_name]))

    def keys(self):
        return self._data.keys()

    def data(self):
        return self._data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return self._n_length

    def __iter__(self):
        return self

    def __next__(self):
        if self._n_iter == 0:
            self._keys = list(self._data.keys())
        if self._n_iter == len(self._keys):
            self._n_iter = 0
            raise StopIteration()
        self._n_iter += 1
        return self._keys[self._n_iter - 1]

    def __contains__(self, item):
        return item in self._data

    def loc(self, _index, _column_name):
        _index = self._index[_index]
        return self._data[_column_name][_index]

    def toCSV(self, csv_fn):
        savecsv(csv_fn, self._data)

    def rowToDict(self, row):
        return {k: self._data[k][row] for k in self.data()}

    def __setitem__(self, key, value):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if self._n_length != len(value):
            warnings.warn("SRTDataFrame add field length of data [{0}] not equal this of [{1}]".format(
                len(value), self._n_length))
            if len(value) > self._n_length:
                for i in range(self._n_length, len(value)):
                    for k in self._data:
                        self._data[k].append(None)
            else:
                for i in range(len(value), self._n_length):
                    value.append(None)
        self._data[key] = value

    def addFields(self, *fields):
        to_fields = []
        for field in fields:
            if isinstance(field, list) or isinstance(field, tuple):
                to_fields += list(field)
            else:
                to_fields.append(field)
        for field in to_fields:
            self._data[field] = [None for _ in range(self._n_length)]
        return self

    def addLine(self, line=None):
        if line is None:
            line = {k: None for k in self._data}
        if isinstance(line, list) or isinstance(line, tuple):
            line = {k: line[i] for i, k in enumerate(self._data)}
        for k in self._data:
            if k in line:
                self._data[k].append(line[k])
            else:
                self._data[k].append(None)
        self._n_length += 1

    def toDict(self):
        return self._data.copy()

    def filterEQ(self, field_name, *data):
        data = list(data)
        sdf = self.copyEmpty()
        for i in range(self._n_length):
            line = self.rowToDict(i)
            if line[field_name] in data:
                sdf.addLine(line)
        return sdf


def sdf_read_csv(csv_fn):
    return SRTDataFrame().read_csv(csv_fn)


def datasCaiFen(datas):
    data_list = []
    for data in datas:
        if isinstance(data, list) or isinstance(data, tuple):
            data_list.extend(data)
        else:
            data_list.append(data)
    return data_list


def concatCSV(*csv_fns, to_csv_fn=None):
    csv_fns = datasCaiFen(csv_fns)
    sdf_list = [SRTDataFrame().read_csv(csv_fn) for csv_fn in csv_fns]
    keys = []
    for sdf in sdf_list:
        for k in sdf:
            if k not in keys:
                keys.append(k)
    sdf_cat = SRTDataFrame().addFields(*keys)
    for sdf in sdf_list:
        for j in range(len(sdf)):
            sdf_cat.addLine(sdf.rowToDict(j))
    if to_csv_fn is not None:
        sdf_cat.toCSV(to_csv_fn)
    return sdf_cat


class SRTFilter:

    def __init__(self):
        self.compare_func = lambda _x, _data: True
        self.compare_name = None
        self.compare_data = None

    def initCompare(self, name, data, func):
        self.compare_func = func
        self.compare_name = name
        self.compare_data = data

    def compare(self, x):
        return self.compare_func(x, self.compare_data)

    @staticmethod
    def eq(name, data):
        f = SRTFilter()

        def func(_x, _data):
            return _x == _data

        f.initCompare(name=name, data=data, func=func)
        return f

    @staticmethod
    def lt(name, data):
        f = SRTFilter()

        def func(_x, _data):
            return _x < _data

        f.initCompare(name=name, data=data, func=func)
        return f

    @staticmethod
    def gt(name, data):
        f = SRTFilter()

        def func(_x, _data):
            return _x > _data

        f.initCompare(name=name, data=data, func=func)
        return f

    @staticmethod
    def lte(name, data):
        f = SRTFilter()

        def func(_x, _data):
            return _x <= _data

        f.initCompare(name=name, data=data, func=func)
        return f

    @staticmethod
    def gte(name, data):
        f = SRTFilter()

        def func(_x, _data):
            return _x >= _data

        f.initCompare(name=name, data=data, func=func)
        return f

    @staticmethod
    def neq(name, data):
        f = SRTFilter()

        def func(_x, _data):
            return _x != _data

        f.initCompare(name=name, data=data, func=func)
        return f


class NONE_CLASS:

    def __init__(self):
        super(NONE_CLASS, self).__init__()


class SRTWriteText:

    def __init__(self, text_fn, mode="w", is_show=False):
        self.text_fn = text_fn
        self.is_show = is_show
        if self.text_fn is None:
            return
        if mode == "w":
            with open(self.text_fn, "w", encoding="utf-8") as f:
                f.write("")

    def write(self, *text, sep=" ", end="\n", is_show=None):
        with open(self.text_fn, "a", encoding="utf-8") as f:
            print(*text, sep=sep, end=end, file=f)
            if is_show is None:
                if self.is_show:
                    print(*text, sep=sep, end=end)
            else:
                if is_show:
                    print(*text, sep=sep, end=end)


class SRTDFColumnCal(SRTDataFrame):

    def __init__(self):
        super(SRTDFColumnCal, self).__init__()

    def fit(self, field_name, fit_func, *args, **kwargs):
        to_list = []
        jdt = Jdt(self._n_length, "SRTDFColumnCal:fit({0})".format(field_name)).start()
        for i in range(self._n_length):
            line = {k: self._data[k][i] for k in self.data()}
            to_list.append(fit_func(line, *args, **kwargs))
            jdt.add()
        jdt.end()
        self._data[field_name] = to_list
        return to_list

    def copyEmpty(self):
        sdf = SRTDFColumnCal()
        sdf._data = {k: [] for k in self._data}
        return sdf


class FN:

    def __init__(self, filename=""):
        self._filename = filename

    def fn(self):
        return self._filename

    def changext(self, ext):
        return changext(self._filename, ext)

    def changedirname(self, dirname):
        return changefiledirname(self._filename, dirname)

    def basename(self):
        return os.path.basename(self._filename)

    def dirname(self):
        return os.path.dirname(self._filename)

    def exists(self):
        return os.path.exists(self._filename)

    def getatime(self):
        return os.path.getatime(self._filename)

    def isfile(self):
        return os.path.isfile(self._filename)

    def realpath(self):
        return os.path.realpath(self._filename)

    def getfilext(self):
        return getfilext(self._filename)

    def getfilenamewithoutext(self):
        return getfilenamewithoutext(self._filename)


class DFN:

    def __init__(self, name="."):
        self.name = name

    def dirname(self):
        self.name = os.path.split(self.name)[0]
        return self

    def filterFileContain(self, *filters, is_full=True):
        return filterFileContain(self.name, *filters, is_full=is_full)

    def filterFileExt(self, ext="", is_full=True):
        return filterFileExt(self.name, ext, is_full=is_full)

    def filterFileEndWith(self, end="", is_full=True):
        return filterFileEndWith(self.name, end=end, is_full=is_full)


class SRTLog:

    def __init__(self, log_fn=None, mode="w", is_print=True):
        self.log_fn = log_fn
        if log_fn is None:
            return
        self.swt = SRTWriteText(text_fn=log_fn, mode=mode)
        self.is_print = is_print

    def _isPrint(self, is_print):
        if self.log_fn is None:
            return
        if is_print is None:
            is_print = self.is_print
        return is_print

    def log(self, *text, sep=" ", end="\n", is_print=None):
        if self.log_fn is None:
            return
        is_print = self._isPrint(is_print)
        if is_print:
            print(*text, sep=sep, end=end)
        if self.log_fn is not None:
            self.swt.write(*text, sep=sep, end=end)

    def wl(self, line, end="\n", is_print=None):
        if self.log_fn is None:
            return line
        self.log(line, end=end, is_print=is_print)
        return line

    def kw(self, key, value, sep=": ", end="\n", is_print=None):
        if self.log_fn is None:
            return value
        if isinstance(value, list) or isinstance(value, tuple):
            to_str = "\"" + "\", \"".join([str(data) for data in value]) + "\""
        else:
            to_str = str(value)
        self.log(key, to_str, sep=sep, end=end, is_print=is_print)
        return value


class CheckLoc:

    def __init__(self, n_row=None, n_column=None, not_rows=None, not_columns=None, ):
        self.n_row = n_row
        self.n_column = n_column
        self.not_rows = not_rows
        self.not_columns = not_columns

    def initLoc(self, n_row=None, n_column=None, not_rows=None, not_columns=None, ):
        self.n_row = n_row
        self.n_column = n_column
        self.not_rows = not_rows
        self.not_columns = not_columns

    def isLoc(self, i_row, i_column):
        def check_is(n_d, i_d, is_none=True):
            if n_d is not None:
                if isinstance(n_d, int):
                    if i_d == n_d:
                        return True
                    else:
                        return False
                elif isinstance(n_d, list):
                    if i_d in n_d:
                        return True
                    else:
                        return False
            else:
                return is_none

        is_row = check_is(self.n_row, i_row) and (not check_is(self.not_rows, i_row, is_none=False))
        is_column = check_is(self.n_column, i_column) and (not check_is(self.not_columns, i_column, is_none=False))

        return is_row and is_column


class TimeName:

    def __init__(self, fmt="%Y%m%dH%H%M%S", _dirname=""):
        self.time = datetime.now()
        self.time_str = self.time.strftime(fmt)
        self._dirname = _dirname

    def thisDirName(self):
        return self._dirname

    def dirname(self, dirname=None, is_mk=False):
        if dirname is None:
            dirname = self._dirname
        to_dirname = os.path.join(dirname, self.time_str)
        if is_mk:
            os.mkdir(to_dirname)
        return to_dirname

    def filename(self, fmt="{}", dirname=None):
        if dirname is None:
            dirname = self._dirname
        fn = fmt.format(self.time_str)
        fn = os.path.join(dirname, fn)
        return fn


class FRW:

    def __init__(self, filename):
        self.filename = filename

    def read(self):
        with open(self.filename, "r", encoding="utf-8") as f:
            return f.read()

    def readcsv(self):
        return readcsv(self.filename)

    def readcsvlines(self):
        return readcsvlines(self.filename)

    def readJson(self):
        return readJson(self.filename)

    def readLines(self, strip_str="\n", is_remove_empty=False):
        lines = readLines(filename=self.filename, strip_str=strip_str)
        if is_remove_empty:
            lines = [line for line in lines if line != ""]
        return lines

    def readLinesList(self, sep=" ", strip_str="\n", ):
        readLinesList(self.filename, sep=sep, strip_str=strip_str, )

    def savecsv(self, data):
        return savecsv(self.filename, data)

    def saveJson(self, data):
        return saveJson(data, self.filename)

    def writeCSVLine(self, line):
        return writeCSVLine(self.filename, line)

    def writeTextLine(self, *texts, mode="a", end="\n"):
        return writeTextLine(self.filename, *texts, mode=mode, end=end)

    def writeTexts(self, *texts, mode="w", end=""):
        return writeTexts(self.filename, *texts, mode=mode, end=end)

    def writeLines(self, *lines, mode="w", sep="\n"):
        return writeLines(self.filename, *lines, mode=mode, sep=sep)


class RunList:

    def __init__(self):
        self.sw = None
        self.run_dict = None
        self.n_run = None
        self.run_list = []
        self.dfn = None
        self.run_func = None
        self.name = None
        self.cmd_line = None

    def add(self, run_dict=None, run=False, *args, **kwargs):
        if run_dict is None:
            run_dict = {}
        self.run_list.append({"run": run, **run_dict, "args": list(args), **kwargs})
        return self.run_list[-1]

    def fit(self, name, dirname, cmd_line=None, is_run=True, run_func=None):
        self.name = name
        self.cmd_line = cmd_line
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        dirname = os.path.join(dirname, name)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        print("** RunList Dirname:", dirname)
        self.dfn = DirFileName(dirname)
        json_fn = self.dfn.fn("{}.json".format(name))
        if os.path.isfile(json_fn):
            self.run_list = readJson(json_fn)
        else:
            saveJson(self.run_list, json_fn)
            runlist_fn = self.dfn.fn("{}_cmd.bat".format(name))
            writeTexts(runlist_fn, "", mode="w")
            if cmd_line is not None:
                for _ in range(len(self.run_list)):
                    writeTexts(runlist_fn, cmd_line, mode="a", end="\n")

        if not is_run:
            return

        if run_func is None:
            warnings.warn("Run function is None.")
            return

        to_json_fn = changext(json_fn, "_run.json")
        if not os.path.isfile(to_json_fn):
            shutil.copyfile(json_fn, to_json_fn)
        self.run_list = readJson(to_json_fn)

        n_run = -1
        run_dict = {}
        for i in range(len(self.run_list)):
            if not self.run_list[i]["run"]:
                n_run = i
                run_dict = self.run_list[i]
                break

        if n_run == -1:
            print("#", "-" * 10, "End Run", n_run + 1, "-" * 10, "#")
            return

        print("#", "-" * 10, "Run", n_run + 1, "->", len(self.run_list), "-" * 10, "#")
        self.n_run = n_run
        self.run_dict = run_dict
        self.sw = SRTWriteText(self.dfn.fn("{}_save.txt".format(self.name)), mode="a")

        run_func(self)

        self.run_list[n_run]["run"] = True
        saveJson(self.run_list, to_json_fn)

    def show(self, i=-1):
        def _show():
            line = "[{}]".format(i + 1)
            print(line)
            for k in run_dict:
                print(" " * len(line), "*", k, ":", run_dict[k])

        if i == -1:
            for i, run_dict in enumerate(self.run_list):
                _show()
        else:
            run_dict = self.run_dict
            _show()

    def saveJsonData(self, name, data):
        saveJson(data, self.dfn.fn(name))
        return data

    def readJsonData(self, name):
        return readJson(self.dfn.fn(name))


class _FieldRecord:

    def __init__(self, name, n=0):
        self.name = name
        self.data = [None for _ in range(n)]
        self.fmt = "{}"

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return len(self.data)

    def toString(self, n, fmt=None):
        if fmt is None:
            fmt = "{}"
        return fmt.format(self.data[n])

    def add(self, data=None):
        self.data.append(data)
        return data


class FieldRecords:

    def __init__(self, *names, n=0):
        self._data_dict = {}
        self._name_n = []
        self.line = {}
        self._length = n

        self.addFields(*names)

        self.func_line_update = None

    def __getitem__(self, item) -> _FieldRecord:
        return self._data_dict[item]

    def addFields(self, *names):
        names = datasCaiFen(names)
        for name in names:
            self._data_dict[name] = _FieldRecord(name, self._length)
            self._name_n.append(name)
            self.line[name] = None
        return names

    def __setitem__(self, key, value):
        if len(key) == 1:
            self._data_dict[key][-1] = value
        else:
            self._data_dict[key[0]][key[1]] = value

    def add(self, *data, dt=None, **kwargs):
        for i, d in enumerate(data):
            self.line[self._name_n[i]] = d
        if dt is not None:
            for k in dt:
                if k in self.line:
                    self.line[k] = dt[k]
        for k in kwargs:
            if k in self.line:
                self.line[k] = kwargs[k]
        return self.addLine()

    def addLine(self, is_none=False):
        for k in self._data_dict:
            if is_none:
                self._data_dict[k].add()
            else:
                self._data_dict[k].add(self.line[k])
        if self.func_line_update is not None:
            self.func_line_update(self.line)
        return self.line

    def clearLine(self):
        for k in self.line:
            self.line[k] = None


class TimeRunning:

    def __init__(self, total):
        self.n = 0
        self.total = total
        self.init_time = time.time()
        self.fmt = ""

    def start(self):
        self.init_time = time.time()
        return self

    def show(self, is_new_line=False):
        fmt = self.fmt
        if is_new_line:
            print("\r", end="")
        else:
            print("\r{}".format(fmt), end="")
        return fmt

    def clearShow(self):
        print("\r", end="")

    def add(self, is_show=True, is_new_line=False):
        time_tmp = time.time()
        self.n += 1
        d_time = (time_tmp - self.init_time) / (self.n + 0.000001)

        def timestr(_time):

            tian = int(_time // (60 * 60 * 24))
            _time = _time % (60 * 60 * 24)
            shi = int(_time // (60 * 60))
            _time = _time % (60 * 60)
            fen = int(_time // 60)
            miao = int(_time % 60)

            if tian >= 1:
                return "{}D {:02d}:{:02d}:{:02d}".format(tian, shi, fen, miao)
            if shi >= 1:
                return "{:02d}:{:02d}:{:02d}".format(shi, fen, miao)
            return "{:02d}:{:02d}".format(fen, miao)

        n = 1 / (d_time + 0.000001)
        n_str = "{:.1f}".format(n)
        if len(n_str) >= 6:
            n_str = ">1000".format(n)
        fmt = "[{}<{}, {}it/s]".format(timestr(time_tmp - self.init_time), timestr(d_time * self.total), n_str)
        self.fmt = fmt
        if is_show:
            self.show(is_new_line=is_new_line)
        return fmt


class TimeDeltaRecord:

    def __init__(self, filename):
        self.sw = SRTWriteText(filename, "a")
        self.sw.write("TimeDeltaRecord")
        self.times = {}
        self.n = 0
        self.delta_times = []
        self.n_save = 1

    def update(self, name):
        self.times[name] = time.time()

    def add(self, n_save=10):
        keys = list(self.times.keys())
        for i in range(len(keys) - 1):
            d_time = self.times[keys[i + 1]] - self.times[keys[i]]
            if i < len(self.delta_times):
                self.delta_times[i][1] += d_time
            else:
                self.delta_times.append(["{}->{}".format(keys[i], keys[i + 1]), d_time])
        self.n += 1
        if self.n >= n_save:
            self.sw.write("{:>2d}.".format(self.n_save), end=" ")
            for i, (name, d_time) in enumerate(self.delta_times):
                d_time = d_time / self.n
                self.sw.write("{}:{:<10.6f}".format(name, d_time), end="  ")
                self.delta_times[i][1] = 0
            self.sw.write()
            self.n_save += 1
            self.n = 0


def printLinesStringTable(lines, loc="<", is_fist_line=True, print_func=print):
    fmt_n = [0 for i in range(len(lines[0]))]
    for line in lines:
        for i, d in enumerate(line):
            if fmt_n[i] < len(d):
                fmt_n[i] = len(d)
    fmts = ["{:" + loc + str(n) + "}" for n in fmt_n]
    for i, line in enumerate(lines):
        print_func(*[fmts[j].format(line[j]) for j in range(len(fmts))], sep=" | ")
        if is_fist_line and (i == 0):
            print_func(*["-" * n for n in fmt_n], sep="-+-")


def main():
    def func1():
        dir_QDS2RGBN_gee_2_deal = DirFileName(r"G:\ImageData\QingDao\20211023\qd20211023\QDS2RGBN_gee_2_deal")
        print(dir_QDS2RGBN_gee_2_deal.fn("sdafsd", "sfsad", "dsfasd"))

        # df = pd.read_csv(r"F:\Week\20231112\Temp\tmp1.csv")
        # printListTable(df.values.tolist(), precision=2, rcl=">")

        sdf = SRTDataFrame().read_csv(r"F:\ProjectSet\Shadow\Analysis\11\qd\QJY_1\qd_qjy_1-back.csv", is_auto_type=True)

        fn = numberfilename(r"F:\ProjectSet\Shadow\Hierarchical\Images\temp\tmp.txt")
        with open(fn, "w", encoding="utf-8") as f:
            f.write(fn)

    def func2():
        rl = RunList()

        for city_name in ["qd", "bj", "cd"]:
            for train_type in [
                {"spectral": "CNN", "texture": False},
                {"spectral": "Transformer", "texture": False},
                {"spectral": "CNN", "texture": True},
                {"spectral": "Transformer", "texture": True},
            ]:
                rl.add({"type": "training", "city_name": city_name, "model": train_type})
                for imdc_mod in [2, 10, 90]:
                    rl.add({"type": "imdc", "city_name": city_name, "imdc": {"models": imdc_mod}})

        rl.run_list.clear()
        rl.add({"city_name": "bj", "model": {'spectral': 'CNN', 'texture': False},
                "dirname": r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240630H105741", "imdc": 89})
        rl.add({"city_name": "bj", "model": {'spectral': 'Transformer', 'texture': False},
                "dirname": r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240630H110340", "imdc": 89})
        rl.add({"city_name": "bj", "model": {'spectral': 'CNN', 'texture': True},
                "dirname": r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240630H111158", "imdc": 89})
        rl.add({"city_name": "bj", "model": {'spectral': 'Transformer', 'texture': True},
                "dirname": r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240630H114524", "imdc": 89})

        rl.add({"city_name": "cd", "model": {'spectral': 'CNN', 'texture': False},
                "dirname": r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240630H122133", "imdc": 89})
        rl.add({"city_name": "cd", "model": {'spectral': 'Transformer', 'texture': False},
                "dirname": r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240630H122728", "imdc": 89})
        rl.add({"city_name": "cd", "model": {'spectral': 'CNN', 'texture': True},
                "dirname": r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240630H123521", "imdc": 89})
        rl.add({"city_name": "cd", "model": {'spectral': 'Transformer', 'texture': True},
                "dirname": r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240630H130543", "imdc": 89})

        rl.add({"city_name": "qd", "model": {'spectral': 'CNN', 'texture': False},
                "dirname": r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240630H133802", "imdc": 89})
        rl.add({"city_name": "qd", "model": {'spectral': 'Transformer', 'texture': False},
                "dirname": r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240630H134415", "imdc": 89})
        rl.add({"city_name": "qd", "model": {'spectral': 'CNN', 'texture': True},
                "dirname": r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240630H135241", "imdc": 89})
        rl.add({"city_name": "qd", "model": {'spectral': 'Transformer', 'texture': True},
                "dirname": r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240630H142608", "imdc": 89})

        rl.show()

        model_dirname = r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods"
        model_name = "VHL_ST"

        class DeepLearning:

            def __init__(self, *args, **kwargs):
                self.name = "name"

            def main(self, *args, **kwargs):
                return

            def train(self, *args, **kwargs):
                print("training", *args)
                return ""

            def imdc(self, *args, **kwargs):
                print("imdc", *args)
                return

        def run_func(_rl: RunList):
            run_dict = _rl.run_dict
            if run_dict["type"] == "training":
                time.sleep(1)
                current_time = datetime.now()
                city_name = None
                if run_dict["city_name"] == "qd":
                    city_name = "QingDao"
                elif run_dict["city_name"] == "bj":
                    city_name = "BeiJing"
                elif run_dict["city_name"] == "cd":
                    city_name = "ChengDu"

                dl = DeepLearning(city_name=run_dict["city_name"])
                dl.main(
                    model_name="VHL_ST",
                    build_model_dict=run_dict,
                    class_names=("NOT_KNOW", "IS", "VEG", "SOIL", "WAT"),
                    win_size=(21, 21),
                    model_dirname=model_dirname,
                    epochs=100,
                )
                to_dirname = dl.train(run_dict)

                sw = SRTWriteText(_rl.dfn.fn("{}_save.txt".format(_rl.name)), mode="a")

                sw.write("{}\n{} DL VHL4 {} {}\n".format(
                    current_time.strftime("%Y年%m月%d日%H:%M:%S"), city_name,
                    run_dict["model"], to_dirname
                ))

                _rl.saveJsonData("build_model_dict", run_dict)
                _rl.saveJsonData("to_dirname", to_dirname)

            elif run_dict["type"] == "imdc":

                build_model_dict = _rl.readJsonData("build_model_dict")
                to_dirname = _rl.readJsonData("to_dirname")

                dl = DeepLearning(city_name=run_dict["city_name"])
                dl.main(
                    model_name=model_name,
                    build_model_dict=build_model_dict,
                    class_names=("NOT_KNOW", "IS", "VEG", "SOIL", "WAT"),
                    win_size=(21, 21),
                    model_dirname=model_dirname,
                    epochs=100,
                )
                mod_fn = os.path.join(to_dirname, "{}_epoch{}.pth".format(model_name, run_dict["imdc"]["models"]))
                print("Imdc: ", mod_fn)
                dl.imdc(mod_fn, run_dict)

        rl.fit(
            name="Test1",
            dirname=r"F:\ProjectSet\Shadow\Hierarchical\Run\Temp",
            cmd_line=r'''python -c "import sys; sys.path.append(r'F:\PyCodes'); from SRTCodes.Utils import main; main()" %* ''',
            is_run=True,
            run_func=run_func,
        )

    def func3():
        jdt = Jdt(120, "Test").start()
        for i in range(120):
            time.sleep(0.1)
            jdt.add()
        jdt.end()

    def func4():
        fr = FieldRecords()
        fr[1:2, 3] = 6

    func4()
    return


if __name__ == "__main__":
    main()
