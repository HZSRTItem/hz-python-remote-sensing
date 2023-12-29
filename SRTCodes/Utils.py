# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : Utils.py
@Time    : 2023/6/12 10:16
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : BaseCodes of Utils
-----------------------------------------------------------------------------"""
import csv
import json
import math
import os
import random
import shutil
import time


def getRandom(x0, x1):
    return x0 + random.random() * (x1 - x0)


def changext(fn, ext=""):
    fn1, ext1 = os.path.splitext(fn)
    return fn1 + ext


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


def changefiledirname(filename, dirname):
    filename = os.path.split(filename)[1]
    return os.path.join(dirname, filename)


def readJson(json_fn):
    with open(json_fn, "r", encoding="utf-8") as f:
        return json.load(f)


def saveJson(d, json_fn):
    with open(json_fn, "w", encoding="utf-8") as f:
        json.dump(d, f)


def readLines(filename, strip_str="\n"):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if strip_str is not None:
            for i in range(len(lines)):
                lines[i] = lines[i].strip(strip_str)
        return lines


def filterFileExt(dirname=".", ext=""):
    filelist = []
    for f in os.listdir(dirname):
        if os.path.splitext(f)[1] == ext:
            filelist.append(os.path.join(dirname, f))
    return filelist


def filterFileContain(dirname=".", *filters):
    out_fns = []
    filter_list = []
    for filter_ele in filters:
        if isinstance(filter_ele, list) or isinstance(filter_ele, tuple) :
            filter_list += list(filter_ele)
        else:
            filter_list.append(str(filter_ele))
    for fn in os.listdir(dirname):
        n = 0
        for fiter_str in filter_list:
            if fiter_str in fn:
                n+=1
        if n == len(filter_list):
            out_fns.append(os.path.join(dirname, fn))
    return out_fns


def writeTexts(filename, *texts, mode="w", end=""):
    with open(filename, mode, encoding="utf-8") as f:
        for text in texts:
            f.write(str(text))
        f.write(end)


def copyFile(filename, to_filename):
    shutil.copyfile(filename, to_filename)


class DirFileName:
    """ Directory file name """

    def __init__(self, dirname=None, init_week_dir=False):
        self.dirname = dirname
        if dirname is None:
            self.dirname = os.getcwd()
        self._week_dir = r"F:\Week"
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


ofn = DirFileName()


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

    def start(self):
        """ 开始进度条 """
        self.is_run = True
        self._print()

    def add(self, n=1):
        """ 添加n个进度

        :param n: 进度的个数
        :return:
        """
        if self.is_run:
            self.n_current += n
            if self.n_current > self.n_print * self.n_split:
                self.n_print += 1
                if self.n_print > self.n_cols:
                    self.n_print = self.n_cols
            self._print()

    def setDesc(self, desc):
        """ 添加打印信息 """
        self.desc = desc

    def _print(self):
        des_info = "\r{0}: {1:>3d}% |".format(self.desc, int(self.n_current / self.total * 100))
        des_info += "*" * self.n_print + "-" * (self.n_cols - self.n_print)
        des_info += "| {0}/{1}".format(self.n_current, self.total)
        print(des_info, end="")

    def end(self):
        """ 结束进度条 """
        self.n_split = float(self.total) / float(self.n_split)
        self.n_current = 0
        self.n_print = 0
        self.is_run = False
        print()


class RumTime:

    def __init__(self, n_all=0):
        self.n_all = n_all
        self.n_current = 0
        self.strat_time = time.time()
        self.current_time = time.time()

    def strat(self):
        self.n_current = 0
        self.strat_time = time.time()
        self.current_time = time.time()

    def add(self, n=1):
        self.n_current += 1
        self.current_time = time.time()

    def printInfo(self):
        out_s = f"+ {self.n_current}"
        # time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
        out_s += " RUN:"
        out_s += RumTime.fmtTime(self.current_time - self.strat_time)
        if self.n_all != 0:
            out_s += " ALL:"
            t1 = (self.current_time - self.strat_time) / (self.n_current + 0.0000001) * self.n_all
            out_s += RumTime.fmtTime(t1)
        print(out_s)

    def end(self):
        print("end")

    @classmethod
    def fmtTime(cls, t):
        hours = t // 3600
        minutes = (t - 3600 * hours) // 60
        seconds = t - 3600 * hours - minutes * 60
        return f"({int(hours)}:{int(minutes)}:{seconds:.2f})"


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


def printDict(front_str, d):
    print(front_str)
    for k in d:
        print("  \"{0}\": {1}".format(k, d[k]))


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


def main():
    # dir_QDS2RGBN_gee_2_deal = DirFileName(r"G:\ImageData\QingDao\20211023\qd20211023\QDS2RGBN_gee_2_deal")
    # print(dir_QDS2RGBN_gee_2_deal.fn("sdafsd", "sfsad", "dsfasd"))
    #
    # df = pd.read_csv(r"F:\Week\20231112\Temp\tmp1.csv")
    # printListTable(df.values.tolist(), precision=2, rcl=">")
    pass


if __name__ == "__main__":
    main()
