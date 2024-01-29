# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SRTReadWrite.py
@Time    : 2023/01/29 16:53:13
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : 读取SRT类型的文件
           + srtinfo: *.srti
           + srtdata: *.srtd
-----------------------------------------------------------------------------"""
import csv

import numpy as np
import os


class SRTFileRW:
    """ 
    SRT 文件读写基类，实现了文件流的管理
    """

    def __init__(self, srt_filename) -> None:
        self.srt_filename = srt_filename
        self.f_stream = None
        pass

    def initFromFile(self, srt_file=None):
        self.srt_filename = srt_file

    def open(self, open_mode="r"):
        self.close()
        if open_mode == "rb" or open_mode == "wb":
            self.f_stream = open(self.srt_filename, open_mode)
        else:
            self.f_stream = open(self.srt_filename, open_mode, encoding="utf-8")

    def close(self):
        if self.f_stream is not None:
            self.f_stream.close()
            self.f_stream = None


class SRTInfoFileRW(SRTFileRW):
    """
    SRT information file read write
    """

    def __init__(self, srt_filename=None) -> None:
        """ build srt info file rw
        \
        :param srt_filename: SRT info file name
        """
        super(SRTInfoFileRW, self).__init__(srt_filename)
        self.mark = ""

    def close(self):
        """
        close SRT info file
        """
        super(SRTInfoFileRW, self).close()
        self.mark = ""

    def open(self, open_mode="r"):
        """ open SRT info file
        \
        :param open_mode: read:"r" write:"w" others like `open()`
        """
        super(SRTInfoFileRW, self).open()
        self.mark = ""

    def readLine(self):
        """
        read a line from file. not read Annotation line.
        If read mark line, record a mark and read next line.
        \
        :return: line
        """
        line = self.f_stream.readline()
        while line != "":
            if line.strip() == "":
                line = self.f_stream.readline()
            elif line[:1] == "#":
                line = self.f_stream.readline()
            elif line[:1] == ">":
                self.mark = line[1:].strip()
                line = self.f_stream.readline()
            else:
                return line[:-1]
        return line

    def readLines(self, mark=None):
        """
        read mark as `mark` all lines from file.
        If `mark is None`, return all lines
        \
        :param mark: `mark`
        :return: lines
        """
        self.open()
        line = self.readLine()
        lines = []
        if mark is None:
            while line != "":
                lines.append(line)
                line = self.readLine()
        else:
            while line != "":
                if self.mark == mark:
                    lines.append(line)
                line = self.readLine()
        self.close()
        return lines

    def readAsDict(self, marks: list = None):
        """
        Read Info file as dict[mark]:info.
        If `marks is not None`, read mark info in marks.
        \
        :param marks: read file info as marks
        :return: dict[mark]:info
        """
        self.open()
        line = self.f_stream.readline()
        d = {"": []}
        if marks is None:
            while line != "":
                if line.strip() == "":
                    line = self.f_stream.readline()
                elif line[:1] == "#":
                    line = self.f_stream.readline()
                elif line[:1] == ">":
                    self.mark = line[1:].strip()
                    if self.mark not in d:
                        d[self.mark] = []
                    line = self.f_stream.readline()
                else:
                    d[self.mark].append(line[:-1])
                    line = self.f_stream.readline()
        else:
            while line != "":
                if line.strip() == "":
                    line = self.f_stream.readline()
                elif line[:1] == "#":
                    line = self.f_stream.readline()
                elif line[:1] == ">":
                    self.mark = line[1:].strip()
                    if self.mark in marks:
                        if self.mark not in d:
                            d[self.mark] = []
                    line = self.f_stream.readline()
                else:
                    if self.mark in marks:
                        d[self.mark].append(line[:-1])
                    line = self.f_stream.readline()
        if "" in d:
            if not d[""]:
                d.pop("")
        # for k in d:
        #     if len(d[k]) == 1:
        #         d[k] = d[k][0]
        #     if not d[k]:
        #         d[k] = None
        self.close()
        return d

    def saveAsDict(self, d: dict):
        """
        save srt information to file.
        \
        :param d: srt information dict
        """
        self.close()
        self.open(open_mode="w")
        for k in d:
            self.writeLineMark(str(k))
            self.writeLine()
            if isinstance(d[k], list) or isinstance(d[k], tuple):
                for line in d[k]:
                    self.writeLine(str(line))
            elif isinstance(d[k], dict):
                for kk in d[k]:
                    self.writeLine(str(kk) + ": " + str(d[k][kk]))
            else:
                self.writeLine(str(d[k]))
            self.writeLine()
        self.close()

    def writeLine(self, line=""):
        """
        write info line to file
        \
        :param line: line string
        """
        self.f_stream.write(line)
        self.f_stream.write("\n")

    def writeLineAnnotation(self, line=""):
        """
        write annotation line to file
        \
        :param line: line string
        """
        self.f_stream.write("# ")
        self.f_stream.write(line)
        self.f_stream.write("\n")

    def writeLineMark(self, line=""):
        """
        write mark line to file
        \
        :param line: line string
        """
        self.f_stream.write("> ")
        self.f_stream.write(line)
        self.f_stream.write("\n")


class SRTDataFileRW(SRTFileRW):

    def __init__(self, srt_filename: str = None) -> None:
        super(SRTDataFileRW, self).__init__(srt_filename)
        self.d_type = "float32"

    def readAsNpArray(self, srt_d_fn: str = None):
        if srt_d_fn is not None:
            self.srt_filename = srt_d_fn
        self.open("rb")
        d_type = np.frombuffer(self.f_stream.read(1), dtype=np.int8)[0]
        d_shape_size = np.frombuffer(self.f_stream.read(1), dtype=np.int8)[0]
        d_shape = np.frombuffer(self.f_stream.read(d_shape_size * 4), dtype=np.int32)
        if d_type == 1:
            d = np.frombuffer(self.f_stream.read(), dtype=np.int64)
        elif d_type == 2:
            d = np.frombuffer(self.f_stream.read(), dtype=np.float32)
        else:
            raise "not find data type index as " + str(d_type)
        d = np.reshape(d, newshape=d_shape)
        d = np.array(d)
        self.close()
        return d

    def write(self, d: np.ndarray, srt_d_fn: str = None, d_type="float32"):
        if srt_d_fn is not None:
            self.srt_filename = srt_d_fn
        self.open("wb")
        if d_type == "int64":
            self.f_stream.write(b"\x01")
        elif d_type == "float32":
            self.f_stream.write(b"\x02")
        else:
            raise "not find data type as " + d_type
        self.f_stream.write(len(d.shape).to_bytes(1, "little"))
        np.array(d.shape).astype("int32").tofile(self.f_stream)
        d.astype(d_type).tofile(self.f_stream)
        self.close()


def readCsvAsDict(csv_file):
    dt = {}
    with open(csv_file, "r", encoding="utf-8-sig") as f:
        d = csv.DictReader(f)
        for fn in d.fieldnames:
            dt[fn] = []
        for row in d:
            for fn in d.fieldnames:
                v = row[fn]
                if fn.upper() == "SRT":
                    dt[fn].append(int(v))
                elif fn.upper() == "X":
                    dt[fn].append(float(v))
                elif fn.upper() == "Y":
                    dt[fn].append(float(v))
                else:
                    dt[fn].append(v)
    return dt


class SRTDataSet(SRTDataFileRW):
    """

    """

    def __init__(self, data_dir=None):
        super(SRTDataSet, self).__init__()
        self.data_dir = data_dir
        self.data_des_filename = None
        self.data_des = {}
        self.c_fn = None

    def init(self):
        if self.data_dir is None:
            return None
        if not os.path.isdir(self.data_dir):
            raise "Input data directory is not exist: " + self.data_dir
        if self.data_des_filename is None:
            data_des_filename = os.path.join(self.data_dir, "data_des.csv")
            if os.path.isfile(data_des_filename):
                self.data_des_filename = data_des_filename
            else:
                raise "Input data directory is not existing file: data_des.csv. dir: " + self.data_dir
        self.data_des = readCsvAsDict(self.data_des_filename)
        self.setCfn()

    def setCfn(self, c_fn: str = "category", is_int=True):
        """ set category field name
        \
        :param c_fn: category field name
        :param is_int: whether convert to int
        :return:  have field name ? `True` or `False`
        """
        if self.data_des is None:
            return False
        for k in self.data_des:
            if k.lower() == c_fn:
                self.c_fn = k
                if is_int:
                    self.data_des[k] = list(map(int, self.data_des[k]))
                return True
        raise "category field name `{0}` not find in data description".format(c_fn)

    def __getitem__(self, index):
        d = self.readAsNpArray(os.path.join(self.data_dir, str(self.data_des["SRT"][index]) + ".srtd"))
        return d

    def __len__(self):
        return self.length()

    def length(self):
        len0 = []
        for k in self.data_des:
            len0.append(len(self.data_des[k]))
        return max(len0)

    def setDataDir(self, data_dir=None):
        if data_dir is None:
            if self.data_dir is not None:
                if not os.path.isdir(self.data_dir):
                    os.mkdir(self.data_dir)
        else:
            if not os.path.isdir(data_dir):
                os.mkdir(data_dir)
            self.data_dir = data_dir

    def addField(self, field=None):
        if field is None:
            return None
        if isinstance(field, list):
            for field0 in field:
                if field0 in self.data_des:
                    print("warning: {0} in data_des".format(field0))
                else:
                    self.data_des[field0] = []
        else:
            if field in self.data_des:
                print("warning: {0} in data_des".format(field))
            else:
                self.data_des[field] = []

    def setDataDesFilename(self, data_des_filename=None):
        if data_des_filename is None:
            if self.data_dir is not None:
                self.data_des_filename = os.path.join(self.data_dir, "data_des.csv")
        else:
            self.data_des_filename = data_des_filename

    def addOneDes(self, data_des_0: list):
        for i, k in enumerate(self.data_des):
            self.data_des[k].append(data_des_0[i])

    def saveOneData(self, d, index=-1):
        to_f = os.path.join(self.data_dir, str(self.data_des["SRT"][index]) + ".srtd")
        self.write(d, to_f)


def main():
    pass


if __name__ == "__main__":
    main()
