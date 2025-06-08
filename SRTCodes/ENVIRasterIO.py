# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ENVIRasterIO.py
@Time    : 2023/6/23 15:53
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PytorchGeo of ENVIRasterIO
-----------------------------------------------------------------------------"""
import json
import os

import numpy as np

from SRTCodes.RasterIO import GEORaster


class ENVIRaster(GEORaster):
    """ ENVI Raster """

    def __init__(self, dat_fn=None, hdr_fn=None):
        super(ENVIRaster, self).__init__()

        self.dat_fn = None
        self.hdr_fn = None
        self.hdr = None
        self.names = []

        self.initENVIRaster(dat_fn, hdr_fn)
        self.coor_trans_info = None

    def initENVIRaster(self, dat_fn, hdr_fn):
        self.dat_fn = dat_fn
        self.hdr_fn = hdr_fn
        self.hdr = None

        self.initRaster()
        self.initGEORaster()

        if self.dat_fn is None and self.hdr_fn is None:
            return False

        if self.dat_fn is not None:
            tof1 = os.path.splitext(self.dat_fn)[0]
            if os.path.isfile(tof1 + ".dat"):
                self.dat_fn = tof1 + ".dat"
            elif os.path.isfile(tof1):
                self.dat_fn = tof1
            else:
                raise Exception("ENVIRaster init error: can not find data file.")

        if self.hdr_fn is None:
            self.hdr_fn = os.path.splitext(self.dat_fn)[0] + ".hdr"

        self._readHDR()
        self._initENVIRaster()
        self._initGeo()

    def _readHDR(self):
        lines = []
        with open(self.hdr_fn, "r", encoding="utf-8") as fr:
            line = fr.readline()
            if line.strip() != "ENVI":
                print("Warning: hdr file header line is not equal \"ENVI\". " + self.hdr_fn)
            for line in fr:
                if line.find("=") != -1:
                    lines.append(line)
                else:
                    lines[-1] += line
        self.hdr = {}
        for line in lines:
            line1 = line.split("=", 1)
            self.hdr[line1[0].strip()] = line1[1].strip()

        if "band names" in self.hdr:
            line = self.hdr["band names"]
            line = line.strip("{}")
            lines = line.split(",")
            self.names = [k.strip() for k in lines]

    def _initENVIRaster(self):
        self.n_rows = int(self.hdr["lines"])
        self.n_columns = int(self.hdr["samples"])
        self.n_channels = int(self.hdr["bands"])

    def _initGeo(self):
        if "map info" in self.hdr:
            lines = self.hdr["map info"].split(",")
            self.coor_trans_info = (float(lines[3]), float(lines[4]), float(lines[5]), float(lines[6]))
        if "coordinate system string" in self.hdr:
            self.srs = self.hdr["coordinate system string"].strip('{}')

    def readAsArray(self, interleave="r,c,b", *args, **kwargs):
        if "interleave" not in self.hdr:
            raise Exception("ENVIRaster read shape error: not find interleave. " + self.hdr_fn)
        """
1 = Byte: 8-bit unsigned integer
2 = Integer: 16-bit signed integer
3 = Long: 32-bit signed integer
4 = Floating-point: 32-bit single-precision
5 = Double-precision: 64-bit double-precision floating-point
6 = Complex: Real-imaginary pair of single-precision floating-point
9 = Double-precision complex: Real-imaginary pair of double precision floating-point
12 = Unsigned integer: 16-bit
13 = Unsigned long integer: 32-bit
14 = 64-bit long integer (signed)
15 = 64-bit unsigned long integer (unsigned)
        """
        n_data = self.n_rows * self.n_columns * self.n_channels
        with open(self.dat_fn, "rb") as frb:
            if self.hdr["data type"] == "1":
                self.d = np.frombuffer(frb.read(), dtype="byte", count=n_data)
            elif self.hdr["data type"] == "2":
                self.d = np.frombuffer(frb.read(), dtype="int16", count=n_data)
            elif self.hdr["data type"] == "3":
                self.d = np.frombuffer(frb.read(), dtype="int32", count=n_data)
            elif self.hdr["data type"] == "4":
                self.d = np.frombuffer(frb.read(), dtype="float32", count=n_data)
            elif self.hdr["data type"] == "5":
                self.d = np.frombuffer(frb.read(), dtype="float64", count=n_data)
            elif self.hdr["data type"] == "6":
                self.d = None
            elif self.hdr["data type"] == "9":
                self.d = None
            elif self.hdr["data type"] == "12":
                self.d = np.frombuffer(frb.read(), dtype="uint16", count=n_data)
            elif self.hdr["data type"] == "13":
                self.d = np.frombuffer(frb.read(), dtype="uint32", count=n_data)
            elif self.hdr["data type"] == "14":
                self.d = np.frombuffer(frb.read(), dtype="int64", count=n_data)
            elif self.hdr["data type"] == "15":
                self.d = np.frombuffer(frb.read(), dtype="uint64", count=n_data)
            else:
                raise Exception("Can not find \"data type\"=" + self.hdr["data type"])

        self.d = np.array(self.d)

        if self.hdr["interleave"].lower() == "bsq":
            self.d = self.d.reshape([self.n_channels, self.n_rows, self.n_columns])
            if interleave == "r,c,b":
                self.d = np.transpose(self.d, axes=(1, 2, 0))
        elif self.hdr["interleave"].lower() == "bip":
            self.d = self.d.reshape([self.n_rows, self.n_columns, self.n_channels])
            if interleave == "b,r,c":
                self.d = np.transpose(self.d, axes=(2, 0, 1))
        elif self.hdr["interleave"].lower() == "bil":
            self.d = self.d.reshape([self.n_rows, self.n_channels, self.n_columns])
            if interleave == "b,r,c":
                self.d = np.transpose(self.d, axes=(1, 0, 2))
            if interleave == "r,c,b":
                self.d = np.transpose(self.d, axes=(0, 2, 1))

        return self.d

    def save(self, imd: np.ndarray, dat_fn=None, hdrs=None, *args, **kwargs):
        if self.hdr is None:
            raise Exception("ENVIRaster hdr error: can not init hdr.")
        hdr_fn = self.hdr_fn
        if dat_fn is None:
            dat_fn = self.dat_fn
        else:
            hdr_fn = os.path.splitext(dat_fn)[0] + ".hdr"

        hdr = self.hdr.copy()
        if hdrs is not None:
            for k in hdrs:
                hdr[k] = hdrs[k]
        for k in kwargs:
            hdr[k] = kwargs[k]

        with open(hdr_fn, "w", encoding="utf-8") as f:
            f.write("ENVI\n")
            for k in hdr:
                f.write(k)
                f.write(" = ")
                f.write(hdr[k])
                f.write("\n")

        with open(dat_fn, "wb") as f:
            imd.tofile(f)

    def hdrToJson(self, json_file):
        with open(json_file, "w", encoding="utf-8") as fw:
            json.dump(self.hdr, fw)
        return self.hdr

    def coorRaster2Geo(self, row=0, column=0):
        if self.coor_trans_info is not None:
            x = column * self.coor_trans_info[2] + self.coor_trans_info[0]
            y = -row * self.coor_trans_info[3] + self.coor_trans_info[1]
            return x, y
        else:
            return None

    def coorGeo2Raster(self, x=0.0, y=0.0):
        if self.coor_trans_info is not None:
            row = (self.coor_trans_info[1] - y) / self.coor_trans_info[3]
            column = (x - self.coor_trans_info[0]) / self.coor_trans_info[2]
            return row, column
        else:
            return None


def main():
    pass


if __name__ == "__main__":
    main()
