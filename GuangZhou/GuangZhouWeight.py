# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : GuangZhouWeight.py
@Time    : 2023/7/20 18:57
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PytorchGeo of GuangZhouWieght
-----------------------------------------------------------------------------"""
import random

import numpy as np
import pandas as pd

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.NumpyUtils import printArrayUnique
from SRTCodes.SRTCollection import SRTCollection
from SRTCodes.Utils import printList


class CategoryWeight:

    def __init__(self, name, year=0, resolution=0, ua=(0.0, 0.0), pa=(0.0, 0.0), oa=0.0):
        self.name = name
        self.year = year
        self.resolution = resolution
        self.category = None
        self.ua = ua
        self.pa = pa
        self.oa = oa

    def print(self):
        print("* {0}:".format(self.name))
        print("  + Year: {0}".format(self.year))
        print("  + Resolution: {0}".format(self.resolution))
        print("  + UA: {0}".format(self.ua))
        print("  + PA: {0}".format(self.pa))
        print("  + OA: {0}".format(self.oa))
        printArrayUnique(self.category)


class CategoryWeightCollection(SRTCollection):

    def __init__(self):
        super(CategoryWeightCollection, self).__init__()
        self._dict = {}

    def add(self, name, year=0, resolution=0, ua=(0.0, 0.0), pa=(0.0, 0.0), oa=0.0):
        self._dict[name] = CategoryWeight(name, year=year, resolution=resolution, ua=ua, pa=pa, oa=oa)
        self._n_next.append(name)

    def __getitem__(self, name) -> CategoryWeight:
        return self._dict[name]

    def print(self):
        for k in self._dict:
            self._dict[k].print()


class GuangZhouWeight:

    def __init__(self, csv_fn):
        self._df = pd.read_csv(csv_fn)
        self.cw_coll = CategoryWeightCollection()
        self.r_30_10 = None
        self.category={}

    def add(self, name, year, resolution, to_c_func, ua=(0.0, 0.0), pa=(0.0, 0.0), oa=0.0):
        self.cw_coll.add(name=name, year=year, resolution=resolution, ua=ua, pa=pa, oa=oa)
        self.cw_coll[name].category = to_c_func(self._df[name]).values
        self.cw_coll[name].category = self.cw_coll[name].category * 1
        self.cw_coll[name].category = self.cw_coll[name].category.astype("int")

    def print(self):
        printList("DataFrame Keys: ", list(self._df.keys()))
        self.cw_coll.print()

    def toCsv(self, csv_fn):
        df = self._df.copy()
        for k in self.cw_coll:
            df["_CW_{0}".format(k)] = self.cw_coll[k].category
        df.to_csv(csv_fn)

    def categoryNames(self, *c_names):
        for k in c_names:
            self.category[k] = []


def fromglc10_2017_toCategory(x):
    return x == 80


def gaia_toCategory(x):
    return x >= 1


def msmt_toCategory(x):
    return x == 2


def ghsl_toCategory(x):
    return x >= 3


def duofenbianlvgailv(x1: np.ndarray):
    x1 = x1.astype("int").ravel()
    n1 = np.bincount(x1, minlength=2)
    n1 = n1.astype("float")
    n1 = n1 / np.sum(n1)
    return n1


def main():
    spl_csv_fn = r"G:\GraduationProject\AutoSample\Samples\gba_random_2537_spl.csv"
    # spl_csv_fn = r"G:\GraduationProject\AutoSample\Samples\Test\gz_test_spl_1_spl.csv"
    gzw = GuangZhouWeight(spl_csv_fn)
    # "fromglc10_2017", "gaia", "msmt", "ghsl",
    gzw.add("fromglc10_2017", 2017, 10, fromglc10_2017_toCategory,
            ua=(0.828082808, 0.925083612), pa=(0.942622951, 0.783569405), oa=0.867097121334409)
    gzw.add("gaia", 2018, 30, gaia_toCategory,
            ua=(0.784076158, 0.900426743), pa=(0.928278689, 0.717280453), oa=0.828087167070218)
    gzw.add("msmt", 2015, 30, msmt_toCategory,
            ua=(0.775313808, 0.925395629), pa=(0.949282787, 0.695750708), oa=0.828894269572236)
    gzw.add("ghsl", 2014, 30, ghsl_toCategory,
            ua=(0.691805656, 0.954118874), pa=(0.977459016, 0.518413598), oa=0.759483454398709)
    gzw.r_30_10 = [[0.964031440, 0.03596856], [0.199664589, 0.800335411], [0.916712222, 0.083287778]]

    gzw.print()
    gzw.toCsv(r"G:\GraduationProject\AutoSample\Samples\Test\gz_test_spl_1_spl_1.csv")

    pass


def method_name3():
    gr = GDALRaster(r"G:\GraduationProject\Data\imd\gba_imdc.xml")
    # with open(r"G:\GraduationProject\AutoSample\Samples\gz_t1.csv", "w", encoding="utf-8", newline="") as f:
    #     cw = csv.writer(f)
    #     cw.writerow(["category", "gaia 0", "gaia 1", "msmt 0", "msmt 1", "ghsl 0", "ghsl 1"])
    #     for i in range(10000):
    #         d = gr.readAsArrayCenter(x_row_center=random.randint(10, gr.n_rows - 10),
    #                                  y_column_center=random.randint(10, gr.n_columns - 10),
    #                                  win_row_size=3, win_column_size=3,
    #                                  is_geo=False)
    #         d1 = fromglc10_2017_toCategory(d[0])
    #         d2 = duofenbianlvgailv(gaia_toCategory(d[1]))
    #         d3 = duofenbianlvgailv(msmt_toCategory(d[2]))
    #         d4 = duofenbianlvgailv(ghsl_toCategory(d[3]))
    #         line = [int(d1[1, 1]), d2[0], d2[1], d3[0], d3[1], d4[0], d4[1]]
    #         cw.writerow(line)
    #         if i%1000 == 0:
    #             print(i)
    gailv = {
        "gaia": [{"n": 0, "arr": np.zeros(2)}, {"n": 0, "arr": np.zeros(2)}, {"n": 0, "arr": np.zeros(2)}],
        "msmt": [{"n": 0, "arr": np.zeros(2)}, {"n": 0, "arr": np.zeros(2)}, {"n": 0, "arr": np.zeros(2)}],
        "ghsl": [{"n": 0, "arr": np.zeros(2)}, {"n": 0, "arr": np.zeros(2)}, {"n": 0, "arr": np.zeros(2)}],
    }
    for i in range(100000):
        d = gr.readAsArrayCenter(x_row_center=random.randint(10, gr.n_rows - 10),
                                 y_column_center=random.randint(10, gr.n_columns - 10),
                                 win_row_size=3, win_column_size=3,
                                 is_geo=False)
        d1 = duofenbianlvgailv(fromglc10_2017_toCategory(d[0]))
        d2 = [gaia_toCategory(d[1]), msmt_toCategory(d[2]), ghsl_toCategory(d[3])]
        for j, k in enumerate(gailv):
            d3 = d2[j]
            if not d3[1, 1]:
                gailv[k][0]["n"] += 1
                gailv[k][0]["arr"] += d1
            else:
                gailv[k][1]["n"] += 1
                gailv[k][1]["arr"] += d1
            gailv[k][2]["n"] += 1
            gailv[k][2]["arr"] += d1

        if i % 1000 == 0:
            print(i)
    for k in gailv:
        for j in range(len(gailv[k])):
            gailv[k][j]["arr"] /= gailv[k][j]["n"]
    for k in gailv:
        print(k, end=" 0,")
    for k in gailv:
        print(k, end=" 1,")
    print()
    for i in range(3):
        for k in gailv:
            print(gailv[k][i]["arr"][0], end=",")
        for k in gailv:
            print(gailv[k][i]["arr"][1], end=",")
        print()


def method_name2():
    # spl_csv_fn = r"G:\GraduationProject\AutoSample\Samples\gba_random_2537_spl.csv" 0.942622951, 0.783569405
    spl_csv_fn = r"G:\GraduationProject\AutoSample\Samples\Test\gz_test_spl_1_spl.csv"
    gzw = GuangZhouWeight(spl_csv_fn)
    # "fromglc10_2017", "gaia", "msmt", "ghsl",
    gzw.add("fromglc10_2017", 2017, 10, fromglc10_2017_toCategory,
            ua=(0.828082808, 0.925083612), pa=(0.942622951, 0.783569405), oa=0.867097121334409)
    gzw.add("gaia", 2018, 30, gaia_toCategory,
            ua=(0.784076158, 0.900426743), pa=(0.928278689, 0.717280453), oa=0.828087167070218)
    gzw.add("msmt", 2015, 30, msmt_toCategory,
            ua=(0.775313808, 0.925395629), pa=(0.949282787, 0.695750708), oa=0.828894269572236)
    gzw.add("ghsl", 2014, 30, ghsl_toCategory,
            ua=(0.691805656, 0.954118874), pa=(0.977459016, 0.518413598), oa=0.759483454398709)
    gzw.print()
    gzw.toCsv(r"G:\GraduationProject\AutoSample\Samples\Test\gz_test_spl_1_spl_1.csv")


def method_name():
    gr = GDALRaster(r"G:\GraduationProject\Data\imd\gba_gaia_c.dat")
    d = gr.readAsArray()
    xd, xn = np.unique(d, return_counts=True)
    df = pd.DataFrame({"xd": xd, "xn": xn})
    df.to_csv(r"G:\GraduationProject\Data\huangxin\stats2.csv")
    print(df)


if __name__ == "__main__":
    main()
