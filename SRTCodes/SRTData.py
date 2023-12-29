# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SRTData.py
@Time    : 2023/7/1 14:58
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : BaseCodes of SRTData
-----------------------------------------------------------------------------"""
import os.path

import numpy as np
import pandas as pd


class SRTKeyValue:

    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value


class SRTKeyValues:

    def __init__(self, is_unique=False):
        self._collection = []
        self._keys = []
        self.is_unique = is_unique

    def add(self, key, value=None):
        if self.is_unique:
            if key in self._keys:
                raise Exception("This SRTKeyValueCollection key have to unique.")
        if key not in self._keys:
            self._keys.append(key)
        self._collection.append(SRTKeyValue(key=key, value=value))

    def __setitem__(self, key, value):
        for kv in self._collection:
            if kv.key == key:
                kv.value = value

    def __getitem__(self, key):
        if key in self._keys:
            d = list(filter(lambda x: x.key == key, self._collection))
            if self.is_unique:
                return d[0]
            else:
                return d
        else:
            if isinstance(key, int):
                return self._collection[key]
            else:
                return None

    def __contains__(self, key):
        return key in self._keys

    def __len__(self):
        return len(self._collection)


class SRTKeyValueCollection:

    def __init__(self):
        self._collections = []

    def add(self, *args, **kwargs):
        d = {}
        for i in range(0, len(args) - 1, 2):
            d[args[i]] = args[i + 1]
        for k in kwargs:
            d[k] = kwargs[k]
        self._collections.append(d)

    def __len__(self):
        return len(self._collections)

    def __getitem__(self, item):
        return self._collections[item]


def filterEQ_is(filter_list, key, value):
    ret_list = []
    for d in filter_list:
        if d[key] == value:
            ret_list.append(d)
    return ret_list


class SRTDataset:

    def __init__(self):
        self._data = {}
        self._categorys = []
        self.datalist = []
        self.category_list = []
        self.addCategory("NOT_KNOW")

    def addNDArray(self, category, data: np.ndarray):
        categorys = category
        if isinstance(category, int) or isinstance(category, str):
            categorys = [category for i in range(data.shape[0])]
        for i in range(data.shape[0]):
            c_code = self.addCategory(categorys[i])
            self.datalist.append(data[i])
            self.category_list.append(c_code)

    def addCategory(self, category, c_code=None):
        if isinstance(category, str):
            c1 = self.filterCName(category)
            if not c1:
                c_code = self.newCCode()
                self._categorys.append({"name": category, "code": c_code})
            else:
                c_code = c1[0]["code"]
        elif isinstance(category, int):
            c_code = category
            c1 = self.filterCCode(category)
            if not c1:
                self._categorys.append({"name": self.newCName("CATEGORY"), "code": c_code})
        return c_code

    def filterCName(self, name):
        return filterEQ_is(self._categorys, "name", name)

    def filterCCode(self, ccode):
        return filterEQ_is(self._categorys, "code", ccode)

    def newCCode(self):
        ccode = 0
        while True:
            for d in self._categorys:
                if d["code"] == ccode:
                    ccode += 1
                    continue
            break
        return ccode

    def newCName(self, name):
        ccode = 1
        while True:
            for d in self._categorys:
                if d["name"] == name:
                    name = "{0}_{1}".format(name, ccode)
                    ccode += 1
                    continue
            break
        return name

    def addNPY(self, category, filename):
        filename = self.addNameData(filename)
        self.addNDArray(category, self._data[filename])

    def addNameData(self, filename, read_type=None):
        filename = os.path.abspath(filename)
        if read_type is None:
            ext = os.path.splitext(filename)[1]
            if ext.startswith("."):
                ext = ext[1:]
            read_type = ext
        if filename not in self._data:
            if read_type == "npy":
                self._data[filename] = np.load(filename)
            elif read_type == "csv":
                self._data[filename] = pd.read_csv(filename)
        return filename

    def addCSV(self, csv_fn, category_field, filename_field=None, idx_field=None):
        csv_fn = self.addNameData(csv_fn)
        df: pd.DataFrame = self._data[csv_fn]
        if (filename_field is not None) and (idx_field is not None):
            for i in range(len(df)):
                filename = df.loc[i][filename_field]
                idx = df.loc[i][idx_field]
                data = self._data[self.addNameData(filename)]
                category = df.loc[i][category_field]
                self.category_list.append(self.addCategory(category))
                self.datalist.append(data[idx])
        else:
            ks = list(df.keys())
            ks.remove(category_field)
            for i in range(len(df)):
                category = df.loc[i][category_field]
                self.category_list.append(self.addCategory(category))
                self.datalist.append(df.loc[i][ks].values)

    def __len__(self):
        return len(self.category_list)

    def __getitem__(self, item):
        return self.datalist[item], self.category_list[item]


class DataFrameFilter:

    def __init__(self, csv_fn):
        self.df = pd.read_csv(csv_fn)

    def eq(self, name, value):
        self.df = self.df[self.df[name] == value]

    def gt(self, name, value):
        self.df = self.df[self.df[name] > value]

    def lt(self, name, value):
        self.df = self.df[self.df[name] < value]

    def ge(self, name, value):
        self.df = self.df[self.df[name] >= value]

    def le(self, name, value):
        self.df = self.df[self.df[name] <= value]

    def get(self):
        return self.df


def DFFilter_eq(df, name, value):
    return df[df[name] == value]


def DFFilter_ge(df, name, value):
    return df[df[name] >= value]


def DFFilter_le(df, name, value):
    return df[df[name] <= value]


def DFFilter_gt(df, name, value):
    return df[df[name] > value]


def DFFilter_lt(df, name, value):
    return df[df[name] < value]


def main():
    sds = SRTDataset()
    sds.addNDArray(1, np.random.random((123, 7, 7)))
    sds.addNDArray("sdfasdf", np.random.random((123, 7, 7)))
    sds.addNDArray(60, np.random.random((123, 7, 7)))
    pass


if __name__ == "__main__":
    main()
