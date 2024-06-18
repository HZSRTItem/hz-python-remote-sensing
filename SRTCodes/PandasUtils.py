# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : PandasUtils.py
@Time    : 2024/1/29 10:57
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of PandasUtils
-----------------------------------------------------------------------------"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.image import imread, imsave


class DataFrameDictSort:

    def __init__(self, df):
        self.df_dict = df.to_dict("records")

    def sort(self, by=None, ascending=None, filter_list=None, filter_func=None):
        if by is None:
            by = []
        filters_loc = []
        if filter_list is not None:
            filters = np.ones(len(self.df_dict), dtype=bool)
            for f in filter_list:
                f = np.array(f)
                filters &= f
            filters_loc = np.where(filters)[0]
        elif filter_func is not None:
            for i, line in enumerate(self.df_dict):
                if filter_func(line):
                    filters_loc.append(i)

        to_df_dict = []
        for i in filters_loc:
            to_df_dict.append(self.df_dict[i])
        to_df_dict = self._sort(by, ascending, to_df_dict)
        j = 0
        for i in filters_loc:
            self.df_dict[i] = to_df_dict[j]
            j += 1
        return self

    def toDF(self):
        return pd.DataFrame(self.df_dict)

    def _sort(self, by, ascending, df_dict):
        df = pd.DataFrame(df_dict)
        df = df.sort_values(by=by, ascending=ascending)
        return df.to_dict("records")


def filterDF(df, *filters, **filter_maps):
    filter_dict = {}
    for f in filters:
        filter_dict[f[0]] = f[1]
    for f in filter_maps:
        filter_dict[f] = filter_maps[f]
    for k, f in filter_dict.items():
        df = df[df[k] == f]
    return df


def splitDf(_df, field_name, *df_splits):
    df_list = []
    for df_split in df_splits:
        df_list_tmp = []
        for data in df_split:
            df_tmp = _df[_df[field_name] == data]
            if len(df_tmp) != 0:
                df_list_tmp.append(df_tmp)
        if len(df_list_tmp) != 0:
            df_list.append(pd.concat(df_list_tmp))
        else:
            df_list.append(None)
    return tuple(df_list)


def main():
    im = imread(r"F:\ProjectSet\Shadow\MkTu\Imdc\ShadowMkTuImdc3.jpg")
    print(type(im))  # ---><class 'numpy.ndarray'>
    print(im.shape)  # --->(720, 1280, 3)
    print(im.size)  # --->2764800
    im = im[0:1025, 0:3650, :]
    imsave(r"F:\ProjectSet\Shadow\MkTu\Imdc\ShadowMkTuImdc4.jpg", im, dpi=300)
    plt.imshow(im)
    plt.show()
    pass


if __name__ == "__main__":
    main()
