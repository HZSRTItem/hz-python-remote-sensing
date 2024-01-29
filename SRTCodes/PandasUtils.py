# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : PandasUtils.py
@Time    : 2024/1/29 10:57
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of PandasUtils
-----------------------------------------------------------------------------"""


def filterDF(df, *filters, **filter_maps):
    filter_dict = {}
    for f in filters:
        filter_dict[f[0]] = f[1]
    for f in filter_maps:
        filter_dict[f] = filter_maps[f]
    for k, f in filter_dict.items():
        df = df[df[k] == f]
    return df


def main():
    pass


if __name__ == "__main__":
    main()
