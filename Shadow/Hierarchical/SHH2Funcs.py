# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2Funcs.py
@Time    : 2024/6/8 21:36
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2Funcs
-----------------------------------------------------------------------------"""
import numpy as np
import pandas as pd

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.OGRUtils import sampleSpaceUniform
from SRTCodes.Utils import FRW, DirFileName, FN


def main():
    def func1(csv_fn):
        df = pd.read_csv(csv_fn)
        coors2, out_index_list = sampleSpaceUniform(df[["X", "Y"]].values.tolist(), x_len=500, y_len=500,
                                                    is_trans_jiaodu=True, ret_index=True)
        is_save = np.zeros(len(df))
        is_save[out_index_list] = 1
        df["IS_SAVE"] = is_save
        df = df[df["IS_SAVE"] == 1]
        print(len(out_index_list))
        df.to_csv(FN(csv_fn).changext("_ssu2.csv"), index=False)

        return

    func1(r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\2\sh2_spl27_2_cd_train1.csv")

    return


def method_name1():
    # data = {}
    #
    # def tongjishuliang(name, csv_fn, c_field_name="CNAME"):
    #     df = pd.read_csv(csv_fn)
    #     if "TEST" in df:
    #         df = df[df["TEST"] == 1]
    #     data[name] = pd.value_counts(df[c_field_name]).to_dict()
    #
    # tongjishuliang("bj_test", r"F:\Week\20240609\Data\samples\bj_test.csv")
    # tongjishuliang("bj_train", r"F:\Week\20240609\Data\samples\bj_train.csv")
    # tongjishuliang("cd_test", r"F:\Week\20240609\Data\samples\cd_test.csv")
    # tongjishuliang("cd_train", r"F:\Week\20240609\Data\samples\cd_train.csv")
    # tongjishuliang("qd_test", r"F:\Week\20240609\Data\samples\qd_test.csv")
    # tongjishuliang("qd_train", r"F:\Week\20240609\Data\samples\qd_train.csv")
    #
    # df = pd.DataFrame(data).T
    # print(df)
    # df.to_csv(r"F:\Week\20240609\Data\samples\spl_counts.csv")
    dfn = DirFileName(r"F:\Week\20240609\Data\samples")
    to_dict = FRW(r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\1\20240610H112357_bjtest.json").readJson()
    df = pd.DataFrame(to_dict["acc"])
    df.to_csv(dfn.fn("cd.csv"))
    print(df)


def method_name():
    gr = GDALRaster(r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240607H100838\OPTASDE_epoch36_imdc1.tif")
    data = gr.readAsArray()
    d, n = np.unique(data, return_counts=True)
    print(n * 1.0 / np.sum(n))


if __name__ == "__main__":
    main()
