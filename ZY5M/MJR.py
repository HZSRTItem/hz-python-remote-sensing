# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : MJR.py
@Time    : 2025/5/31 21:08
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2025, ZhengHan. All rights reserved.
@Desc    : PyCodes of MJR
-----------------------------------------------------------------------------"""
import pandas as pd
from matplotlib import pyplot as plt


def main():
    def func1():
        csv_fn = r"F:\Week\20250601\Data\sample_huanghekou.csv"
        df = pd.read_csv(csv_fn)
        name_df = df.groupby("Name").mean()
        print((name_df-name_df.std())/name_df.mean())

        plt.rcParams["font.family"] = ["SimHei"]
        name_df.T[["互花米草", "土壤", "柽柳", "盐地碱蓬", "芦苇"]].plot()
        plt.show()

        return

    return func1()


if __name__ == "__main__":
    main()
