# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : Temp.py
@Time    : 2024/7/5 17:28
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of Temp
-----------------------------------------------------------------------------"""
import os
from shutil import copyfile

import pandas as pd

from SRTCodes.Utils import Jdt


def main():
    def func1():
        dirname = r"F:\PyCodes"
        to_dirname = r"F:\Week\20240811\Code\PyCodes"
        for root, dirs, files in os.walk(dirname):
            for file in files:
                fn = os.path.join(root, file)
                to_fn = fn.replace(dirname, to_dirname)
                if os.path.splitext(fn)[1] == ".py":
                    to_dirname_tmp = os.path.dirname(to_fn)
                    if not os.path.isdir(to_dirname_tmp):
                        print(to_dirname_tmp)
                        os.mkdir(to_dirname_tmp)
                    copyfile(fn, to_fn)

    def func2():
        read_csv_fn = r"F:\材料\2025事业编人数.xlsx"
        df = pd.read_excel(read_csv_fn)
        print(df.to_dict("records")[0])
        print(df.keys())

        to_csv_fn = r"F:\材料\2025事业编省属2.xlsx"
        to_df = pd.read_excel(to_csv_fn)
        print(to_df)
        print(to_df.keys())

        filter_names = {"类别1": "省属", "类别2": "省属"}
        for name in filter_names:
            df = df[df[name] == filter_names[name]]

        find_field_names = ["主管部门", "事业单位", "岗位名称", ]

        # to_df2 = pd.merge(to_df, df, on=find_field_names , how='left')

        df_list = df.to_dict("records")
        to_df_list = to_df.to_dict("records")

        jdt = Jdt(len(to_df_list)).start()
        for data in to_df_list:
            for find_data in df_list:
                is_find = True
                for name in find_field_names:
                    data1 = str(find_data[name]).strip()
                    data2 = str(data[name]).strip()
                    if data1 != data2:
                        is_find = False
                        break
                if is_find:
                    data["审核通过人数"] = find_data["审核通过人数"]
                    break
            jdt.add()
        jdt.end()
        to_df2 = pd.DataFrame(to_df_list)

        print(to_df2)
        to_df2.to_excel(r"F:\材料\2025事业编省属18日数量.xlsx", index=False)

    def func3():
        for i in range(1, 100):
            print("{:3d} -> ".format(i), end="")
            for j in range(10):
                print("x{}:{} ".format(j, "{:<3d}".format(i * j)[-2:]), end="")
            print()

    return func3()


if __name__ == "__main__":
    main()
