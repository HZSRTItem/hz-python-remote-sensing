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


def main():
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
    pass


if __name__ == "__main__":
    main()
