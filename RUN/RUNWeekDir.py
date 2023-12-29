# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : RUNWeekDir.py
@Time    : 2023/12/27 16:38
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of RUNWeekDir
-----------------------------------------------------------------------------"""
import os


def main():
    week_dirname = r"F:\Week"
    fn_list = []
    for fn in os.listdir(week_dirname):
        fn2 = os.path.join(week_dirname, fn)
        if os.path.isdir(fn2):
            if fn.isdigit():
                fn_list.append(fn)
    week_dirname_this = os.path.join(week_dirname, max(fn_list))
    crof_fn = r"F:\code\share\OF\CROF.txt"
    is_find = False
    with open(crof_fn, "r", encoding="utf-8") as f:
        text = f.read()
        to_text = ""
        lines = text.split("\n")
        for line in lines:
            tmp = line.split(":")[0].strip()
            if tmp == "week":
                is_find = True
                to_text += "{0}: explorer {1}\n".format("week", week_dirname_this)
            else:
                to_text += line + "\n"
    if not is_find:
        to_text += "{0}: {1}\n".format("week", week_dirname_this)
    with open(crof_fn + "-back", "w", encoding="utf-8") as f:
        f.write(text)
    with open(crof_fn, "w", encoding="utf-8") as f:
        f.write(to_text)


    pass


if __name__ == "__main__":
    main()
