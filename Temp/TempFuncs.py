# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : TempFuncs.py
@Time    : 2023/12/29 20:56
@Author  : Zheng Han
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of TempFuncs
-----------------------------------------------------------------------------"""
import os

import matplotlib.colors as mcolors

from Draw.m_color_data import CSS4_COLORS


def main():
    def sort_color(colors):
        names = sorted(colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
        return {name: colors[name] for name in names}

    colors = sort_color(CSS4_COLORS)
    with open(r"F:\PyCodes\Draw\CSS4_COLORS.css", "w", encoding="utf-8") as f:
        f.write("mcolor {\n")
        for k, v in colors.items():
            print("    color: {0};".format(k, v), file=f)
        f.write("}\n")

    pass


def method_name1():
    dirname = r"F:\PyCodes"
    to_dirname = r"F:\PyCodes"
    find_str = "tourensong@gmail.com"
    to_str = "tourensong@gmail.com"
    for root, dirs, files in os.walk(dirname):
        for file in files:
            fn = os.path.join(root, file)
            to_fn = fn.replace(dirname, to_dirname)
            if os.path.splitext(fn)[1] == ".py":
                fr = open(fn, "r", encoding="utf-8")
                lines = []
                for line in fr:
                    if find_str in line:
                        print(fn, "->", to_fn)
                        print("   ", line)
                        line = line.replace(find_str, to_str)
                        print("   ", line)
                    lines.append(line)
                fr.close()
                fw = open(to_fn, "w", encoding="utf-8")
                fw.writelines(lines)
                fw.close()


if __name__ == "__main__":
    main()
