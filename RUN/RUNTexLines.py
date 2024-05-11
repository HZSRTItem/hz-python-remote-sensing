# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : RUNTexLines.py
@Time    : 2024/5/8 20:15
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of RUNTexLines
-----------------------------------------------------------------------------"""
from SRTCodes.Utils import readText


class RUNTexLines_main:

    def __init__(self):
        self.name = "texlines"
        self.description = "Get directory size "
        self.argv = []

    def run(self, argv):
        self.argv = argv

        filename = None
        to_filename = None
        n_line = 100

        i = 1
        while i < len(argv):
            if (argv[i] == "-o") and (i < len(argv) - 1):
                to_filename = argv[i + 1]
                i += 1
            elif (argv[i] == "-n") and (i < len(argv) - 1):
                n_line = int(argv[i+1])
                i += 1
            elif filename is not None:
                filename = argv[i]
            i += 1

        if filename is not None:
            text = readText(filename)
        else:
            text = ""
            while not text.endswith("\n\n"):
                text += input()

        f = None
        if to_filename is not None:
            f = open(to_filename, "w", encoding="utf-8")

        n_line_tmp = 0
        text_tmp = ""
        is_new_line = False
        lines = text.split(" ")
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff':
                n_line_tmp += 2

            else:
                n_line += 1

        if to_filename is not None:
            f.close()

    def usage(self):
        print("{0} [opt:text_file] [opt:-o] [opt:-n]\n"
              "    [opt:text_file]: file to lines\n"
              "    [opt:-o]: out put file\n"
              "    [opt:-n]: number of line default:100".format(self.name, self.description))


def main():
    pass


if __name__ == "__main__":
    main()
