# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : Run.py
@Time    : 2023/12/4 20:34
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of Run
-----------------------------------------------------------------------------"""
import sys

from RUN.RUNFucs import QJYTxt_main, DFColumnCount_main
from RUN.RUNgetdirsize import RUNgetdirsize_main


class GetDirSize_main:

    def __init__(self):
        self.name = "dirsize"
        self.description = "Get directory size "
        self.argv = []

    def run(self, argv):
        RUNgetdirsize_main(argv)

    def usage(self):
        print("{0} [opt: dirname] [opt: --sf] [--help]".format(self.name, self.description))


class Help_main:

    def __init__(self):
        self.show_help = None
        self.name = "help"
        self.description = "Get help information for mark of exe"
        self.argv = []

    def run(self, argv):
        self.argv = argv
        if len(argv) == 1:
            self.usage()
        else:
            self.show_help = argv[1]

    def usage(self):
        print("{0} mark \n @Description: {1}".format(self.name, self.description))


class SRTRun:

    def __init__(self):
        self.name = "srt_run"
        self.description = "Some self-developed exe about Utils. \n(C)Copyright 2023, ZhengHan. All rights reserved."
        self.exes = {}

    def usage(self):
        print("{0} mark/--h [options]\n@Description:\n{1}\n@Args:\n    mark: mark of exe\n"
              "    --h: get help of this\n@Marks:".format(self.name, "    " + self.description.replace("\n", "\n    ")))
        for k in self.exes:
            print("    {0}: {1}".format(k, self.exes[k].description))

    def add(self, exe):
        if exe.name in self.exes:
            raise Exception("mark:{0} has in this".format(exe.name))
        self.exes[exe.name] = exe

    def run(self, mark, argv):
        mark = mark.lower()

        self.exes[mark].run(argv)

        if mark == "help" and self.exes["help"].show_help is not None:
            if self.exes["help"].show_help in self.exes:
                print("`{0}` help information are as follows:\n".format(self.exes["help"].show_help))
                self.exes[self.exes["help"].show_help].usage()
            else:
                print("Show help fault.\nCan not find mark: `{0}`".format(self.exes["help"].show_help))


def main_run(argv):
    srt_run = SRTRun()
    srt_run.add(Help_main())
    srt_run.add(GetDirSize_main())
    srt_run.add(QJYTxt_main())
    srt_run.add(DFColumnCount_main())

    if len(argv) == 1:
        srt_run.usage()
    else:
        if argv[1] == "--h":
            srt_run.usage()
            return
        if argv[1] in srt_run.exes:
            srt_run.run(argv[1], argv[1:])
        else:
            print("Can not find mark:`{0}` ".format(argv[1]))
            srt_run.usage()


if __name__ == "__main__":
    main_run(sys.argv)
