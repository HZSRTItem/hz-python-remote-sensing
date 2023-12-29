# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SRTCodesExe.py
@Time    : 2023/7/1 11:05
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : BaseCodes of SRTCodesExe
-----------------------------------------------------------------------------"""
import json
import os
import sys
import time
from shutil import copyfile

exe_dir = os.path.dirname(os.path.realpath(sys.argv[0]))


def getCodeFileName(filename):
    filename = os.path.split(filename)[1]
    fn, ext = os.path.splitext(filename)
    fns = fn.split("_")
    # %Y-%m-%d %H:%M:%S
    t = time.strptime(fns[-1], "%Y%m%d%H%M%S")
    fn = "_".join(fns[:-1]) + ext
    return fn, t


def IsY(front_str="Whether or not? [y/n]:", is_y=False):
    if is_y:
        return True
    line = input(front_str)
    if line.startswith("y"):
        return True
    else:
        return False


class SRTCodeFile:

    def __init__(self, filename):
        fn, t = getCodeFileName(filename)
        self.filename = fn
        self.file_time = [t]
        self.file_names = [filename]

    def add(self, filename):
        fn, t = getCodeFileName(filename)
        if fn == self.filename:
            self.file_time.append(t)
            self.file_names.append(filename)

    def getCodeFilename(self, n=-1, code_dir=""):
        return os.path.join(code_dir, self.file_names[n])


class SRTCodeFileCollection:

    def __init__(self, code_dir):
        self.code_dir = code_dir
        self.code_files = {}
        self._d_next = []
        self._n_iter = 0

        for f in os.listdir(self.code_dir):
            ff = os.path.join(self.code_dir, f)
            if os.path.isfile(ff):
                fn, t = getCodeFileName(f)
                if fn in self.code_files:
                    self.code_files[fn].add(f)
                else:
                    self.code_files[fn] = SRTCodeFile(f)
                    self._d_next.append(fn)

    def __getitem__(self, item) -> SRTCodeFile:
        return self.code_files[item]

    def __iter__(self):
        return self

    def __next__(self):
        if self._n_iter == len(self._d_next):
            self._n_iter = 0
            raise StopIteration()
        else:
            self._n_iter += 1
            return self._d_next[self._n_iter - 1]

    def __len__(self):
        return len(self._d_next)

    def isin(self, fn):
        return fn in self.code_files

    def print(self):
        for f in self.code_files:
            print("{0}: {1}".format(f, len(self.code_files[f].file_time)))


class SRTCodesExeManager:

    def __init__(self):
        self.is_return = False
        self.init_json_fn = os.path.join(exe_dir, "srtcodes_py_init.json")
        self.d_init = {}
        self.code_dir = ""
        self.code_files = []
        self.to_dir = os.getcwd()

        if not os.path.isfile(self.init_json_fn):
            getInitJsonFromUser(self.init_json_fn)
            self.is_return = True
            return

        self.readJson(self.init_json_fn)

        if "CodeFiles" not in self.d_init:
            print("Can not find srt code files.")
            self.is_return = True
            return

        self.code_dir = self.d_init["CodeFilesDirectory"]
        self.code_files = self.d_init["CodeFiles"]

        self.cfc = SRTCodeFileCollection(self.code_dir)
        pass

    def add(self, argv):
        if len(argv) <= 2:
            self.addUasge()
            return

        filename = os.path.abspath(argv[2])
        fn = os.path.split(filename)[1]
        if not self.cfc.isin(fn):
            print("Can not find file \"{0}\" in code "
                  "files collection.".format(fn))
            return

        if fn in self.code_files:
            print("File \"{0}\" have in srt code files".format(fn))
            return

        self.code_files.append(fn)
        pass

    def addUasge(self):
        print("add uasge: add [filename]")

    def delete(self, argv):
        if len(argv) <= 2:
            self.deleteUsage()
            return

        filename = os.path.abspath(argv[2])
        fn = os.path.split(filename)[1]

        if fn not in self.code_files:
            print("Can not find file \"{0}\" in code "
                  "files collection.".format(fn))
            return

        self.code_files.remove(fn)

    def deleteUsage(self):
        print("delete usage: delete [filename]")

    def load(self, argv):
        is_y = False
        load_files = []
        for i in range(2, len(argv)):
            if argv[i] == "--y":
                is_y = True
            else:
                load_files.append(argv[i])
        if len(load_files) == 0:
            load_files = self.code_files.copy()

        load_files_exist = []
        for f in load_files:
            filename = os.path.abspath(f)
            fn = os.path.split(filename)[1]

            if fn not in self.code_files:
                print("Warning: Can not find file \"{0}\" in code "
                      "files collection.".format(fn))
                load_files_exist.append(False)
            else:
                load_files_exist.append(True)
                if os.path.isfile(filename):
                    print("Warning: file will be overwrite " + filename)

        print("The following files will be loaded to "
              "dectory\n  ->{0}".format(self.to_dir))
        for i, f in enumerate(load_files):
            if load_files_exist[i]:
                print("  * {0}".format(f))

        if not IsY(is_y=is_y):
            print("Not load.")
            return

        for i, f in enumerate(load_files):
            if load_files_exist[i]:
                filename = self.cfc[f].getCodeFilename(code_dir=self.code_dir)
                to_filename = os.path.join(self.to_dir, f)
                copyfile(filename, to_filename)

    def loadUsage(self):
        print("load usage: load [opt: *filename]")

    def update(self, argv):
        lines = []
        for f in os.listdir(self.to_dir):
            ff = os.path.join(self.to_dir, f)
            if os.path.isfile(ff):
                if self.cfc.isin(f):
                    lines.append(
                        ["srt_cfm update {0} -i {1} --y".format(ff, f)])
        if len(lines) == 0:
            print("Do not have file in collection. Directory: " + self.to_dir)
            return

        for line in lines:
            print(line)

        pass

    def show(self, argv):

        load_files = []
        for i in range(2, len(argv)):
            if argv[i] == "--y":
                is_y = True
            else:
                load_files.append(argv[i])

        if len(load_files) == 0:
            load_files = self.code_files.copy()

        load_files_exist = []
        i_line = 0
        for f in load_files:
            filename = os.path.abspath(f)
            fn = os.path.split(filename)[1]

            if not self.cfc.isin(fn):
                print("Warning: Can not find file \"{0}\" in code "
                      "files collection.".format(fn))
                load_files_exist.append(False)
            else:
                load_files_exist.append(True)
                i_line += 1

        if i_line == 0:
            print("Do not have file in collection. Directory: " + self.to_dir)
            return

        print("Show Code Files:")
        for i, f in enumerate(load_files):
            if load_files_exist[i]:
                code_f: SRTCodeFile = self.cfc[f]
                print(" {0:>2d}. {1}[{2}]".format(
                    i + 1, f, len(code_f.file_time)))
                for j, t in enumerate(code_f.file_time):
                    print("    * {0}: {1}".format(
                        j + 1, time.strftime("%Y-%m-%d %H:%M:%S", t)))

    def showUsage(self):
        print("show usage: show [opt: *filename]")

    def readJson(self, json_fn=None):
        if json_fn is None:
            json_fn = self.init_json_fn
        with open(json_fn, 'r', encoding='utf-8') as f:
            self.d_init = json.load(f)

    def saveJson(self, json_fn=None):
        if json_fn is None:
            json_fn = self.init_json_fn
        with open(json_fn, 'w', encoding='utf-8') as f:
            d = {"CodeFilesDirectory": self.code_dir,
                 "CodeFiles": self.code_files}
            json.dump(d, f)


def usage():
    print("srtcodes_py [add/delete/load/update/show]\n"
          "    add: filename\n"
          "    delete: filename\n"
          "    load: [opt: *filename]\n"
          "    update\n"
          "    show: [opt: *filename]")


def main(argv):
    if len(argv) <= 1:
        usage()
        return

    sce = SRTCodesExeManager()
    # sce.cfc.print()

    if sce.is_return:
        usage()
        return

    if argv[1] == "add":
        sce.add(argv)
    elif argv[1] == "delete":
        sce.delete(argv)
    elif argv[1] == "load":
        sce.load(argv)
    elif argv[1] == "update":
        sce.update(argv)
    elif argv[1] == "show":
        sce.show(argv)

    sce.saveJson()
    pass


def getInitJsonFromUser(init_json_fn):
    while True:
        code_dir = input("Please input codes files dirname:\n")
        if os.path.isdir(code_dir):
            break
        else:
            print("Can not find input directory\n")
    d = {"CodeFilesDirectory": code_dir, "CodeFiles": []}
    with open(init_json_fn, 'w', encoding='utf-8') as f:
        json.dump(d, f)


if __name__ == "__main__":
    main(sys.argv)
