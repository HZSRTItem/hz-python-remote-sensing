# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : FileCompare.py
@Time    : 2023/8/29 18:22
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of FileCompare
-----------------------------------------------------------------------------"""
import os.path


class FileCompareTest:

    def __init__(self):
        self.d = {}

    def add(self, filename):
        dirname, fn = os.path.split(filename)
        if fn in self.d:
            self.d[fn].append(dirname)
        else:
            self.d[fn] = [dirname]

    def main(self):
        filenames = list(self.d.keys())
        filenames.sort()
        for i, k in enumerate(filenames):
            print("{0}. {1}:".format(i + 1, k))
            for dirname in self.d[k]:
                print("  * {0}".format(dirname))

    def method_name2(self):
        for filename in self.d:
            print(len(self.d[filename]))
            if len(self.d[filename]) == 2:
                print(filename, self.d[filename])

    def method_name(self):
        for filename in self.d:
            fn1 = os.path.join(self.d[filename][0], filename)
            is_eq = True
            for dirname in self.d[filename][1:]:
                fn2 = os.path.join(dirname, filename)
                is_eq = is_eq and self.isFileEqual(fn1, fn2)
            if not is_eq:
                print("-" * 60)
                print(filename, is_eq)
                for dirname in self.d[filename][1:]:
                    fn2 = os.path.join(dirname, filename)
                    print("fc", fn1, " ", fn2)
                    print(self.isFileEqual(fn1, fn2))
                fn1 = os.path.join(self.d[filename][1], filename)
                fn2 = os.path.join(self.d[filename][2], filename)
                print("fc", fn1, " ", fn2)
                print(self.isFileEqual(fn1, fn2))

    def isFileEqual(self, fn1, fn2):

        with open(fn1, "r", encoding="utf-8") as f:
            d1 = f.read()
        with open(fn2, "r", encoding="utf-8") as f:
            d2 = f.read()
        return d1 == d2


def main():
    dirnames = [
        r"F:\ProjectSet\PycharmEnvs\BaseCodes\SRTCodes",
        r"F:\ProjectSet\PytorchGeo\SRTCodes",
        r"F:\ProjectSet\PycharmEnvs\GEOCodes\SRTCodes",
    ]
    filename = r"ENVIRasterClassification.py"

    fct = FileCompareTest()

    for dirname in dirnames:
        for f in os.listdir(dirname):
            ff = os.path.join(dirname, f)
            if os.path.isfile(ff):
                fct.add(ff)
    fct.main()

    pass


if __name__ == "__main__":
    main()
