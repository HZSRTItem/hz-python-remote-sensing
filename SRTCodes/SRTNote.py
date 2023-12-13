# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SRTNote.py
@Time    : 2023/10/5 16:41
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of SRTNote
-----------------------------------------------------------------------------"""


class SRTNote:

    def __init__(self, filename):
        self.filename = filename


class SRTMarkDownWrite:

    def __init__(self):
        self.filename = None
        self._fs = None

    def open(self, filename, mode="w", encoding="utf-8"):
        self._fs = open(filename, mode=mode, encoding=encoding)

    def close(self):
        if self._fs is not None:
            self._fs.close()





def main():
    pass


if __name__ == "__main__":
    main()
