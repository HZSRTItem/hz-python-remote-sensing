# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SRTCollection.py
@Time    : 2023/7/4 10:14
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : BaseCodes of SRTCollection
-----------------------------------------------------------------------------"""


class SRTCollection:

    def __init__(self):
        self._n_iter = 0
        self._n_next = []

    def __len__(self):
        return len(self._n_next)

    def __iter__(self):
        return self

    def __next__(self):
        if self._n_iter == len(self._n_next):
            self._n_iter = 0
            raise StopIteration()
        self._n_iter += 1
        return self._n_next[self._n_iter - 1]

    def __contains__(self, item):
        return item in self._n_next

    def __getitem__(self, item):
        return self._n_next[item]


def main():
    pass


if __name__ == "__main__":
    main()
