# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SRTModel.py
@Time    : 2023/12/9 21:53
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of SRTModel
-----------------------------------------------------------------------------"""


class SRTModelInit:

    def __init__(self):
        self.name = "MODEL"

    def train(self, *args, **kwargs):
        train_args = {"name": self.name}
        return train_args

    def predict(self, *args, **kwargs):
        return 0

    def score(self, *args, **kwargs):
        return 0

    def load(self, *args, **kwargs):
        return True

    def save(self, *args, **kwargs):
        return True


def main():
    pass


if __name__ == "__main__":
    main()
