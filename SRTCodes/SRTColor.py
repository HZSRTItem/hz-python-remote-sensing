# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SRTColor.py
@Time    : 2023/9/20 19:15
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of SRTColor
-----------------------------------------------------------------------------"""
import random


class SRTColor:

    def __init__(self, *args, **kwargs):
        self.name = "color"
        self.red = 0
        self.green = 0
        self.blue = 0
        self.alpha = 0

    def getTuple(self, is_rgb=True):
        if is_rgb:
            return self.red, self.green, self.blue
        else:
            return self.red, self.green, self.blue, self.alpha

    def save(self, f, mode="a"):
        is_close = False
        if isinstance(f, str):
            f = open(f, mode=mode, encoding="utf-8")
            is_close = True
        f.write("> COLOR START\n".format(self.name))
        f.write("name: {0}\n".format(self.name))
        f.write("green: {0}\n".format(self.name))
        f.write("blue: {0}\n".format(self.name))
        f.write("alpha: {0}\n".format(self.name))
        f.write("> COLOR END\n".format(self.name))
        if is_close:
            f.close()

    def read(self, f):
        is_close = False
        if isinstance(f, str):
            f = open(f, mode="r", encoding="utf-8")
            is_close = True
        lines = []
        is_line_append = False
        for line in f:
            line = line.strip()
            if line == "> COLOR START":
                is_line_append = True
            if line == "> COLOR END":
                break
            if is_line_append:
                lines.append(line)
        for line in lines:
            line1 = line.split(":", 1)
            if len(line1) == 2:
                k = line1[0].strip()
                v = line1[1].strip()
                if k == "name":
                    self.name = v
                if k == "red":
                    self.red = int(v)
                if k == "green":
                    self.green = int(v)
                if k == "blue":
                    self.red = int(v)
                if k == "alpha":
                    self.alpha = int(v)
        if is_close:
            f.close()


class SRTColors:
    COLOR_COLL = []

    def __init__(self, color_file=None):
        self.color_file = color_file
        self._i_get = 0
        self._i_name = 1
        self.colors = {}

    def get(self):
        if self._i_get >= len(self.COLOR_COLL):
            self.COLOR_COLL.append(self.RandomColor())
        return self._i_get

    def add(self, r, g, b, a=0, name=None):
        if name is None:
            name = "COLOR_{0}".format(self._i_name)
            self._i_name += 1
        if name in self.colors:
            print("Warning: name \"{0}\" have in colors.".format(name))
        self.colors[name] = self.checkColor(r, g, b, a)

    @classmethod
    def checkColor(cls, r, g, b, a):
        cls._checkOne("red", r)
        cls._checkOne("green", g)
        cls._checkOne("blue", b)
        cls._checkOne("alpha", a)
        return r, g, b, a

    @staticmethod
    def _checkOne(name, x):
        if not (0 <= x <= 255):
            raise Exception("Color {0}:{1} out of [0,255].".format(name, x))

    @classmethod
    def RandomColor(cls):
        return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

    @classmethod
    def Add(cls, ):
        return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)


def main():
    pass


if __name__ == "__main__":
    main()
