# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowAreas.py
@Time    : 2024/7/25 14:05
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of ShadowAreas
-----------------------------------------------------------------------------"""
import math

import matplotlib.pyplot as plt
from mpl_toolkits.axisartist import AxesZero
from matplotlib import rcParams

# plt.rc('font', family='Times New Roman')
# plt.rc('text', usetex=True)
# plt.rc('font', size=12)
# plt.rc('mathtext', default='regular')
FONT_SIZE = 16
#
# config = {
#     "font.family": 'serif',
#     "mathtext.fontset": 'stix',
#     "font.serif": ['SimSun'],  # simsun字体中文版就是宋体
# }
# rcParams.update(config)


def coorsXY(xys):
    datas = []
    for xy in xys:
        if hasattr(xy, "__len__") and hasattr(xy, "__getitem__"):
            for i in range(len(xy)):
                datas.append(xy[i])
        else:
            datas.append(xy)
    x, y = [], []
    for i in range(0, len(datas) - 1, 2):
        x.append(datas[i])
        y.append(datas[i + 1])
    return x, y


def calculateBearing(x1, y1, x2, y2):
    delta_x = x2 - x1
    delta_y = y2 - y1
    bearing_rad = math.atan2(delta_x, delta_y)
    bearing_deg = math.degrees(bearing_rad)
    return bearing_deg


def angleToRadian(angle=1.0):
    return angle * math.pi / 180.0


def coorsTrans(x, y, theta=0.0):
    if theta == 0:
        return x, y
    rad = angleToRadian(theta)
    xys = [(x[i], y[i]) for i in range(len(x))]
    x_list, y_list = [], []
    for x, y in xys:
        x1 = x * math.cos(rad) + y * math.sin(rad)
        y1 = -x * math.sin(rad) + y * math.cos(rad)
        x_list.append(x1)
        y_list.append(y1)
    return x_list, y_list


def coorsListAdd(_list, data):
    return [d + data for d in _list]


class SHADC:
    """ SHADrawCoors """

    def __init__(self, *coors):
        self.x = []
        self.y = []
        self.x, self.y = coorsXY(coors)

    def fill(self, *args, **kwargs):
        plt.fill(self.x, self.y, *args, **kwargs)
        return self

    def line(self, *args, **kwargs):
        plt.plot(self.x, self.y, *args, **kwargs)
        return self

    def initRt(self, x0, x1, y0, y1):
        coors = (x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0)
        self.x, self.y = coorsXY(coors)
        return self

    def addArrow(self, n=1, fx=1, d1=0.03, d2=0.05, *args, **kwargs):
        for i in range(len(self.x) - 1):
            x1, y1 = self.x[i], self.y[i]
            x2, y2 = self.x[i + 1], self.y[i + 1]
            dx, dy = (x2 - x1) / (n + 1), (y2 - y1) / (n + 1)
            azimuth = calculateBearing(x1, y1, x2, y2)
            for j in range(n):
                j = j + 1
                x, y = x1 + dx * j, y1 + dy * j
                x_list = [0, d1 / 2, -d1 / 2]
                y_list = [d2 / 2, -d2 / 2, -d2 / 2]
                x_list, y_list = coorsTrans(x_list, y_list, azimuth + fx * 180)
                x_list, y_list = coorsListAdd(x_list, x), coorsListAdd(y_list, y)
                plt.fill(x_list, y_list, *args, **kwargs)
        return self

    def addX(self, dx):
        self.x = coorsListAdd(self.x, dx)
        return self

    def addY(self, dy):
        self.y = coorsListAdd(self.y, dy)
        return self


def main():
    plt.figure(figsize=(9.5, 9.5))
    plt.subplots_adjust(top=0.96, bottom=0.04, left=0.04, right=0.96, hspace=0.01, wspace=0.01)

    show11()

    ax = plt.subplot(141, axes_class=AxesZero)
    ax.set_aspect('equal', adjustable='box')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks([])
    plt.yticks([])

    plt.show()


def show11(number_fig="a", is_chinese=False):
    if not is_chinese:
        ax = plt.subplot(141, axes_class=AxesZero)
        ax.set_aspect('equal', adjustable='box')
        plt.xlim([0, 1.0])
        plt.ylim([0, (1.2 + 0.9) / 2.4])
        plt.xticks([])
        plt.yticks([])
        dy = 0.30
        y0 = 0.0
        as_line = SHADC(0.35, 0.6, 0.95, y0).addY(dy).line(
            color="#BF1D2D", label="AS incident wave").addArrow(2, fx=0, color="#BF1D2D")
        SHADC(0.58, y0 + 0.02, 0.58, y0 - 0.05).addY(dy).line(color="#BF1D2D")
        SHADC(0.95, y0 + 0.02, 0.95, y0 - 0.05).addY(dy).line(color="#BF1D2D")
        de_line = SHADC(0.65, 0.6, 0.05, y0).addY(dy).line(
            color="#293890", label="DE incident wave").addArrow(2, fx=0, color="#293890")
        SHADC(0.42, y0 + 0.02, 0.42, y0 - 0.05).addY(dy).line(color="#293890")
        SHADC(0.05, y0 + 0.02, 0.05, y0 - 0.05).addY(dy).line(color="#293890")
        building = SHADC().initRt(0.42, 0.58, y0, 0.66 - dy, ).addY(dy).line(
            color="black").fill(color="lightgrey", label="Building")
        ground = SHADC(0.02, y0, 0.98, y0).addY(dy).line(color="black")
        plt.text(0.58 + 0.02, dy - 0.05, "AS Shadow Area", )
        plt.text(0.7, 0.40, "East", )
        plt.text(0.05 + 0.02, dy - 0.05, "DE Shadow Area", )
        plt.text(0.2, 0.40, "West", )
        plt.text(0.05, 0.75, "({0})".format(number_fig), fontdict={"size": 16})
        plt.legend(loc="lower center", frameon=False)
    else:
        ax = plt.subplot(221, axes_class=AxesZero)
        ax.set_aspect('equal', adjustable='box')
        plt.xlim([0, 1.0])
        plt.ylim([0, (1.2 + 0.9) / 2.4])
        plt.xticks([])
        plt.yticks([])
        dy = 0.30
        y0 = 0.0
        as_line = SHADC(0.35, 0.6, 0.95, y0).addY(dy).line(
            color="#BF1D2D", label="升轨卫星入射波").addArrow(2, fx=0, color="#BF1D2D")
        SHADC(0.58, y0 + 0.02, 0.58, y0 - 0.05).addY(dy).line(color="#BF1D2D")
        SHADC(0.95, y0 + 0.02, 0.95, y0 - 0.05).addY(dy).line(color="#BF1D2D")
        de_line = SHADC(0.65, 0.6, 0.05, y0).addY(dy).line(
            color="#293890", label="降轨卫星入射波").addArrow(2, fx=0, color="#293890")
        SHADC(0.42, y0 + 0.02, 0.42, y0 - 0.05).addY(dy).line(color="#293890")
        SHADC(0.05, y0 + 0.02, 0.05, y0 - 0.05).addY(dy).line(color="#293890")
        building = SHADC().initRt(0.42, 0.58, y0, 0.66 - dy, ).addY(dy).line(
            color="black").fill(color="lightgrey", label="建筑物")
        ground = SHADC(0.02, y0, 0.98, y0).addY(dy).line(color="black")
        plt.text(0.58 + 0.02, dy - 0.05, "升轨SAR阴影区域", fontdict={"size": 14})
        plt.text(0.7, 0.40, "东", fontdict={"size": 14})
        plt.text(0.05 + 0.02, dy - 0.05, "降轨SAR阴影区域", fontdict={"size": 14})
        plt.text(0.2, 0.40, "西", fontdict={"size": 14})
        plt.text(0.05, 0.75, "({0})".format(number_fig), fontdict={"size": 14})
        plt.legend(loc="lower center", frameon=False, prop={"size": 14})


if __name__ == "__main__":
    main()
