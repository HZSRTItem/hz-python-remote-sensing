# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowDirection.py
@Time    : 2023/9/18 8:55
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of ShadowDirection

-----------------------------------------------------------------------------"""
import math

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axisartist.axislines import AxesZero
from mpl_toolkits.mplot3d import Axes3D

from SRTCodes.Utils import angleToRadian, radianToAngle

plt.rc('font', family='Times New Roman')
FONT_SIZE = 14


def calculateBearing(x1, y1, x2, y2):
    delta_x = x2 - x1
    delta_y = y2 - y1
    bearing_rad = math.atan2(delta_x, delta_y)
    bearing_deg = math.degrees(bearing_rad)
    return bearing_deg


def is_in_poly(p, poly):
    px, py = p
    is_in = False
    for i, corner in enumerate(poly):
        next_i = i + 1 if i + 1 < len(poly) else 0
        x1, y1 = corner
        x2, y2 = poly[next_i]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in


def distanceCoor(x1, x2, y1, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def findMinWhere(xulie):
    dmin = xulie[0]
    i0 = 0
    for i, d in enumerate(xulie):
        if d < dmin:
            dmin = d
            i0 = i
    return i0


def findMaxWhere(xulie):
    dmax = xulie[0]
    i0 = 0
    for i, d in enumerate(xulie):
        if d > dmax:
            dmax = d
            i0 = i
    return i0


def intersectLine2(line1, line2):
    a1, a2 = line1.a, line2.a
    b1, b2 = line1.b, line2.b
    c1, c2 = line1.c, line2.c
    x = (c2 * b1 - c1 * b2) / (a1 * b2 - a2 * b1 + 0.000000001)
    y = (c1 * a2 - c2 * a1) / (a1 * b2 - a2 * b1 + 0.000000001)
    # if line1.x0 == 0.5 and line1.y0 == 392.5:
    #     plt.scatter(line1.x0, line1.y0)
    #     line1.plot(-100, 1000)
    #     line2.plot(color="green")
    #     plt.scatter(x, y)
    if line1.isIn(x, y) and line2.isIn(x, y):
        return x, y
    else:
        return None


class SHDLine:

    def __init__(self):
        self.k = 0
        self.x0 = 0.0
        self.y0 = 0.0
        self.distance_c = 0.0
        self.distance_1 = 0.0

        self.a = 0
        self.b = 0
        self.c = 0

        self.x_min = None
        self.y_min = None
        self.x_max = None
        self.y_max = None
        self.z_min = None
        self.z_max = None

        self.normal_azimuth = None
        self.azimuth = None

    def _distanceCal(self):
        self.distance_c = self.y0 - self.k * self.x0
        self.distance_1 = math.sqrt(self.k ** 2 + 1)

    def _calABC(self):
        self.a = self.k
        self.b = -1
        self.c = self.y0 - self.k * self.x0

    def _calNormalAzimuth(self):
        self.normal_azimuth = radianToAngle(math.atan(-self.k))

    def _calAzimuth(self):
        y0, y1 = self.y(0), self.y(1)
        self.azimuth = calculateBearing(0, y0, 1, y1)

    def azimuthCoor(self, alpha, x0=0.0, y0=0.0):
        self.k = 1 / (math.tan(angleToRadian(alpha)) + 0.000001)
        self.x0 = x0
        self.y0 = y0
        self._distanceCal()
        self._calABC()
        self._calNormalAzimuth()
        self._calAzimuth()
        return self

    def twoPoint(self, x1, y1, x2, y2, is_range=False, ):
        self.k = (y2 - y1) / (x2 - x1)
        self.x0 = x1
        self.y0 = y1
        if is_range:
            self.x_min = min(x1, x2)
            self.y_min = min(y1, y2)
            self.x_max = max(x1, x2)
            self.y_max = max(y1, y2)
        self._distanceCal()
        self._calABC()
        self._calNormalAzimuth()
        self._calAzimuth()
        return self

    def plot(self, x0=None, x1=None, color="red", **kwargs):
        if x0 is None:
            x0 = self.x_min
        if x0 is None:
            x0 = 0.0
        if x1 is None:
            x1 = self.x_max
        if x1 is None:
            x1 = 1.0
        # plt.scatter(self.x0, self.y0)
        plt.plot([x0, x1], [self.y(x0), self.y(x1)], color=color, **kwargs)
        # plt.xlim(x0, x1)
        # plt.ylim(x0, x1)

    def y(self, x):
        return self.k * (x - self.x0) + self.y0

    def pointDistance(self, x1, y1):
        return abs(self.k * x1 - y1 + self.distance_c) / self.distance_1

    def show(self):
        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')  # 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')  # 指定下边的边作为 x 轴   指定左边的边为 y 轴
        ax.spines['bottom'].set_position(('data', 0))  # 指定 data  设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
        ax.spines['left'].set_position(('data', 0))
        ax.set_aspect('equal', adjustable='box')

    def isIn(self, x, y):
        if self.x_min is not None:
            if x < self.x_min:
                return False
        if self.y_min is not None:
            if y < self.y_min:
                return False
        if self.x_max is not None:
            if x > self.x_max:
                return False
        if self.y_max is not None:
            if y > self.y_max:
                return False
        return True

    def intersect(self, line):
        return intersectLine2(self, line)


class SHDSatellite:

    def __init__(self):
        self.height = None
        self.anchor_coor = None
        self.flight_azimuth = None
        self.normal_azimuth = None
        self.line = SHDLine()

    def init(self, anchor_coor, flight_azimuth, height):
        if anchor_coor is None:
            anchor_coor = [0.0, 0.0]
        self.line.azimuthCoor(flight_azimuth, anchor_coor[0], anchor_coor[1])
        self.flight_azimuth = flight_azimuth
        self.anchor_coor = anchor_coor
        self.height = height
        self.normal_azimuth = self.line.normal_azimuth

    def calIncidentAngle(self, x, y):
        return math.atan2(self.height, (self.line.pointDistance(x, y) + 0.00001))


class SHDBuildingLines:

    def __init__(self):
        self.lines = []
        self.buildings = []

    def addLines(self, lines, buildings):
        self.lines += list(lines)
        self.buildings += list(buildings)

    def addLine(self, line, building):
        self.lines.append(line)
        self.buildings.append(building)

    def incidentIntersect(self, line1: SHDLine):
        line_intersect = []
        coors = []
        distance = []
        idx_lines = []
        building_coll = []
        for i, line2 in enumerate(self.lines):
            coor = intersectLine2(line2)
            if coor is not None:
                line_intersect.append(line2)
                coors.append(coor)
                distance.append(line1.pointDistance(coor[0], coor[1]))
                idx_lines.append(i + 1)
                building_coll.append(self.buildings[i])
        # if line1.x0 == 0.5 and line1.y0 == 392.5:
        #     plt.show()
        if not line_intersect:
            return None, None
        n = findMinWhere(distance)
        return building_coll[n], idx_lines[n]

    def __getitem__(self, item) -> SHDLine:
        return self.lines[item]


class SHDBuilding:

    def __init__(self, polygon=None, first_line_outside="right", name=None, height=0.0):
        if polygon is None:
            polygon = []
        self.name = name
        self.polygon = polygon
        self.first_line_outside = first_line_outside
        self.lines = []
        self.height = 0.0
        if polygon is not None:
            self.init(polygon, first_line_outside, height)

    def init(self, polygon: list, first_line_outside="right", height=0.0):
        self.polygon = polygon
        self.first_line_outside = first_line_outside
        self.height = height
        for i in range(len(polygon) - 1):
            # print(polygon[i][0], polygon[i][1], polygon[i + 1][0], polygon[i + 1][1])
            self.lines.append(
                SHDLine().twoPoint(polygon[i][0], polygon[i][1], polygon[i + 1][0], polygon[i + 1][1], is_range=True))
            self.lines[-1].z_min = 0.0
            self.lines[-1].z_max = self.height

    def incidentIntersect(self, line1: SHDLine):
        line_intersect = []
        coors = []
        distance = []
        idx_lines = []
        for i, line2 in enumerate(self.lines):
            coor = intersectLine2(line2)
            if coor is not None:
                line_intersect.append(line2)
                coors.append(coor)
                distance.append(line1.pointDistance(coor[0], coor[1]))
                idx_lines.append(i + 1)
        if not line_intersect:
            return None
        n = findMinWhere(distance)
        return idx_lines[n], distance[n]

    def isContain(self, x, y):
        return is_in_poly([x, y], self.polygon)

    def plot(self):
        for line in self.lines:
            line.plot()


class SHDBuildings:

    def __init__(self):
        self.buildings = {}
        self.lines = SHDBuildingLines()

    def add(self, polygon=None, first_line_outside="right", height=0.0):
        n = len(self.buildings) + 1
        self.buildings[n] = SHDBuilding(polygon, first_line_outside=first_line_outside, name=n, height=height)
        for line in self.buildings[n].lines:
            self.lines.addLine(line, n)

    def findBuilding(self, x, y, incident_angle, azimuth):
        for k in self.buildings:
            if self.buildings[k].isContain(x, y):
                return k, -1
        line1 = SHDLine().azimuthCoor(azimuth, x, y)
        n_building, n_line = self.lines.incidentIntersect(line1)
        if n_building is None:
            return None, None
        line = self.lines[n_line - 1]
        coor = intersectLine2(line1)
        d1 = distanceCoor(x, y, coor[0], coor[1])
        h1 = d1 * math.tan(angleToRadian(incident_angle))
        if h1 < line.z_max:
            return n_building, n_line
        else:
            return None, None

    def __getitem__(self, item):
        return self.buildings[item]

    def plot(self):
        for k in self.buildings:
            self.buildings[k].plot()


class ShadowDirection:

    def __init__(self):
        self.y_n = None
        self.x_n = None
        self.satellite = SHDSatellite()
        self.buildings = SHDBuildings()

        self.origin_coor = [0.0, 0.0]
        self.resolution = 0.0
        self.x_min = 0.0
        self.y_min = 0.0
        self.x_max = 0.0
        self.y_max = 0.0

        self.incident_angle_image = np.array([])
        self.image1 = np.array([])

    def calIncidentAngleImage(self):
        self.calImageSize()
        self.incident_angle_image = np.zeros((self.x_n, self.y_n))
        for i in range(self.x_n):
            for j in range(self.y_n):
                self.incident_angle_image[i, j] = self.satellite.calIncidentAngle(self.x(i), self.y(j))
                self.incident_angle_image[i, j] = radianToAngle(self.incident_angle_image[i, j])

    def calImageSize(self):
        self.x_n = int((self.x_max - self.x_min) / self.resolution)
        self.y_n = int((self.y_max - self.y_min) / self.resolution)
        self.image1 = np.zeros((self.x_n, self.y_n))

    def x(self, idx):
        return self.x_min + self.resolution * (idx + 0.5)

    def y(self, idx):
        return self.y_min + self.resolution * (idx + 0.5)

    def imshow_incident_angle_image(self):
        plt.imshow(self.incident_angle_image)

    def plotSatelliteLine(self, x0, x1):
        self.satellite.line.plot(x0, x1)
        self.satellite.line.show()

    def show(self):
        plt.xlim([self.x_min, self.x_max])
        plt.ylim([self.y_min, self.y_max])

    def main(self):
        self.origin_coor = [0.0, 0.0]
        self.resolution = 1
        self.x_min = 0.0
        self.y_min = 0.0
        self.x_max = 1000
        self.y_max = 1000

        self.satellite.init([9634673.443298217, 0], 360 - 8.9, 7000000)
        self.calImageSize()
        self.calIncidentAngleImage()
        # self.imshow_incident_angle_image()
        # plt.show()
        self.buildings.add([[400, 534], [565, 756], [645, 534], [450, 462], [400, 534]], height=60)
        self.buildings.plot()
        # self.show()
        # plt.show()

        for i in range(self.x_n):
            for j in range(self.y_n):
                if i == 0 and j == 392:
                    pass
                x1, y1 = self.x(i), self.y(j)
                n_building, n_line = self.buildings.findBuilding(
                    x1, y1, self.incident_angle_image[i, j], self.satellite.normal_azimuth)
                if n_building is not None:
                    if n_line == -1:
                        self.image1[i, j] = n_building
                    else:
                        self.image1[i, j] = n_building * 100 + n_line

        plt.imshow(self.image1.T)
        # self.imshow_incident_angle_image()
        # self.plotSatelliteLine(9634673.443298217 - 10, 9634673.443298217 + 10)

        plt.show()


class SHSketchMapLine(SHDLine):

    def __init__(self):
        super(SHSketchMapLine, self).__init__()

        self.fx = 1

    def init(self, k, x0, y0):
        self.k = k
        self.x0 = x0
        self.y0 = y0
        self._distanceCal()
        self._calABC()
        self._calNormalAzimuth()
        self._calAzimuth()
        return self

    def x(self, y):
        return (y - self.b) / self.k

    def vK(self):
        return -1 / self.k

    def xAddDistance(self, x0, distance=1.0, fx=1):
        y0 = self.y(x0)
        x1, y1 = self.xyAddDistance(distance, fx, x0, y0)
        return x1, y1

    def xyAddDistance(self, distance, fx, x0, y0):
        t = math.sqrt(distance * distance / (1 + self.k * self.k))
        x1 = x0 + fx * t
        y1 = y0 + fx * t * self.k
        return x1, y1

    def yAddDistance(self, y0, distance=1.0, fx=1):
        x0 = self.x(y0)
        x1, y1 = self.xyAddDistance(distance, fx, x0, y0)
        return x1, y1

    def initXJiaoCoor(self, x0, y0, theta, fx=1, fy=1):
        x1 = x0 + fx * 1
        y1 = y0 + fy * math.tan(angleToRadian(theta))
        self.twoPoint(x0, y0, x1, y1)


def varToList(xys):
    datas = []
    for xy in xys:
        if hasattr(xy, "__len__") and hasattr(xy, "__getitem__"):
            for i in range(len(xy)):
                datas.append(xy[i])
        else:
            datas.append(xy)
    return tuple(datas)


class ShadowSketchMap:
    """
    https://matplotlib.org/stable/api/index.html
    """

    def __init__(self):
        self.name = ""
        self.vlri_coors = []

    def plotLine_2p(self, *xys, color="red"):
        line, x_max, x_min = self.line2p(xys)
        line.plot(x0=x_min, x1=x_max, color=color)
        return line

    def line2p(self, xys):
        args = varToList(xys)
        x1, y1, x2, y2, x_min, x_max = 0, 0, 0, 0, None, None,
        if len(args) == 4:
            x1, y1, x2, y2 = args
        elif len(args) == 5:
            x1, y1, x2, y2, x_min = args
        elif len(args) >= 6:
            x1, y1, x2, y2, x_min, x_max = args[:6]
        line = SHSketchMapLine()
        line.twoPoint(x1, y1, x2, y2, is_range=True)
        return line, x_max, x_min

    def lineK(self, *xys):
        args = varToList(xys)
        k = 0
        if len(args) >= 4:
            x1, y1, x2, y2 = args[:4]
            k = (y2 - y1) / (x2 - x1)
        return k

    def plotLine_kb(self, k, x0, y0, x_min=0, x_max=1, color="red"):
        line = self.lineKB(k, x0, y0)
        line.plot(x0=x_min, x1=x_max, color=color)
        return line

    def plotLine(self, line: SHSketchMapLine, x0=None, x1=None, y0=None, y1=None,
                 scalex=True, scaley=True, data=None, **kwargs):
        if (x0 is not None) and (x1 is not None) and (y0 is None) and (y1 is None):
            y0 = line.y(x0)
            y1 = line.y(x1)
        elif (x0 is None) and (x1 is None) and (y0 is not None) and (y1 is not None):
            x0 = line.x(y0)
            x1 = line.x(y1)
        args = self.plotXY(x0, y0, x1, y1)
        plt.plot(*args, scalex=scalex, scaley=scaley, data=data, **kwargs)

    def plotLine3D(self, ax: Axes3D, line: SHSketchMapLine, x0=None, x1=None, y0=None, y1=None, **kwargs):
        if (x0 is not None) and (x1 is not None) and (y0 is None) and (y1 is None):
            y0 = line.y(x0)
            y1 = line.y(x1)
        elif (x0 is None) and (x1 is None) and (y0 is not None) and (y1 is not None):
            x0 = line.x(y0)
            x1 = line.x(y1)
        ax.plot3D([x0, x1], [y0, y1], [0, 0], **kwargs)

    def lineKB(self, k, x0, y0):
        line = SHSketchMapLine()
        line.init(k, x0, y0)
        return line

    def coorAxis(self, ax, x0=None, x1=None, y0=None, y1=None):
        for direction in ["xzero", "yzero"]:
            ax.axis[direction].set_axisline_style("->", size=1.5)
            ax.axis[direction].set_visible(False)
        # for direction in ["left", "right", "bottom", "top"]:
        #     ax.axis[direction].set_visible(False)
        if (x0 is None) and (y0 is None) and (y0 is None) and (y1 is None):
            plt.arrow(-1, 0, 1.9, 0, length_includes_head=True, head_width=0.025, head_length=0.06, fc='black')
            plt.arrow(0, -0.8, 0, 1.9, length_includes_head=True, head_width=0.025, head_length=0.06, fc='black')
        else:
            if x0 is None:
                x0 = -1.0
            if x1 is None:
                x1 = 1.0
            if y0 is None:
                y0 = -1.0
            if y1 is None:
                y1 = 1
            plt.arrow(x0, 0, x1 - x0, 0, length_includes_head=True, head_width=0.025, head_length=0.06, fc='black')
            plt.arrow(0, y0, 0, y1 - y0, length_includes_head=True, head_width=0.025, head_length=0.06, fc='black')

    def plotXY(self, *xys):
        args = varToList(xys)
        x, y = [], []
        for i in range(0, len(args) - 1, 2):
            x.append(args[i])
            y.append(args[i + 1])
        return x, y

    def plotRectangle(self, *xys, scalex=True, scaley=True, data=None, **kwargs):
        args = self.plotXY(*xys)
        plt.plot(*args, scalex=scalex, scaley=scaley, data=data, **kwargs)

    def plotArrow(self, line: SHSketchMapLine, x_min=0.0, x_max=1.0, fx=1, **kwargs):
        y_min = line.y(x_min)
        y_max = line.y(x_max)
        if fx == 1:
            plt.arrow(x_min, y_min, x_max - x_min, y_max - y_min, **kwargs)
        elif fx == -1:
            plt.arrow(x_max, y_max, x_min - x_max, y_min - y_max, **kwargs)
        line.fx = fx
        return line

    def plotArrow3D(self, ax: Axes3D, line: SHSketchMapLine, x_min=0.0, x_max=1.0, fx=1, **kwargs):
        y_min = line.y(x_min)
        y_max = line.y(x_max)
        if fx == 1:
            ax.arrow(x_min, y_min, x_max - x_min, y_max - y_min, **kwargs)
        elif fx == -1:
            ax.arrow(x_max, y_max, x_min - x_max, y_min - y_max, **kwargs)
        line.fx = fx
        return line

    def plotLinePointDistance(self, line: SHSketchMapLine, x0, distance=1.0, fx=1,
                              scalex=True, scaley=True, data=None, **kwargs):
        x1, y1 = line.xAddDistance(x0, distance, fx=fx)
        y0 = line.y(x0)
        args = self.plotXY(x0, y0, x1, y1)
        plt.plot(*args, scalex=scalex, scaley=scaley, data=data, **kwargs)
        return x0, y0, x1, y1

    def plotLinePointDistance3D(self, ax: Axes3D, line: SHSketchMapLine, x0, *args, distance=1.0, fx=1, **kwargs):
        x1, y1 = line.xAddDistance(x0, distance, fx=fx)
        y0 = line.y(x0)
        ax.plot3D([x0, x1], [y0, y1], zs=[0, 0], **kwargs)

    def vLineRectangleIntersect(self, line: SHSketchMapLine, rectangle_coors):
        lines = [SHSketchMapLine().init(line.vK(), x, y) for x, y in rectangle_coors[:-1]]
        lines_select = []
        lines_n = []
        lines_d = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                lines_select.append((lines[i], lines[j]))
                lines_n.append((i, j))
                lines_d.append(abs(lines[i].c - lines[j].c))
        n = findMaxWhere(lines_d)
        n_select = lines_n[n]
        self.vlri_coors = []
        for i in range(len(lines)):
            if i not in n_select:
                self.vlri_coors.append(rectangle_coors[i])
        return lines_select[n]

    def rectangleTrans(self, *xys, theta=0.0):
        if theta == 0:
            return xys
        rad = angleToRadian(theta)
        xys_out = []
        for x, y in xys:
            x1 = x * math.cos(rad) + y * math.sin(rad)
            y1 = -x * math.sin(rad) + y * math.cos(rad)
            xys_out.append((x1, y1))
        return xys_out

    def plotRectangle3D(self, *xyzs, ax: Axes3D = None, height=0.6, theta=0.0, args=(), **kwargs):
        if ax is None:
            ax = plt.axes(projection='3d')
        # if theta != 0:
        #     xys = []
        #     for xyz in xyzs:
        #         xys.append(xyz[:2])
        #     xys = self.rectangleTrans(*tuple(xys), theta=theta)
        #     xys = [list(xy) for xy in xys]
        #     for i, xyz in enumerate(xyzs):
        #         if len(xyz) >= 3:
        #             xys[i].append(xyz[2])
        #     xyzs = xys
        xyzs = list(xyzs)
        x_list, y_list, z_list = [], [], []
        for i in range(len(xyzs)):
            x_list.append(xyzs[i][0])
            y_list.append(xyzs[i][1])
            if len(xyzs[i]) == 2:
                z_list.append(0)
            else:
                z_list.append(xyzs[i][2])
        ax.plot3D(x_list, y_list, z_list, **kwargs)
        for i in range(len(xyzs) - 1):
            xs = [x_list[i], x_list[i], x_list[i + 1], x_list[i + 1]]
            ys = [y_list[i], y_list[i], y_list[i + 1], y_list[i + 1]]
            zs = [0, height, height, 0]
            ax.plot3D(xs, ys, zs, **kwargs)

    def fill(self, xys, *args, **kwargs):
        xys = varToList(xys)
        x, y = self.plotXY(xys)
        # plt.scatter(x, y)
        plt.fill(x, y, *args, **kwargs)

    def plotLineThree(self, ax, line: SHSketchMapLine, x0, d, h, fx=0, **kwargs):
        y0 = line.y(x0)
        xys = self.rectangleTrans((-d / 2.0, 0), (d / 2.0, 0), (0, h), theta=line.azimuth + fx * 180)
        x, y = self.plotXY(*tuple(xys))
        x, y = np.array(x), np.array(y)
        x, y = x + x0, y + y0
        three_xz = plt.Polygon(xy=[[x[i], y[i]] for i in range(len(x))], **kwargs)
        ax.add_patch(three_xz)

    def main(self):
        self.draw3D()

    def draw3D(self):
        self.name = ""
        ax: Axes3D = plt.figure().add_subplot(projection='3d')
        x0, y0 = 0.3, 0.2
        building_coors = self.rectangleTrans((x0, y0), (-x0, y0), (-x0, -y0), (x0, -y0), (x0, y0), theta=-5)
        self.plotRectangle3D(*tuple(building_coors), ax=ax, color="gray")
        ax.plot3D([-1, 1], [0, 0], [0, 0], color="black")
        ax.plot3D([0, 0], [-1, 1], [0, 0], color="black")

        def sh_as(fx=1):
            line_as = self.lineKB(self.lineK(-0.2493, -0.05480094, -0.25319841, -0.03558636), -0.8, 0)
            self.plotLine3D(ax, line_as, -0.96, -0.60)
            line1, line2 = self.vLineRectangleIntersect(line_as, building_coors)
            self.plotLinePointDistance3D(ax, line1, line1.x0, distance=0.8, fx=fx, color="green")
            self.plotLinePointDistance3D(ax, line2, line2.x0, distance=0.8, fx=fx, color="green")
            # self.plotLinePointDistance3D(ax, line1, line1.x0, distance=0.8, fx=-fx, color="green")
            # self.plotLinePointDistance3D(ax, line2, line2.x0, distance=0.8, fx=-fx, color="green")

        def sh_de(fx=-1):
            line_de = self.lineKB(self.lineK(0.24939, -0.05480094, 0.25319841, -0.03558636), 0.8, 0)
            self.plotLine3D(ax, line_de, 0.60, 0.96)
            line1, line2 = self.vLineRectangleIntersect(line_de, building_coors)
            self.plotLinePointDistance3D(ax, line1, line1.x0, distance=0.8, fx=fx, color="red")
            self.plotLinePointDistance3D(ax, line2, line2.x0, distance=0.8, fx=fx, color="red")
            # self.plotLinePointDistance3D(ax, line1, line1.x0, distance=0.8, fx=-fx, color="red")
            # self.plotLinePointDistance3D(ax, line2, line2.x0, distance=0.8, fx=-fx, color="red")

        def sh_opt():
            line_sun = SHSketchMapLine().init(0.3, 0, -0.6)
            # self.plotLine(line_sun, -1, 1)
            line1, line2 = self.vLineRectangleIntersect(line_sun, building_coors)
            self.plotLinePointDistance3D(ax, line1, line1.x0, distance=1, fx=-1, color="blue")
            self.plotLinePointDistance3D(ax, line2, line2.x0, distance=1, fx=-1, color="blue")

        sh_as(1)
        # sh_de(-1)
        # sh_opt()
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)
        # ax.set_zlim(0, 1)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        # Make legend, set axes limits and labels
        # ax.legend()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # Customize the view angle so it's easier to see that the scatter points lie
        # on the plane y=0
        ax.view_init(elev=20., azim=-35, roll=0)
        # plt.text(-0.1, -0.1, "O", fontdict={"size": FONT_SIZE})
        plt.show()

    def draw2D(self):

        def same_sign(num1, num2):
            return (num1 * num2) > 0

        def plotDraw2D(subplot_number=111, theta=0.0,
                       is_as=False, is_de=False, is_opt=False, fig=None,
                       number_fig="a"
                       ):
            if fig is None:
                fig = plt.figure(figsize=(6, 6))
                plt.subplots_adjust(top=0.96, bottom=0.04, left=0.04, right=0.96, hspace=0.2, wspace=0.2)

            ax = fig.add_subplot(subplot_number, axes_class=AxesZero)
            ax.set_aspect('equal', adjustable='box')
            self.coorAxis(ax)

            x0, y0 = 0.3, 0.3
            building_coors = self.rectangleTrans((x0, y0), (-x0, y0), (-x0, -y0), (x0, -y0), (x0, y0), theta=theta)
            self.plotRectangle(*tuple(building_coors), color="black")
            is_draw = True

            line_as = self.lineKB(self.lineK(-0.2493, -0.05480094, -0.25319841, -0.03558636), -0.8, 0)
            line_de = self.lineKB(self.lineK(0.24939, -0.05480094, 0.25319841, -0.03558636), 0.8, 0)
            line_sun = SHSketchMapLine().init(0.3, 0, -0.9)

            def sh_as(fx=1):
                self.plotArrow(line_as, -0.95, -0.66, fx=-1, length_includes_head=True,
                               head_width=0.025, head_length=0.06, fc='blue', color="blue")
                line1, line2 = self.vLineRectangleIntersect(line_as, building_coors)
                distance1 = line_as.pointDistance(line1.x0, line1.y0) - 0.1
                distance2 = line_as.pointDistance(line2.x0, line2.y0) - 0.1
                coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance2, fx=fx, color="darkblue")
                coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance1, fx=fx, color="darkblue")
                self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[0]), alpha=0.5,
                          color="darkblue", label="Ascending shadow")
                if is_draw:
                    distance1 = line_as.pointDistance(line1.x0, line1.y0) - 0.1
                    distance2 = line_as.pointDistance(line2.x0, line2.y0) - 0.1
                    coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance1, fx=-fx, color="royalblue")
                    coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance2, fx=-fx, color="royalblue")
                    self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[1]), alpha=0.5,
                              color="royalblue", label="Ascending double-bounce")
                plt.text(-1.12, 0.8, "Ascending orbit", fontdict={"size": FONT_SIZE})
                return line1, line2

            def sh_de(fx=-1):
                self.plotArrow(line_de, 0.68, 0.95, fx=-1, length_includes_head=True,
                               head_width=0.025, head_length=0.06, fc='red', color="red")
                line1, line2 = self.vLineRectangleIntersect(line_de, building_coors)
                distance1 = line_de.pointDistance(line1.x0, line1.y0) - 0.1
                distance2 = line_de.pointDistance(line2.x0, line2.y0) - 0.1
                coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance2, fx=fx, color="darkred")
                coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance1, fx=fx, color="darkred")
                self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[1]), alpha=0.5,
                          color="darkred", label="Descending shadow")
                if is_draw:
                    distance1 = line_de.pointDistance(line1.x0, line1.y0) - 0.1
                    distance2 = line_de.pointDistance(line2.x0, line2.y0) - 0.1
                    coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance1, fx=-fx, color="red")
                    coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance2, fx=-fx, color="red")
                    self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[0]), alpha=0.5,
                              color="red", label="Descending double-bounce")
                plt.text(0.2, -0.7, "Descending orbit", fontdict={"size": FONT_SIZE})
                return line1, line2

            def sh_opt():
                # self.plotLine(line_sun, -1, 1, color="orange")
                line1, line2 = self.vLineRectangleIntersect(line_sun, building_coors)
                distance1 = line_sun.pointDistance(line1.x0, line1.y0) - 0.1
                distance2 = line_sun.pointDistance(line2.x0, line2.y0) - 0.1
                coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance2, fx=-1, color="dimgrey")
                coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance1, fx=-1, color="dimgrey")
                self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[0]), alpha=0.5,
                          color="dimgrey", label="Optical shadow")
                return line1, line2

            line_as_1, line_as_2, line_de_1, line_de_2, line_opt_1, line_opt_2 = (None for _ in range(6))
            if is_as:
                line_as_1, line_as_2 = sh_as(1)
            if is_de:
                line_de_1, line_de_2 = sh_de(-1)
            if is_opt:
                line_opt_1, line_opt_2 = sh_opt()

            if is_as and is_opt:
                coor1 = line_as_1.intersect(line_de_1)
                coor2 = line_as_1.intersect(line_de_2)
                coor3 = line_as_2.intersect(line_de_1)
                coor4 = line_as_2.intersect(line_de_2)
                coors = [coor for coor in [coor1, coor2, coor3, coor4] if -1 < coor[0] < 1 and -1 < coor[1] < 1]

                def plot_coor_line(coor_init):
                    _coor_line = [coor_init]
                    for coor in building_coors[:4]:
                        if same_sign(coor[1], _coor_line[0][1]):
                            _coor_line.append(coor)
                    x, y = self.plotXY(*tuple(_coor_line))
                    plt.scatter(x, y, color="red", s=60)
                    _coor_line.append(coor_init)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.plot(x, y)
                    plt.fill(x, y, color="dimgray")

                plot_coor_line(coors[0])
                # plot_coor_line(coors[1])

            self.fill(tuple(building_coors), color="gray")

            plt.xlim([-1.2, 1.2])
            plt.ylim([-1.2, 1.2])
            plt.xticks([])
            plt.yticks([])

            plt.text(0.05, 0.9, "North", fontdict={"size": FONT_SIZE})
            plt.text(0.9, 0.05, "East", fontdict={"size": FONT_SIZE})
            plt.text(-0.2, 0, "Building", fontdict={"size": FONT_SIZE})
            plt.text(-1.1, 0.96, "({0})".format(number_fig), fontdict={"size": 16})

        def func1(filename, theta, ):
            plotDraw2D(111, theta, is_as=True)
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plt.savefig(r"F:\ProjectSet\Shadow\MkTu\SketchMap\as_{0}.svg".format(filename), dpi=300)
            plt.show()
            plotDraw2D(111, theta, is_de=True)
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plt.savefig(r"F:\ProjectSet\Shadow\MkTu\SketchMap\de_{0}.svg".format(filename), dpi=300)
            plt.show()
            plotDraw2D(111, theta, is_opt=True)
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plt.savefig(r"F:\ProjectSet\Shadow\MkTu\SketchMap\opt_{0}.svg".format(filename), dpi=300)
            plt.show()

        def func2(filename, theta):
            fig_this = plt.figure(figsize=(9, 9))
            plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98, hspace=0.01, wspace=0.01)

            plotDraw2D(331, 0, is_as=True, fig=fig_this, number_fig="a")
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plotDraw2D(332, 0, is_de=True, fig=fig_this, number_fig="b")
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plotDraw2D(333, 0, is_opt=True, fig=fig_this, number_fig="c")
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plt.savefig(r"F:\ProjectSet\Shadow\MkTu\SketchMap\{0}.svg".format(filename), dpi=300)

            plotDraw2D(334, theta, is_as=True, fig=fig_this, number_fig="d")
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plotDraw2D(335, theta, is_de=True, fig=fig_this, number_fig="e")
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plotDraw2D(336, theta, is_opt=True, fig=fig_this, number_fig="f")
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plt.savefig(r"F:\ProjectSet\Shadow\MkTu\SketchMap\{0}.svg".format(filename), dpi=300)

            plotDraw2D(337, -theta, is_as=True, fig=fig_this, number_fig="g")
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plotDraw2D(338, -theta, is_de=True, fig=fig_this, number_fig="h")
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plotDraw2D(339, -theta, is_opt=True, fig=fig_this, number_fig="i")
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plt.savefig(r"F:\ProjectSet\Shadow\MkTu\SketchMap\{0}.svg".format(filename), dpi=300)
            plt.show()

        # func1("fig2  20", 20)
        # func1("fig2 -20", -20)
        # func1("fig2  0", 0)
        # func2("fig3", 16)
        plotDraw2D(111, 0, True, True, True, number_fig="c")
        plt.legend(loc="lower left", prop={"size": 12}, frameon=False)
        # plt.savefig(r"F:\ProjectSet\Shadow\MkTu\SketchMap\fig2.svg", dpi=300)
        plt.show()

    def draw2D2(self):

        def same_sign(num1, num2):
            return (num1 * num2) > 0

        def plotDraw2D(subplot_number=111, theta=0.0,
                       is_as=False, is_de=False, is_opt=False, fig=None,
                       number_fig="a", is_draw=True
                       ):
            if fig is None:
                fig = plt.figure(figsize=(6, 6))
                plt.subplots_adjust(top=0.96, bottom=0.04, left=0.04, right=0.96, hspace=0.2, wspace=0.2)

            ax = fig.add_subplot(subplot_number, axes_class=AxesZero)
            ax.set_aspect('equal', adjustable='box')
            self.coorAxis(ax)

            x0, y0 = 0.3, 0.3
            building_coors = self.rectangleTrans((x0, y0), (-x0, y0), (-x0, -y0), (x0, -y0), (x0, y0), theta=theta)
            self.plotRectangle(*tuple(building_coors), color="black")

            line_as = self.lineKB(self.lineK(-0.2493, -0.05480094, -0.25319841, -0.03558636), -0.8, 0)
            line_de = self.lineKB(self.lineK(0.24939, -0.05480094, 0.25319841, -0.03558636), 0.8, 0)
            line_sun = SHSketchMapLine().init(0.3, 0, -0.9)

            def sh_as(fx=1):
                self.plotArrow(line_as, -0.95, -0.66, fx=-1, length_includes_head=True,
                               head_width=0.025, head_length=0.06, fc='blue', color="blue")
                line1, line2 = self.vLineRectangleIntersect(line_as, building_coors)
                distance1 = line_as.pointDistance(line1.x0, line1.y0) - 0.1
                distance2 = line_as.pointDistance(line2.x0, line2.y0) - 0.1
                coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance2, fx=fx, color="darkblue")
                coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance1, fx=fx, color="darkblue")
                self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[0]), alpha=0.5,
                          color="darkblue", label="Ascending shadow")
                if is_draw:
                    distance1 = line_as.pointDistance(line1.x0, line1.y0) - 0.1
                    distance2 = line_as.pointDistance(line2.x0, line2.y0) - 0.1
                    coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance1, fx=-fx, color="royalblue")
                    coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance2, fx=-fx, color="royalblue")
                    self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[1]), alpha=0.5,
                              color="royalblue", label="Ascending double-bounce")
                plt.text(-1.12, 0.8, "Ascending orbit", fontdict={"size": FONT_SIZE})
                return line1, line2

            def sh_de(fx=-1):
                self.plotArrow(line_de, 0.68, 0.95, fx=-1, length_includes_head=True,
                               head_width=0.025, head_length=0.06, fc='red', color="red")
                line1, line2 = self.vLineRectangleIntersect(line_de, building_coors)
                distance1 = line_de.pointDistance(line1.x0, line1.y0) - 0.1
                distance2 = line_de.pointDistance(line2.x0, line2.y0) - 0.1
                coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance2, fx=fx, color="darkred")
                coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance1, fx=fx, color="darkred")
                self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[1]), alpha=0.5,
                          color="darkred", label="Descending shadow")
                if is_draw:
                    distance1 = line_de.pointDistance(line1.x0, line1.y0) - 0.1
                    distance2 = line_de.pointDistance(line2.x0, line2.y0) - 0.1
                    coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance1, fx=-fx, color="red")
                    coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance2, fx=-fx, color="red")
                    self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[0]), alpha=0.5,
                              color="red", label="Descending double-bounce")
                plt.text(0.2, -0.7, "Descending orbit", fontdict={"size": FONT_SIZE})
                return line1, line2

            def sh_opt():
                # self.plotLine(line_sun, -1, 1, color="orange")
                line1, line2 = self.vLineRectangleIntersect(line_sun, building_coors)
                distance1 = line_sun.pointDistance(line1.x0, line1.y0) - 0.1
                distance2 = line_sun.pointDistance(line2.x0, line2.y0) - 0.1
                coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance2, fx=-1, color="dimgrey")
                coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance1, fx=-1, color="dimgrey")
                self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[0]), alpha=0.5,
                          color="dimgrey", label="Optical shadow")
                return line1, line2

            line_as_1, line_as_2, line_de_1, line_de_2, line_opt_1, line_opt_2 = (None for _ in range(6))
            if is_as:
                line_as_1, line_as_2 = sh_as(1)
            if is_de:
                line_de_1, line_de_2 = sh_de(-1)
            if is_opt:
                line_opt_1, line_opt_2 = sh_opt()

            if is_as and is_opt:
                coor1 = line_as_1.intersect(line_de_1)
                coor2 = line_as_1.intersect(line_de_2)
                coor3 = line_as_2.intersect(line_de_1)
                coor4 = line_as_2.intersect(line_de_2)
                coors = [coor for coor in [coor1, coor2, coor3, coor4] if -1 < coor[0] < 1 and -1 < coor[1] < 1]

                def plot_coor_line(coor_init):
                    _coor_line = [coor_init]
                    for coor in building_coors[:4]:
                        if same_sign(coor[1], _coor_line[0][1]):
                            _coor_line.append(coor)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.scatter(x, y, color="red", s=60)
                    _coor_line.append(coor_init)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.plot(x, y)
                    plt.fill(x, y, color="black", label="Shadow overlap area")

                plot_coor_line(coors[0])
                # plot_coor_line(coors[1])

            self.fill(tuple(building_coors), color="lightgrey")

            plt.xlim([-1.2, 1.2])
            plt.ylim([-1.2, 1.2])
            plt.xticks([])
            plt.yticks([])

            plt.text(0.05, 0.9, "North", fontdict={"size": FONT_SIZE})
            plt.text(0.9, 0.05, "East", fontdict={"size": FONT_SIZE})
            plt.text(-0.2, 0, "Building", fontdict={"size": FONT_SIZE})
            plt.text(-1.1, 0.96, "({0})".format(number_fig), fontdict={"size": 16})

        def func1(filename, theta, ):
            plotDraw2D(111, theta, is_as=True)
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plt.savefig(r"F:\ProjectSet\Shadow\MkTu\SketchMap\as_{0}.svg".format(filename), dpi=300)
            plt.show()
            plotDraw2D(111, theta, is_de=True)
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plt.savefig(r"F:\ProjectSet\Shadow\MkTu\SketchMap\de_{0}.svg".format(filename), dpi=300)
            plt.show()
            plotDraw2D(111, theta, is_opt=True)
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plt.savefig(r"F:\ProjectSet\Shadow\MkTu\SketchMap\opt_{0}.svg".format(filename), dpi=300)
            plt.show()

        def func2(theta_small, theta_big):
            fig_this = plt.figure(figsize=(9, 9))
            plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98, hspace=0.01, wspace=0.01)

            plotDraw2D(331, 0, True, True, True, is_draw=False, fig=fig_this, number_fig="a")
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plotDraw2D(332, -theta_small, True, True, True, is_draw=False, fig=fig_this, number_fig="b")
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plotDraw2D(333, theta_small, True, True, True, is_draw=False, fig=fig_this, number_fig="c")
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)

            plotDraw2D(334, 0, True, True, True, is_draw=False, fig=fig_this, number_fig="d")
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plotDraw2D(335, theta_big, True, True, True, is_draw=False, fig=fig_this, number_fig="e")
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plotDraw2D(336, -theta_big, True, True, True, is_draw=False, fig=fig_this, number_fig="f")
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)

            plotDraw2D(337, 0, True, True, True, is_draw=True, fig=fig_this, number_fig="g")
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plotDraw2D(338, -10, True, True, True, is_draw=True, fig=fig_this, number_fig="h")
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plotDraw2D(339, 10, True, True, True, is_draw=True, fig=fig_this, number_fig="i")
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)

        func2(6, 20)
        # plotDraw2D(111, 6, True, True, True, is_draw=False, number_fig="a")
        # plt.legend(loc="lower left", prop={"size": 12}, frameon=False)
        # plt.savefig(r"F:\ProjectSet\Shadow\MkTu\SketchMap\fig2.svg", dpi=300)
        plt.show()

    def drawASDE(self):

        def same_sign(num1, num2):
            return (num1 * num2) > 0

        def plotDraw2D(subplot_number=111, theta=0.0,
                       is_as=False, is_de=False, is_opt=False, fig=None,
                       number_fig="a", is_draw=True):

            if fig is None:
                fig = plt.figure(figsize=(2, 3))
                plt.subplots_adjust(top=0.96, bottom=0.04, left=0.04, right=0.96, hspace=0.2, wspace=0.2)

            ax = fig.add_subplot(subplot_number, axes_class=AxesZero)
            ax.set_aspect('equal', adjustable='box')
            self.coorAxis(ax, y0=-0.7)

            x0, y0 = 0.3, 0.2
            building_coors = self.rectangleTrans((x0, y0), (-x0, y0), (-x0, -y0), (x0, -y0), (x0, y0), theta=theta)
            self.plotRectangle(*tuple(building_coors), color="black")

            line_as = self.lineKB(self.lineK(-0.2493, -0.05480094, -0.25319841, -0.03558636), -0.8, 0)
            line_de = self.lineKB(self.lineK(0.24939, -0.05480094, 0.25319841, -0.03558636), 0.8, 0)
            line_sun = SHSketchMapLine().init(0.3, 0, -0.9)

            def sh_as(fx=1):
                self.plotArrow(line_as, -0.95, -0.66, fx=-1, length_includes_head=True,
                               head_width=0.025, head_length=0.06, fc='blue', color="blue")
                line1, line2 = self.vLineRectangleIntersect(line_as, building_coors)
                distance1 = line_as.pointDistance(line1.x0, line1.y0) - 0.1
                distance2 = line_as.pointDistance(line2.x0, line2.y0) - 0.1
                coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance2, fx=fx, color="darkblue")
                coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance1, fx=fx, color="darkblue")
                self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[0]), alpha=0.5,
                          color="darkblue", label="Ascending shadow")
                if is_draw:
                    distance1 = line_as.pointDistance(line1.x0, line1.y0) - 0.1
                    distance2 = line_as.pointDistance(line2.x0, line2.y0) - 0.1
                    coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance1, fx=-fx, color="royalblue")
                    coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance2, fx=-fx, color="royalblue")
                    self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[1]), alpha=0.5,
                              color="royalblue", label="Ascending double-bounce")
                plt.text(-1.12, 0.8, "Ascending orbit", fontdict={"size": FONT_SIZE})
                return line1, line2

            def sh_de(fx=-1):
                self.plotArrow(line_de, 0.68, 0.95, fx=-1, length_includes_head=True,
                               head_width=0.025, head_length=0.06, fc='red', color="red")
                line1, line2 = self.vLineRectangleIntersect(line_de, building_coors)
                distance1 = line_de.pointDistance(line1.x0, line1.y0) - 0.1
                distance2 = line_de.pointDistance(line2.x0, line2.y0) - 0.1
                coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance2, fx=fx, color="darkred")
                coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance1, fx=fx, color="darkred")
                self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[1]), alpha=0.5,
                          color="darkred", label="Descending shadow")
                if is_draw:
                    distance1 = line_de.pointDistance(line1.x0, line1.y0) - 0.1
                    distance2 = line_de.pointDistance(line2.x0, line2.y0) - 0.1
                    coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance1, fx=-fx, color="red")
                    coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance2, fx=-fx, color="red")
                    self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[0]), alpha=0.5,
                              color="red", label="Descending double-bounce")
                plt.text(0.2, -0.7, "Descending orbit", fontdict={"size": FONT_SIZE})
                return line1, line2

            def sh_opt():
                # self.plotLine(line_sun, -1, 1, color="orange")
                line1, line2 = self.vLineRectangleIntersect(line_sun, building_coors)
                distance1 = line_sun.pointDistance(line1.x0, line1.y0) - 0.1
                distance2 = line_sun.pointDistance(line2.x0, line2.y0) - 0.1
                coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance2, fx=-1, color="dimgrey")
                coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance1, fx=-1, color="dimgrey")
                color_str = "dimgrey"
                alpha = 0.5
                if (not is_as) and (not is_de) and is_opt:
                    color_str = "black"
                    alpha = 1
                self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[0]), alpha=alpha,
                          color=color_str, label="Optical shadow")
                return line1, line2

            line_as_1, line_as_2, line_de_1, line_de_2, line_opt_1, line_opt_2 = (None for _ in range(6))
            if is_as:
                line_as_1, line_as_2 = sh_as(1)
            if is_de:
                line_de_1, line_de_2 = sh_de(-1)
            if is_opt:
                line_opt_1, line_opt_2 = sh_opt()

            if is_as and is_de:
                coor1 = line_as_1.intersect(line_de_1)
                coor2 = line_as_1.intersect(line_de_2)
                coor3 = line_as_2.intersect(line_de_1)
                coor4 = line_as_2.intersect(line_de_2)
                coors = [coor for coor in [coor1, coor2, coor3, coor4] if -1 < coor[0] < 1 and -1 < coor[1] < 1]

                def plot_coor_line(coor_init):
                    _coor_line = [coor_init]
                    for coor in building_coors[:4]:
                        if same_sign(coor[1], _coor_line[0][1]):
                            _coor_line.append(coor)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.scatter(x, y, color="red", s=60)
                    _coor_line.append(coor_init)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.plot(x, y)
                    plt.fill(x, y, color="black", label="Shadow overlap area")

                plot_coor_line(coors[0])
                # plot_coor_line(coors[1])

            if is_opt and is_as and (not is_de):
                coor1 = line_opt_1.intersect(line_as_1)
                coor2 = line_opt_1.intersect(line_as_2)
                coor3 = line_opt_2.intersect(line_as_1)
                coor4 = line_opt_2.intersect(line_as_2)
                coors = [coor for coor in [coor1, coor2, coor3, coor4] if -1 < coor[0] < 1 and -1 < coor[1] < 1]

                def plot_coor_line(coor_init):
                    _coor_line = [coor_init]
                    for coor in building_coors[:4]:
                        if same_sign(coor[1], _coor_line[0][1]):
                            _coor_line.append(coor)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.scatter(x, y, color="red", s=60)
                    _coor_line.append(coor_init)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.plot(x, y)
                    plt.fill(x, y, color="black", label="Shadow overlap area")

                plot_coor_line(coors[0])
                # plot_coor_line(coors[1])

            if is_opt and is_de and (not is_as):
                coor1 = line_opt_1.intersect(line_de_1)
                coor2 = line_opt_1.intersect(line_de_2)
                coor3 = line_opt_2.intersect(line_de_1)
                coor4 = line_opt_2.intersect(line_de_2)
                coors = [coor for coor in [coor1, coor2, coor3, coor4] if -1 < coor[0] < 1 and -1 < coor[1] < 1]

                def plot_coor_line(coor_init, ):
                    _coor_line = [coor_init]
                    for coor in building_coors[:4]:
                        if same_sign(coor[1], _coor_line[0][1]):
                            _coor_line.append(coor)
                        elif coor[0] < 0:
                            _coor_line.append(coor)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.scatter(x, y, color="red", s=60)
                    _coor_line.append(coor_init)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.plot(x, y)
                    plt.fill(x, y, color="black", label="Shadow overlap area")

                # plot_coor_line(coors[0])
                plot_coor_line(coors[2])
                # plot_coor_line(coors[3])

            self.fill(tuple(building_coors), color="lightgrey")

            plt.xlim([-1.2, 1.2])
            plt.ylim([-0.9, 1.2])
            plt.xticks([])
            plt.yticks([])

            plt.text(0.05, 0.9, "North", fontdict={"size": FONT_SIZE})
            plt.text(0.9, 0.05, "East", fontdict={"size": FONT_SIZE})
            plt.text(-0.22, -0.05, "Building", fontdict={"size": FONT_SIZE})
            plt.text(-1.1, 0.96, "({0})".format(number_fig), fontdict={"size": 16})

        fig_this = plt.figure(figsize=(12, 7.3))
        plt.subplots_adjust(top=0.96, bottom=0.04, left=0.02, right=0.98, hspace=0.01, wspace=0.01)

        def func1(theta):
            plotDraw2D(231, theta, False, False, True, is_draw=False, fig=fig_this, number_fig="a")
            plotDraw2D(232, theta, True, False, True, is_draw=False, fig=fig_this, number_fig="b")
            plotDraw2D(233, theta, False, True, True, is_draw=False, fig=fig_this, number_fig="c")
            plotDraw2D(235, theta, True, True, True, is_draw=False, fig=fig_this, number_fig="e")
            plt.legend(bbox_to_anchor=(1.05, 0.0), loc=3, borderaxespad=0, prop={"size": FONT_SIZE}, frameon=False)
            plotDraw2D(234, theta, True, True, False, is_draw=False, fig=fig_this, number_fig="d")

        func1(0)
        # plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
        plt.savefig(r"F:\ProjectSet\Shadow\MkTu\SketchMap\as_de_opt.jpg", dpi=300)
        plt.show()

    def drawDoubleBounds(self):

        def same_sign(num1, num2):
            return (num1 * num2) > 0

        def plotDraw2D(subplot_number=111, theta=0.0,
                       is_as=False, is_de=False, is_opt=False, fig=None,
                       number_fig="a", is_draw=True):

            if fig is None:
                fig = plt.figure(figsize=(2, 3))
                plt.subplots_adjust(top=0.96, bottom=0.04, left=0.04, right=0.96, hspace=0.2, wspace=0.2)

            ax = fig.add_subplot(subplot_number, axes_class=AxesZero)
            ax.set_aspect('equal', adjustable='box')
            self.coorAxis(ax, y0=-0.7)

            x0, y0 = 0.3, 0.2
            building_coors = self.rectangleTrans((x0, y0), (-x0, y0), (-x0, -y0), (x0, -y0), (x0, y0), theta=theta)
            self.plotRectangle(*tuple(building_coors), color="black")

            line_as = self.lineKB(self.lineK(-0.2493, -0.05480094, -0.25319841, -0.03558636), -0.8, 0)
            line_de = self.lineKB(self.lineK(0.24939, -0.05480094, 0.25319841, -0.03558636), 0.8, 0)
            line_sun = SHSketchMapLine().init(0.3, 0, -0.9)

            def sh_as(fx=1):
                self.plotArrow(line_as, -0.95, -0.66, fx=-1, length_includes_head=True,
                               head_width=0.025, head_length=0.06, fc='blue', color="blue")
                line1, line2 = self.vLineRectangleIntersect(line_as, building_coors)
                if not is_draw:
                    distance1 = line_as.pointDistance(line1.x0, line1.y0) - 0.1
                    distance2 = line_as.pointDistance(line2.x0, line2.y0) - 0.1
                    coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance2, fx=fx, color="darkblue")
                    coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance1, fx=fx, color="darkblue")
                    self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[0]), alpha=0.5,
                              color="darkblue", label="Ascending shadow")
                if is_draw:
                    distance1 = line_as.pointDistance(line1.x0, line1.y0) - 0.1
                    distance2 = line_as.pointDistance(line2.x0, line2.y0) - 0.1
                    coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance1, fx=-fx, color="royalblue")
                    coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance2, fx=-fx, color="royalblue")
                    self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[1]), alpha=0.5,
                              color="royalblue", label="Ascending double-bounce")
                plt.text(-1.12, 0.8, "Ascending orbit", fontdict={"size": FONT_SIZE})
                return line1, line2

            def sh_de(fx=-1):
                self.plotArrow(line_de, 0.68, 0.95, fx=-1, length_includes_head=True,
                               head_width=0.025, head_length=0.06, fc='red', color="red")
                line1, line2 = self.vLineRectangleIntersect(line_de, building_coors)
                if not is_draw:
                    distance1 = line_de.pointDistance(line1.x0, line1.y0) - 0.1
                    distance2 = line_de.pointDistance(line2.x0, line2.y0) - 0.1
                    coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance2, fx=fx, color="darkred")
                    coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance1, fx=fx, color="darkred")
                    self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[1]), alpha=0.5,
                              color="darkred", label="Descending shadow")
                if is_draw:
                    distance1 = line_de.pointDistance(line1.x0, line1.y0) - 0.1
                    distance2 = line_de.pointDistance(line2.x0, line2.y0) - 0.1
                    coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance1, fx=-fx, color="red")
                    coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance2, fx=-fx, color="red")
                    self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[0]), alpha=0.5,
                              color="red", label="Descending double-bounce")
                plt.text(0.2, -0.7, "Descending orbit", fontdict={"size": FONT_SIZE})
                return line1, line2

            def sh_opt():
                # self.plotLine(line_sun, -1, 1, color="orange")
                line1, line2 = self.vLineRectangleIntersect(line_sun, building_coors)
                distance1 = line_sun.pointDistance(line1.x0, line1.y0) - 0.1
                distance2 = line_sun.pointDistance(line2.x0, line2.y0) - 0.1
                coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance2, fx=-1, color="dimgrey")
                coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance1, fx=-1, color="dimgrey")
                color_str = "dimgrey"
                alpha = 0.5
                if (not is_as) and (not is_de) and is_opt:
                    color_str = "black"
                    alpha = 1
                self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[0]), alpha=alpha,
                          color=color_str, label="Optical shadow")
                return line1, line2

            line_as_1, line_as_2, line_de_1, line_de_2, line_opt_1, line_opt_2 = (None for _ in range(6))
            if is_as:
                line_as_1, line_as_2 = sh_as(1)
            if is_de:
                line_de_1, line_de_2 = sh_de(-1)
            if is_opt:
                line_opt_1, line_opt_2 = sh_opt()

            if is_as and is_de:
                coor1 = line_as_1.intersect(line_de_1)
                coor2 = line_as_1.intersect(line_de_2)
                coor3 = line_as_2.intersect(line_de_1)
                coor4 = line_as_2.intersect(line_de_2)
                coors = [coor for coor in [coor1, coor2, coor3, coor4] if -1 < coor[0] < 1 and -1 < coor[1] < 1]

                def plot_coor_line(coor_init):
                    _coor_line = [coor_init]
                    for coor in building_coors[:4]:
                        if same_sign(coor[1], _coor_line[0][1]):
                            _coor_line.append(coor)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.scatter(x, y, color="red", s=60)
                    _coor_line.append(coor_init)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.plot(x, y)
                    plt.fill(x, y, color="black", label="Shadow overlap area")

                plot_coor_line(coors[0])
                # plot_coor_line(coors[1])

            if is_opt and is_as and (not is_de):
                coor1 = line_opt_1.intersect(line_as_1)
                coor2 = line_opt_1.intersect(line_as_2)
                coor3 = line_opt_2.intersect(line_as_1)
                coor4 = line_opt_2.intersect(line_as_2)
                coors = [coor for coor in [coor1, coor2, coor3, coor4] if -1 < coor[0] < 1 and -1 < coor[1] < 1]

                def plot_coor_line(coor_init):
                    _coor_line = [coor_init]
                    for coor in building_coors[:4]:
                        if same_sign(coor[1], _coor_line[0][1]):
                            _coor_line.append(coor)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.scatter(x, y, color="red", s=60)
                    _coor_line.append(coor_init)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.plot(x, y)
                    plt.fill(x, y, color="black", label="Shadow overlap area")

                plot_coor_line(coors[0])
                # plot_coor_line(coors[1])

            if is_opt and is_de and (not is_as):
                coor1 = line_opt_1.intersect(line_de_1)
                coor2 = line_opt_1.intersect(line_de_2)
                coor3 = line_opt_2.intersect(line_de_1)
                coor4 = line_opt_2.intersect(line_de_2)
                coors = [coor for coor in [coor1, coor2, coor3, coor4] if -1 < coor[0] < 1 and -1 < coor[1] < 1]

                def plot_coor_line(coor_init, ):
                    _coor_line = [coor_init]
                    for coor in building_coors[:4]:
                        if same_sign(coor[1], _coor_line[0][1]):
                            _coor_line.append(coor)
                        elif coor[0] < 0:
                            _coor_line.append(coor)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.scatter(x, y, color="red", s=60)
                    _coor_line.append(coor_init)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.plot(x, y)
                    plt.fill(x, y, color="black", label="Shadow overlap area")

                # plot_coor_line(coors[0])
                plot_coor_line(coors[2])
                # plot_coor_line(coors[3])

            self.fill(tuple(building_coors), color="lightgrey")

            plt.xlim([-1.2, 1.2])
            plt.ylim([-1.2, 1.2])
            plt.xticks([])
            plt.yticks([])

            plt.text(0.05, 0.9, "North", fontdict={"size": FONT_SIZE})
            plt.text(0.9, 0.05, "East", fontdict={"size": FONT_SIZE})
            plt.text(-0.22, -0.05, "Building", fontdict={"size": FONT_SIZE})
            plt.text(-1.1, 0.96, "({0})".format(number_fig), fontdict={"size": 16})

        fig_this = plt.figure(figsize=(12, 4))
        plt.subplots_adjust(top=0.96, bottom=0.04, left=0.02, right=0.98, hspace=0.01, wspace=0.01)

        def plot_double_bounce(subplot_number, theta, fig, number_fig=None):

            ax = fig.add_subplot(subplot_number, axes_class=AxesZero)
            ax.set_aspect('equal', adjustable='box')

            x0, y0 = 0.2, 0.6
            dx, dy = -0.6, 0.1
            building_coors = self.rectangleTrans((x0, y0), (-x0, y0), (-x0, -y0), (x0, -y0), (x0, y0), theta=theta)

            x, y = self.plotXY(*tuple(building_coors))
            x = np.array(x) + dx
            y = np.array(y) + dy
            plt.plot(x, y, color="black")
            plt.plot([-x0 + dx - 0.1, 1.0], [-y0 + dy, -y0 + dy], color="black")
            plt.fill(x, y, color="lightgrey", label="Building")

            incidence_angle = 38
            x_double_bounce = 0.2
            color_double_bounce = "midnightblue"

            line1 = SHSketchMapLine()
            line1.initXJiaoCoor(x_double_bounce, -y0 + dy, incidence_angle)
            line1.plot(x_double_bounce, 1.0, color=color_double_bounce,
                       label="Electromagnetic wave")

            line2 = SHSketchMapLine()
            line2.initXJiaoCoor(x_double_bounce, -y0 + dy, incidence_angle, fx=-1)
            line2.plot(x0 + dx, x_double_bounce, color=color_double_bounce)

            line3 = SHSketchMapLine()
            line3.initXJiaoCoor(x0 + dx, line2.y(x0 + dx), incidence_angle, fx=1)
            line3.plot(x0 + dx, 0.8, color=color_double_bounce)

            self.plotLineThree(ax, line1, (x_double_bounce + 1.0) / 2.0-0.05,     0.08, 0.12, color=color_double_bounce, fx=1)
            self.plotLineThree(ax, line2, (x0 + dx + x_double_bounce) / 2.0-0.05, 0.08, 0.12, color=color_double_bounce, fx=1)
            self.plotLineThree(ax, line3, (x0 + dx + 0.8) / 2.0-0.05,             0.08, 0.12, color=color_double_bounce, fx=0)

            plt.xlim([-1.2, 1.2])
            plt.ylim([-1.2, 1.2])

            plt.xticks([])
            plt.yticks([])

            if number_fig is not None:
                plt.text(-1.1, 0.96, "({0})".format(number_fig), fontdict={"size": 16})

        def func1(theta):
            # plotDraw2D(131, theta, False, False, True, is_draw=False, fig=fig_this, number_fig="a")
            plot_double_bounce(131, theta, fig=fig_this, number_fig="a")
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plotDraw2D(132, theta, True, False, False, is_draw=True, fig=fig_this, number_fig="b")
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plotDraw2D(133, theta, False, True, False, is_draw=True, fig=fig_this, number_fig="c")
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)

        func1(0)
        # plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
        plt.savefig(r"F:\ProjectSet\Shadow\MkTu\4.1Details\double_bounds.jpg", dpi=300)
        plt.show()

    def drawASDE2(self):

        def same_sign(num1, num2):
            return (num1 * num2) > 0

        def plotDraw2D(subplot_number=111, theta=0.0,
                       is_as=False, is_de=False, is_opt=False, fig=None,
                       number_fig="a", is_draw=True):

            if fig is None:
                fig = plt.figure(figsize=(2, 3))
                plt.subplots_adjust(top=0.96, bottom=0.04, left=0.04, right=0.96, hspace=0.2, wspace=0.2)

            ax = fig.add_subplot(subplot_number, axes_class=AxesZero)
            ax.set_aspect('equal', adjustable='box')
            self.coorAxis(ax, y0=-0.7)

            x0, y0 = 0.3, 0.2
            building_coors = self.rectangleTrans((x0, y0), (-x0, y0), (-x0, -y0), (x0, -y0), (x0, y0), theta=theta)
            self.plotRectangle(*tuple(building_coors), color="black")

            line_as = self.lineKB(self.lineK(-0.2493, -0.05480094, -0.25319841, -0.03558636), -0.8, 0)
            line_de = self.lineKB(self.lineK(0.24939, -0.05480094, 0.25319841, -0.03558636), 0.8, 0)
            line_sun = SHSketchMapLine().init(0.3, 0, -0.9)

            def sh_as(fx=1):
                self.plotArrow(line_as, -0.95, -0.66, fx=-1, length_includes_head=True,
                               head_width=0.025, head_length=0.06, fc='blue', color="blue")
                line1, line2 = self.vLineRectangleIntersect(line_as, building_coors)
                distance1 = line_as.pointDistance(line1.x0, line1.y0) - 0.1
                distance2 = line_as.pointDistance(line2.x0, line2.y0) - 0.1
                coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance2, fx=fx, color="darkblue")
                coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance1, fx=fx, color="darkblue")
                self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[0]), alpha=0.5,
                          color="darkblue", label="Ascending shadow")
                if is_draw:
                    distance1 = line_as.pointDistance(line1.x0, line1.y0) - 0.1
                    distance2 = line_as.pointDistance(line2.x0, line2.y0) - 0.1
                    coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance1, fx=-fx, color="royalblue")
                    coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance2, fx=-fx, color="royalblue")
                    self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[1]), alpha=0.5,
                              color="royalblue", label="Ascending double-bounce")
                plt.text(-1.12, 0.8, "Ascending orbit", fontdict={"size": FONT_SIZE})
                return line1, line2

            def sh_de(fx=-1):
                self.plotArrow(line_de, 0.68, 0.95, fx=-1, length_includes_head=True,
                               head_width=0.025, head_length=0.06, fc='red', color="red")
                line1, line2 = self.vLineRectangleIntersect(line_de, building_coors)
                distance1 = line_de.pointDistance(line1.x0, line1.y0) - 0.1
                distance2 = line_de.pointDistance(line2.x0, line2.y0) - 0.1
                coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance2, fx=fx, color="darkred")
                coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance1, fx=fx, color="darkred")
                self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[1]), alpha=0.5,
                          color="darkred", label="Descending shadow")
                if is_draw:
                    distance1 = line_de.pointDistance(line1.x0, line1.y0) - 0.1
                    distance2 = line_de.pointDistance(line2.x0, line2.y0) - 0.1
                    coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance1, fx=-fx, color="red")
                    coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance2, fx=-fx, color="red")
                    self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[0]), alpha=0.5,
                              color="red", label="Descending double-bounce")
                plt.text(0.2, -0.7, "Descending orbit", fontdict={"size": FONT_SIZE})
                return line1, line2

            def sh_opt():
                # self.plotLine(line_sun, -1, 1, color="orange")
                line1, line2 = self.vLineRectangleIntersect(line_sun, building_coors)
                distance1 = line_sun.pointDistance(line1.x0, line1.y0) - 0.1
                distance2 = line_sun.pointDistance(line2.x0, line2.y0) - 0.1
                coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance2, fx=-1, color="dimgrey")
                coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance1, fx=-1, color="dimgrey")
                color_str = "dimgrey"
                alpha = 0.5
                if (not is_as) and (not is_de) and is_opt:
                    color_str = "black"
                    color_str = "dimgrey"
                    alpha = 1
                    alpha = 0.5
                self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[0]), alpha=alpha,
                          color=color_str, label="Optical shadow")
                return line1, line2

            line_as_1, line_as_2, line_de_1, line_de_2, line_opt_1, line_opt_2 = (None for _ in range(6))
            if is_as:
                line_as_1, line_as_2 = sh_as(1)
            if is_de:
                line_de_1, line_de_2 = sh_de(-1)
            if is_opt:
                line_opt_1, line_opt_2 = sh_opt()

            if is_as and is_de:
                coor1 = line_as_1.intersect(line_de_1)
                coor2 = line_as_1.intersect(line_de_2)
                coor3 = line_as_2.intersect(line_de_1)
                coor4 = line_as_2.intersect(line_de_2)
                coors = [coor for coor in [coor1, coor2, coor3, coor4] if -1 < coor[0] < 1 and -1 < coor[1] < 1]

                def plot_coor_line(coor_init):
                    _coor_line = [coor_init]
                    for coor in building_coors[:4]:
                        if same_sign(coor[1], _coor_line[0][1]):
                            _coor_line.append(coor)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.scatter(x, y, color="red", s=60)
                    _coor_line.append(coor_init)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.plot(x, y)
                    plt.fill(x, y, color="black", label="Shadow overlap area")

                plot_coor_line(coors[0])
                # plot_coor_line(coors[1])

            if is_opt and is_as and (not is_de):
                coor1 = line_opt_1.intersect(line_as_1)
                coor2 = line_opt_1.intersect(line_as_2)
                coor3 = line_opt_2.intersect(line_as_1)
                coor4 = line_opt_2.intersect(line_as_2)
                coors = [coor for coor in [coor1, coor2, coor3, coor4] if -1 < coor[0] < 1 and -1 < coor[1] < 1]

                def plot_coor_line(coor_init):
                    _coor_line = [coor_init]
                    for coor in building_coors[:4]:
                        if same_sign(coor[1], _coor_line[0][1]):
                            _coor_line.append(coor)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.scatter(x, y, color="red", s=60)
                    _coor_line.append(coor_init)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.plot(x, y)
                    plt.fill(x, y, color="black", label="Shadow overlap area")

                plot_coor_line(coors[0])
                # plot_coor_line(coors[1])

            if is_opt and is_de and (not is_as):
                coor1 = line_opt_1.intersect(line_de_1)
                coor2 = line_opt_1.intersect(line_de_2)
                coor3 = line_opt_2.intersect(line_de_1)
                coor4 = line_opt_2.intersect(line_de_2)
                coors = [coor for coor in [coor1, coor2, coor3, coor4] if -1 < coor[0] < 1 and -1 < coor[1] < 1]

                def plot_coor_line(coor_init, ):
                    _coor_line = [coor_init]
                    for coor in building_coors[:4]:
                        if same_sign(coor[1], _coor_line[0][1]):
                            _coor_line.append(coor)
                        elif coor[0] < 0:
                            _coor_line.append(coor)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.scatter(x, y, color="red", s=60)
                    _coor_line.append(coor_init)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.plot(x, y)
                    plt.fill(x, y, color="black", label="Shadow overlap area")

                # plot_coor_line(coors[0])
                plot_coor_line(coors[2])
                # plot_coor_line(coors[3])

            self.fill(tuple(building_coors), color="lightgrey")

            plt.xlim([-1.2, 1.2])
            plt.ylim([-0.9, 1.2])
            plt.xticks([])
            plt.yticks([])

            plt.text(0.05, 0.9, "North", fontdict={"size": FONT_SIZE})
            plt.text(0.9, 0.05, "East", fontdict={"size": FONT_SIZE})
            plt.text(-0.22, -0.05, "Building", fontdict={"size": FONT_SIZE})
            plt.text(-1.1, 0.96, "({0})".format(number_fig), fontdict={"size": 16})

        fig_this = plt.figure(figsize=(12, 7.3))
        plt.subplots_adjust(top=0.98, bottom=0.08, left=0.03, right=0.97, hspace=0.01, wspace=0.01)

        def func1(theta):
            plotDraw2D(231, theta, False, False, True, is_draw=False, fig=fig_this, number_fig="a")
            plotDraw2D(232, theta, True, False, False, is_draw=False, fig=fig_this, number_fig="b")
            plotDraw2D(233, theta, False, True, False, is_draw=False, fig=fig_this, number_fig="c")
            plotDraw2D(235, theta, False, True, True, is_draw=False, fig=fig_this, number_fig="e")
            plotDraw2D(234, theta, True, False, True, is_draw=False, fig=fig_this, number_fig="d")
            plotDraw2D(236, theta, True, True, True, is_draw=False, fig=fig_this, number_fig="f")
            plt.legend(ncol=4, bbox_to_anchor=(-0.5, -0.1), loc="center",
                       borderaxespad=0, prop={"size": FONT_SIZE}, frameon=False)

        func1(0)
        # plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
        plt.savefig(r"F:\ProjectSet\Shadow\MkTu\4.1Details\as_de_opt2.jpg", dpi=300)
        plt.show()

    def drawDoubleBounds2(self):

        def same_sign(num1, num2):
            return (num1 * num2) > 0

        def plotDraw2D(subplot_number=111, theta=0.0,
                       is_as=False, is_de=False, is_opt=False, fig=None,
                       number_fig="a", is_draw=False):

            if fig is None:
                fig = plt.figure(figsize=(2, 3))
                plt.subplots_adjust(top=0.96, bottom=0.04, left=0.04, right=0.96, hspace=0.2, wspace=0.2)

            ax = fig.add_subplot(subplot_number, axes_class=AxesZero)
            ax.set_aspect('equal', adjustable='box')
            self.coorAxis(ax, y0=-0.7)

            x0, y0 = 0.3, 0.2
            building_coors = self.rectangleTrans((x0, y0), (-x0, y0), (-x0, -y0), (x0, -y0), (x0, y0), theta=theta)
            self.plotRectangle(*tuple(building_coors), color="black")

            line_as = self.lineKB(self.lineK(-0.2493, -0.05480094, -0.25319841, -0.03558636), -0.8, 0)
            line_de = self.lineKB(self.lineK(0.24939, -0.05480094, 0.25319841, -0.03558636), 0.8, 0)
            line_sun = SHSketchMapLine().init(0.3, 0, -0.9)

            def sh_as(fx=1):
                self.plotArrow(line_as, -0.95, -0.66, fx=-1, length_includes_head=True,
                               head_width=0.025, head_length=0.06, fc='blue', color="blue")
                line1, line2 = self.vLineRectangleIntersect(line_as, building_coors)
                if not is_draw:
                    distance1 = line_as.pointDistance(line1.x0, line1.y0) - 0.1
                    distance2 = line_as.pointDistance(line2.x0, line2.y0) - 0.1
                    coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance2, fx=fx, color="darkblue")
                    coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance1, fx=fx, color="darkblue")
                    self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[0]), alpha=0.5,
                              color="darkblue", label="Ascending shadow")
                if not is_draw:
                    distance1 = line_as.pointDistance(line1.x0, line1.y0) - 0.1
                    distance2 = line_as.pointDistance(line2.x0, line2.y0) - 0.1
                    coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance1, fx=-fx, color="royalblue")
                    coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance2, fx=-fx, color="royalblue")
                    self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[1]), alpha=0.5,
                              color="royalblue", label="Ascending double-bounce")
                plt.text(-1.12, 0.8, "Ascending orbit", fontdict={"size": FONT_SIZE})
                return line1, line2

            def sh_de(fx=-1):
                self.plotArrow(line_de, 0.68, 0.95, fx=-1, length_includes_head=True,
                               head_width=0.025, head_length=0.06, fc='red', color="red")
                line1, line2 = self.vLineRectangleIntersect(line_de, building_coors)
                if not is_draw:
                    distance1 = line_de.pointDistance(line1.x0, line1.y0) - 0.1
                    distance2 = line_de.pointDistance(line2.x0, line2.y0) - 0.1
                    coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance2, fx=fx, color="darkred")
                    coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance1, fx=fx, color="darkred")
                    self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[1]), alpha=0.5,
                              color="darkred", label="Descending shadow")
                if not is_draw:
                    distance1 = line_de.pointDistance(line1.x0, line1.y0) - 0.1
                    distance2 = line_de.pointDistance(line2.x0, line2.y0) - 0.1
                    coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance1, fx=-fx, color="red")
                    coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance2, fx=-fx, color="red")
                    self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[0]), alpha=0.5,
                              color="red", label="Descending double-bounce")
                plt.text(0.2, -0.7, "Descending orbit", fontdict={"size": FONT_SIZE})
                return line1, line2

            def sh_opt():
                # self.plotLine(line_sun, -1, 1, color="orange")
                line1, line2 = self.vLineRectangleIntersect(line_sun, building_coors)
                distance1 = line_sun.pointDistance(line1.x0, line1.y0) - 0.1
                distance2 = line_sun.pointDistance(line2.x0, line2.y0) - 0.1
                coors1 = self.plotLinePointDistance(line1, line1.x0, distance=distance2, fx=-1, color="dimgrey")
                coors2 = self.plotLinePointDistance(line2, line2.x0, distance=distance1, fx=-1, color="dimgrey")
                color_str = "dimgrey"
                alpha = 0.5
                if (not is_as) and (not is_de) and is_opt:
                    color_str = "black"
                    alpha = 1
                self.fill((coors1, coors2[2], coors2[3], coors2[0], coors2[1], self.vlri_coors[0]), alpha=alpha,
                          color=color_str, label="Optical shadow")
                return line1, line2

            line_as_1, line_as_2, line_de_1, line_de_2, line_opt_1, line_opt_2 = (None for _ in range(6))
            if is_as:
                line_as_1, line_as_2 = sh_as(1)
            if is_de:
                line_de_1, line_de_2 = sh_de(-1)
            if is_opt:
                line_opt_1, line_opt_2 = sh_opt()

            if is_as and is_de:
                coor1 = line_as_1.intersect(line_de_1)
                coor2 = line_as_1.intersect(line_de_2)
                coor3 = line_as_2.intersect(line_de_1)
                coor4 = line_as_2.intersect(line_de_2)
                coors = [coor for coor in [coor1, coor2, coor3, coor4] if -1 < coor[0] < 1 and -1 < coor[1] < 1]

                def plot_coor_line(coor_init):
                    _coor_line = [coor_init]
                    for coor in building_coors[:4]:
                        if same_sign(coor[1], _coor_line[0][1]):
                            _coor_line.append(coor)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.scatter(x, y, color="red", s=60)
                    _coor_line.append(coor_init)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.plot(x, y)
                    plt.fill(x, y, color="black", label="Shadow overlap area")

                plot_coor_line(coors[0])
                # plot_coor_line(coors[1])

            if is_opt and is_as and (not is_de):
                coor1 = line_opt_1.intersect(line_as_1)
                coor2 = line_opt_1.intersect(line_as_2)
                coor3 = line_opt_2.intersect(line_as_1)
                coor4 = line_opt_2.intersect(line_as_2)
                coors = [coor for coor in [coor1, coor2, coor3, coor4] if -1 < coor[0] < 1 and -1 < coor[1] < 1]

                def plot_coor_line(coor_init):
                    _coor_line = [coor_init]
                    for coor in building_coors[:4]:
                        if same_sign(coor[1], _coor_line[0][1]):
                            _coor_line.append(coor)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.scatter(x, y, color="red", s=60)
                    _coor_line.append(coor_init)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.plot(x, y)
                    plt.fill(x, y, color="black", label="Shadow overlap area")

                plot_coor_line(coors[0])
                # plot_coor_line(coors[1])

            if is_opt and is_de and (not is_as):
                coor1 = line_opt_1.intersect(line_de_1)
                coor2 = line_opt_1.intersect(line_de_2)
                coor3 = line_opt_2.intersect(line_de_1)
                coor4 = line_opt_2.intersect(line_de_2)
                coors = [coor for coor in [coor1, coor2, coor3, coor4] if -1 < coor[0] < 1 and -1 < coor[1] < 1]

                def plot_coor_line(coor_init, ):
                    _coor_line = [coor_init]
                    for coor in building_coors[:4]:
                        if same_sign(coor[1], _coor_line[0][1]):
                            _coor_line.append(coor)
                        elif coor[0] < 0:
                            _coor_line.append(coor)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.scatter(x, y, color="red", s=60)
                    _coor_line.append(coor_init)
                    x, y = self.plotXY(*tuple(_coor_line))
                    # plt.plot(x, y)
                    plt.fill(x, y, color="black", label="Shadow overlap area")

                # plot_coor_line(coors[0])
                plot_coor_line(coors[2])
                # plot_coor_line(coors[3])

            self.fill(tuple(building_coors), color="lightgrey")

            plt.xlim([-1.2, 1.2])
            plt.ylim([-1.2, 1.2])
            plt.xticks([])
            plt.yticks([])

            plt.text(0.05, 0.9, "North", fontdict={"size": FONT_SIZE})
            plt.text(0.9, 0.05, "East", fontdict={"size": FONT_SIZE})
            plt.text(-0.22, -0.05, "Building", fontdict={"size": FONT_SIZE})
            plt.text(-1.1, 0.96, "({0})".format(number_fig), fontdict={"size": 16})

        fig_this = plt.figure(figsize=(12, 4))
        plt.subplots_adjust(top=0.96, bottom=0.04, left=0.02, right=0.98, hspace=0.01, wspace=0.01)

        def plot_double_bounce(subplot_number, theta, fig, number_fig=None):

            ax = fig.add_subplot(subplot_number, axes_class=AxesZero)
            ax.set_aspect('equal', adjustable='box')

            x0, y0 = 0.2, 0.6
            dx, dy = -0.6, 0.1
            building_coors = self.rectangleTrans((x0, y0), (-x0, y0), (-x0, -y0), (x0, -y0), (x0, y0), theta=theta)

            x, y = self.plotXY(*tuple(building_coors))
            x = np.array(x) + dx
            y = np.array(y) + dy
            plt.plot(x, y, color="black")
            plt.plot([-x0 + dx - 0.1, 1.0], [-y0 + dy, -y0 + dy], color="black")
            plt.fill(x, y, color="lightgrey", label="Building")

            incidence_angle = 38
            x_double_bounce = 0.2
            color_double_bounce = "midnightblue"

            line1 = SHSketchMapLine()
            line1.initXJiaoCoor(x_double_bounce, -y0 + dy, incidence_angle)
            line1.plot(x_double_bounce, 1.0, color=color_double_bounce,
                       label="Electromagnetic wave")

            line2 = SHSketchMapLine()
            line2.initXJiaoCoor(x_double_bounce, -y0 + dy, incidence_angle, fx=-1)
            line2.plot(x0 + dx, x_double_bounce, color=color_double_bounce)

            line3 = SHSketchMapLine()
            line3.initXJiaoCoor(x0 + dx, line2.y(x0 + dx), incidence_angle, fx=1)
            line3.plot(x0 + dx, 0.8, color=color_double_bounce)

            self.plotLineThree(ax, line1, (x_double_bounce + 1.0) / 2.0-0.05,     0.08, 0.12, color=color_double_bounce, fx=1)
            self.plotLineThree(ax, line2, (x0 + dx + x_double_bounce) / 2.0-0.05, 0.08, 0.12, color=color_double_bounce, fx=1)
            self.plotLineThree(ax, line3, (x0 + dx + 0.8) / 2.0-0.05,             0.08, 0.12, color=color_double_bounce, fx=0)

            plt.xlim([-1.2, 1.2])
            plt.ylim([-1.2, 1.2])

            plt.xticks([])
            plt.yticks([])

            if number_fig is not None:
                plt.text(-1.1, 0.96, "({0})".format(number_fig), fontdict={"size": 16})

        def func1(theta):
            # plotDraw2D(131, theta, False, False, True, is_draw=False, fig=fig_this, number_fig="a")
            plot_double_bounce(131, theta, fig=fig_this, number_fig="a")
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plotDraw2D(132, theta, True, False, False, is_draw=False, fig=fig_this, number_fig="b")
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
            plotDraw2D(133, theta, False, True, False, is_draw=False, fig=fig_this, number_fig="c")
            plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)

        func1(0)
        # plt.legend(loc="lower center", prop={"size": FONT_SIZE}, frameon=False)
        plt.savefig(r"F:\ProjectSet\Shadow\MkTu\4.1Details\double_bounds2.jpg", dpi=300)
        plt.show()


def main():
    # line = SHDLine()
    # line.azimuthCoor(360 - 8.9, 1, 2)
    # line.plot(-10, 10)
    # line.show()
    # plt.show()

    # shd = ShadowDirection()
    # shd.main()
    # plt.colorbar()
    scm = ShadowSketchMap()
    scm.drawDoubleBounds()

    pass


if __name__ == "__main__":
    main()
