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

from SRTCodes.Utils import angleToRadian, radianToAngle


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

    def _distanceCal(self):
        self.distance_c = self.y0 - self.k * self.x0
        self.distance_1 = math.sqrt(self.k ** 2 + 1)

    def _calABC(self):
        self.a = self.k
        self.b = -1
        self.c = self.y0 - self.k * self.x0

    def _calNormalAzimuth(self):
        self.normal_azimuth = radianToAngle(math.atan(-self.k))

    def azimuthCoor(self, alpha, x0=0.0, y0=0.0):
        self.k = 1 / (math.tan(angleToRadian(alpha)) + 0.000001)
        self.x0 = x0
        self.y0 = y0
        self._distanceCal()
        self._calABC()
        self._calNormalAzimuth()
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
        return self

    def plot(self, x0=None, x1=None, color="red"):
        if x0 is None:
            x0 = self.x_min
        if x0 is None:
            x0 = 0.0
        if x1 is None:
            x1 = self.x_max
        if x1 is None:
            x1 = 1.0
        # plt.scatter(self.x0, self.y0)
        plt.plot([x0, x1], [self.y(x0), self.y(x1)], color=color)
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

    @classmethod
    def intersect(cls, line1, line2):
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
            coor = SHDLine.intersect(line1, line2)
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
            coor = SHDLine.intersect(line1, line2)
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
        coor = SHDLine.intersect(line, line1)
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


def main():
    # line = SHDLine()
    # line.azimuthCoor(360 - 8.9, 1, 2)
    # line.plot(-10, 10)
    # line.show()
    # plt.show()

    # shd = ShadowDirection()
    # shd.main()
    plt.colorbar()

    pass


if __name__ == "__main__":
    main()
