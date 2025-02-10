# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : GeoMap.py
@Time    : 2024/7/2 15:24
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of GeoMap
-----------------------------------------------------------------------------"""
import matplotlib.axes
import matplotlib.transforms
import matplotlib.pyplot as plt
import numpy as np
from osgeo.gdal_array import BandReadAsArray
from osgeo import ogr
from osgeo import osr
from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.Utils import datasCaiFen

plt.rc('font', family='Times New Roman')


def getRange(gr, region=None, is_geo=False):
    if not is_geo:
        row0, row1, colum0, column1 = 0, gr.n_rows, 0, gr.n_columns
        if region is not None:
            row0, row1, colum0, column1 = tuple(region)
        x0, y0 = gr.coorRaster2Geo(row0, colum0)
        x1, y1 = gr.coorRaster2Geo(row1, column1)
        if x0 > x1:
            x1, x0 = x0, x1
        if y0 > y1:
            y1, y0 = y0, y1
        return [x0, x1, y0, y1]
    else:
        x0, x1, y0, y1 = tuple(region)
        x0, y0 = gr.coorGeo2Raster(x0, y0, is_int=True)
        x1, y1 = gr.coorRaster2Geo(x1, y1, is_int=True)
        if x0 > x1:
            x1, x0 = x0, x1
        if y0 > y1:
            y1, y0 = y0, y1
        return [x0, x1, y0, y1]


def readChannel(gr: GDALRaster, channel, raster_region, ):
    if isinstance(channel, str):
        channel = gr.names.index(channel)
    xoff, yoff = raster_region[2], raster_region[0]
    win_xsize = raster_region[3] - raster_region[2]
    win_ysize = raster_region[1] - raster_region[0]
    return BandReadAsArray(gr.raster_ds.GetRasterBand(channel), xoff, yoff, win_xsize, win_ysize)


class GMCoorTrans:

    def __init__(self, gr: GDALRaster, geo_region, raster_region):
        self.gr = gr
        self.geo_region = geo_region
        self.raster_region = raster_region

    def y(self, *data):
        row0, row1, colum0, column1 = self.raster_region
        data = datasCaiFen(data)
        to_list = []
        for d in data:
            x, y = self.gr.coorRaster2Geo(row0, colum0 + d)
            to_list.append(x)
        return to_list

    def x(self, *data):
        row0, row1, colum0, column1 = self.raster_region
        data = datasCaiFen(data)
        to_list = []
        for d in data:
            x, y = self.gr.coorRaster2Geo(row0 + d, colum0)
            to_list.append(y)
        return to_list

    def row(self, *data):
        x0, x1, y0, y1 = self.geo_region
        row0, row1, colum0, column1 = self.raster_region
        data = datasCaiFen(data)
        to_list = []
        for d in data:
            row, column = self.gr.coorGeo2Raster(x0, d)
            to_list.append(row - row0)
        return to_list

    def column(self, *data):
        x0, x1, y0, y1 = self.geo_region
        row0, row1, colum0, column1 = self.raster_region
        data = datasCaiFen(data)
        to_list = []
        for d in data:
            row, column = self.gr.coorGeo2Raster(d, y0)
            to_list.append(column - colum0)
        return to_list


class DMS:
    def __init__(self, degrees=0, minutes=0, seconds=0):
        self.degrees = degrees
        self.minutes = minutes
        self.seconds = seconds

    def toDecimalDegrees(self):
        decimal_degrees = self.degrees + self.minutes / 60 + self.seconds / 3600
        return decimal_degrees

    @classmethod
    def fromDecimalDegrees(cls, decimal_degrees):
        degrees = int(decimal_degrees)
        minutes = int((decimal_degrees - degrees) * 60)
        seconds = ((decimal_degrees - degrees) * 60 - minutes) * 60
        return cls(degrees, minutes, seconds)

    def __add__(self, other):
        total_seconds_self = self.degrees * 3600 + self.minutes * 60 + self.seconds
        total_seconds_other = other.degrees * 3600 + other.minutes * 60 + other.seconds
        total_seconds_result = total_seconds_self + total_seconds_other

        degrees = total_seconds_result // 3600
        remaining_seconds = total_seconds_result % 3600
        minutes = remaining_seconds // 60
        seconds = remaining_seconds % 60

        return DMS(degrees, minutes, seconds)

    def __sub__(self, other):
        total_seconds_self = self.degrees * 3600 + self.minutes * 60 + self.seconds
        total_seconds_other = other.degrees * 3600 + other.minutes * 60 + other.seconds
        total_seconds_result = total_seconds_self - total_seconds_other

        degrees = abs(total_seconds_result) // 3600
        remaining_seconds = abs(total_seconds_result) % 3600
        minutes = remaining_seconds // 60
        seconds = remaining_seconds % 60

        return DMS(degrees, minutes, seconds)

    def __str__(self):
        return f"{self.degrees}°{self.minutes}′{self.seconds:.2f}″"

    def __lt__(self, other):
        return (self.degrees, self.minutes, self.seconds) < (other.degrees, other.minutes, other.seconds)

    def __le__(self, other):
        return (self.degrees, self.minutes, self.seconds) <= (other.degrees, other.minutes, other.seconds)

    def __gt__(self, other):
        return (self.degrees, self.minutes, self.seconds) > (other.degrees, other.minutes, other.seconds)

    def __ge__(self, other):
        return (self.degrees, self.minutes, self.seconds) >= (other.degrees, other.minutes, other.seconds)

    def __eq__(self, other):
        return (self.degrees, self.minutes, self.seconds) >= (other.degrees, other.minutes, other.seconds)


def coors(gm_coor, x_major_len, y_major_len, x_minor_len, y_minor_len, fontdict=None, ax=None):
    def getlen(lim, major_len, minor_len):
        lim0, lim1 = DMS.fromDecimalDegrees(lim[0]), DMS.fromDecimalDegrees(lim[1])
        if lim0 > lim1:
            lim0, lim1 = lim1, lim0

        datas = []

        def get_list(_len, ):
            _list = []
            _len = DMS(*_len)
            dms = DMS()
            while dms <= lim1:
                dms = dms + _len
                if lim0 < dms < lim1:
                    _list.append(dms.toDecimalDegrees())
                    datas.append(str(dms))

            return _list

        return get_list(major_len), get_list(minor_len)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    def x_coors():
        if xlim is not None:
            ax.set_xlim(xlim)
        column_lim = ax.get_xlim()

        xticks1, xticks2 = getlen(tuple(gm_coor.y(xlim[0] + 0.5, xlim[1] - 0.5)), x_major_len, x_minor_len)
        xticks1, xticks2 = gm_coor.column(xticks1), gm_coor.column(xticks2)
        ax.set_xticks(xticks1)
        ax.set_xticks(xticks2, minor=True)
        xticks_show = gm_coor.y(xticks1)
        xticks_show = ["{}E".format(toDFM(data)) for data in xticks_show]
        ax.set_xticklabels(xticks_show, fontdict=fontdict)
        ax.tick_params("x", length=10, width=1)
        ax.tick_params("x", which="minor", length=5)

        for k in ax.xaxis.get_majorticklabels():
            k.set_ha("center")
        return column_lim

    def y_coors():
        if ylim is not None:
            ax.set_ylim(ylim)
        row_lim = ax.get_ylim()
        yticks1, yticks2 = getlen(tuple(gm_coor.x(ylim[1] - 0.5, ylim[0] + 0.5)), y_major_len, y_minor_len)
        yticks1, yticks2 = gm_coor.row(yticks1), gm_coor.row(yticks2)
        ax.set_yticks(yticks1)
        ax.set_yticks(yticks2, minor=True)
        yticks_show = gm_coor.x(yticks1)
        yticks_show = ["{}N".format(toDFM(data)) for data in yticks_show]
        ax.set_yticklabels(yticks_show, fontdict=fontdict, )
        ax.tick_params("y", labelrotation=90, length=10, width=1)
        ax.tick_params("y", which="minor", length=5)

        for k in ax.yaxis.get_majorticklabels():
            k.set_va("center")
        return row_lim

    xlim, ylim = x_coors(), y_coors()

    ax = ax.twinx()
    ylim = y_coors()

    ax = ax.twiny()
    xlim = x_coors()


def toDFM(data, is_miao=False):
    degrees = float(data)
    deg = int(degrees)
    minutes = (degrees - deg) * 60
    min_tmp = int(minutes)
    sec = (minutes - min_tmp) * 60
    if is_miao:
        return "{}°{:2d}′{:2d}″".format(int(deg), int(round(minutes)), int(round(sec)))
    return "{}°{:2d}′".format(int(deg), int(minutes))


class GMRaster:

    def __init__(self, raster_fn, ax=None):
        self.raster_fn = raster_fn
        self.gr = GDALRaster(raster_fn)
        self.data = None
        self.geo_region = None
        self.raster_region = None
        self._initRegion()
        self.ax = ax

    def read(self, channels, geo_region=None, raster_region=None,
             min_list=None, max_list=None, color_table=None, _func=None):
        self._initRegion(geo_region=geo_region, raster_region=raster_region)
        if isinstance(channels, int):
            channels = [channels]
        if isinstance(channels, str):
            channels = [channels]

        min_list_tmp, max_list_tmp = [], []
        data = []
        for i, channel in enumerate(channels):

            data_tmp = readChannel(self.gr, channel, self.raster_region)

            if min_list is None:
                min_list_tmp.append(np.min(data_tmp))
            else:
                if isinstance(min_list, list):
                    min_list_tmp.append(min_list[i])
                else:
                    min_list_tmp.append(min_list)

            if max_list is None:
                max_list_tmp.append(np.max(data_tmp))
            else:
                if isinstance(max_list, list):
                    max_list_tmp.append(max_list[i])
                else:
                    max_list_tmp.append(max_list)

            if _func is not None:
                data_tmp = _func(data_tmp)

            if color_table is None:
                data_tmp = np.clip(data_tmp, min_list_tmp[i], max_list_tmp[i], )
                data_tmp = (data_tmp - min_list_tmp[i]) / (max_list_tmp[i] - min_list_tmp[i])

            data.append([data_tmp])

        if len(data) == 1:
            if color_table is not None:
                data = data[0][0]
                to_data = np.zeros((data.shape[0], data.shape[1], 3,), )
                for n, color in color_table.items():
                    to_data[data == n, :] = np.array(color)
                to_data = to_data / 255.0
                self.data = to_data
            else:
                data = data * 3
                self.data = np.concatenate(data)
                self.data = self.data.transpose((1, 2, 0))
        else:
            self.data = np.concatenate(data)
            self.data = self.data.transpose((1, 2, 0))

    def _initRegion(self, geo_region=None, raster_region=None, ):
        if geo_region is None and raster_region is None:
            self.raster_region = [0, self.gr.n_rows, 0, self.gr.n_columns]
            self.geo_region = getRange(self.gr, self.raster_region)
            return
        if geo_region is not None:
            self.geo_region = geo_region
            self.raster_region = getRange(self.gr, geo_region, False)
            return
        if raster_region is not None:
            self.raster_region = raster_region
            self.geo_region = getRange(self.gr, raster_region, True)
            return
        self.raster_region = raster_region
        self.geo_region = geo_region

        return

    def draw(self, n_ex=2,
             x_major_len=(0, 6, 0), y_major_len=(0, 6, 0),
             x_minor_len=(0, 0, 20), y_minor_len=(0, 0, 20),
             fontdict=None):

        if fontdict is None:
            fontdict = {}

        ax=None
        if n_ex is not None:
            fig = plt.figure(figsize=(self.data.shape[1] / self.data.shape[0] * n_ex, n_ex), )
            fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.04, wspace=0.03)
            self.ax = plt.gca()
            ax = plt.gca()

        self.ax.imshow(self.data)

        gm_coor = GMCoorTrans(self.gr, self.geo_region, self.raster_region)
        coors(gm_coor, x_major_len, y_major_len, x_minor_len, y_minor_len, fontdict=fontdict, ax=self.ax)

        return ax

    def scale(self, loc=(0.5, 0.1), length=0.3, offset=0.05, offset2=0.05, offset3=0.3,
              high=0.05, high2=0.016, fontdict=None):
        if fontdict is None:
            fontdict = {'family': 'Times New Roman', "size": 14}
        self.ax: matplotlib.axes.Axes
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        loc1 = (xlim[0] + loc[0] * (xlim[1] - xlim[0]), ylim[0] + loc[1] * (ylim[1] - ylim[0]),)
        length = length * (xlim[1] - xlim[0])
        loc2 = (loc1[0] + length, loc1[1])
        offset1 = offset * length
        loc11 = loc1[0] + offset1, loc1[1]
        loc22 = loc2[0] - offset1 - offset2 * (xlim[1] - xlim[0]), loc2[1]
        self.ax.plot((loc11[0], loc22[0]), (loc11[1], loc22[1]), color='black', )
        high = high * (ylim[1] - ylim[0])
        self.ax.fill(
            [loc1[0], loc1[0], loc2[0], loc2[0], loc1[0]],
            [loc1[1] - high * 0.1, loc1[1] + high, loc2[1] + high, loc2[1] - high * 0.1, loc1[1] - high * 0.1],
            color='white',
        )
        self.ax.text(loc22[0] + 0.01 * (xlim[1] - xlim[0]), loc22[1], "kilometers", fontdict=fontdict)
        x_list = [loc11[0],
                  loc11[0] + (loc22[0] - loc11[0]) / 4.0,
                  loc11[0] + (loc22[0] - loc11[0]) / 2.0,
                  loc22[0]]
        high2 = high2 * (ylim[1] - ylim[0])
        for x in x_list:
            self.ax.plot((x, x), (loc11[1], loc11[1] + high2), color="black")
        if self.gr.dst_srs.IsGeographic():
            resolution = abs(111319.489700 * self.gr.geo_transform[1])
        else:
            resolution = abs(self.gr.geo_transform[1])
        numbers = [0, (loc22[0] - loc11[0]) / 4.0 * resolution, (loc22[0] - loc11[0]) / 2.0 * resolution,
                   loc22[0] * resolution]
        numbers = np.array(numbers) / 1000
        for i, number in enumerate(numbers):
            self.ax.text(x_list[i], loc11[1] + high2 + high * offset3, "{:.1f}".format(float(number)),
                         fontdict=fontdict, horizontalalignment="center")

    def compass(self, loc=(0.5, 0.5), size=1.0, fontdict=None):
        if fontdict is None:
            fontdict = {'family': 'Times New Roman', "size": 14 * size}
        t1 = 0.1
        plt.axes((loc[0], loc[1], (t1 * 2 * 0.1) * size, 0.1 * size))
        plt.fill(
            [0.5, 0.5 - t1, 0.5, 0.5 + t1, 0.5],
            [0.6, 0.05, 0.15, 0.05, 0.6],
            color='black',
        )
        plt.text(0.5, 0.69, "N", fontdict=fontdict, verticalalignment="center", horizontalalignment="center")
        plt.xlim([0.5 - t1, 0.5 + t1])
        plt.ylim([0, 1])

        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)

        plt.xticks([])
        plt.yticks([])


def main():
    gmr = GMRaster(r"F:\G\SHImages\QD_AS_VV.tif")
    gmr.read(1)
    gmr.draw(8, fontdict={"size": 16})
    gmr.scale(
        loc=(0.40, 0.02), length=0.5269,
        offset=0.06, offset2=0.13, offset3=0.3, high=0.08, high2=0.016,
        fontdict={'family': 'Times New Roman', "size": 21}
    )
    gmr.compass(loc=(0.94, 0.76), size=1.8)
    print(plt.gcf().get_axes())
    plt.show()


if __name__ == "__main__":
    main()
