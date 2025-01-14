# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : DrawRaster.py
@Time    : 2024/8/31 14:57
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of DrawRaster
-----------------------------------------------------------------------------"""
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal_array, gdal


class _HZRasterData:

    def __init__(self, raster_fn):
        self.raster_fn = raster_fn
        self.ds = gdal.Open(self.raster_fn, gdal.GA_ReadOnly)
        if self.ds is None:
            raise Exception("Can not find file: {}".format(self.raster_fn))
        self.geo_transform = self.ds.GetGeoTransform()
        self.inv_geo_transform = gdal.InvGeoTransform(self.geo_transform)

    def readXY(self, x, y, win_rows, win_columns, band_list):
        row, column = self.coorGeo2Raster(x, y, is_int=True)
        return self.readRowColumn(row, column, win_rows, win_columns, band_list)

    def readRowColumn(self, row, column, win_rows, win_columns, band_list):
        row = row - int(win_rows / 2)
        column = column - int(win_columns / 2)

        return gdal_array.DatasetReadAsArray(
            self.ds, xoff=column, yoff=row,
            win_xsize=win_columns,
            win_ysize=win_rows,
            band_list=band_list
        )

    def coorGeo2Raster(self, x, y, is_int=False):
        """ Geographical coordinates to image coordinates
        \
        :param is_int:
        :param x: Geographic North Coordinates / Latitude
        :param y: Geographical East Coordinates / Longitude
        :return: Image coordinates
        """
        column = self.inv_geo_transform[0] + x * self.inv_geo_transform[1] + y * self.inv_geo_transform[2]
        row = self.inv_geo_transform[3] + x * self.inv_geo_transform[4] + y * self.inv_geo_transform[5]
        if is_int:
            return int(row), int(column)
        else:
            return row, column

    def coorRaster2Geo(self, row, column):
        """ image coordinates to Geographical coordinates
        \
        :param row: row
        :param column: column
        :return: Geographical coordinates
        """
        x = self.geo_transform[0] + column * self.geo_transform[1] + row * self.geo_transform[2]
        y = self.geo_transform[3] + column * self.geo_transform[4] + row * self.geo_transform[5]
        return x, y


def scaleMinMax(d, d_min=None, d_max=None, is_01=True):
    if (d_min is None) and (d_max is None):
        is_01 = True
    if d_min is None:
        d_min = np.min(d)
    if d_max is None:
        d_max = np.max(d)
    d = np.clip(d, d_min, d_max)
    if is_01:
        d = (d - d_min) / (d_max - d_min)
    return d


def draw(raster_fn, win_rows, win_columns, band_list, x=None, y=None, row=None, column=None, min_list=None,
         max_list=None, color_table=None, figsize=(12, 12), save_filename=None):
    rd = _HZRasterData(raster_fn)

    if (len(band_list) != 1) and (len(band_list) != 3):
        raise Exception("Number of bands is 1 or 3 not {} as {}".format(len(band_list), band_list))

    if (x is not None) and (y is not None):
        data = rd.readXY(x, y, win_rows, win_columns, band_list)
    else:
        if (row is not None) and (column is not None):
            data = rd.readRowColumn(row, column, win_rows, win_columns, band_list)
        else:
            data = None
    if data is None:
        raise Exception("Location information of (x, y) or (row, column) can not find.")

    to_data = np.zeros((3, data.shape[-2], data.shape[-1]))

    if color_table is not None:
        data = data.astype("int")
        if len(data.shape) == 3:
            data = data[0]
        for n in color_table:
            color_data = np.array([color_table[n]]).T
            to_data[:, data == n] = color_data / 255
    else:
        if len(data.shape) == 2:
            data = np.concatenate([[data], [data], [data]])
        if min_list is None:
            min_list = np.min(data, axis=(1, 2))
        else:
            if len(min_list) == 1:
                min_list = list(min_list) * 3
        if max_list is None:
            max_list = np.max(data, axis=(1, 2))
        else:
            if len(max_list) == 1:
                max_list = list(max_list) * 3
        for i in range(3):
            to_data[i] = scaleMinMax(data[i], min_list[i], max_list[i])

    for i in range(3):
        to_data[i] = scaleMinMax(to_data[i], 0, 1)

    plt.figure(figsize=figsize, )
    plt.imshow(np.transpose(to_data, axes=(1, 2, 0)))
    plt.xticks([])
    plt.yticks([])
    if save_filename is not None:
        plt.savefig(save_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.show()


def main():
    # 支持市面上的所有的影像类型，例如: tiff, ENVI, jpg, png, ...
    # python依赖库 gdal matplotlib
    # 实例数据:
    #   链接：https://pan.baidu.com/s/1BzK0SlKLzA170ZdhiCPqOw?pwd=jdks
    #   提取码：jdks
    #   data.tif: 1-4 波段是 SAR 波段 其他波段是哨兵的波段, imdc.tif: 是一个分类结果 1是非不透水面，2是不透水面

    # 示例1: 绘制三波段影像
    #   地理坐标
    draw(
        figsize=(12, 12),  # figsize 越大画出来的图越清楚
        raster_fn=r"data.tif",  # 影像的文件路径
        win_rows=161, win_columns=161,  # 绘图的大小，win_rows行数，win_columns是列数
        band_list=[7, 6, 5],  # 绘制的波段，可以是一个波段，可以是三个波段
        x=112.826016, y=22.619000, row=None, column=None,  # 中心坐标，可以使用地理坐标 x,y 也可以使用行列 row,column，必须设置一个
        min_list=[0.0509996, 0.0794492, 0.0985505],
        max_list=[0.1887, 0.173599, 0.1739],  # 数据拉伸范围，min_list是最小范围，max_list是最大范围，None就是使用截取部分的最大值和最小值
        color_table=None,  # 这个是绘制结果图的时候，设置每个值的颜色
        save_filename=None,  # 保存影像文件路径，例如：image.jpg，None就是不保存
    )
    #   行列坐标
    draw(
        raster_fn=r"data.tif",  # 影像的文件路径
        win_rows=161, win_columns=161,  # 绘图的大小，win_rows行数，win_columns是列数
        band_list=[7, 6, 5],  # 绘制的波段，可以是一个波段，可以是三个波段
        x=None, y=None, row=912, column=1413,  # 中心坐标，可以使用地理坐标 x,y 也可以使用行列 row,column，必须设置一个
        min_list=[0.0509996, 0.0794492, 0.0985505],
        max_list=[0.1887, 0.173599, 0.1739],  # 数据拉伸范围，min_list是最小范围，max_list是最大范围，None就是使用截取部分的最大值和最小值
        color_table=None,  # 这个是绘制结果图的时候，设置每个值的颜色
        save_filename=None,  # 保存影像文件路径，例如：image.jpg，None就是不保存
    )

    # 示例2: 绘制单波段影像
    draw(
        raster_fn=r"data.tif",  # 影像的文件路径
        win_rows=161, win_columns=161,  # 绘图的大小，win_rows行数，win_columns是列数
        band_list=[1],  # 绘制的波段，可以是一个波段，可以是三个波段
        x=112.826016, y=22.619000, row=None, column=None,  # 中心坐标，可以使用地理坐标 x,y 也可以使用行列 row,column，必须设置一个
        min_list=[-21.60658],
        max_list=[-3.12265],  # 数据拉伸范围，min_list是最小范围，max_list是最大范围，None就是使用截取部分的最大值和最小值
        color_table=None,  # 这个是绘制结果图的时候，设置每个值的颜色
        save_filename=None,  # 保存影像文件路径，例如：image.jpg，None就是不保存
    )

    # 示例3: 绘制分类影像
    draw(
        raster_fn=r"imdc.tif",  # 影像的文件路径
        win_rows=161, win_columns=161,  # 绘图的大小，win_rows行数，win_columns是列数
        band_list=[1],  # 绘制的波段，可以是一个波段，可以是三个波段
        x=112.826016, y=22.619000, row=None, column=None,  # 中心坐标，可以使用地理坐标 x,y 也可以使用行列 row,column，必须设置一个
        min_list=None, max_list=None,  # 数据拉伸范围，min_list是最小范围，max_list是最大范围，None就是使用截取部分的最大值和最小值
        color_table={1: (0, 255, 0), 2: (255, 0, 0)},  # 这个是绘制结果图的时候，设置每个值的颜色
        save_filename="image.jpg",  # 保存影像文件路径，例如：image.jpg，None就是不保存
    )

    return


if __name__ == "__main__":
    main()
