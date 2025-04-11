# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : TempFuncs.py
@Time    : 2023/12/29 20:56
@Author  : Zheng Han
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of TempFuncs
-----------------------------------------------------------------------------"""
import os

import matplotlib.colors as mcolors
import numpy as np
from osgeo import gdal

from Draw.m_color_data import CSS4_COLORS
from SRTCodes.GDALRasterIO import GDALRaster


def main():

    def func1():
        dirname = r"F:\ProjectSet\Huo\sun\drive-download-20240508T081752Z-001"
        for fn in os.listdir(dirname):
            if os.path.splitext(fn)[1] == ".tif":
                print(fn)
                fn_open = os.path.join(dirname, fn)
                ds = gdal.Open(fn_open)
                data = ds.ReadAsArray()
                data = (np.clip(data, 200, 3000) - 200) / (3000 - 200) * 255
                data = data.astype("int8")
                driver = gdal.GetDriverByName("GTiff")
                tif_fn = os.path.join(dirname, "1", fn)
                if os.path.isfile(tif_fn):
                    os.remove(tif_fn)
                dst_ds:gdal.Dataset = driver.Create(tif_fn, data.shape[2], data.shape[1], 3, gdal.GDT_Byte)
                dst_ds.GetRasterBand(1).WriteArray(data[3])
                dst_ds.GetRasterBand(2).WriteArray(data[2])
                dst_ds.GetRasterBand(3).WriteArray(data[1])
                driver_jpg = gdal.GetDriverByName("JPEG")
                to_fn = tif_fn.replace(".tif", ".jpg")
                if os.path.isfile(to_fn):
                    os.remove(to_fn)
                to_ds = driver_jpg.CreateCopy(to_fn, dst_ds)
                del to_ds
                del dst_ds
                if os.path.isfile(tif_fn):
                    os.remove(tif_fn)

    def func2():
        return

    return func2()


def method_name3():
    gr = GDALRaster(r"G:\GraduationProject\Images\2021\gba_grid_183_67.tif")
    # d = np.zeros((gr.n_rows, gr.n_columns))
    # gr.save(d, r"F:\Week\20240121\Data\gba_grid_183_67_01.tif", fmt="GTiff")
    # gr = GDALRaster(raster_fn)
    # d = gr.readAsArray()
    # if len(d.shape) == 3:
    #     d = d[n_channel]
    # colored_image = np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
    # for k in map_dict:
    #     colored_image[d == k] = map_dict[k]
    # image = Image.fromarray(colored_image)
    # image.save(to_fn)


def method_name2():
    def sort_color(colors):
        names = sorted(colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
        return {name: colors[name] for name in names}

    colors = sort_color(CSS4_COLORS)
    with open(r"F:\PyCodes\Draw\CSS4_COLORS.css", "w", encoding="utf-8") as f:
        f.write("mcolor {\n")
        for k, v in colors.items():
            print("    color: {0};".format(k, v), file=f)
        f.write("}\n")


def method_name1():
    dirname = r"F:\PyCodes"
    to_dirname = r"F:\PyCodes"
    find_str = "tourensong@gmail.com"
    to_str = "tourensong@gmail.com"
    for root, dirs, files in os.walk(dirname):
        for file in files:
            fn = os.path.join(root, file)
            to_fn = fn.replace(dirname, to_dirname)
            if os.path.splitext(fn)[1] == ".py":
                fr = open(fn, "r", encoding="utf-8")
                lines = []
                for line in fr:
                    if find_str in line:
                        print(fn, "->", to_fn)
                        print("   ", line)
                        line = line.replace(find_str, to_str)
                        print("   ", line)
                    lines.append(line)
                fr.close()
                fw = open(to_fn, "w", encoding="utf-8")
                fw.writelines(lines)
                fw.close()


if __name__ == "__main__":
    main()
