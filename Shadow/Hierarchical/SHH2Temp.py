# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2Temp.py
@Time    : 2024/6/26 15:15
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2Temp
-----------------------------------------------------------------------------"""
import os.path
import time
from datetime import datetime
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageChops
from osgeo import gdal

from SRTCodes.GDALDraw import GDALDrawImages
from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.GDALUtils import RasterRandomCoors, GDALSamplingFast
from SRTCodes.SRTDraw import SRTDrawImages
from SRTCodes.Utils import changext, DirFileName, FRW, saveJson, readJson, SRTWriteText, writeTexts
from Shadow.Hierarchical import SHH2Config
from Shadow.Hierarchical.SHH2ML2 import mapDict


# plt.rcParams['font.sans-serif'] = ['SimHei']

# config = {
#     "font.family": 'serif',
#     "font.size": 18,
#     "mathtext.fontset": 'stix',
#     "font.serif": ['SimSun'],
# }
# plt.rcParams.update(config)
#

def main():
    dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods")
    raster_fn = dfn.fn("20240703H125230", "VHL3_ML_imdc.tif")

    def func1():
        print("raster_fn", raster_fn)
        gr = GDALRaster(raster_fn)
        data = gr.readAsArray()
        data_unique, data_counts = np.unique(data, return_counts=True)
        data_counts = data_counts / data.size
        print("ChengDu", data_unique, data_counts * 6000)

    def func2():
        df = RasterRandomCoors(raster_fn).random(4000)
        df = GDALSamplingFast(raster_fn).samplingDF(df)
        df = df.rename(columns={"FEATURE_1": "VHL3_ML"})
        print(df)
        print(df["VHL3_ML"].value_counts())
        df.to_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\cd\sh2_spl30_cd2_random4000.csv", index=False)

    r"""
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2Temp import main; main()"
    """
    func2()
    return


def method_name5():
    # VHL scatter
    config = {"font.size": 12, }
    plt.rcParams.update(config)
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(top=0.85, bottom=0.15, left=0.15, right=0.85)
    df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\qd\VHL3_ML_accuracy_data2_spl.csv")
    df["VHL_C"] = mapDict(df["CNAME"].tolist(), {
        "IS": 1, "VEG": 2, "SOIL": 1, "WAT": 3,
        "IS_SH": 3, "VEG_SH": 3, "SOIL_SH": 3, "WAT_SH": 3
    })

    def scatter(n, color, label):
        print(n, len(df[df["VHL_C"] == n]))
        plt.scatter(df[df["VHL_C"] == n][x_key], df[df["VHL_C"] == n][y_key], color=color, alpha=0.5, label=label)

    x_key, y_key = "Red", "NIR"
    scatter(1, "red", "IS SOIL")
    scatter(2, "green", "VEG")
    scatter(3, "black", "WATER SHADOW")
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.legend()
    plt.show()


def method_name4():
    def run(is_run=False):
        if not is_run:
            run_dict = [
                {"run": False, "type": "training", "city_name": "qd", "model": {"spectral": "CNN", "texture": False}},
                {"run": False, "type": "imdc", "city_name": "qd", "imdc": {"models": 1}},
                {"run": False, "type": "imdc", "city_name": "qd", "imdc": {"models": 5}},
                {"run": False, "type": "imdc", "city_name": "qd", "imdc": {"models": 10}},
                {"run": False, "type": "imdc", "city_name": "qd", "imdc": {"models": 30}},
                {"run": False, "type": "imdc", "city_name": "qd", "imdc": {"models": 60}},
                {"run": False, "type": "imdc", "city_name": "qd", "imdc": {"models": 90}},
                {"run": False, "type": "training", "city_name": "qd",
                 "model": {"spectral": "Transformer", "texture": False}},
                {"run": False, "type": "training", "city_name": "qd", "model": {"spectral": "CNN", "texture": True}},
                {"run": False, "type": "training", "city_name": "qd",
                 "model": {"spectral": "Transformer", "texture": True}},
            ]
            run_dict = []
            for city_name in ["qd", "bj", "cd"]:
                for train_type in [
                    {"spectral": "CNN", "texture": False},
                    {"spectral": "Transformer", "texture": False},
                    {"spectral": "CNN", "texture": True},
                    {"spectral": "Transformer", "texture": True},
                ]:
                    run_dict.append({"run": False, "type": "training", "city_name": city_name, "model": train_type})
                    for imdc_mod in [2, 10, 90]:
                        run_dict.append(
                            {"run": False, "type": "imdc", "city_name": city_name, "imdc": {"models": imdc_mod}})

            print(run_dict)
            for i in run_dict:
                writeTexts(r"F:\ProjectSet\Shadow\Hierarchical\Run\run_list.txt",
                           "python -c \"import sys; sys.path.append(r'F:\\PyCodes'); "
                           "from Shadow.Hierarchical.SHH2Temp import main; main()\" %* \n",
                           mode="a")
            writeTexts(r"F:\ProjectSet\Shadow\Hierarchical\Run\run_list.txt", "\n", mode="a")
            to_fn = r"F:\ProjectSet\Shadow\Hierarchical\Run\VHL_SHH2MOD_SpectralTextureDouble2.json"
            saveJson(run_dict, to_fn)

        else:

            model_dirname = r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods"

            json_fn = r"F:\ProjectSet\Shadow\Hierarchical\Run\VHL_SHH2MOD_SpectralTextureDouble2.json"

            dfn = DirFileName(os.path.split(json_fn)[0])

            sw = SRTWriteText(r"F:\ProjectSet\Shadow\Hierarchical\Run\save.txt", mode="a")

            to_json_fn = changext(json_fn, "_run.json")

            if not os.path.isfile(to_json_fn):
                copyfile(json_fn, to_json_fn)

            json_dict = readJson(to_json_fn)

            n_run = -1
            run_dict = {}
            for i in range(len(json_dict)):
                if not json_dict[i]["run"]:
                    n_run = i
                    run_dict = json_dict[i]
                    break

            if n_run == -1:
                print("#", "-" * 10, "End Run", n_run, "-" * 10, "#")
                return

            print("#", "-" * 10, "Run", n_run, "->", len(json_dict), "-" * 10, "#")

            if run_dict["type"] == "training":
                current_time = datetime.now()
                time.sleep(1)
                save_dirname = current_time.strftime("%Y%m%dH%H%M%S")
                time.sleep(1)
                current_time = datetime.now()
                city_name = None
                if run_dict["city_name"] == "qd":
                    city_name = "QingDao"
                elif run_dict["city_name"] == "bj":
                    city_name = "BeiJing"
                elif run_dict["city_name"] == "cd":
                    city_name = "ChengDu"

                to_dirname = os.path.join(model_dirname, save_dirname)

                sw.write("{}\n{} DL VHL4 {} {}\n".format(
                    current_time.strftime("%Y年%m月%d日%H:%M:%S"), city_name,
                    run_dict["model"], os.path.join(model_dirname, save_dirname), to_dirname
                ))
                writeTexts(dfn.fn("to_dirname.txt"), to_dirname)
                print("training")

            elif run_dict["type"] == "imdc":
                with open(dfn.fn("to_dirname.txt"), "r", encoding="utf-8") as f:
                    to_dirname = f.read()
                print("Imdc: ", os.path.join(to_dirname, "model{}.pth".format(run_dict["imdc"]["models"])))

            json_dict[n_run]["run"] = True
            saveJson(json_dict, to_json_fn)

    run(False)


def method_name3():
    def show():
        gdi = GDALDrawImages((200, 200))
        qd_name = gdi.addGeoRange(SHH2Config.QD_ENVI_FN, SHH2Config.QD_RANGE_FN)
        bj_name = gdi.addGeoRange(SHH2Config.BJ_ENVI_FN, SHH2Config.BJ_RANGE_FN)
        cd_name = gdi.addGeoRange(SHH2Config.CD_ENVI_FN, SHH2Config.CD_RANGE_FN)

        gdi.addCategoryColor("color", {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255)})
        gdi.addRasterCenterCollection("RGB", bj_name, cd_name, qd_name, channel_list=["Red", "Green", "Blue"])
        gdi.addRasterCenterCollection("NRG", bj_name, cd_name, qd_name, channel_list=["NIR", "Red", "Green"])

        gdi.addAxisDataXY(0, 0, "NRG", 120.082611, 36.363674)
        gdi.addAxisDataXY(0, 1, "NRG", 120.152759, 36.299145)
        gdi.addAxisDataXY(0, 2, "NRG", 120.11769, 36.19676)
        gdi.addAxisDataXY(1, 0, "NRG", 120.373995, 36.090731)
        gdi.addAxisDataXY(1, 1, "NRG", 120.118186, 36.254419)
        gdi.addAxisDataXY(1, 2, "NRG", 120.46047, 36.13073)
        gdi.addAxisDataXY(2, 0, "NRG", 120.14962, 36.18202)
        gdi.addAxisDataXY(2, 1, "NRG", 120.373467, 36.063960)
        gdi.addAxisDataXY(2, 2, "NRG", 120.027511, 36.266115)

        gdi.draw(n_rows_ex=3, n_columns_ex=3)
        plt.show()

    def toimage():
        coors = [
            [120.082611, 36.363674],
            [120.152759, 36.299145],
            [120.11769, 36.19676],
            [120.373995, 36.090731],
            [120.118186, 36.254419],
            [120.46047, 36.13073],
            [120.14962, 36.18202],
            [120.373467, 36.063960],
            [120.027511, 36.266115],
        ]
        dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\TestRegions")
        gr = GDALRaster(SHH2Config.QD_ENVI_FN)
        to_dict = {}
        for i in range(len(coors)):
            x, y = tuple(coors[i])
            row, column = gr.coorGeo2Raster(x, y, is_int=True)
            data = gr.readAsArrayCenter(x, y, 301, 301, is_geo=True)

            x0, y0 = gr.coorRaster2Geo(row - 150, column - 150)
            x1, y1 = gr.coorRaster2Geo(row + 150, column + 150)

            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0
            geo_transform = (x0, gr.geo_transform[1], 0, y1, 0, gr.geo_transform[5])

            to_fn = dfn.fn("TR{}.tif".format(i))
            to_dict[to_fn] = {"X": x, "Y": y}
            gr.save(
                d=data, save_geo_raster_fn=to_fn, fmt="GTiff", dtype=gdal.GDT_Float32,
                start_xy=None, descriptions=gr.names, geo_transform=geo_transform
            )
            print(to_fn)
        FRW(dfn.fn("test_region.json")).saveJson(to_dict)

    toimage()


def method_name2():
    sdi = SRTDrawImages()
    # sdi.addImage(0, 0, r"F:\GraduationDesign\MkTu\C2Images\ASC11QingDaoLayout_rekong.jpg", )
    # sdi.addImage(0, 1, r"F:\GraduationDesign\MkTu\C2Images\ASC22QingDaoLayout_rekong.jpg", )
    # sdi.addImage(0, 2, r"F:\GraduationDesign\MkTu\C2Images\DEC11QingDaoLayout_rekong.jpg", )
    # sdi.addImage(0, 3, r"F:\GraduationDesign\MkTu\C2Images\DEC22QingDaoLayout_rekong.jpg", )
    # sdi.addImage(1, 0, r"F:\GraduationDesign\MkTu\C2Images\ASC11BeiJingLayout_rekong.jpg", )
    # sdi.addImage(1, 1, r"F:\GraduationDesign\MkTu\C2Images\ASC22BeiJingLayout_rekong.jpg", )
    # sdi.addImage(1, 2, r"F:\GraduationDesign\MkTu\C2Images\DEC11BeiJingLayout_rekong.jpg", )
    # sdi.addImage(1, 3, r"F:\GraduationDesign\MkTu\C2Images\DEC22BeiJingLayout_rekong.jpg", )
    # sdi.addImage(2, 0, r"F:\GraduationDesign\MkTu\C2Images\ASC11ChengDuLayout_rekong.jpg", )
    # sdi.addImage(2, 1, r"F:\GraduationDesign\MkTu\C2Images\ASC22ChengDuLayout_rekong.jpg", )
    # sdi.addImage(2, 2, r"F:\GraduationDesign\MkTu\C2Images\DEC11ChengDuLayout_rekong.jpg", )
    # sdi.addImage(2, 3, r"F:\GraduationDesign\MkTu\C2Images\DEC22ChengDuLayout_rekong.jpg", )

    sdi.addImage(0, 0, r"F:\GraduationDesign\MkTu\HA\ASHQingDaoLayout_rekong.jpg", )
    sdi.addImage(0, 1, r"F:\GraduationDesign\MkTu\HA\ASAQingDaoLayout_rekong.jpg", )
    sdi.addImage(0, 2, r"F:\GraduationDesign\MkTu\HA\DEHQingDaoLayout_rekong.jpg", )
    sdi.addImage(0, 3, r"F:\GraduationDesign\MkTu\HA\DEAQingDaoLayout_rekong.jpg", )
    sdi.addImage(1, 0, r"F:\GraduationDesign\MkTu\HA\ASHBeiJingLayout_rekong.jpg", )
    sdi.addImage(1, 1, r"F:\GraduationDesign\MkTu\HA\ASABeiJingLayout_rekong.jpg", )
    sdi.addImage(1, 2, r"F:\GraduationDesign\MkTu\HA\DEHBeiJingLayout_rekong.jpg", )
    sdi.addImage(1, 3, r"F:\GraduationDesign\MkTu\HA\DEABeiJingLayout_rekong.jpg", )
    sdi.addImage(2, 0, r"F:\GraduationDesign\MkTu\HA\ASHChengDuLayout_rekong.jpg", )
    sdi.addImage(2, 1, r"F:\GraduationDesign\MkTu\HA\ASAChengDuLayout_rekong.jpg", )
    sdi.addImage(2, 2, r"F:\GraduationDesign\MkTu\HA\DEHChengDuLayout_rekong.jpg", )
    sdi.addImage(2, 3, r"F:\GraduationDesign\MkTu\HA\DEAChengDuLayout_rekong.jpg", )

    k = 3.5
    sdi.draw(
        n_columns_ex=k, n_rows_ex=2.0 / 2.7 * k,
        row_names=["青  \n岛  ", "北  \n京  ", "成  \n都  "],
        # column_names=["升轨 C11", "升轨 C22", "降轨 C11", "降轨 C22"],
        column_names=["升轨极化熵 H", "升轨平均散射角 α", "降轨极化熵 H", "降轨平均散射角 α"],
        fontdict=None
    )
    plt.savefig(r"F:\GraduationDesign\MkTu\HA\HAImage.jpg", dpi=300)
    plt.show()


def method_name1():
    def func1(fn):
        to_fn = changext(fn, "_rekong.jpg")
        dataset = gdal.Open(fn)
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        image_data = dataset.ReadAsArray()
        if len(image_data.shape) == 3:
            image_data = np.transpose(image_data, (1, 2, 0))
        image = Image.fromarray(image_data)
        background = Image.new(image.mode, image.size, (255, 255, 255))
        diff = ImageChops.difference(image, background)
        bbox = diff.getbbox()
        if bbox:
            cropped_image = image.crop(bbox)
        else:
            cropped_image = image  # 如果没有找到边界框，则不进行裁剪
        cropped_image.save(to_fn)

    fns = [
        # r"F:\GraduationDesign\MkTu\C2Images\ASC11BeiJingLayout.jpg",
        # r"F:\GraduationDesign\MkTu\C2Images\ASC11ChengDuLayout.jpg",
        # r"F:\GraduationDesign\MkTu\C2Images\ASC11QingDaoLayout.jpg",
        # r"F:\GraduationDesign\MkTu\C2Images\ASC22BeiJingLayout.jpg",
        # r"F:\GraduationDesign\MkTu\C2Images\ASC22ChengDuLayout.jpg",
        # r"F:\GraduationDesign\MkTu\C2Images\ASC22QingDaoLayout.jpg",
        # r"F:\GraduationDesign\MkTu\C2Images\DEC11BeiJingLayout.jpg",
        # r"F:\GraduationDesign\MkTu\C2Images\DEC11ChengDuLayout.jpg",
        # r"F:\GraduationDesign\MkTu\C2Images\DEC11QingDaoLayout.jpg",
        # r"F:\GraduationDesign\MkTu\C2Images\DEC22BeiJingLayout.jpg",
        # r"F:\GraduationDesign\MkTu\C2Images\DEC22ChengDuLayout.jpg",
        # r"F:\GraduationDesign\MkTu\C2Images\DEC22QingDaoLayout.jpg",

        r"F:\GraduationDesign\MkTu\HA\ASABeiJingLayout.jpg",
        r"F:\GraduationDesign\MkTu\HA\ASAChengDuLayout.jpg",
        r"F:\GraduationDesign\MkTu\HA\ASAQingDaoLayout.jpg",
        r"F:\GraduationDesign\MkTu\HA\ASHBeiJingLayout.jpg",
        r"F:\GraduationDesign\MkTu\HA\ASHChengDuLayout.jpg",
        r"F:\GraduationDesign\MkTu\HA\ASHQingDaoLayout.jpg",
        r"F:\GraduationDesign\MkTu\HA\DEABeiJingLayout.jpg",
        r"F:\GraduationDesign\MkTu\HA\DEAChengDuLayout.jpg",
        r"F:\GraduationDesign\MkTu\HA\DEAQingDaoLayout.jpg",
        r"F:\GraduationDesign\MkTu\HA\DEHBeiJingLayout.jpg",
        r"F:\GraduationDesign\MkTu\HA\DEHChengDuLayout.jpg",
        r"F:\GraduationDesign\MkTu\HA\DEHQingDaoLayout.jpg",
    ]
    for fn in fns:
        func1(fn)


if __name__ == "__main__":
    main()
