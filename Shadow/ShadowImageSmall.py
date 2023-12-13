# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowImageSmall.py
@Time    : 2023/10/6 8:55
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of ShadowImageSmall
-----------------------------------------------------------------------------"""
import csv
import os.path

import numpy as np
from PIL import Image
from osgeo import gdal

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.GDALUtils import gdalStratifiedRandomSampling
from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.Utils import readJson, CoorInPoly, Jdt, changext
from Shadow.ShadowDraw import cal_10log10
from Shadow.ShadowTraining import ShadowCategoryTrainImdcOne, trainRF_nocv


class ShadowImageSmall:

    def __init__(self, raster_fn, init_d=0.0):
        self.raster_fn = raster_fn
        self.gr = GDALRaster(raster_fn)
        self.ground_true_imd = np.ones([self.gr.n_rows, self.gr.n_columns]) * init_d
        self.mod_dir = r"F:\ProjectSet\Shadow\BeiJing\Temp\1\Mods"

    def geojsonToGTImd(self, json_fn):
        d = readJson(json_fn)
        feats = d["features"]
        coor_isin_list = []
        for feat in feats:
            coor_in = CoorInPoly()
            for coor in feat["geometry"]["coordinates"][0]:
                coor_in.addCoor(coor[0], coor[1])
            d0 = (feat["properties"]["CATEGORY"], coor_in)
            coor_isin_list.append(d0)
        jdt = Jdt(self.ground_true_imd.shape[0] * self.ground_true_imd.shape[1],
                  desc="ShadowImageSmall->geojsonToGTImd")
        jdt.start()
        for i in range(self.ground_true_imd.shape[0]):
            for j in range(self.ground_true_imd.shape[1]):
                x, y = self.gr.coorRaster2Geo(i + 0.5, j + 0.5)
                n = self.ground_true_imd[i, j]
                for category, coor_in in coor_isin_list:
                    if coor_in.t(x, y):
                        n = category
                        break
                self.ground_true_imd[i, j] = n
                jdt.add()
        jdt.end()

    def saveGTImage(self, to_raster_fn):
        self.gr.save(self.ground_true_imd.astype("int16"), to_raster_fn, dtype=gdal.GDT_Int16)

    def readGTImage(self, gt_raster_fn):
        gr = GDALRaster(gt_raster_fn)
        self.ground_true_imd = gr.readAsArray()

    def sampling(self, gt_raster_fn, to_csv_fn):
        spls = gdalStratifiedRandomSampling(gt_raster_fn, spls_n={0: 600, 2: 200, 11: 600, 31: 200, 41: 200})
        spls = self._changeCategory1(spls, {0: 2, 2: 5, 11: 1, 31: 3, 41: 4})
        with open(to_csv_fn, "w", encoding="utf-8", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(["X", "Y", "CATEGORY"])
            cw.writerows(spls)

    def _changeCategory1(self, spls, change_dict):
        for i in range(len(spls)):
            if spls[i][2] in change_dict:
                spls[i][2] = change_dict[spls[i][2]]
        return spls

    def trainImdcOne(self, spl_fn, raster_fn):
        # spl_fn = r"F:\ProjectSet\Shadow\BeiJing\Samples\1\sh_bj_spl_summary2_600_train.csv"
        # spl_fn = r"F:\ProjectSet\Shadow\BeiJing\Samples\3\sh_bj_3_shnosh2.csv"
        # spl_fn = r"F:\ProjectSet\Shadow\BeiJing\Samples\5\sh_bj_5_imdc1800_2.csv"
        # raster_fn = r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_envi.dat"

        scti = ShadowCategoryTrainImdcOne(self.mod_dir)
        scti.addGDALRaster(raster_fn)
        scti.initSIC(raster_fn)

        scti.addCSVFile(spl_fn, is_spl=False)
        scti.csv_spl.fieldNameCategory("CNAME")  # CNAME
        scti.csv_spl.fieldNameTag("TAG")
        scti.csv_spl.addCategoryNames(["NOT_KNOW", "IS", "VEG", "SOIL", "WAT", "SH"])
        # scti.csv_spl.addCategoryNames(["SH", "NO_SH"])
        scti.csv_spl.readData()

        scti.featureCallBack("AS_VV", cal_10log10)
        scti.featureCallBack("AS_VH", cal_10log10)
        scti.featureCallBack("AS_C11", cal_10log10)
        scti.featureCallBack("AS_C22", cal_10log10)
        scti.featureCallBack("AS_Lambda1", cal_10log10)
        scti.featureCallBack("AS_Lambda2", cal_10log10)
        scti.featureCallBack("AS_SPAN", cal_10log10)
        scti.featureCallBack("AS_Epsilon", cal_10log10)
        scti.featureCallBack("DE_VV", cal_10log10)
        scti.featureCallBack("DE_VH", cal_10log10)
        scti.featureCallBack("DE_C11", cal_10log10)
        scti.featureCallBack("DE_C22", cal_10log10)
        scti.featureCallBack("DE_Lambda1", cal_10log10)
        scti.featureCallBack("DE_Lambda2", cal_10log10)
        scti.featureCallBack("DE_SPAN", cal_10log10)
        scti.featureCallBack("DE_Epsilon", cal_10log10)

        scti.featureScaleMinMax("Blue", 299.76996, 2397.184)
        scti.featureScaleMinMax("Green", 345.83414, 2395.735)
        scti.featureScaleMinMax("Red", 177.79654, 2726.7026)
        scti.featureScaleMinMax("NIR", 87.66086, 3498.4321)
        scti.featureScaleMinMax("NDVI", -0.6, 0.9)
        scti.featureScaleMinMax("NDWI", -0.7, 0.8)

        scti.featureScaleMinMax("AS_VV", -24.609674, 5.9092603)
        scti.featureScaleMinMax("AS_VH", -31.865038, -5.2615275)
        scti.featureScaleMinMax("AS_C11", -22.61998, 5.8634768)
        scti.featureScaleMinMax("AS_C22", -28.579813, -5.2111626)
        scti.featureScaleMinMax("AS_Lambda1", -21.955856, 6.124724)
        scti.featureScaleMinMax("AS_Lambda2", -29.869734, -8.284683)
        scti.featureScaleMinMax("AS_SPAN", -25.869734, 10.284683)
        scti.featureScaleMinMax("AS_Epsilon", -10.0, 26.0)
        # scti.featureScaleMinMax("AS_Mu", 0.0, 1.0)
        scti.featureScaleMinMax("AS_RVI", 0, 2.76234)
        # scti.featureScaleMinMax("AS_m", 0.0, 1.0)
        # scti.featureScaleMinMax("AS_Beta", 0.0, 1.0)

        scti.featureScaleMinMax("DE_VV", -27.851603, 5.094706)
        scti.featureScaleMinMax("DE_VH", -35.427082, -5.4092093)
        scti.featureScaleMinMax("DE_C11", -26.245598, 4.9907513)
        scti.featureScaleMinMax("DE_C22", -32.042320, -5.322515)
        scti.featureScaleMinMax("DE_Lambda1", -25.503738, 5.2980003)
        scti.featureScaleMinMax("DE_Lambda2", -33.442368, -8.68537)
        scti.featureScaleMinMax("DE_SPAN", -27.869734, 13.284683)
        scti.featureScaleMinMax("DE_Epsilon", -6.0, 20.0)
        # scti.featureScaleMinMax("DE_Mu", 0.0, 1.0)
        scti.featureScaleMinMax("DE_RVI", 0, 2.76234)
        # scti.featureScaleMinMax("DE_m", 0.0, 1.0)
        # scti.featureScaleMinMax("DE_Beta", 0.0, 1.0)

        scti.sicAddCategory("IS", (255, 0, 0))
        scti.sicAddCategory("VEG", (0, 255, 0))
        scti.sicAddCategory("SOIL", (255, 255, 0))
        scti.sicAddCategory("WAT", (0, 0, 255))
        scti.sicAddCategory("SH", (0, 0, 0))

        # scti.sicAddCategory("SH", (255, 255,255))
        # scti.sicAddCategory("NO_SH", (0, 0, 0))

        scti.setSample()

        scti.fitFeatureNames(
            "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
            # "AS_VV", "AS_VH", "AS_C11", "AS_C12_imag", "AS_C12_real", "AS_C22", "AS_Lambda1", "AS_Lambda2",
            # "AS_SPAN", "AS_Epsilon", "AS_Mu", "AS_RVI", "AS_m", "AS_Beta",
            "DE_VV", "DE_VH", "DE_C11", "DE_C12_imag", "DE_C12_real", "DE_C22", "DE_Lambda1", "DE_Lambda2",
            "DE_SPAN", "DE_Epsilon", "DE_Mu", "DE_RVI", "DE_m", "DE_Beta"
        )
        # "NOT_KNOW", "IS", "VEG", "SOIL", "WAT", "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH",
        scti.fitCategoryNames("IS", "VEG", "SOIL", "WAT",
                              # "SH"
                              )
        scti.fitCMNames("IS", "VEG", "SOIL", "WAT",
                        # "SH"
                        )

        # scti.fitFeatureNames(
        #     "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
        #     "AS_VV", "AS_VH", "AS_C11", "AS_C12_imag", "AS_C12_real", "AS_C22", "AS_Lambda1", "AS_Lambda2",
        #     "AS_SPAN", "AS_Epsilon", "AS_Mu", "AS_RVI", "AS_m", "AS_Beta",
        #     "DE_VV", "DE_VH", "DE_C11", "DE_C12_imag", "DE_C12_real", "DE_C22", "DE_Lambda1", "DE_Lambda2",
        #     "DE_SPAN", "DE_Epsilon", "DE_Mu", "DE_RVI", "DE_m", "DE_Beta"
        # )
        # scti.fitCMNames("SH", "NO_SH")

        scti.trainFunc(trainRF_nocv)
        # scti.trainFunc(trainSvm_nocv)

        scti.fit(is_sh_to_no=False)

    def testGT(self, gt_raster_fn, imdc_fn, map_dict=None):
        gr_gt = GDALRaster(gt_raster_fn)
        gt_d = gr_gt.readAsArray().ravel()
        gr_imd = GDALRaster(imdc_fn)
        imd = gr_imd.readAsArray().ravel()
        print("gt_d", gt_d.shape, "gr_imd", imd.shape)
        if map_dict is not None:
            for i in range(len(gt_d)):
                if gt_d[i] in map_dict:
                    gt_d[i] = map_dict[gt_d[i]]
        names = ["IS", "VEG", "SOIL", "WAT",
                 "SH"
                 ]
        cm = ConfusionMatrix(n_class=len(names), class_names=names)
        cm.addData(gt_d, imd)
        print(cm.fmtCM())
        ...

    def showImdc1(self, raster_fn, to_fn, n_channel=0, map_dict=None):
        if map_dict is None:
            map_dict = {}
        gr = GDALRaster(raster_fn)
        d = gr.readAsArray()
        if len(d.shape) == 3:
            d = d[n_channel]
        colored_image = np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
        for k in map_dict:
            colored_image[d == k] = map_dict[k]
        image = Image.fromarray(colored_image)
        image.save(to_fn)


def main():
    sis_bj_1 = ShadowImageSmall(r"F:\ProjectSet\Shadow\BeiJing\Temp\1\bj_test_1.tif", 21)

    # sis_bj_1.geojsonToGTImd(r"F:\ProjectSet\Shadow\BeiJing\Temp\1\bj_test_1_cate_1.json")
    # sis_bj_1.saveGTImage(r"F:\ProjectSet\Shadow\BeiJing\Temp\1\bj_test_1_gt1.dat")
    # sis_bj_1.sampling(r"F:\ProjectSet\Shadow\BeiJing\Temp\1\bj_test_1_g2t.dat",
    #                   r"F:\ProjectSet\Shadow\BeiJing\Temp\1\bj_test_1_g2t_spl1.csv")
    # sis_bj_1.trainImdcOne(r"F:\ProjectSet\Shadow\BeiJing\Temp\1\bj_test_1_g2t_spl1_2.csv",
    #                       r"F:\ProjectSet\Shadow\BeiJing\Temp\1\bj_test_1_envi.dat")
    # sis_bj_1.testGT(r"F:\ProjectSet\Shadow\BeiJing\Temp\1\bj_test_1_g2t.dat",
    #                 r"F:\ProjectSet\Shadow\BeiJing\Temp\1\Mods\20231006H110156\model_name_imdc.dat",
    #                 map_dict={0: 2, 2: 5, 11: 1, 31: 3, 41: 4})
    # sis_bj_1.showImdc1(r"F:\ProjectSet\Shadow\BeiJing\Temp\1\bj_test_1_g2t.dat",
    #                    r"F:\ProjectSet\Shadow\BeiJing\Temp\1\bj_test_1_g2t_image.png",
    #                    map_dict={0: (0, 255, 0), 2: (0, 0, 0), 11: (255, 0, 0), 31: (255, 255, 0), 41: (0, 0, 255)})

    def show_imdc(mod_dirname, map_dict):
        raster_fn = os.path.join(r"F:\ProjectSet\Shadow\BeiJing\Temp\1\Mods", mod_dirname, "model_name_imdc.dat")
        to_fn = changext(raster_fn, "_image.png")
        sis_bj_1.showImdc1(raster_fn, to_fn, map_dict=map_dict)

    map_dict_0 = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), 5: (0, 0, 0)}
    show_imdc("20231006H110727", map_dict_0)

    pass


if __name__ == "__main__":
    main()
