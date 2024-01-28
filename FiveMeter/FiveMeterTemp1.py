# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : FiveMeterTemp1.py
@Time    : 2024/1/18 14:21
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of FiveMeterTemp1
-----------------------------------------------------------------------------"""

import os
import time

import joblib
import numpy as np
import pandas as pd
from osgeo import gdal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.NumpyUtils import conv2dDim1
from SRTCodes.Utils import Jdt, DirFileName


class JPZ5MNicTrainImage:

    def __init__(self):
        self.dfn = DirFileName(r"F:\Week\20240121\Data")
        self.spl_fn = self.dfn.fn(r"samples1_spl4.csv")
        self.to_csv_fn = self.dfn.fn(r"samples1_spl4.csv")
        self.imdc_dirname = self.dfn.fn(r"Imdc")
        self.model_dirname = self.dfn.fn(r"Mods")
        self.raster_fn = self.dfn.fn("image2.tif")
        self.mod1_fn = self.dfn.fn(r"Mods\mod1.txt")

        self.df = pd.read_csv(self.spl_fn)
        self.category_name = "CATEGORY"

        self.train_feat_names = ["SAMPLE_7", "SAMPLE_8", "SAMPLE_9", "SAMPLE_10",
                                 "SAMPLE_1", "SAMPLE_2", "SAMPLE_3", "SAMPLE_4",
                                 "VV", "VH",
                                 "NDVI", "NDWI", "ND1", "ND2", "ND4",
                                 "ND6", ]
        self.clf = None

        self.time_str = None

    def train(self):
        filename = os.path.join(self.model_dirname, time.strftime("%Y%m%dH%H%M%S") + ".mod")
        print(filename)
        read_model_name = None

        print(self.df.keys())
        print(len(self.df))
        x = self.df[self.train_feat_names].values
        y = self.df[self.category_name].values
        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=3620, shuffle=True)
        x_train, x_test, y_train, y_test = x, x, y, y
        if read_model_name is None:
            print("train")
            self.clf = RandomForestClassifier(
                n_estimators=120,
                criterion='gini',
                max_depth=6,
                min_samples_leaf=10,
                min_samples_split=10,
            )
            self.clf.fit(x_train, y_train)
        else:
            print("read_model_name")
            self.clf = joblib.load(read_model_name)

        print("Test")

        y_pred = self.clf.predict(x_test)

        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        print("Mod", self.clf)
        print("Can Shu", self.clf.estimator_params)
        if read_model_name is None:
            joblib.dump(self.clf, filename)
            with open(self.mod1_fn, "w", encoding="utf-8") as f:
                f.write(filename)
        print(self.clf.score(x_test, y_test))

    def extFeat(self):
        self.df["NDVI"] = self.nd("N", "R")
        self.df["NDWI"] = self.nd("G", "N")
        self.df["ND6"] = self.nd("B", "N")
        self.df["ND1"] = self.nd("R", "G")
        self.df["ND2"] = self.nd("R", "B")
        self.df["ND4"] = self.nd("G", "B")
        if self.to_csv_fn is not None:
            self.df.to_csv(self.to_csv_fn, index=False)

    def nd(self, name1, name2):
        name_change = {"B": "SAMPLE_1", "G": "SAMPLE_2", "R": "SAMPLE_3", "N": "SAMPLE_4", }
        name1, name2 = name_change[name1], name_change[name2]
        return (self.df[name1] - self.df[name2]) / (self.df[name1] + self.df[name2])

    def getImdcFN(self, fn):
        fn = os.path.split(fn)[1]
        fn = os.path.splitext(fn)[0]
        if self.time_str is None:
            self.time_str = time.strftime("%Y%m%dH%H%M%S")
        time_str = self.time_str
        return os.path.join(self.imdc_dirname, "{0}_{1}_imdc.tif".format(fn, time_str))

    def imdcFeatExt(self, imd_data: np.ndarray):
        """
        self.df["NDVI"] = self.nd("N", "R")
        self.df["NDWI"] = self.nd("G", "N")
        self.df["ND1"] = self.nd("R", "G")
        self.df["ND2"] = self.nd("R", "B")
        self.df["ND4"] = self.nd("G", "B")
        self.df["ND6"] = self.nd("B", "N")

        :param imd_data:
        :return:
        """

        def normal_feat(d1, d2):
            return (d1 - d2) / (d1 + d2)

        b, g, r, n = imd_data[0, :, :], imd_data[1, :, :], imd_data[2, :, :], imd_data[3, :, :],
        data_list = [
            imd_data[6:, :, :],
            imd_data[:4, :, :],
            imd_data[4:6, :, :],
            [normal_feat(n, r)],
            [normal_feat(g, n)],
            [normal_feat(r, g)],
            [normal_feat(r, b)],
            [normal_feat(g, b)],
            [normal_feat(b, n)],
        ]

        d = np.concatenate(data_list)
        return d

    def addRasterChannel(self):
        gr = GDALRaster(self.raster_fn)
        data = gr.readAsArray()
        data_tmp = [data]
        print(data.shape)

        def conv2d(kernel):
            for i in range(4):
                print(i)
                data_tmp.append([conv2dDim1(data[i], kernel)])
                print(i)

        conv2d([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        data_tmp = np.concatenate(data_tmp)
        print(data_tmp.shape)
        gr.save(data_tmp, self.raster_fn+"tmp",fmt="GTiff")

    def imdc(self):
        model_fn = None
        if model_fn is None:
            with open(self.mod1_fn, "r", encoding="utf-8") as f:
                filename = f.read()
                model_fn = filename.strip()
        print("model_fn:", model_fn)
        raster_fn_list = [self.raster_fn]

        for raster_fn in raster_fn_list:
            print("raster_fn:", raster_fn)

            self.clf = joblib.load(model_fn)
            im_gr = GDALRaster(raster_fn)
            im_data = im_gr.readAsArray()
            im_data = self.imdcFeatExt(im_data)
            # im_data = self.imdcFeatExt(im_data)

            jdt = Jdt(im_data.shape[1], "Imdc")
            jdt.start()
            imdc = np.zeros((im_data.shape[1], im_data.shape[2]))
            for i in range(im_data.shape[1]):
                spl = im_data[:, i, :].T
                y = self.clf.predict(spl)
                jdt.add()
                imdc[i, :] = y
            jdt.end()

            to_fn = self.getImdcFN(raster_fn)
            print("to_fn:", to_fn)
            im_gr.save(imdc.astype("int8"), to_fn, dtype=gdal.GDT_Byte, fmt="GTiff")

    def main(self):
        self.train()
        # self.extFeat()
        self.imdc()
        # self.addRasterChannel()


def main():
    JPZ5MNicTrainImage().main()
    pass


if __name__ == "__main__":
    main()
