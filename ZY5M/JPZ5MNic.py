# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : JPZ5MNic.py
@Time    : 2024/1/9 18:53
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of JPZ5MNic

sampleSpaceUniform
-----------------------------------------------------------------------------"""
import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.GDALUtils import samplingToCSV
from SRTCodes.OGRUtils import sampleSpaceUniform
from SRTCodes.Utils import readcsv, listMap, filterFileExt, readJson, saveJson, Jdt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号


class JPZ5MNicTrainImage:

    def __init__(self):
        self.spl_fn = r"F:\ProjectSet\Huo\jpz5m4nian\Temp\JPZ5M2-20240110T142025Z-001\JPZ5M2\jpz5m4_t1_0.csv_cat.csv"
        self.spl_fn = r"F:\ProjectSet\Huo\jpz5m4nian\samples\2\sample21_1.csv"
        # self.spl_fn = r"G:\Downloads\JPZ5M2-20240110T150216Z-001\JPZ5M2\jpz5m4_t2_0.csv_cat.csv"
        self.spl_fn = r"F:\ProjectSet\Huo\jpz5m4nian\samples\2\sample21_2.csv"

        self.imdc_dirname = r"K:\zhongdianyanfa\jpz5m_2\Imdc"

        self.df = pd.read_csv(self.spl_fn)
        self.category_name = "category"
        self.model_dirname = r"F:\ProjectSet\Huo\jpz5m4nian\Mod"
        self.to_csv_fn = r"F:\ProjectSet\Huo\jpz5m4nian\samples\2\sample21_2.csv"

        self.train_feat_names = ["B", "G", "R", "N", "NDVI", "NDWI", "ND1", "ND2", "ND4", "ND6"]
        self.clf = None

        self.time_str = None

    def train(self):
        filename = os.path.join(self.model_dirname, time.strftime("%Y%m%dH%H%M%S") + ".mod")
        print(filename)

        read_model_name = r"F:\ProjectSet\Huo\jpz5m4nian\Mod\20240110H230122.mod"
        read_model_name = r"F:\ProjectSet\Huo\jpz5m4nian\Mod\20240110H231326.mod"
        read_model_name = None

        # self.extFeat()
        print(self.df.keys())
        print(len(self.df))
        x = self.df[self.train_feat_names].values
        y = self.df[self.category_name].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=3620, shuffle=True)
        if read_model_name is None:
            print("train")
            self.clf = RandomForestClassifier(
                n_estimators=120,
                criterion='gini',
                max_depth=5,
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
            imd_data[:4, :, :],
            [normal_feat(n, r)],
            [normal_feat(g, n)],
            [normal_feat(r, g)],
            [normal_feat(r, b)],
            [normal_feat(g, b)],
            [normal_feat(b, n)],
        ]

        d = np.concatenate(data_list)
        return d

    def imdc(self):
        model_fn = r"F:\ProjectSet\Huo\jpz5m4nian\Mod\20240110H232317.mod"
        model_fn = r"F:\ProjectSet\Huo\jpz5m4nian\Mod\20240110H232052.mod"
        raster_fn_list = [
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_21_im0.tif",
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_21_im1.tif",
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_21_im2.tif",
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_21_im3.tif",
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_21_im4.tif",
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_21_im5.tif",
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_21_im6.tif",
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_21_im7.tif",
            r"K:\zhongdianyanfa\jpz5m_2\Images\jpz5m_21_im8.tif",
        ]
        for raster_fn in raster_fn_list:
            print("raster_fn:", raster_fn)

            self.clf = joblib.load(model_fn)
            im_gr = GDALRaster(raster_fn)
            im_data = im_gr.readAsArray()
            im_data = self.imdcFeatExt(im_data)

            jdt = Jdt(im_data.shape[1], "JPZ5MNic Imdc")
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
        # self.train()
        # self.extFeat()
        self.imdc()


def reportFuncs():
    def filter_grids():
        csv_fn = r"F:\ProjectSet\Huo\jpz5m4nian\Temp\jpz_region1_cal2.csv"
        df = pd.read_csv(csv_fn)
        del df[".geo"]
        df.to_csv(r"F:\ProjectSet\Huo\jpz5m4nian\Report\jpz_region1_cal2_nogeo.csv")
        return

    def cat_grids(*grids_fns, to_fn=None):
        to_json_dict = None
        for fn in grids_fns:
            if to_json_dict is None:
                to_json_dict = readJson(fn)
            else:
                json_dict = readJson(fn)
                i_find = 0
                for feat in to_json_dict["features"]:
                    find_d = feat["properties"]["id"]

                    for feat2 in json_dict["features"]:
                        find_d2 = feat2["properties"]["id"]
                        if find_d == find_d2:
                            for k in feat2["properties"]:
                                to_json_dict["features"][i_find]["properties"][k] = feat2["properties"][k]
                            break

                    i_find += 1
        saveJson(to_json_dict, to_fn)

    def fun2():
        cat_grids(
            r"F:\ProjectSet\Huo\jpz5m4nian\Temp\ZY5MCitysGrids-20240113T143733Z-001\ZY5MCitysGrids\jinbian_grids_area.geojson",
            r"F:\ProjectSet\Huo\jpz5m4nian\Temp\ZY5MCitysGrids-20240113T143733Z-001\ZY5MCitysGrids\jinbian_grids_wbei18.geojson",
            r"F:\ProjectSet\Huo\jpz5m4nian\Temp\ZY5MCitysGrids-20240113T143733Z-001\ZY5MCitysGrids\jinbian_grids_wbei21.geojson",
            to_fn=r"F:\ProjectSet\Huo\jpz5m4nian\Temp\ZY5MCitysGrids-20240113T143733Z-001\ZY5MCitysGrids\jinbian_grids.geojson"
        )

        cat_grids(
            r"F:\ProjectSet\Huo\jpz5m4nian\Temp\ZY5MCitysGrids-20240113T143733Z-001\ZY5MCitysGrids\mdw_grids_area.geojson",
            r"F:\ProjectSet\Huo\jpz5m4nian\Temp\ZY5MCitysGrids-20240113T143733Z-001\ZY5MCitysGrids\mdw_grids_wbei18.geojson",
            r"F:\ProjectSet\Huo\jpz5m4nian\Temp\ZY5MCitysGrids-20240113T143733Z-001\ZY5MCitysGrids\mdw_grids_wbei21.geojson",
            to_fn=r"F:\ProjectSet\Huo\jpz5m4nian\Temp\ZY5MCitysGrids-20240113T143733Z-001\ZY5MCitysGrids\mdw_grids.geojson"
        )

        cat_grids(
            r"F:\ProjectSet\Huo\jpz5m4nian\Temp\ZY5MCitysGrids-20240113T143733Z-001\ZY5MCitysGrids\xl_grids__area.geojson",
            r"F:\ProjectSet\Huo\jpz5m4nian\Temp\ZY5MCitysGrids-20240113T143733Z-001\ZY5MCitysGrids\xl_grids_wbei18.geojson",
            r"F:\ProjectSet\Huo\jpz5m4nian\Temp\ZY5MCitysGrids-20240113T143733Z-001\ZY5MCitysGrids\xl_grids_wbei21.geojson",
            to_fn=r"F:\ProjectSet\Huo\jpz5m4nian\Temp\ZY5MCitysGrids-20240113T143733Z-001\ZY5MCitysGrids\xl_grids.geojson"
        )

    def plotScatter(csv_fn):
        fig = plt.figure(figsize=(8, 8))
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, hspace=0.4, wspace=0.3)
        plt.subplot(331)

        def scatter_show(csv_fn_plot, n, name=""):
            df = pd.read_csv(csv_fn_plot)
            df["del_is"] = df["ais21"] - df["ais17"]
            df["del_veg"] = df["aveg21"] - df["aveg17"]
            df["del_wl"] = df["awl21"] - df["awl17"]
            df["del_wbei"] = df["wbei_22"] - df["wbei_18"]

            plt.subplot(330 + n)
            plt.title(name)
            plt.scatter(df["del_is"], df["del_wbei"], s=2, color="red")
            plt.xlabel("不透水面变化面积(km2)")
            plt.ylabel("WBEI")

            plt.subplot(330 + n + 1)
            plt.title(name)
            plt.scatter(df["del_veg"], df["del_wbei"], s=2, color="green")
            plt.xlabel("植被变化面积(km2)")
            plt.ylabel("WBEI")

            plt.subplot(330 + n + 2)
            plt.title(name)
            plt.scatter(df["del_wl"], df["del_wbei"], s=2, color="blue")
            plt.xlabel("湿地变化面积(km2)")
            plt.ylabel("WBEI")

        scatter_show(
            r"F:\ProjectSet\Huo\jpz5m4nian\Temp\ZY5MCitysGrids-20240113T143733Z-001\ZY5MCitysGrids\jinbian_grids.csv",
            1, name="金边")
        scatter_show(
            r"F:\ProjectSet\Huo\jpz5m4nian\Temp\ZY5MCitysGrids-20240113T143733Z-001\ZY5MCitysGrids\xl_grids.csv", 4,
            name="暹粒")
        scatter_show(
            r"F:\ProjectSet\Huo\jpz5m4nian\Temp\ZY5MCitysGrids-20240113T143733Z-001\ZY5MCitysGrids\mdw_grids.csv", 7,
            name="马德望")
        plt.show()

    plotScatter(r"F:\ProjectSet\Huo\jpz5m4nian\Temp\ZY5MCitysGrids-20240113T143733Z-001\ZY5MCitysGrids\xl_grids.csv")


def main():
    def func1():
        # 样本空间均匀
        csv_fn = r"F:\ProjectSet\Huo\jpz5m4nian\samples\jpz5m_nic_grids_2_gt100_2.csv"
        d = readcsv(csv_fn)
        print(d.keys())
        x, y = listMap(d["X"], float), listMap(d["Y"], float)
        coors = list(zip(x, y))
        print(len(coors))
        coors, find_list = sampleSpaceUniform(coors, x_len=1000, y_len=1000, is_trans_jiaodu=True, is_jdt=True,
                                              ret_index=True)
        coors = np.array(coors)
        print(coors.shape)
        df = pd.read_csv(csv_fn)
        df = df.loc[find_list]
        df.to_csv(csv_fn + "_ssu.csv", index=False)

    def func2():
        df_filelist = filterFileExt(
            r"G:\Downloads\JPZ5M2-20240110T150216Z-001\JPZ5M2", ".csv")
        df_list = []
        for df_fn in df_filelist:
            if "_cat.csv" in df_fn:
                continue
            print(df_fn)
            df_list.append(pd.read_csv(df_fn))
        df = pd.concat(df_list)
        to_fn = df_filelist[0] + "_cat.csv"
        print(to_fn)
        df.to_csv(to_fn, index=False)

    def filter_yaosu(json_fn, to_json_fn, field_name, filter_d):
        d = readJson(json_fn)
        to_dict = {'type': d['type'], 'name': d['name'], 'crs': d['crs'], 'features': []}
        jdt = Jdt(len(d["features"]), "filter_yaosu")
        jdt.start()
        for i, feat1 in enumerate(d["features"]):
            if feat1["properties"][field_name] in filter_d:
                to_dict["features"].append(feat1)
            jdt.add()
        jdt.end()
        saveJson(to_dict, to_json_fn)

    def add_length_number(json_fn, to_json_fn, divide_d):
        d = readJson(json_fn)
        to_dict = {'type': d['type'], 'name': d['name'], 'crs': d['crs'], 'features': []}
        field_name = "length"
        jdt = Jdt(len(d["features"]), "filter_yaosu")
        jdt.start()
        n_all = 0
        for i, feat1 in enumerate(d["features"]):
            n = int(feat1["properties"][field_name] / divide_d)
            if n == 0:
                n = 1
            feat1["properties"]["n_random"] = n
            to_dict["features"].append(feat1)
            n_all += n
            jdt.add()
        jdt.end()
        saveJson(to_dict, to_json_fn)
        print("n", n_all)
        return

    def func3():
        csv_fn = r"F:\ProjectSet\Huo\jpz5m4nian\samples\jpz5m4_spl_coll1_csv_cat_veg.csv"
        samplingToCSV(
            csv_fn=csv_fn,
            gr=GDALRaster(r"F:\ProjectSet\Huo\jpz5m4nian\drive-download-20240110T003717Z-001\imdc17.vrt"),
            to_csv_fn=csv_fn + "_imdc17.csv"
        )

    def func4():

        gr1 = GDALRaster(r"G:\Downloads\drive-download-20240112T035911Z-001\jpz5m_17_3_imclass1_tif.tif")
        d1 = gr1.readAsArray()
        print(d1.shape)

        gr2 = GDALRaster(r"G:\Downloads\drive-download-20240112T035911Z-001\jpz5m_17_3_imclass22_dat.dat")
        d2 = gr2.readAsArray()
        print(d2.shape)

        d = (d1 != d2) * 1

        gr1.save(d.astype("int8"), r"G:\Downloads\drive-download-20240112T035911Z-001\jpz5m_17_3_imclass1_tif_t.dat",
                 fmt="ENVI", dtype=gdal.GDT_Byte)

    # func2()
    # filter_yaosu(
    #     json_fn=r"G:\ShapeData\cambodia-latest-free.shp\gis_osm_roads_free_1.geojson",
    #     to_json_fn=r"F:\ProjectSet\Huo\jpz5m4nian\samples\gis_osm_roads_free_1_1.geojson",
    #     field_name="fclass",
    #     filter_d=["footway", "pedestrian", "primary", "secondary", "secondary_link",
    #               "service", "tertiary", "tertiary_link", "trunk"]
    # )
    # add_length_number(r"F:\ProjectSet\Huo\jpz5m4nian\samples\jpz5m_nic_grids_2_gt100.geojson",
    #                   "F:\ProjectSet\Huo\jpz5m4nian\samples\jpz5m_nic_grids_2_gt100_1.geojson", 500)
    # func1()
    # func3()
    # func2()
    # JPZ5MNicTrainImage().main()
    reportFuncs()


if __name__ == "__main__":
    main()
