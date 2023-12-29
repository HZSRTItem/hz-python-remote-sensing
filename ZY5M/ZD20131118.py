# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ZD20131118.py
@Time    : 2023/11/18 19:48
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of ZD20131118
-----------------------------------------------------------------------------"""
import os.path

import numpy as np
import pandas as pd

from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.Utils import readLines, getfilext, readcsv, savecsv
from ZY5M.ZY5MWarp import ZY5MGDALRasterWarp


def vlookup_ZD20131118(find_list, df: pd.DataFrame, field_name):
    lines = [df.keys().tolist()]
    for find_d in find_list:
        for i in range(len(df)):
            if df[field_name][i] == find_d:
                lines.append(df.loc[i].values.tolist())
                break
    return pd.DataFrame(lines)


def main():
    dirname1 = r"M:\Dataset\AOIS"
    dirname2 = r"K:\zhongdianyanfa\jpz_2m\tmp"

    def cmd_run(filename, filelist):
        with open(os.path.join(dirname1, filename + "_run.txt"), "w", encoding="utf-8") as fw:
            for fn in filelist:
                print("copy /Y {0}".format(to_s(fn)), file=fw)
            filelist_fn = os.path.join(dirname1, filename + "_filelist.txt")
            with open(filelist_fn, "w", encoding="utf-8") as f:
                for fn in filelist:
                    fn2 = os.path.join(dirname2, os.path.split(fn)[1])
                    print("gdal_translate -of ENVI {0} {1}".format(to_s(fn2), to_s(fn2 + ".dat")), file=fw)
                    f.write(fn2 + ".dat\n")

            print("gdal_merge_r -of ENVI -o {0}_envi.dat -filelist {1}".format(filename, filelist_fn), file=fw)
            print("gdal_translate -a_srs EPSG:4326 -of GTiff -co COMPRESS=DEFLATE -co PREDICTOR=2 -co ZLEVEL=9 "
                  "{0}_envi.dat {0}_gtif.tif".format(filename), file=fw)

    to_dict = {}
    for fn in os.listdir(dirname1):
        if os.path.splitext(fn)[1] == ".tif":
            name = fn.split("-")[0]
            if name not in to_dict:
                to_dict[name] = []
            to_dict[name].append(os.path.join(dirname1, fn))
    for k in to_dict:
        print(k, len(to_dict[k]))
        cmd_run(k, to_dict[k])
    print(to_dict)


def to_s(line):
    return "\"" + line + "\""


def method_name5():
    def tmp1(csv_fn):
        print(csv_fn)
        df = readcsv(csv_fn)
        to_df = {"X": [], "Y": [], "CATEGORY": [], "SAMPLE_1": []}
        for i in range(len(df["X"])):
            if df["SAMPLE_1"][i] != "":
                for k in to_df:
                    to_df[k].append(df[k][i])
        print(len(to_df["X"]))
        savecsv(os.path.splitext(csv_fn)[0] + "_spl.csv", to_df)

    # tmp1(r"M:\jpz2m\liz\imdc_spl\jpz2m_imdc_spl_xl17_2.csv")
    # tmp1(r"M:\jpz2m\liz\imdc_spl\jpz2m_imdc_spl_mdw17_2.csv")
    # tmp1(r"M:\jpz2m\liz\imdc_spl\jpz2m_imdc_spl_jb21_2.csv")
    # tmp1(r"M:\jpz2m\liz\imdc_spl\jpz2m_imdc_spl_jb17_2.csv")
    # tmp1(r"M:\jpz2m\liz\imdc_spl\jpz2m_imdc_spl_admj23_2.csv")
    # tmp1(r"M:\jpz2m\liz\imdc_spl\jpz2m_imdc_spl_admj22_2.csv")
    # tmp1(r"M:\jpz2m\liz\imdc_spl\jpz2m_imdc_spl_admj21_2.csv")
    # tmp1(r"M:\jpz2m\liz\imdc_spl\jpz2m_imdc_spl_admj20_2.csv")
    # tmp1(r"M:\jpz2m\liz\imdc_spl\jpz2m_imdc_spl_admj19_2.csv")
    # tmp1(r"M:\jpz2m\liz\imdc_spl\jpz2m_imdc_spl_admj18_2.csv")
    # tmp1(r"M:\jpz2m\liz\imdc_spl\jpz2m_imdc_spl_admj17_2.csv")
    # tmp1(r"M:\jpz2m\liz\imdc_spl\jpz2m_imdc_spl_admj16_2.csv")
    # tmp1(r"M:\jpz2m\liz\imdc_spl\jpz2m_imdc_spl_admj14_2.csv")
    def tmp2(csv_fn, pred_k):
        df = pd.read_csv(csv_fn)
        cm = ConfusionMatrix(2, ["IS", "NOIS"])
        cate = np.array(df["CATEGORY"]) + 1
        pred = np.array(df[pred_k]) + 1
        cm.addData(cate.tolist(), pred.tolist())
        print(cm.fmtCM())
        print(cm.OA())
        print(cm.UA()[1])
        print(cm.PA()[1])
        print(cm.getKappa() / 100)

    # tmp2(r"M:\jpz2m\liz\JPZ-20231121T095144Z-001\SHP\sample_noc2_gisa2.csv", "GISA10")
    def tmp3(csv_fn, pred_k):
        # print(pred_k)
        df = readcsv(csv_fn)
        cm = ConfusionMatrix(2, ["NOIS", "IS"])
        cate = []
        pred = []
        for i in range(len(df["X"])):
            if df[pred_k][i] != "":
                cate.append(eval(df["CATEGORY"][i]) + 1)
                pred.append(eval(df[pred_k][i]) + 1)
        cm.addData(cate, pred)
        print(cm.OA())
        print(cm.UA()[1])
        print(cm.PA()[1])
        print(cm.getKappa())

    # 'X', 'Y', 'CATEGORY', 'SRT', 'admj23', 'jb17', 'jb21', 'mdw17', 'xl17', 'GAIS', 'ESA'
    # tmp3(r"M:\jpz2m\liz\2\spl1.csv", 'ESA')
    # tmp3(r"M:\jpz2m\liz\2\spl1.csv", 'GAIS')
    # tmp3(r"M:\jpz2m\liz\2\spl1.csv", 'admj23')
    # tmp3(r"M:\jpz2m\liz\2\spl1.csv", 'jb17')
    # tmp3(r"M:\jpz2m\liz\2\spl1.csv", 'jb21')
    # tmp3(r"M:\jpz2m\liz\2\spl1.csv", 'mdw17')
    # tmp3(r"M:\jpz2m\liz\2\spl1.csv", 'xl17')
    tmp3(r"M:\jpz2m\liz\2\city.csv", 'ESA')
    tmp3(r"M:\jpz2m\liz\2\city.csv", 'GAIS')
    tmp3(r"M:\jpz2m\liz\2\city.csv", 'admj23')
    tmp3(r"M:\jpz2m\liz\2\city.csv", 'jb17')
    tmp3(r"M:\jpz2m\liz\2\city.csv", 'jb21')
    tmp3(r"M:\jpz2m\liz\2\city.csv", 'mdw17')
    tmp3(r"M:\jpz2m\liz\2\city.csv", 'xl17')


def method_name4():
    filelist = [r"M:\jpz2m\Samples\Random\jpz2m_random2_14.csv",
                r"M:\jpz2m\Samples\Random\jpz2m_random2_16.csv",
                r"M:\jpz2m\Samples\Random\jpz2m_random2_17.csv",
                r"M:\jpz2m\Samples\Random\jpz2m_random2_18.csv",
                r"M:\jpz2m\Samples\Random\jpz2m_random2_19.csv",
                r"M:\jpz2m\Samples\Random\jpz2m_random2_20.csv",
                r"M:\jpz2m\Samples\Random\jpz2m_random2_21.csv",
                r"M:\jpz2m\Samples\Random\jpz2m_random2_22.csv",
                r"M:\jpz2m\Samples\Random\jpz2m_random2_23.csv"]

    def extNDVI(red_list, nir_list):
        red_list = [float(d) for d in red_list]
        nir_list = [float(d) for d in nir_list]
        ndvi = (np.array(nir_list) - np.array(red_list)) / (np.array(nir_list) + np.array(red_list) + 0.0000001)
        return ndvi.tolist()

    def extNDWI(green_list, nir_list):
        green_list = [float(d) for d in green_list]
        nir_list = [float(d) for d in nir_list]
        ndvi = (np.array(green_list) - np.array(nir_list)) / (np.array(green_list) + np.array(nir_list) + 0.0000001)
        return ndvi.tolist()

    # to_dict = None
    # for fn in filelist:
    #     year = fn.split("_")[-1].split(".")[0]
    #     print(year)
    #     tmp_dict = readcsv(fn)
    #     to_dict = {k: [] for k in tmp_dict}
    #     for i in range(len(tmp_dict["X"])):
    #         if tmp_dict["SAMPLE_1"][i] != "":
    #             for k in tmp_dict:
    #                 to_dict[k].append(tmp_dict[k][i])
    #     savecsv(os.path.splitext(fn)[0] + "_2.csv", to_dict)
    #
    # if to_dict is None:
    #     to_dict = {"X": tmp_dict["X"], "Y": tmp_dict["Y"], "id": tmp_dict["id"]}
    # # to_dict["Blue_{0}".format(year)] = tmp_dict["SAMPLE_1"]
    # # to_dict["Green_{0}".format(year)] = tmp_dict["SAMPLE_2"]
    # # to_dict["Red_{0}".format(year)] = tmp_dict["SAMPLE_3"]
    # # to_dict["NIR_{0}".format(year)] = tmp_dict["SAMPLE_4"]
    #
    # to_dict["NDVI_{0}".format(year)] = extNDVI(tmp_dict["SAMPLE_3"], tmp_dict["SAMPLE_4"])
    # savecsv(r"M:\jpz2m\Samples\jpz2m_spl2.csv", to_dict)
    # tmp_dict = readcsv(r"M:\jpz2m\Samples\jpz2m_spl2.csv")
    # to_dict = {k: [] for k in tmp_dict}
    # for i in range(len(tmp_dict["X"])):
    #     if tmp_dict["Blue_2019"][i] != "":
    #         for k in tmp_dict:
    #             to_dict[k].append(tmp_dict[k][i])
    # savecsv(r"M:\jpz2m\Samples\jpz2m_spl3.csv", to_dict)
    #
    year_list = [14, 16, 17, 18, 19, 20, 21, 22, 23]

    def tmp1():
        csv_fn = r"M:\jpz2m\Samples\Random\jpz2m_random2.csv"
        df = pd.read_csv(csv_fn)
        to_dict = {"X": df["X"].tolist(), "Y": df["Y"].tolist(), "id": df["id"].tolist()}
        for year in year_list:
            to_dict["NDVI_{0}".format(year)] = extNDVI(
                df["Red_{0}".format(year)].tolist(), df["NIR_{0}".format(year)].tolist())
        to_dict_df = pd.DataFrame(to_dict)
        to_dict_df.to_csv(r"M:\jpz2m\Samples\Random\jpz2m_random2_ndvi.csv", index=False)
        print(to_dict_df)

    tmp1()

    def tmp2():
        year_list = [2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
        csv_fn = r"M:\jpz2m\Samples\jpz2m_spl2000_1.csv"
        df = pd.read_csv(csv_fn)
        to_dict = {"X": df["X"].tolist(), "Y": df["Y"].tolist(), "id": df["id"].tolist()}
        print(df)
        for year in year_list:
            d = (df["NDVI_{0}".format(year)] > 0.3) * 1
            to_dict["NDVI_N_{0}".format(year)] = d.tolist()
        to_dict_df = pd.DataFrame(to_dict)
        to_dict_df.to_csv(r"M:\jpz2m\Samples\jpz2m_spl2000_2.csv", index=False)
        print(to_dict_df)

    # tmp2()
    def tmp3():
        year_list = [2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
        fields = ["Blue", "Green", "Red", "NIR"]
        csv_fn = r"M:\jpz2m\Samples\jpz2m_spl2000.csv"
        df = pd.read_csv(csv_fn)
        to_dict = {"X": df["X"].tolist(), "Y": df["Y"].tolist(), "id": df["id"].tolist()}
        for year in year_list:
            d_tmp = np.zeros(len(df))
            for field in fields:
                d_tmp += df["{0}_{1}".format(field, year)].values
            to_dict["SUM_{0}".format(year)] = d_tmp.tolist()
        to_dict_df = pd.DataFrame(to_dict)
        to_dict_df.to_csv(r"M:\jpz2m\Samples\jpz2m_spl2000_3.csv", index=False)
        print(to_dict_df)

    def tmp4():
        csv_fn = r"M:\jpz2m\Samples\jpz2m_spl2000.csv"
        df = pd.read_csv(csv_fn)
        df_esa = pd.read_csv(r"M:\jpz2m\Samples\jpz2m_spl1_esa_1.csv")
        df_esa_find = pd.merge(df_esa, df, how="left", on="id")
        print(df_esa_find)
        df_esa_find.to_csv(r"M:\jpz2m\Samples\jpz2m_spl1_esa_1_1.csv")

    # tmp4()
    def tmp5():
        year_list = [2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
        csv_fn = r"M:\jpz2m\Samples\jpz2m_spl2000.csv"
        df = pd.read_csv(csv_fn)
        to_dict = {"X": df["X"].tolist(), "Y": df["Y"].tolist(), "id": df["id"].tolist()}
        for year in year_list:
            to_dict["NDWI_{0}".format(year)] = extNDWI(
                df["Red_{0}".format(year)].tolist(), df["NIR_{0}".format(year)].tolist())
        to_dict_df = pd.DataFrame(to_dict)
        to_dict_df.to_csv(r"M:\jpz2m\Samples\jpz2m_spl2000_4.csv", index=False)
        print(to_dict_df)

    # tmp5()
    def tmp6():
        year_list = [2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
        csv_fn = r"M:\jpz2m\Samples\jpz2m_spl2000_4.csv"
        df = pd.read_csv(csv_fn)
        to_dict = {"X": df["X"].tolist(), "Y": df["Y"].tolist(), "id": df["id"].tolist()}
        print(df)
        for year in year_list:
            d = (df["NDWI_{0}".format(year)] > 0.3) * 1
            to_dict["NDWI_N_{0}".format(year)] = d.tolist()
        to_dict_df = pd.DataFrame(to_dict)
        to_dict_df.to_csv(r"M:\jpz2m\Samples\jpz2m_spl2000_4_1.csv", index=False)
        print(to_dict_df)

    # tmp6()
    # year_list = [2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    # for year in year_list:
    #     print(r"copy M:\jpz2m\Images\{0}\jpz2m_{0}_2.tif M:\jpz2m\Images".format(year))
    def tmp7():
        to_dict = None
        for fn in filelist:
            year = fn.split("_")[-1].split(".")[0]
            print("20" + year)
            tmp_dict = readcsv(fn)
            if to_dict is None:
                to_dict = {"X": tmp_dict["X"], "Y": tmp_dict["Y"], "id": tmp_dict["id"]}
            to_dict["Blue_{0}".format(year)] = tmp_dict["SAMPLE_1"]
            to_dict["Green_{0}".format(year)] = tmp_dict["SAMPLE_2"]
            to_dict["Red_{0}".format(year)] = tmp_dict["SAMPLE_3"]
            to_dict["NIR_{0}".format(year)] = tmp_dict["SAMPLE_4"]
        savecsv(r"M:\jpz2m\Samples\Random\jpz2m_random2.csv", to_dict)
    # tmp7()


def method_name3():
    filelist = [r"M:\jpz2m\Samples\jpz2m_spl1_2014.csv",
                r"M:\jpz2m\Samples\jpz2m_spl1_2016.csv",
                r"M:\jpz2m\Samples\jpz2m_spl1_2017.csv",
                r"M:\jpz2m\Samples\jpz2m_spl1_2018.csv",
                r"M:\jpz2m\Samples\jpz2m_spl1_2019.csv",
                r"M:\jpz2m\Samples\jpz2m_spl1_2020.csv",
                r"M:\jpz2m\Samples\jpz2m_spl1_2021.csv",
                r"M:\jpz2m\Samples\jpz2m_spl1_2022.csv",
                r"M:\jpz2m\Samples\jpz2m_spl1_2023.csv"]

    def extNDVI(red_list, nir_list):
        red_list = [float(d) for d in red_list]
        nir_list = [float(d) for d in nir_list]
        ndvi = (np.array(nir_list) - np.array(red_list)) / (np.array(nir_list) + np.array(red_list) + 0.0000001)
        return ndvi.tolist()

    def extNDWI(green_list, nir_list):
        green_list = [float(d) for d in green_list]
        nir_list = [float(d) for d in nir_list]
        ndvi = (np.array(green_list) - np.array(nir_list)) / (np.array(green_list) + np.array(nir_list) + 0.0000001)
        return ndvi.tolist()

    # to_dict = None
    # for fn in filelist:
    #     year = fn.split("_")[-1].split(".")[0]
    #     print(year)
    #     tmp_dict = readcsv(fn)
    #     to_dict = {k: [] for k in tmp_dict}
    #     for i in range(len(tmp_dict["X"])):
    #         if tmp_dict["SAMPLE_1"][i] != "":
    #             for k in tmp_dict:
    #                 to_dict[k].append(tmp_dict[k][i])
    #     savecsv(os.path.splitext(fn)[0] + "_2.csv", to_dict)
    #
    # if to_dict is None:
    #     to_dict = {"X": tmp_dict["X"], "Y": tmp_dict["Y"], "id": tmp_dict["id"]}
    # # to_dict["Blue_{0}".format(year)] = tmp_dict["SAMPLE_1"]
    # # to_dict["Green_{0}".format(year)] = tmp_dict["SAMPLE_2"]
    # # to_dict["Red_{0}".format(year)] = tmp_dict["SAMPLE_3"]
    # # to_dict["NIR_{0}".format(year)] = tmp_dict["SAMPLE_4"]
    #
    # to_dict["NDVI_{0}".format(year)] = extNDVI(tmp_dict["SAMPLE_3"], tmp_dict["SAMPLE_4"])
    # savecsv(r"M:\jpz2m\Samples\jpz2m_spl2.csv", to_dict)
    # tmp_dict = readcsv(r"M:\jpz2m\Samples\jpz2m_spl2.csv")
    # to_dict = {k: [] for k in tmp_dict}
    # for i in range(len(tmp_dict["X"])):
    #     if tmp_dict["Blue_2019"][i] != "":
    #         for k in tmp_dict:
    #             to_dict[k].append(tmp_dict[k][i])
    # savecsv(r"M:\jpz2m\Samples\jpz2m_spl3.csv", to_dict)
    def tmp1():
        year_list = [2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
        csv_fn = r"M:\jpz2m\Samples\jpz2m_spl2000.csv"
        df = pd.read_csv(csv_fn)
        to_dict = {"X": df["X"].tolist(), "Y": df["Y"].tolist(), "id": df["id"].tolist()}
        for year in year_list:
            to_dict["NDVI_{0}".format(year)] = extNDVI(
                df["Red_{0}".format(year)].tolist(), df["NIR_{0}".format(year)].tolist())
        to_dict_df = pd.DataFrame(to_dict)
        to_dict_df.to_csv(r"M:\jpz2m\Samples\jpz2m_spl2000_1.csv", index=False)
        print(to_dict_df)

    def tmp2():
        year_list = [2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
        csv_fn = r"M:\jpz2m\Samples\jpz2m_spl2000_1.csv"
        df = pd.read_csv(csv_fn)
        to_dict = {"X": df["X"].tolist(), "Y": df["Y"].tolist(), "id": df["id"].tolist()}
        print(df)
        for year in year_list:
            d = (df["NDVI_{0}".format(year)] > 0.3) * 1
            to_dict["NDVI_N_{0}".format(year)] = d.tolist()
        to_dict_df = pd.DataFrame(to_dict)
        to_dict_df.to_csv(r"M:\jpz2m\Samples\jpz2m_spl2000_2.csv", index=False)
        print(to_dict_df)

    # tmp2()
    def tmp3():
        year_list = [2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
        fields = ["Blue", "Green", "Red", "NIR"]
        csv_fn = r"M:\jpz2m\Samples\jpz2m_spl2000.csv"
        df = pd.read_csv(csv_fn)
        to_dict = {"X": df["X"].tolist(), "Y": df["Y"].tolist(), "id": df["id"].tolist()}
        for year in year_list:
            d_tmp = np.zeros(len(df))
            for field in fields:
                d_tmp += df["{0}_{1}".format(field, year)].values
            to_dict["SUM_{0}".format(year)] = d_tmp.tolist()
        to_dict_df = pd.DataFrame(to_dict)
        to_dict_df.to_csv(r"M:\jpz2m\Samples\jpz2m_spl2000_3.csv", index=False)
        print(to_dict_df)

    def tmp4():
        csv_fn = r"M:\jpz2m\Samples\jpz2m_spl2000.csv"
        df = pd.read_csv(csv_fn)
        df_esa = pd.read_csv(r"M:\jpz2m\Samples\jpz2m_spl1_esa_1.csv")
        df_esa_find = pd.merge(df_esa, df, how="left", on="id")
        print(df_esa_find)
        df_esa_find.to_csv(r"M:\jpz2m\Samples\jpz2m_spl1_esa_1_1.csv")

    # tmp4()
    def tmp5():
        year_list = [2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
        csv_fn = r"M:\jpz2m\Samples\jpz2m_spl2000.csv"
        df = pd.read_csv(csv_fn)
        to_dict = {"X": df["X"].tolist(), "Y": df["Y"].tolist(), "id": df["id"].tolist()}
        for year in year_list:
            to_dict["NDWI_{0}".format(year)] = extNDWI(
                df["Red_{0}".format(year)].tolist(), df["NIR_{0}".format(year)].tolist())
        to_dict_df = pd.DataFrame(to_dict)
        to_dict_df.to_csv(r"M:\jpz2m\Samples\jpz2m_spl2000_4.csv", index=False)
        print(to_dict_df)

    # tmp5()
    def tmp6():
        year_list = [2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
        csv_fn = r"M:\jpz2m\Samples\jpz2m_spl2000_4.csv"
        df = pd.read_csv(csv_fn)
        to_dict = {"X": df["X"].tolist(), "Y": df["Y"].tolist(), "id": df["id"].tolist()}
        print(df)
        for year in year_list:
            d = (df["NDWI_{0}".format(year)] > 0.3) * 1
            to_dict["NDWI_N_{0}".format(year)] = d.tolist()
        to_dict_df = pd.DataFrame(to_dict)
        to_dict_df.to_csv(r"M:\jpz2m\Samples\jpz2m_spl2000_4_1.csv", index=False)
        print(to_dict_df)

    tmp6()
    # year_list = [2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    # for year in year_list:
    #     print(r"copy M:\jpz2m\Images\{0}\jpz2m_{0}_2.tif M:\jpz2m\Images".format(year))


def method_name2():
    zy5m_grw = ZY5MGDALRasterWarp()
    year = 2023
    zy5m_grw.warpImage(
        coor_csv_fn=r"M:\jpz2m\Images\{0}\spl{0}.csv".format(year),
        raster_fn=r"M:\jpz2m\Images\{0}\jpz2m_{0}.dat".format(year),
        to_fn=r"M:\jpz2m\Images\{0}\jpz2m_{0}_2.tif".format(year),
    )


def method_name():
    # 影像融合
    filelist_fn = r"M:\jpz2m\filelist.txt"
    cc_fn = r"M:\jpz2m\geometry3\geometry3_kuo_tindex.shp"
    cc_name = "geometry3_kuo_tindex"
    to_dirname = r"M:\jpz2m\Images"

    def add_yh(filename):
        return "\"" + filename + "\""

    filelist = readLines(filelist_fn, strip_str="\n")
    for fn in filelist:
        dirname = os.path.split(fn)[0]
        mss_fn, pan_fn = "", ""
        for fn2 in os.listdir(dirname):
            fn2 = os.path.join(dirname, fn2)
            if getfilext(fn2) == ".tiff":
                if "MSS" in fn2:
                    mss_fn = fn2
                if "MUX" in fn2:
                    mss_fn = fn2
                if "PAN" in fn2:
                    pan_fn = fn2
        # print("gdalwarp -overwrite -t_srs EPSG:4326 -r bilinear -rpc -of vrt", add_yh(mss_fn), add_yh(mss_fn + ".vrt"))
        # print("gdalwarp -overwrite -t_srs EPSG:4326 -r bilinear -rpc -of vrt", add_yh(pan_fn), add_yh(pan_fn + ".vrt"))
        # print("gdalwarp -overwrite -r near -cutline", add_yh(cc_fn), "-cl", cc_name, "-crop_to_cutline -of ENVI",
        #       add_yh(mss_fn + ".vrt"), add_yh(mss_fn + ".dat"))
        # print("gdalwarp -overwrite -r near -cutline", add_yh(cc_fn), "-cl", cc_name, "-crop_to_cutline -of ENVI",
        #       add_yh(pan_fn + ".vrt"), add_yh(pan_fn + ".dat"))
        # print("gdal_pansharpen_r -r nearest -of ENVI", add_yh(pan_fn + ".dat"), add_yh(mss_fn + ".dat"),
        #       add_yh(mss_fn + "_rh.dat"))
        to_fn = add_yh(mss_fn + "_rh.dat")
        to_fn2 = add_yh(os.path.join(to_dirname, "jpz2m_" + to_fn.split("\\")[2] + ".dat"))
        print("gdal_translate -of ENVI", to_fn, to_fn2)


if __name__ == "__main__":
    main()
