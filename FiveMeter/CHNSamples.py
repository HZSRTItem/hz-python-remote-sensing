# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : CHNSamples.py
@Time    : 2024/9/12 19:59
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of CHNSamples
-----------------------------------------------------------------------------"""
import random

import pandas as pd

from SRTCodes.GDALUtils import GDALRastersSampling


def main():
    def func1():
        grs = GDALRastersSampling(
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_30N_115E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_35N_100E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_35N_105E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_35N_110E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_35N_115E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_35N_120E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_40N_90E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_40N_95E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_40N_100E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_40N_105E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_40N_110E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_40N_115E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_40N_120E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_45N_90E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_45N_95E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_45N_100E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_45N_110E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_45N_115E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_45N_120E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_45N_125E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_45N_130E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_50N_120E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_50N_125E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_50N_130E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_50N_135E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_55N_120E.tif",
            r"F:\ChinaNorthIS\Run\GISA10m\GISA-10m_55N_125E.tif",
        )
        df = pd.read_csv(r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_2.csv")
        data = grs.samplingXY(df["X"].tolist(), df["Y"].tolist(), is_trans=True)
        df["GAIA10m"] = data.ravel()
        df.to_csv(r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_2_GISA10.csv", index=False)
        print(df.head(10))

    def func2():
        spls = [
            *pd.read_csv(r"F:\ChinaNorthIS\Run\Samples\1\chn_spl21_GAIA10m.csv").to_dict("records"),
            *pd.read_csv(r"F:\ChinaNorthIS\Run\Samples\1\chn_spl31_GAIA10m.csv").to_dict("records"),
        ]
        i = 1
        for spl in spls:
            spl["SRT"] = i
            i += 1
        df = pd.DataFrame(spls)
        print(len(df))
        print(df.head(10))
        df.to_csv(r"F:\ChinaNorthIS\Run\Samples\1\chn_spl_GAIA10m.csv", index=False)

    def func3():
        category_field_name = "GAIA10m"
        category_select = {0: 10000, 1: 10000}
        df = pd.read_csv(r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_1_2_GISA10.csv")
        samples = df.to_dict("records")
        random.shuffle(samples)
        datas = []
        for data_select, n in category_select.items():
            print(data_select, n )
            n_tmp = 0
            for spl in samples:
                if n_tmp < n:
                    if spl[category_field_name] == data_select:
                        datas.append(spl)
                        n_tmp += 1
                else:
                    break

        df = pd.DataFrame(datas)
        print("len df", len(df))
        print(df.head(10))
        df.to_csv(r"F:\ChinaNorthIS\Run\Samples\2\sdspl2_1_2_GISA10_1.csv", index=False)

    return func3()


if __name__ == "__main__":
    main()
