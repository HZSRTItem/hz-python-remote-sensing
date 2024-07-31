# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowWrite.py
@Time    : 2024/7/27 16:43
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of ShadowWrite
-----------------------------------------------------------------------------"""
import pandas as pd
from matplotlib import pyplot as plt

from SRTCodes.GDALDraw import GDALDrawImage
from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.NumpyUtils import update10Log10
from SRTCodes.Utils import DirFileName


def main():
    dfn = DirFileName(r"F:\ProjectSet\Shadow\Release")
    sar_mean_dfn = DirFileName(r"F:\ASDEWrite\Images\SARMean")

    def func1():
        fn = dfn.fn("ChengDuImages", "SH_CD_envi.dat")
        gr = GDALRaster(fn)

        def read_data(_name):
            _data = gr.readGDALBand(_name)
            print("{:>10} {:>10.3f} {:>10.3f} {:>10.3f}".format(_name, _data.min(), _data.max(), _data.mean()))
            return _data

        def mean(_data1, _data2):
            # _data = (_data1 + _data2) / 2
            # _data = update10Log10(_data)
            _data1 = update10Log10(_data1)
            _data2 = update10Log10(_data2)
            _data = (_data1 + _data2) / 2
            print("{:>10} {:>10.3f} {:>10.3f} {:>10.3f}".format(" ", _data.min(), _data.max(), _data.mean()))
            return _data

        as_vv = read_data("AS_VV")
        as_vh = read_data("AS_VH")
        de_vv = read_data("DE_VV")
        de_vh = read_data("DE_VH")
        as_mean = mean(as_vv, as_vh)
        de_mean = mean(de_vv, de_vh)

        gr.save(as_mean, sar_mean_dfn.fn("cd_as_mean2.tif"), fmt="GTiff")
        gr.save(de_mean, sar_mean_dfn.fn("cd_de_mean2.tif"), fmt="GTiff")

    def func2():
        gdi = GDALDrawImage((17, 17))
        qd = gdi.addGR(dfn.fn("BeiJingImages", "SH_BJ_envi.dat"), dfn.fn("BeiJingImages", "SH_BJ_look_envi.range"))
        bj = gdi.addGR(dfn.fn("QingDaoImages", "SH_QD_envi.dat"), dfn.fn("QingDaoImages", "SH_QD_look_envi.range"))
        cd = gdi.addGR(dfn.fn("ChengDuImages", "SH_CD_envi.dat"), dfn.fn("ChengDuImages", "SH_CD_look_envi.range"))
        gdi.addRCC("RGB", qd, bj, cd, channel_list=["Red", "Green", "Blue"])
        gdi.addRCC("NRG", qd, bj, cd, channel_list=["NIR", "Red", "Green"])
        gdi.addRCC(
            "AS_SAR",
            sar_mean_dfn.fn("bj_as_mean2.tif"),
            sar_mean_dfn.fn("qd_as_mean2.tif"),
            sar_mean_dfn.fn("cd_as_mean2.tif"),
            channel_list=[0])
        gdi.addRCC(
            "DE_SAR",
            sar_mean_dfn.fn("bj_de_mean2.tif"),
            sar_mean_dfn.fn("qd_de_mean2.tif"),
            sar_mean_dfn.fn("cd_de_mean2.tif"),
            channel_list=[0])
        gdi.addRCC(
            "GI",
            r"F:\ProjectSet\Shadow\MkTu\4.1Details\BingImages\bj_googleimages.mbtiles",
            r"F:\ProjectSet\Shadow\MkTu\4.1Details\BingImages\cd_googleimages.mbtiles",
            r"F:\ProjectSet\Shadow\MkTu\4.1Details\BingImages\qd_googleimages_wgs84.dat",
            channel_list=[0, 1, 2], win_size=(300, 300)
        )

        fig = plt.figure(figsize=(12, 9), )
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, hspace=0.04, wspace=0.03)
        n_rows, n_columns = 1,2

        def column(n, x, y):
            plt.subplot(n_rows, n_columns, n)
            gdi.readDraw("NRG", x, y, min_list=[100, 100, 100], max_list=[3000, 2000, 2000])
            plt.subplot(n_rows, n_columns, n + n_rows * 1)
            gdi.readDraw("GI", x, y, is_trans=False, min_list=[0, 0, 0], max_list=[255, 255, 255])
            # plt.subplot(n_rows, n_columns, n + n_rows * 2)
            # gdi.readDraw("AS_SAR", x, y, min_list=[-16], max_list=[3.5])
            # plt.subplot(n_rows, n_columns, n + n_rows * 3)
            # gdi.readDraw("DE_SAR", x, y, min_list=[-16], max_list=[3.5])

        column(1, 120.3077294,36.0667493, )
        # column(2, 116.431408, 39.892027, )
        # column(3, 116.431408, 39.892027, )
        # column(4, 116.431408, 39.892027, )

        plt.show()

    return func2()


def method_name1():
    csv_fn = r"F:\ASDEWrite\Result\QingDao\qd_data_spl.csv"
    # csv_fn = r"F:\ASDEWrite\Result\BeiJing\HSPL_BJ_select.csv"
    # csv_fn = r"F:\ASDEWrite\Result\ChengDu\HSPL_CD_select.csv"
    csv_fns = [
        r"F:\ASDEWrite\Result\QingDao\qd_data_spl.csv",
        r"F:\ASDEWrite\Result\BeiJing\HSPL_BJ_select.csv",
        r"F:\ASDEWrite\Result\ChengDu\HSPL_CD_select.csv",
    ]

    def func1():
        df = pd.read_csv(csv_fn)
        df = df[df["TEST"] == 1]
        df = df[df["OS"] == 1]
        counts_dict = df["CNAME"].value_counts().to_dict()
        names = [
            'IS', 'IS_SH', 'IS_AS_SH', 'IS_DE_SH',
            'VEG', 'VEG_SH', 'VEG_AS_SH', 'VEG_DE_SH',
            'SOIL', 'SOIL_SH', 'SOIL_AS_SH', 'SOIL_DE_SH',
            'WAT', 'WAT_SH', 'WAT_AS_SH', 'WAT_DE_SH'
        ]

        def print_line(*_names):
            print(*[counts_dict[name] for name in _names])

        print_line('IS', 'IS_SH', 'IS_AS_SH', 'IS_DE_SH', )
        print_line('VEG', 'VEG_SH', 'VEG_AS_SH', 'VEG_DE_SH')
        print_line('SOIL', 'SOIL_SH', 'SOIL_AS_SH', 'SOIL_DE_SH', )
        print_line('WAT', 'WAT_SH', 'WAT_AS_SH', 'WAT_DE_SH')

    def func2():
        for _csv_fn in csv_fns:
            df = pd.read_csv(_csv_fn)
            df = df[df["TEST"] == 1]
            for _spl_type in ["HS", "OS", "NS"]:
                _df = df[df[_spl_type] == 1]
                counts_dict = _df["CNAME"].value_counts().to_dict()

                def _sum(*_names):
                    return sum(counts_dict[name] for name in _names if name in counts_dict)

                print(
                    _spl_type,
                    _sum("IS", "VEG", "SOIL", "WAT"),
                    _sum("IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH"),
                    _sum("IS_AS_SH", "VEG_AS_SH", "SOIL_AS_SH", "WAT_AS_SH"),
                    _sum("IS_DE_SH", "VEG_DE_SH", "SOIL_DE_SH", "WAT_DE_SH"),
                )

    def func3():
        for _csv_fn in csv_fns:
            df = pd.read_csv(_csv_fn)
            df = df[df["TEST"] == 0]
            for _spl_type in ["TEST_IS", "TEST_SH"]:
                _df = df[df[_spl_type] == 1]
                counts_dict = _df["CNAME"].value_counts().to_dict()

                def _sum(*_names):
                    return sum(counts_dict[name] for name in _names if name in counts_dict)

                print(
                    _spl_type,
                    _sum("IS", "IS_SH"),
                    _sum("VEG", "VEG_SH"),
                    _sum("SOIL", "SOIL_SH"),
                    _sum("WAT", "WAT_SH"),
                )

    func3()


if __name__ == "__main__":
    main()
