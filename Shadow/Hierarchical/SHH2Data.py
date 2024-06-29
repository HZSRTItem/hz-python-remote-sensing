# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2Data.py
@Time    : 2024/6/20 17:22
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2Data
-----------------------------------------------------------------------------"""

import numpy as np
from osgeo import gdal

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.NumpyUtils import eig, update10EDivide10, eig2
from SRTCodes.Utils import DirFileName
from Shadow.Hierarchical import SHH2Config
from Shadow.ShadowGeoDraw import _10log10
from Shadow.ShadowRaster import ShadowRasterGLCM


def main():
    method_name2()
    return


def method_name6():
    gr = GDALRaster(SHH2Config.BJ_ENVI_FN)

    def data_deal(x, x_min, x_max):
        x = np.clip(x, x_min, x_max)
        x = (x - x_min) / (x_max - x_min) * 255
        return x

    data = np.array([
        data_deal(gr.readGDALBand("NIR"), 99.0098, 3231.99),
        data_deal(gr.readGDALBand("Red"), 187.016, 2345.98),
        data_deal(gr.readGDALBand("Green"), 390.027, 2019.96),
        # data_deal(gr.readGDALBand("Blue"), 355.984, 2053.96),
        # np.ones((gr.n_rows, gr.n_columns)) * 255,
    ]).astype("int8")
    to_fn = r"F:\ProjectSet\Shadow\Hierarchical\Images\SH22_BJ_NRG.tif"
    gr.save(data, to_fn, fmt="GTiff", dtype=gdal.GDT_Byte, descriptions=["Red", "Green", "Blue"])
    # ds: gdal.Dataset = gdal.Open(to_fn, gdal.GA_Update)
    # red_band = ds.GetRasterBand(3)
    # red_band.SetRasterColorInterpretation(gdal.GCI_RedBand)
    # green_band = ds.GetRasterBand(2)
    # green_band.SetRasterColorInterpretation(gdal.GCI_GreenBand)
    # blue_band = ds.GetRasterBand(1)
    # blue_band.SetRasterColorInterpretation(gdal.GCI_BlueBand)


def method_name5():
    dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Images")
    grs = {
        # "QingDao": SHH2Config.QD_GR(),
        # "BeiJing": SHH2Config.BJ_GR(),
        # "ChengDu": SHH2Config.CD_GR(),

        "QingDao": GDALRaster(SHH2Config.QD_LOOK_FN),
        "BeiJing": GDALRaster(SHH2Config.BJ_LOOK_FN),
        "ChengDu": GDALRaster(SHH2Config.CD_LOOK_FN),
    }
    for city_name, gr in grs.items():
        to_dfn = DirFileName(dfn.fn(city_name, "SH22", "Channels"))
        to_dfn.mkdir()
        for name in gr.names:
            to_fn = to_dfn.fn("{}_{}.tif".format(city_name, name))
            print("gdaladdo", to_fn)
            data = gr.readGDALBand(name)
            gr.save(data, to_fn, fmt="GTiff", descriptions=[name])


def method_name4():
    def func1():
        r"""
        "E:\ImageData\GLCM\Opt\beijing_gray_mean"
        "E:\ImageData\GLCM\Opt\chengdu_gray_mean"
        "E:\ImageData\GLCM\Opt\qingdao_gray_mean"
        """

        names = ['OPT_mean', 'OPT_var', 'OPT_hom', 'OPT_con', 'OPT_dis', 'OPT_ent', 'OPT_asm', 'OPT_cor']

        gr = GDALRaster(SHH2Config.BJ_ENVI_FN, open_type=gdal.GA_Update)
        gr_glcm = GDALRaster(r"E:\ImageData\GLCM\Opt\beijing_gray_mean")
        print(gr_glcm.names)

        for name in names:
            data1 = gr.readGDALBand(name)
            data2 = gr_glcm.readGDALBand(name)
            # gr.updateData(name, data2.astype("float32"))
            print(name)
            print("data1", np.min(data1), np.max(data1), np.mean(data1))
            print("data2", np.min(data2), np.max(data2), np.mean(data2))
            print("-" * 60)

    def func2():
        dfn = DirFileName(r"E:\ImageData\GLCM")

        def city(city_name, gr):
            for name in ["AS_VV", "AS_VH", "DE_VV", "DE_VH"]:
                gr_sar = GDALRaster(dfn.fn(city_name, "{}_{}_mean".format(city_name, name)))
                print("-" * 30, city_name, name, "-" * 30)
                print(gr_sar.gdal_raster_fn)
                print(gr.gdal_raster_fn)
                for gr_name in gr_sar.names:
                    print("-" * 60)
                    data1 = gr.readGDALBand(gr_name)
                    data2 = gr_sar.readGDALBand(gr_name)
                    print(gr_name)
                    # gr.updateData(gr_name, data2.astype("float32"))
                    print("data1", np.min(data1), np.max(data1), np.mean(data1))
                    print("data2", np.min(data2), np.max(data2), np.mean(data2))

        city("QD", GDALRaster(SHH2Config.QD_ENVI_FN, open_type=gdal.GA_Update))
        city("BJ", GDALRaster(SHH2Config.BJ_ENVI_FN, open_type=gdal.GA_Update))
        city("CD", GDALRaster(SHH2Config.CD_ENVI_FN, open_type=gdal.GA_Update))

    def func3():
        dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Images\HA")

        def city(city_name, gr):
            for name in ["AS_H", "AS_Alpha", "AS_A", "DE_H", "DE_Alpha", "DE_A", ]:
                fn = dfn.fn(city_name, "{}.dat".format(name))
                gr_update = GDALRaster(fn)
                print("-" * 30, city_name, name, "-" * 30)
                print(fn)
                print(gr.gdal_raster_fn)
                for gr_name in gr_update.names:
                    print("-" * 60)
                    data1 = gr.readGDALBand(gr_name)
                    data2 = gr_update.readGDALBand(1)
                    # gr.updateData(gr_name, data2.astype("float32"))
                    print("data1", np.min(data1), np.max(data1), np.mean(data1))
                    print("data2", np.min(data2), np.max(data2), np.mean(data2))

        city("QD", GDALRaster(SHH2Config.QD_ENVI_FN, open_type=gdal.GA_Update))
        city("BJ", GDALRaster(SHH2Config.BJ_ENVI_FN, open_type=gdal.GA_Update))
        city("CD", GDALRaster(SHH2Config.CD_ENVI_FN, open_type=gdal.GA_Update))

    def func4():

        def city(city_name, gr):
            dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Images\{}\SH22\Channels".format(city_name))
            for name in ["AS_VV", "AS_VH", "DE_VV", "DE_VH"]:
                fn = dfn.fn("{}_{}.tif".format(city_name, name))
                gr_update = GDALRaster(fn)
                print("-" * 30, city_name, name, "-" * 30)
                print(fn)
                print(gr.gdal_raster_fn)
                for gr_name in gr_update.names:
                    print("-" * 30, gr_name, "-" * 30)
                    data1 = gr.readGDALBand(gr_name)
                    data2 = gr_update.readGDALBand(1)
                    # if city_name == "BeiJing":
                    #     data2 = _10log10(data2)
                    # gr.updateData(gr_name, data2.astype("float32"))
                    print("data1", np.min(data1), np.max(data1), np.mean(data1))
                    print("data2", np.min(data2), np.max(data2), np.mean(data2))

        city("QingDao", GDALRaster(SHH2Config.QD_ENVI_FN, open_type=gdal.GA_Update))
        city("BeiJing", GDALRaster(SHH2Config.BJ_ENVI_FN, open_type=gdal.GA_Update))
        city("ChengDu", GDALRaster(SHH2Config.CD_ENVI_FN, open_type=gdal.GA_Update))

    func4()


def method_name3():
    # raster_fn = SHH2Config.CD_ENVI_FN
    # gr = GDALRaster(raster_fn)
    #
    # red = gr.readGDALBand("Red")
    # green = gr.readGDALBand("Green")
    # blue = gr.readGDALBand("Blue")
    #
    # gray = red * 0.3 + green * 0.59 + blue * 0.11
    #
    # gr.save(gray, r"E:\ImageData\GLCM\Opt\chengdu_gray.dat")
    srg = ShadowRasterGLCM()
    # srg.meanFourDirection("qingdao_gray", "OPT_", r"E:\ImageData\GLCM\Opt")
    # srg.meanFourDirection("beijing_gray", "OPT_", r"E:\ImageData\GLCM\Opt")
    # srg.meanFourDirection("chengdu_gray", "OPT_", r"E:\ImageData\GLCM\Opt")
    dfn = DirFileName(r"E:\ImageData\GLCM")

    def func1(city_name):
        srg.meanFourDirection("{}_AS_VV".format(city_name), "AS_VV_", dfn.fn(city_name))
        srg.meanFourDirection("{}_AS_VH".format(city_name), "AS_VH_", dfn.fn(city_name))
        srg.meanFourDirection("{}_DE_VV".format(city_name), "DE_VV_", dfn.fn(city_name))
        srg.meanFourDirection("{}_DE_VH".format(city_name), "DE_VH_", dfn.fn(city_name))

    func1("QD")
    func1("BJ")
    func1("CD")


def method_name2():
    gr = GDALRaster(SHH2Config.QD_ENVI_FN)

    def read(name):
        _data = gr.readGDALBand(name)
        print(name, _data.min(), _data.max())
        _data = update10EDivide10(_data)
        print(name, _data.min(), _data.max())
        return _data

    as1 = read("AS_H")
    as2 = read("AS_Alpha")
    de1 = read("DE_H")
    de2 = read("DE_Alpha")

    e1, e2, v11, v12, v21, v22 = eig2(as1 * as1, de2 * as2, as2 * de2, de1 * de1)
    # print(e1.min(), e1.max())
    print(e2.min(), e2.max())
    # e1 = _10log10(e1)
    e2 = _10log10(e2)
    # print(e1.min(), e1.max())
    print(e2.min(), e2.max())
    dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Analysis\4")
    # gr.save(e1, dfn.fn("c2_e1.dat"))
    gr.save(e2, dfn.fn("ha_e2.dat"))
    # gr.save(v11, dfn.fn("v11.dat"))
    # gr.save(v12, dfn.fn("v12.dat"))
    # gr.save(v21, dfn.fn("v21.dat"))
    # gr.save(v22, dfn.fn("v22.dat"))


def method_name1():
    def func1():
        gr = GDALRaster(r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\SHH2_CD2_envi.dat")
        c11_data = update10EDivide10(gr.readGDALBand("AS_C11"))
        c12_imag_data = gr.readGDALBand("AS_C12_imag")
        c12_real_data = gr.readGDALBand("AS_C12_real")
        c22_data = update10EDivide10(gr.readGDALBand("AS_C22"))

        # dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Temp")
        # gr.save(c11_data, dfn.fn("AS_C11.tif"), fmt="GTiff", dtype=gdal.GDT_Float32)
        # gr.save(c12_imag_data, dfn.fn("AS_C12_imag.tif"), fmt="GTiff", dtype=gdal.GDT_Float32)
        # gr.save(c12_real_data, dfn.fn("AS_C12_real.tif"), fmt="GTiff", dtype=gdal.GDT_Float32)
        # gr.save(c22_data, dfn.fn("AS_C22.tif"), fmt="GTiff", dtype=gdal.GDT_Float32)

        def data_des(_data):
            print(_data.min(), _data.max(), _data.mean())

        data_des(c11_data)
        data_des(c12_imag_data)
        data_des(c12_real_data)
        data_des(c22_data)

        lamd1, lamd2, vec1, vec2 = eig(c11_data, 2 * c12_real_data, 2 * c12_imag_data, 4 * c22_data, True)
        e1, e2, v11, v12, v21, v22 = eig2(
            c11_data,
            2 * (c12_real_data + c12_imag_data * 1j),
            2 * (c12_real_data - c12_imag_data * 1j),
            4 * c22_data,
        )

        lamd1_data = update10EDivide10(gr.readGDALBand("AS_Lambda1"))
        lamd2_data = update10EDivide10(gr.readGDALBand("AS_Lambda2"))

        i = 100
        j = 200

        c2 = np.array([
            [c11_data[i, j], 2 * c12_real_data[i, j] + 2 * (c12_imag_data[i, j] * 1j)],
            [2 * c12_real_data[i, j] - 2 * (c12_imag_data[i, j] * 1j), 4 * c22_data[i, j]],
        ])
        print(c2)
        eigenvalue, featurevector = np.linalg.eig(c2)

        d = np.diag([e1[i, j], e2[i, j]])
        v = np.array([[v11[i, j], v12[i, j]], [v21[i, j], v22[i, j]]])

        print("eigenvalue", eigenvalue)
        print("featurevector", featurevector)

        print("lamd1_data[i, j], lamd2_data[i, j]", lamd1_data[i, j], lamd2_data[i, j])
        print("lamd1[i, j], lamd2[i, j], vec1[i, j], vec2[i, j]", lamd1[i, j], lamd2[i, j], vec1[i, j], vec2[i, j])

        alp1 = np.arccos(abs(featurevector[0, 0]))
        alp2 = np.arccos(abs(featurevector[0, 1]))
        print(np.rad2deg(alp1), np.rad2deg(alp2))

        alp1 = np.arccos(abs(vec1[i, j]))
        alp2 = np.arccos(abs(vec2[i, j]))

        print(np.rad2deg(alp1), np.rad2deg(alp2))

        return

    def func2():
        import sympy
        a, b, c, d = sympy.symbols('a b c d')
        A = sympy.Matrix([[a, b + c * 1j], [b - c * 1j, d]])
        eig = A.eigenvals()
        print(eig)
        vec1, vec2 = A.eigenvects()

        def func21(_vec):
            print("0:", _vec[0])
            print("1:", _vec[1])
            print("2:", _vec[2][0])

        func21(vec1)
        func21(vec2)

        return

    def func3():
        eig(1, 2, 3, 4)

    # func2()
    func1()


if __name__ == "__main__":
    main()
