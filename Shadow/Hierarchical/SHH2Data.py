# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2Data.py
@Time    : 2024/6/20 17:22
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2Data
-----------------------------------------------------------------------------"""
import os

import numpy as np
from osgeo import gdal

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.NumpyUtils import eig, update10EDivide10, eig2, update10Log10, calPCA, calHSV
from SRTCodes.Utils import DirFileName, readJson
from Shadow.Hierarchical import SHH2Config
from Shadow.ShadowGeoDraw import _10log10
from Shadow.ShadowRaster import ShadowRasterGLCM


def show1(_name, _data):
    print("{:>6} {:>15.3f}  {:>15.3f}".format(_name, _data.min(), _data.max()))
    return _data


def main():
    gr = GDALRaster(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\qd_1.tif")
    gr_save = GDALRaster(SHH2Config.QD_ENVI_FN)
    data = gr.readGDALBand(1)
    print(data.shape)
    to_data = np.zeros((gr_save.n_rows, gr_save.n_columns))
    print(to_data.shape)
    to_data = data[:to_data.shape[0], :to_data.shape[1]]
    gr.save(
        to_data,
        r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\data1.tif",
        dtype=gdal.GDT_Float32,
        fmt="GTiff",
    )
    return


def method_name7():
    dfn = DirFileName(r"F:\ProjectSet\Shadow\ASDEIndex\Images\2").mkdir()
    gr = GDALRaster(SHH2Config.QD_ENVI_FN)

    def read(_name):
        _data = show1(_name, gr.readGDALBand(_name))
        _data = show1(_name, update10EDivide10(_data))
        return _data

    as1 = read("AS_VV")
    as2 = read("AS_VH")
    de1 = read("DE_VV")
    de2 = read("DE_VH")
    # data = np.sqrt(as1 * as2 * de1 * de2)
    # show1("E3", data)
    # gr.save(data.astype("float32"), dfn.fn("e3.tif"), fmt="GTiff", dtype=gdal.GDT_Float32)
    # data = update10Log10(data)
    # show1("E3_10log10", data)
    # gr.save(data.astype("float32"), dfn.fn("e3_10log10.tif"), fmt="GTiff", dtype=gdal.GDT_Float32)
    data = update10Log10(np.sqrt(as1 * as2))
    show1("E32", data)
    gr.save(data.astype("float32"), dfn.fn("e32.tif"), fmt="GTiff", dtype=gdal.GDT_Float32)


def featExt():
    # Water index
    gr = GDALRaster(SHH2Config.QD_ENVI_FN)
    print(gr.names)
    gr_water9 = GDALRaster(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\Water9.tif")
    gr_data1 = GDALRaster(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\data1.tif")
    to_dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\Index")
    range_dict = readJson(SHH2Config.QD_RANGE_FN)
    eps = 0.0000001
    data_dict = {}

    def add_data(_name, _data):
        data_dict[_name] = _data
        show1(_name, _data)
        return _data

    def read(_name):
        _data = gr.readGDALBand(_name)
        show1(_name, _data)
        return _data

    def norm(_data1, _data2):
        return (_data1 - _data2) / (_data1 + _data2 + eps)

    def cal_pca(_data):
        _data_shape = _data.shape
        to_data = calPCA(np.reshape(_data, (_data_shape[0], -1)))
        to_data = np.reshape(to_data[2].T, _data_shape)
        return to_data

    def data_range(name, _data):
        _data = np.clip(_data, range_dict[name]["min"], range_dict[name]["max"])
        _data = (_data - range_dict[name]["min"]) / (range_dict[name]["max"] - range_dict[name]["min"])
        return _data

    data_blue = read("Blue")
    data_green = read("Green")
    data_red = read("Red")
    data_nir = read("NIR")
    data_swir1 = read("SWIR1")
    data_swir2 = read("SWIR2")

    print("-" * 60)
    # A method for extracting small water bodies based on DEM and remote sensing images
    # The use of normalized difference water index (NDWI) in the delineation of open water features
    ndwi = add_data("NDWI", norm(data_green, data_nir))
    # Extracting Miyun reservoir’s water area and monitoring its change based on a revised normalized different water index
    rndwi = add_data("RNDWI", norm(data_swir2, data_red))
    mndwi = add_data("MNDWI", norm(data_green, data_swir1))
    mbwi = add_data("MBWI", 2 * data_green - data_red - data_nir - data_swir1 - data_swir2)

    print("-" * 60)
    ndvi = add_data("NDVI", norm(data_nir, data_red))
    savi = add_data("SAVI", 1.5 * ndvi + 0.5)

    print("-" * 60)
    # ASI: An artificial surface Index for Landsat 8 imagery." International Journal of Applied Earth Observation and Geoinformation
    ui = add_data("NDBI", norm(data_swir2, data_nir))
    ndbi = add_data("NDBI", norm(data_swir1, data_nir))
    nbi = add_data("NBI", data_red * data_swir2 / data_nir)
    mbi = add_data("MBI", (data_swir1 * data_red - data_swir2 * data_swir2) / (data_red + data_nir + data_swir1))

    print("-" * 60)
    water9 = gr_water9.readGDALBand(1)
    data1 = gr_data1.readGDALBand(1)
    sei = add_data("SEI", norm(data1 + water9, data_green + data_nir))
    csi = add_data("CSI", sei - ndvi)
    rgb = np.concatenate([
        [data_range("Red", data_red)],
        [data_range("Green", data_green)],
        [data_range("Blue", data_blue)],
    ])
    hsv = calHSV(rgb)
    nsvdi = add_data("NSVDI", norm(hsv[1], hsv[2]))
    # pca = cal_pca(rgb)
    # hsi = calHSI(rgb)
    # si = add_data("SI", (pca[0] - hsi[2]) * (1 + hsi[1]) / (pca[0] + hsi[2] + hsi[1]))

    print(data_dict.keys())

    for name in data_dict:
        to_fn = to_dfn.fn("{}.tif".format(name))
        print(to_fn)
        if os.path.isfile(to_fn):
            os.remove(to_fn)
        gr.save(data_dict[name].astype("float32"), to_fn, fmt="GTiff", dtype=gdal.GDT_Float32, descriptions=name)


def dataConcat():
    data_names_dict = {}

    names1 = [
        'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDWI', 'OPT_asm', 'OPT_con', 'OPT_cor',
        'OPT_dis', 'OPT_ent', 'OPT_hom', 'OPT_mean', 'OPT_var', 'AS_VV', 'AS_VH', 'AS_angle', 'AS_VHDVV', 'AS_C11',
        'AS_C12_imag', 'AS_C12_real', 'AS_C22', 'AS_Lambda1', 'AS_Lambda2', 'AS_SPAN', 'AS_Epsilon', 'AS_Mu',
        'AS_RVI', 'AS_m', 'AS_Beta', 'AS_H', 'AS_A', 'AS_Alpha', 'AS_VH_asm', 'AS_VH_con', 'AS_VH_cor',
        'AS_VH_dis', 'AS_VH_ent', 'AS_VH_hom', 'AS_VH_mean', 'AS_VH_var', 'AS_VV_asm', 'AS_VV_con', 'AS_VV_cor',
        'AS_VV_dis', 'AS_VV_ent', 'AS_VV_hom', 'AS_VV_mean', 'AS_VV_var', 'DE_VV', 'DE_VH', 'DE_angle', 'DE_VHDVV',
        'DE_C11', 'DE_C12_imag', 'DE_C12_real', 'DE_C22', 'DE_SPAN', 'DE_Lambda1', 'DE_Lambda2', 'DE_Epsilon',
        'DE_Mu', 'DE_RVI', 'DE_m', 'DE_Beta', 'DE_H', 'DE_A', 'DE_Alpha', 'DE_VH_asm', 'DE_VH_con', 'DE_VH_cor',
        'DE_VH_dis', 'DE_VH_ent', 'DE_VH_hom', 'DE_VH_mean', 'DE_VH_var', 'DE_VV_asm', 'DE_VV_con', 'DE_VV_cor',
        'DE_VV_dis', 'DE_VV_ent', 'DE_VV_hom', 'DE_VV_mean', 'DE_VV_var'
    ]

    for name in names1:
        fn = r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\Channels\QingDao_{}.tif".format(name)
        if not os.path.isfile(fn):
            raise Exception("Not find {}".format(fn))
        data_names_dict[name] = fn

    names2 = ['NDWI', 'RNDWI', 'MNDWI', 'MBWI', 'NDVI', 'SAVI', 'NDBI', 'NBI', 'MBI', 'SEI', 'CSI', 'NSVDI']
    for name in names2:
        fn = r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\Index\{}.tif".format(name)
        if not os.path.isfile(fn):
            raise Exception("Not find {}".format(fn))
        data_names_dict[name] = fn

    data = None
    to_names = []
    gr = None
    for i, name in enumerate(data_names_dict):
        fn = data_names_dict[name]
        gr = GDALRaster(fn)
        print("{:>2d}. {:<15} {}".format(i + 1, name, fn))
        if data is None:
            data = np.zeros((len(data_names_dict), gr.n_rows, gr.n_columns), dtype="float32")
        data[i] = gr.readGDALBand(1).astype("float32")
        to_names.append(name)
    print(data.shape)
    gr.save(data,
            r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\SHH2_QD2_envi.dat",
            dtype=gdal.GDT_Float32, descriptions=to_names)

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
        "QingDao": SHH2Config.QD_GR(),
        # "BeiJing": SHH2Config.BJ_GR(),
        # "ChengDu": SHH2Config.CD_GR(),

        # "QingDao": GDALRaster(SHH2Config.QD_LOOK_FN),
        # "BeiJing": GDALRaster(SHH2Config.BJ_LOOK_FN),
        # "ChengDu": GDALRaster(SHH2Config.CD_LOOK_FN),
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
    dataConcat()
