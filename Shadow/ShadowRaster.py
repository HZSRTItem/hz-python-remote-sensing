# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowRaster.py
@Time    : 2023/11/7 15:46
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of ShadowRaster
-----------------------------------------------------------------------------"""
import os.path

import numpy as np
from matplotlib import pyplot as plt
from osgeo import gdal

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.GDALUtils import GDALRasterHist, vrtAddDescriptions
from SRTCodes.Utils import readLines
from Shadow.ShadowDraw import cal_10log10


class ShadowRasterGLCM:

    def __init__(self):
        # "mean", "variance", "homogeneity", "contrast", "dissimilarity", "entropy", "second moment", "correlation"
        self.des = ["mean", "var", "hom", "con", "dis", "ent", "asm", "cor"]
        self.glcm_fn_ext = ["_01", "_10", "_11", "_1_1"]
        ...

    def meanFourDirection(self, glcm_fn, front_str, dirname=None, to_fn=None):
        print(glcm_fn)

        if to_fn is None:
            to_fn = glcm_fn + "_mean"
        if dirname is not None:
            glcm_fns = [os.path.join(dirname, glcm_fn) + glcm_fn_ext for glcm_fn_ext in self.glcm_fn_ext]
            to_fn = os.path.join(dirname, to_fn)
        else:
            glcm_fns = [glcm_fn + glcm_fn_ext for glcm_fn_ext in self.glcm_fn_ext]

        def mean_one_d(filenames, front_str, to_fn):
            des = [front_str + des0 for des0 in self.des]
            grs = [GDALRaster(filename) for filename in filenames]
            out_d = np.zeros((8, grs[0].n_rows, grs[0].n_columns))
            for i in range(8):
                print(des[i])
                d_tmp = np.zeros((4, grs[0].n_rows, grs[0].n_columns))
                for j in range(4):
                    d_tmp[j] = grs[j].readGDALBand(i + 1)
                out_d[i] = np.mean(d_tmp, axis=0)
            grs[0].save(out_d, to_fn, descriptions=des)

        mean_one_d(glcm_fns, front_str, to_fn)

    def sar10Log10(self, fn, to_fn):
        gr = GDALRaster(fn)
        d = gr.readAsArray()
        d = 10 * np.log10(d + 0.00001)
        gr.save(d, to_fn)

    def imChangeRange(self, fn, to_fn, d_min, d_max):
        gr = GDALRaster(fn)
        d = gr.readAsArray()
        d = np.clip(d, a_min=d_min, a_max=d_max)
        gr.save(d, to_fn)

    def updateToRelease(self, fn, release_dirname):
        gr = GDALRaster(fn)
        for name in gr.names:
            to_fn = os.path.join(release_dirname, name + ".dat")
            print(to_fn)
            d = gr.readGDALBand(name)
            gr.save(d, to_fn)


def main():
    release_dirname = r"F:\ProjectSet\Shadow\Release\BeiJingImages"
    vrt_fn = r"F:\ProjectSet\Shadow\Release\BJ_SH.vrt"

    with open(r"F:\ProjectSet\Shadow\Release\filelist.txt", "r", encoding="utf-8") as f:
        envi_fns = []
        descriptions = []
        for line in f:
            line = line.strip()
            envi_fns.append(os.path.join(release_dirname, line + ".dat"))
            descriptions.append(line)
        gdal.BuildVRT(vrt_fn, envi_fns, options=["-separate", "-r", "bilinear"])
        vrtAddDescriptions(vrt_fn, descriptions=descriptions)


def cdGLCM():
    sh_glcm = ShadowRasterGLCM()
    tmp_dirname = r"F:\ProjectSet\Shadow\ChengDu\Image\GLCM"
    release_dirname = r"F:\ProjectSet\Shadow\ChengDu\Image\GLCM\release"

    def d_dirname(fn):
        return os.path.join(tmp_dirname, fn)

    def rele_dirname(fn):
        return os.path.join(release_dirname, fn)

    # sh_glcm.sar10Log10(rele_dirname(r"F:\ProjectSet\Shadow\ChengDu\Image\5\CD_SH5_AS_VH.dat"), d_dirname("tmp_AS_VH"))
    # sh_glcm.sar10Log10(rele_dirname(r"F:\ProjectSet\Shadow\ChengDu\Image\5\CD_SH5_AS_VV.dat"), d_dirname("tmp_AS_VV"))
    # sh_glcm.sar10Log10(rele_dirname(r"F:\ProjectSet\Shadow\ChengDu\Image\5\CD_SH5_DE_VH.dat"), d_dirname("tmp_DE_VH"))
    # sh_glcm.sar10Log10(rele_dirname(r"F:\ProjectSet\Shadow\ChengDu\Image\5\CD_SH5_DE_VV.dat"), d_dirname("tmp_DE_VV"))

    # sh_glcm.imChangeRange(d_dirname("sh_cd_pca"), d_dirname("cd_im5_pca"), -2267.13696, 1603.20044)
    # sh_glcm.imChangeRange(d_dirname("tmp_AS_VH"), d_dirname("cd_im5_as_vh"), -23.82608, -1.48292)
    # sh_glcm.imChangeRange(d_dirname("tmp_AS_VV"), d_dirname("cd_im5_as_vv"), -17.15184, 7.85603)
    # sh_glcm.imChangeRange(d_dirname("tmp_DE_VH"), d_dirname("cd_im5_de_vh"), -24.79699, -0.86977)
    # sh_glcm.imChangeRange(d_dirname("tmp_DE_VV"), d_dirname("cd_im5_de_vv"), -16.29003, 5.35793)

    # sh_glcm.meanFourDirection("cd_im5_pca", "OPT_", tmp_dirname)
    # sh_glcm.meanFourDirection("cd_im5_as_vh", "AS_VV_", tmp_dirname)
    # sh_glcm.meanFourDirection("cd_im5_as_vv", "AS_VH_", tmp_dirname)
    # sh_glcm.meanFourDirection("cd_im5_de_vh", "DE_VV_", tmp_dirname)
    # sh_glcm.meanFourDirection("cd_im5_de_vv", "DE_VH_", tmp_dirname)

    sh_glcm.updateToRelease(d_dirname("cd_im5_pca_mean"),   release_dirname)
    sh_glcm.updateToRelease(d_dirname("cd_im5_as_vh_mean"),  release_dirname)
    sh_glcm.updateToRelease(d_dirname("cd_im5_as_vv_mean"),  release_dirname)
    sh_glcm.updateToRelease(d_dirname("cd_im5_de_vh_mean"),  release_dirname)
    sh_glcm.updateToRelease(d_dirname("cd_im5_de_vv_mean"),  release_dirname)


def bjGLCM():
    sh_glcm = ShadowRasterGLCM()
    tmp_dirname = r"F:\ProjectSet\Shadow\BeiJing\Image\GLCM"
    release_dirname = r"F:\ProjectSet\Shadow\BeiJing\Image\GLCM\release"

    def d_dirname(fn):
        return os.path.join(tmp_dirname, fn)

    def rele_dirname(fn):
        return os.path.join(release_dirname, fn)

    # sh_glcm.sar10Log10(rele_dirname("BJ_SH3_AS_VH.dat"), d_dirname("tmp_AS_VH"))
    # sh_glcm.sar10Log10(rele_dirname("BJ_SH3_AS_VV.dat"), d_dirname("tmp_AS_VV"))
    # sh_glcm.sar10Log10(rele_dirname("BJ_SH3_DE_VH.dat"), d_dirname("tmp_DE_VH"))
    # sh_glcm.sar10Log10(rele_dirname("BJ_SH3_DE_VV.dat"), d_dirname("tmp_DE_VV"))

    # sh_glcm.imChangeRange(d_dirname("bj_sh_pca1"), d_dirname("bj_sh_pca2"), -2279.11475, 3659.58594)
    # sh_glcm.imChangeRange(d_dirname("tmp_AS_VH"), d_dirname("bj_as_vh"), -24.94139, -1.38980)
    # sh_glcm.imChangeRange(d_dirname("tmp_AS_VV"), d_dirname("bj_as_vv"), -17.10382, 8.56375)
    # sh_glcm.imChangeRange(d_dirname("tmp_DE_VH"), d_dirname("bj_de_vh"), -24.52231, 0.63422)
    # sh_glcm.imChangeRange(d_dirname("tmp_DE_VV"), d_dirname("bj_de_vv"), -16.55685, 7.42070)

    # sh_glcm.meanFourDirection("bj_sh_pca2", "OPT_", tmp_dirname)
    # sh_glcm.meanFourDirection("bj_as_vh", "AS_VV_", tmp_dirname)
    # sh_glcm.meanFourDirection("bj_as_vv", "AS_VH_", tmp_dirname)
    # sh_glcm.meanFourDirection("bj_de_vh", "DE_VV_", tmp_dirname)
    # sh_glcm.meanFourDirection("bj_de_vv", "DE_VH_", tmp_dirname)

    sh_glcm.updateToRelease(d_dirname("bj_sh_pca2_mean"), release_dirname)
    sh_glcm.updateToRelease(d_dirname("bj_as_vh_mean"), release_dirname)
    sh_glcm.updateToRelease(d_dirname("bj_as_vv_mean"), release_dirname)
    sh_glcm.updateToRelease(d_dirname("bj_de_vh_mean"), release_dirname)
    sh_glcm.updateToRelease(d_dirname("bj_de_vv_mean"), release_dirname)


def qdGLCM():
    sh_glcm = ShadowRasterGLCM()

    tmp_dirname = r"F:\ProjectSet\Shadow\QingDao\Image\Image1\glcm"
    release_dirname = r"F:\ProjectSet\Shadow\Release\BeiJingImages"

    def im_dirname(fn):
        return os.path.join(r"F:\ProjectSet\Shadow\QingDao\Image\Image1\1", fn)

    def d_dirname(fn):
        return os.path.join(tmp_dirname, fn)

    def rele_dirname(fn):
        return os.path.join(release_dirname, fn)

    # sh_glcm.sar10Log10(im_dirname("QD_SH1_AS_VV.dat"),d_dirname("tmp_AS_VV"))
    # sh_glcm.sar10Log10(im_dirname("QD_SH1_AS_VH.dat"),d_dirname("tmp_AS_VH"))
    # sh_glcm.sar10Log10(im_dirname("QD_SH1_DE_VV.dat"),d_dirname("tmp_DE_VV"))
    # sh_glcm.sar10Log10(im_dirname("QD_SH1_DE_VH.dat"),d_dirname("tmp_DE_VH"))

    # sh_glcm.imChangeRange(d_dirname("qd_glcm_pca"), d_dirname("qd_glcm_pca2"), -1365.07080, 3322.94230)
    # sh_glcm.imChangeRange(d_dirname("tmp_AS_VH"), d_dirname("tmp_AS_VH_qd"), -35.41401, -1.06227)
    # sh_glcm.imChangeRange(d_dirname("tmp_AS_VV"), d_dirname("tmp_AS_VV_qd"), -26.69723, 7.61192)
    # sh_glcm.imChangeRange(d_dirname("tmp_DE_VH"), d_dirname("tmp_DE_VH_qd"), -37.32133, -0.68132)
    # sh_glcm.imChangeRange(d_dirname("tmp_DE_VV"), d_dirname("tmp_DE_VV_qd"), -28.40360, 5.67388)
    #
    # sh_glcm.meanFourDirection("qd_glcm_pca2", "OPT_", tmp_dirname)
    # sh_glcm.meanFourDirection("tmp_AS_VH_qd", "AS_VH_", tmp_dirname)
    # sh_glcm.meanFourDirection("tmp_AS_VV_qd", "AS_VV_", tmp_dirname)
    # sh_glcm.meanFourDirection("tmp_DE_VH_qd", "DE_VH_", tmp_dirname)
    # sh_glcm.meanFourDirection("tmp_DE_VV_qd", "DE_VV_", tmp_dirname)

    sh_glcm.updateToRelease(d_dirname(r"qd_glcm_pca2_mean"), r"F:\ProjectSet\Shadow\QingDao\Image\Image1\glcm\qd_glcm")
    sh_glcm.updateToRelease(d_dirname(r"tmp_AS_VH_qd_mean"), r"F:\ProjectSet\Shadow\QingDao\Image\Image1\glcm\qd_glcm")
    sh_glcm.updateToRelease(d_dirname(r"tmp_AS_VV_qd_mean"), r"F:\ProjectSet\Shadow\QingDao\Image\Image1\glcm\qd_glcm")
    sh_glcm.updateToRelease(d_dirname(r"tmp_DE_VH_qd_mean"), r"F:\ProjectSet\Shadow\QingDao\Image\Image1\glcm\qd_glcm")
    sh_glcm.updateToRelease(d_dirname(r"tmp_DE_VV_qd_mean"), r"F:\ProjectSet\Shadow\QingDao\Image\Image1\glcm\qd_glcm")


def method_name():
    grh = GDALRasterHist()
    # 'Blue', 'Green', 'Red', 'NIR', 'NDVI', 'NDWI',
    # 'Gray_asm', 'Gray_contrast', 'Gray_corr', 'Gray_var', 'Gray_idm', 'Gray_savg', 'Gray_svar',
    #     'Gray_sent', 'Gray_ent', 'Gray_dvar', 'Gray_dent', 'Gray_imcorr1', 'Gray_imcorr2', 'Gray_maxcorr',
    #     'Gray_diss', 'Gray_inertia', 'Gray_shade', 'Gray_prom',
    # 'AS_VV', 'AS_VH', 'AS_VHDVV', 'AS_C11', 'AS_C12_imag', 'AS_C12_real', 'AS_C22', 'AS_Lambda1', 'AS_Lambda2',
    #     'AS_SPAN', 'AS_Epsilon', 'AS_Mu', 'AS_RVI', 'AS_m', 'AS_Beta',
    # 'AS_GLCM_asm', 'AS_GLCM_contrast', 'AS_GLCM_corr', 'AS_GLCM_var', 'AS_GLCM_idm', 'AS_GLCM_savg', 'AS_GLCM_svar',
    #     'AS_GLCM_sent', 'AS_GLCM_ent', 'AS_GLCM_dvar', 'AS_GLCM_dent', 'AS_GLCM_imcorr1', 'AS_GLCM_imcorr2',
    #     'AS_GLCM_maxcorr', 'AS_GLCM_diss', 'AS_GLCM_inertia', 'AS_GLCM_shade', 'AS_GLCM_prom',
    # 'DE_VV', 'DE_VH', 'DE_VHDVV', 'DE_C11', 'DE_C12_imag', 'DE_C12_real', 'DE_C22', 'DE_Lambda1', 'DE_Lambda2',
    #     'DE_SPAN', 'DE_Epsilon', 'DE_Mu', 'DE_RVI', 'DE_m', 'DE_Beta',
    # 'DE_GLCM_asm', 'DE_GLCM_contrast', 'DE_GLCM_corr', 'DE_GLCM_var', 'DE_GLCM_idm', 'DE_GLCM_savg', 'DE_GLCM_svar',
    #     'DE_GLCM_sent', 'DE_GLCM_ent', 'DE_GLCM_dvar', 'DE_GLCM_dent', 'DE_GLCM_imcorr1', 'DE_GLCM_imcorr2',
    #     'DE_GLCM_maxcorr', 'DE_GLCM_diss', 'DE_GLCM_inertia', 'DE_GLCM_shade', 'DE_GLCM_prom'
    vrt_qd = r"F:\ProjectSet\Shadow\Release\QD_SH.vrt"
    grh.show('Gray_contrast', vrt_qd, )
    plt.legend()
    plt.show()


def method_name2():
    # 测试所有影像的大小
    lines = readLines(r"F:\ProjectSet\Shadow\QingDao\Image\Image1\filelist.txt", "\n")
    size = None
    for line in lines:
        gr = GDALRaster(line)
        if size is None:
            size = (gr.n_channels, gr.n_rows, gr.n_columns)
        else:
            if not size == (gr.n_channels, gr.n_rows, gr.n_columns):
                print(line)


def rasterConcat():
    lines = readLines(r"F:\ProjectSet\Shadow\Release\ChengDuImages\filelist.txt", "\n")
    d = None
    gr = None
    des_list = []
    for i, line in enumerate(lines):
        des, fn = tuple(line.split(","))
        des_list.append(des)
        print(des, fn)
        gr = GDALRaster(fn)
        if d is None:
            d = np.zeros((len(lines), gr.n_rows, gr.n_columns))
        d[i] = gr.readAsArray()
    gr.save(d, r"F:\ProjectSet\Shadow\Release\ChengDuImages\SH_CD_envi.dat", descriptions=des_list)


def raster10Log10():
    fn = r"F:\ProjectSet\Shadow\Release\ChengDuImages\SH_CD_envi.dat"
    to_fn = r"F:\ProjectSet\Shadow\Release\ChengDuImages\SH_CD_look_envi.dat"

    gr = GDALRaster(fn)
    d = np.zeros((gr.n_channels, gr.n_rows, gr.n_columns))
    name_list = ['AS_VV', 'AS_VH', 'AS_C11', 'AS_C22', 'AS_Lambda1', 'AS_Lambda2', 'AS_SPAN', 'AS_Epsilon',
                 'DE_VV', 'DE_VH', 'DE_C11', 'DE_C22', 'DE_Lambda1', 'DE_Lambda2', 'DE_SPAN', 'DE_Epsilon']

    print(gr.names)
    for i, name in enumerate(gr):
        d_tmp = gr.readGDALBand(name)
        print(name)
        if name in name_list:
            d_tmp = cal_10log10(d_tmp)
        d[i] = d_tmp
    gr.save(d, to_fn, descriptions=gr.names)


if __name__ == "__main__":
    raster10Log10()
    ...
