# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHHClasses.py
@Time    : 2024/3/3 20:23
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHHClasses
-----------------------------------------------------------------------------"""
import os.path

from SRTCodes.GDALRasterClassification import GDALModelDataCategory
from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.GDALUtils import GDALRastersSampling
from SRTCodes.SRTSample import SRTSample
from SRTCodes.Utils import getRandom, Jdt, DirFileName, SRTDataFrame
from Shadow.Hierarchical import SHHConfig
from Shadow.Hierarchical.SHHMLFengCeng import read_geo_raster
from Shadow.Hierarchical.ShadowHSample import ShadowHierarchicalSampleCollection, initSHHGRS


def _coorsToSHHSC(coors):
    shh_sc = ShadowHierarchicalSampleCollection()
    for i in range(len(coors)):
        spl = SRTSample()
        spl.x = coors[i][0]
        spl.y = coors[i][1]
        spl["X"] = spl.x
        spl["Y"] = spl.y
        shh_sc.addSample(spl)
    return shh_sc


class SHHT_RandomCoor:

    def __init__(self, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.coors = []

    def generate(self, n, mode=None):
        if mode is None:
            self.coors = []
        for i in range(n):
            self.coors.append(self.randomXY())

    def __getitem__(self, item):
        return self.coors[item]

    def __len__(self):
        return len(self.coors)

    def toSHHSC(self) -> ShadowHierarchicalSampleCollection:
        return _coorsToSHHSC(self.coors)

    def randomXY(self):
        return [getRandom(self.x_min, self.x_max), getRandom(self.y_min, self.y_max), ]


class SHHGDALSampling:

    def __init__(self, *name_rss):
        self.name = ""
        self.rasters_samplings = {
            "shh1_im": GDALRastersSampling(*SHHConfig.SHH2_IMAGE1_FNS),
            "shh1_glcm": GDALRastersSampling(*SHHConfig.SHH2_IMAGE1_GLCM_FNS),
            "shh1_esa21": GDALRastersSampling(*SHHConfig.SHH2_IMAGE1_ESA21_FNS),
        }

    def samplingSHH13_CSV(self, csv_fn, to_csv_fn, is_jdt=True, no_data=0.0):
        jdt_name = "SHHGDALSampling::samplingSHH13_CSV"
        sdf = SRTDataFrame().read_csv(csv_fn, is_auto_type=True)
        rs_names = ["shh1_im", "shh1_glcm", "shh1_esa21"]

        field_names = []
        for name in rs_names:
            field_names.extend(self.rasters_samplings[name].getNames())
        sdf.addFields(*field_names)

        jdt = Jdt(len(sdf), jdt_name).start(is_jdt)
        for i in range(len(sdf)):
            x, y = sdf["X"][i], sdf["Y"][i]
            d_line = []
            for rs_name in rs_names:
                rasters_sampling: GDALRastersSampling = self.rasters_samplings[rs_name]
                d = rasters_sampling.sampling(x, y, no_data=no_data, is_none=False)
                d_line.extend(d.ravel().tolist())
            jdt.add(is_jdt=is_jdt)
            for j, field_name in enumerate(field_names):
                sdf[field_name][i] = d_line[j]
        jdt.end(is_jdt=is_jdt)

        sdf.toCSV(to_csv_fn)


class SHHT_GDALRandomCoor:

    def __init__(self):
        self.random_coor = SHHT_RandomCoor()
        self.rasters_sampling = GDALRastersSampling()
        self.coors = []

    def initRandomCoor(self, city_type="qd1", random_coor=None):
        if random_coor is not None:
            self.random_coor = random_coor
        else:
            if city_type == "qd1":
                # 119.97155,36.40607
                # 120.48067,36.05027
                self.random_coor = SHHT_RandomCoor(119.97155, 120.48067, 36.05027, 36.40607)

    def initGRSampling(self, grs_type="qd_sh1", raster_fns=None):
        self.rasters_sampling = initSHHGRS(grs_type, raster_fns)

    def sampling(self, n, filter_func=None, win_row_size=1, win_column_size=1, is_jdt=True, *args, **kwargs):
        i = 0
        self.coors = []
        jdt = Jdt(n, "SHHT_GDALRandomCoor:sampling")
        if is_jdt:
            jdt.start()
        while i < n:
            x, y = self.random_coor.randomXY()
            d = self.rasters_sampling.sampling(x, y, win_row_size=win_row_size, win_column_size=win_column_size, )
            if filter_func(d, *args, **kwargs):
                self.coors.append([x, y])
                i += 1
                if is_jdt:
                    jdt.add()
        if is_jdt:
            jdt.end()

    def toSHHSC(self) -> ShadowHierarchicalSampleCollection:
        return _coorsToSHHSC(self.coors)


class SHHModelImdc(GDALModelDataCategory):

    def __init__(self, *raster_fns):
        super().__init__(*raster_fns)
        self.shh_fi = SHHConfig.SHHFNImages.images1()
        dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Images")
        self.glcm_fn_qd = dfn.fn(r"QingDao\glcm\qd_sh2_1_gray_envi_mean")
        self.glcm_fn_cd = dfn.fn(r"ChengDu\glcm\cd_sh2_1_gray_envi_mean")
        self.glcm_fn_bj = dfn.fn(r"BeiJing\glcm\bj_sh2_1_gray_envi_mean")
        self.city_names = []

    def initSHH1(self):
        self.addRasters(self.shh_fi.qd, self.shh_fi.bj, self.shh_fi.cd)

    def initSHH1GLCM(self, city_name, imdc_keys):
        if city_name == "qd":
            self.addData(read_geo_raster(self.shh_fi.qd, imdc_keys, self.glcm_fn_qd))
            self.city_names.append("qd")
            self.gr = GDALRaster(self.shh_fi.qd)
        elif city_name == "bj":
            self.addData(read_geo_raster(self.shh_fi.bj, imdc_keys, self.glcm_fn_bj))
            self.city_names.append("bj")
            self.gr = GDALRaster(self.shh_fi.bj)
        elif city_name == "cd":
            self.addData(read_geo_raster(self.shh_fi.cd, imdc_keys, self.glcm_fn_cd))
            self.city_names.append("cd")
            self.gr = GDALRaster(self.shh_fi.cd)

    def imdc(self, model, to_dirname, name="", data_deal=None, color_table=None, description="Category"):
        if color_table is None:
            color_table = {}
        to_fns = [os.path.join(to_dirname, "{0}_{1}_imdc.tif".format(city_name, name)) for city_name in self.city_names]
        self.fit(to_fns, model, data_deal=data_deal, is_jdt=True, color_table=color_table, description=description)
        return to_fns

