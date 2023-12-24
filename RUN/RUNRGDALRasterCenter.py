# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : RUNRGDALRasterCenter.py
@Time    : 2023/12/24 11:38
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of RUNRGDALRasterCenter
-----------------------------------------------------------------------------"""
from Shadow.ShadowGeoDraw import DrawShadowImage_0


class RUN_GRC_DrawShadowImage_0(DrawShadowImage_0):

    def __init__(self, rows, columns, x, y, raster_fn, to_dirname, width=0.0, height=0.0, is_expand=False):
        super(RUN_GRC_DrawShadowImage_0, self).__init__(
            rows, columns, x, y, raster_fn, to_dirname, width=width, height=height, is_expand=is_expand)


class RUNRGDALRasterCenter_main:

    def __init__(self):
        self.name = "center_image"
        self.description = "Convert a raster coor center to image."
        self.argv = []
        self.mark_dict = {
            "shqd": {"rows": 100, "columns": 100,
                     "raster_fn": r"F:\ProjectSet\Shadow\Release\ChengDuImages\SH_CD_look_envi.dat",
                     "width": 0.0, "height": 0.0, "is_expand": False}
        }

    def run(self, argv):
        self.argv = argv
        raster_fn = r"F:\ProjectSet\Shadow\Release\QingDaoImages\SH_QD_envi.dat"
        to_dirname = r"F:\ProjectSet\Shadow\MkTu\4.1Details\1"

    def drawShdow(self, row, colum, x, y, raster_fn, to_dirname, width=6, height=6, is_expand=True):
        # dsi = DrawShadowImage_0(60, 60, 120.384609, 36.106485, raster_fn=raster_fn, to_dirname=to_dirname,
        #                         width=6, height=6, is_expand=True)
        dsi = DrawShadowImage_0(row, colum, x, y, raster_fn, to_dirname, width=width,
                                height=height, is_expand=is_expand)
        dsi.drawOptical("RGB", channel_list=[2, 1, 0])
        dsi.drawOptical("NRB", channel_list=[3, 2, 1])
        dsi.drawIndex("NDVI", d_min=-0.6, d_max=0.9)
        dsi.drawIndex("NDWI", d_min=-0.7, d_max=0.8)
        dsi.drawSAR("AS_VV", d_min=-24.609674, d_max=5.9092603)
        dsi.drawSAR("AS_VH", d_min=-31.865038, d_max=-5.2615275)
        dsi.drawSAR("AS_C11", d_min=-22.61998, d_max=5.8634768)
        dsi.drawSAR("AS_C22", d_min=-28.579813, d_max=-5.2111626)
        dsi.drawSAR("AS_Lambda1", d_min=-21.955856, d_max=6.124724)
        dsi.drawSAR("AS_Lambda2", d_min=-29.869734, d_max=-8.284683)
        dsi.drawSAR("DE_VV", d_min=-27.851603, d_max=5.094706)
        dsi.drawSAR("DE_VH", d_min=-35.427082, d_max=-5.4092093)
        dsi.drawSAR("DE_C11", d_min=-26.245598, d_max=4.9907513)
        dsi.drawSAR("DE_C22", d_min=-32.04232, d_max=-5.322515)
        dsi.drawSAR("DE_Lambda1", d_min=-25.503738, d_max=5.2980003)
        dsi.drawSAR("DE_Lambda2", d_min=-33.442368, d_max=-8.68537)


    def usage(self):
        print("{0} x,y  [-rc r,c] [mark|image:band] [-o filename] [-td to_dirname] [-of JPEG]\n"
              "    {1}\n"
              "    x,y: coor of raster center\n"
              "    [-rc r,c]: number of row column default:100,100\n"
              "    [mark|image:band]: mark `shqd|shbj|sjcd` or image:band|name\n"
              "    [-o filename]: save image file name\n"
              "    [-td to_dirname]: save dirname\n"
              "    [-of JPEG]: save dirname"
              "".format(self.name, self.description))


def main():
    pass


if __name__ == "__main__":
    main()
