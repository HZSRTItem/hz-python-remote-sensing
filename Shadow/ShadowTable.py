# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowTable.py
@Time    : 2023/12/21 22:10
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of ShadowTable
-----------------------------------------------------------------------------"""
import os
import time

from SRTCodes.GDALRasterClassification import GDALRasterClassificationAccuracy


class ShadowGDALRasterClassificationAccuracy(GDALRasterClassificationAccuracy):

    def __init__(self, name="ShadowTableAccuracy"):
        super().__init__()
        self.name = name
        self.save_dirname = r"F:\ProjectSet\Shadow\Table"
        self.model_dir = self.save_dirname
        self.addCategoryCode(IS_SH=1, VEG_SH=2, SOIL_SH=3, WAT_SH=4)

    def timeModelDir(self):
        dir_name = time.strftime("%Y%m%dH%H%M%S")
        self.model_dir = os.path.join(self.model_dir, dir_name)
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
        return dir_name

    def run(self, mod_dir, ):
        dirname = self.timeModelDir()
        save_csv_fn = os.path.join(self.model_dir, "")

        excel_fn = r"F:\ProjectSet\Shadow\QingDao\Sample\Test\3\青岛测试样本3.xlsx"  # sheet_name="选出600个样本作为总体精度"
        excel_fn = r"F:\Week\20230917\Data\看了青岛的样本.xlsx"  # sheet_name="Sheet1" NOSH
        raster_fn = r"F:\ProjectSet\Shadow\QingDao\Mods\20230707H200910\SPL_NOSH-RF-TAG-OPTICS-AS_C2-AS_LAMD-DE_C2_imdc.dat"

        grca = GDALRasterClassificationAccuracy()
        grca.addCategoryCode(IS_SH=1, VEG_SH=2, SOIL_SH=3, WAT_SH=4)
        grca.addSampleExcel(excel_fn, c_column_name="CNAME", sheet_name="NOSH")

        # grca.fit(raster_fn)
        grca.openSaveCSVFileName(r"F:\Week\20230917\Data\test1.csv")
        grca.openSaveCMFileName(r"F:\Week\20230917\Data\test1_cm.txt")
        grca.fitModelDirectory(os.path.dirname(raster_fn))
        grca.closeSaveCSVFileName()
        grca.closeSaveCMFileName()



def main():
    pass


if __name__ == "__main__":
    main()
