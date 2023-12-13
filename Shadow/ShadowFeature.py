# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowFeature.py
@Time    : 2023/7/20 10:08
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : GEOCodes of ShadowFeature
-----------------------------------------------------------------------------"""
import numpy as np

from SRTCodes.SRTFeature import SRTFeatureExtraction


class ShadowFeatureExtractionASDET(SRTFeatureExtraction):

    def __init__(self, new_feat_name="FEATURE"):
        super().__init__(new_feat_name=new_feat_name, cal_feat_names=[
            "AS_VV_real", "AS_VV_imag", "AS_VH_real", "AS_VH_imag",
            "DE_VV_real", "DE_VV_imag", "DE_VH_real", "DE_VH_imag",
        ])

    def fit(self, *args, **kwargs):
        fs = self.features
        n_row = fs["AS_VV_real"].shape[0]
        n_column = fs["AS_VV_real"].shape[1]

        A = (fs["AS_VV_real"] + fs["DE_VV_real"]) + (fs["AS_VV_imag"] + fs["DE_VV_imag"]) * 1j
        B = (fs["AS_VV_real"] - fs["DE_VV_real"]) + (fs["AS_VV_imag"] - fs["DE_VV_imag"]) * 1j
        C = (fs["AS_VH_real"] + fs["DE_VH_real"]) + (fs["AS_VH_imag"] + fs["DE_VH_imag"]) * 1j
        D = ((fs["AS_VH_real"] - fs["DE_VH_real"]) + (fs["AS_VH_imag"] - fs["DE_VH_imag"]) * 1j) * 1j
        d = np.concatenate([A, B, C, D])

        for i in range(n_row):
            for j in range(n_column):
                x = np.array([d[:, i, j]])
                y = np.dot(x.T, x)




def main():
    pass


if __name__ == "__main__":
    main()
