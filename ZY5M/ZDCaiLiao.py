# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ZDCaiLiao.py
@Time    : 2024/2/23 19:34
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of ZDCaiLiao
-----------------------------------------------------------------------------"""
import pandas as pd

from SRTCodes.ModelTraining import ConfusionMatrix


def main():
    def func1():
        df = pd.read_excel(r"F:\ProjectSet\Huo\zhondianyanfa\cailiao\zdcailaio.xlsx", sheet_name="huizong2")
        print(df.keys())

        def cal_cm_17_21(name_17, name_21):
            def cal_cm(field_name):
                cm = ConfusionMatrix(2, ["NOIS", "IS"])
                y1, y2 = df["category"].values + 1, df[field_name].values + 1
                cm.addData(y1, y2)
                return {"OA": cm.OA(), "UA": cm.UA("IS"), "PA": cm.PA("IS"), "Kappa": cm.getKappa()}

            return {"2017": cal_cm(name_17), "2021": cal_cm(name_21)}

        cm_dict = {
            "Dynami": cal_cm_17_21('Dynami_17', 'Dynami_21'),
            "GAIA30": cal_cm_17_21('GAIA30_17', 'GAIA30_21'),
            "GISA30": cal_cm_17_21('GISA30_17', 'GISA30_21'),
            "EsriLa": cal_cm_17_21('EsriLa_17', 'EsriLa_21'),
            "Imdc": cal_cm_17_21('Imdc_17', 'Imdc_21'),
        }
        print(cm_dict)

        print("Name", "OA", "UA", "PA", "Kappa", "OA", "UA", "PA", "Kappa", sep=",")
        for name in cm_dict:
            print(name, end=",")
            for k in cm_dict[name]["2017"]:
                print(cm_dict[name]["2017"][k], end=",")
            for k in cm_dict[name]["2021"]:
                print(cm_dict[name]["2021"][k], end=",")
            print()

        return

    func1()


if __name__ == "__main__":
    main()
