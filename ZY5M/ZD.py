# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ZD.py
@Time    : 2023/10/16 20:08
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of ZD
-----------------------------------------------------------------------------"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SRTCodes.Utils import readJson, saveJson, changext, printList


def main():
    def trans_1_2(json_fn, to_excel_fn=None):
        if to_excel_fn is None:
            to_excel_fn = changext(json_fn, "_feats.xlsx")
        d = readJson(json_fn)
        for i in range(len(d["features"])):
            d["features"][i]["properties"]["SRT"] = i
        feats = [feat["properties"] for feat in d["features"]]
        df = pd.DataFrame(feats)
        print(df)
        # saveJson(d,json_fn)
        df.to_excel(to_excel_fn, index=False)

    def test1_1():
        df2 = pd.read_excel(r"D:\SpecialProjects\Huo\IS_JiJin\20231016\laos\laos_result.xlsx")
        df3 = df2[['SRT', 'area_is15', 'area_is16', 'area_is17', 'area_is18', 'area_is19', 'area_is20', 'area_is21',
                   'ENGTYPE_1', 'area_is22', 'NL_NAME_1',
                   'HASC_1', 'CC_1', 'TYPE_1', 'NAME_1', 'COUNTRY',
                   'GID_0', 'ISO_1', 'VARNAME_1',
                   'GID_1', 'area_admin']]
        print(df3.keys())
        df3.to_excel(r"D:\SpecialProjects\Huo\IS_JiJin\20231016\laos\laos_result2.xlsx", index=False)

    def to_trans1(json_fn, excel_fn, to_json_fn, sheet_name=None):
        d = readJson(json_fn)
        df = pd.read_excel(excel_fn, sheet_name=sheet_name)
        for i in range(len(df)):
            srt = df["SRT"][i]
            for j in range(len(d["features"])):
                if d["features"][j]["properties"]["SRT"] == srt:
                    d["features"][j]["properties"] = {"SRT": srt}
                    for k in df:
                        d["features"][j]["properties"][k] = df[k][i]
                    break

        with open(to_json_fn, "w", encoding="utf-8") as f:
            print(d, file=f)

        with open(to_json_fn, "r", encoding="utf-8") as f:
            d1 = f.read()
            d2 = d1.replace("'", '"')

        with open(to_json_fn, "w", encoding="utf-8") as f:
            f.write(d2)

    def concat_geojson(json_fn_list, to_json_fn):
        json_list = [readJson(fn) for fn in json_fn_list]
        json_out = json_list[0]
        for json0 in json_list[1:]:
            json_out["features"] += json0["features"]
        saveJson(json_out, to_json_fn)

    def merge_geojson(json_fn1, json_fn2, to_json_fn, merge_key):
        d1 = readJson(json_fn1)
        d2 = readJson(json_fn2)
        for i, feat1 in enumerate(d1["features"]):
            merge_key_d1 = feat1["properties"][merge_key]
            feat2 = {}
            for feat in d2["features"]:
                if feat["properties"][merge_key] == merge_key_d1:
                    feat2 = feat
                    break
            for k in feat2["properties"]:
                d1["features"][i]["properties"][k] = feat2["properties"][k]
        saveJson(d1, to_json_fn)

    def show_properties_names(json_fn):
        d = readJson(json_fn)
        knames = []
        for feat1 in d["features"]:
            for k in feat1["properties"]:
                if k not in knames:
                    knames.append(k)
        printList("show_properties_names", knames)

    def change_properties_names(json_fn, to_json_fn, map_dict=None):
        if map_dict is None:
            return
        d = readJson(json_fn)
        for i, feat1 in enumerate(d["features"]):
            for k in map_dict:
                if k in feat1["properties"]:
                    value = d["features"][i]["properties"].pop(k)  # 先删除旧键，并获取对应的值
                    d["features"][i]["properties"][map_dict[k]] = value  # 添加新键和对应的值
        saveJson(d, to_json_fn)

    def calLanMei1(to_csv_fn, csv_fn_list, fields_dict):
        d_list = []
        for i, csv_fn in enumerate(csv_fn_list):
            df = pd.read_csv(csv_fn)
            print(csv_fn)
            d0 = df.sum().to_dict()
            for k in fields_dict:
                d0[k] = fields_dict[k][i]
            d_list.append(d0)
        pd.DataFrame(d_list).to_csv(to_csv_fn, index=False)

    def test1():
        d = readJson(r"F:\ProjectSet\Huo\zhondianyanfa\Cities\Cities.json")
        d1 = []
        for feat in d["features"]:
            d1.append(feat["properties"])
        pd.DataFrame(d1).to_csv(r"F:\ProjectSet\Huo\zhondianyanfa\Cities\Cities.csv")

    # trans_1_2(r"D:\SpecialProjects\Huo\IS_JiJin\20231016\laos\result.geojson",
    #           r"D:\SpecialProjects\Huo\IS_JiJin\20231016\laos\laos_result.xlsx")
    # to_trans1(r"D:\SpecialProjects\Huo\IS_JiJin\20231016\laos\result.geojson",
    #           r"D:\SpecialProjects\Huo\IS_JiJin\20231016\laos\laos_result2.xlsx",
    #           r"D:\SpecialProjects\Huo\IS_JiJin\20231016\laos\laos_result2_json2.geojson",
    #           sheet_name="end")
    # concat_geojson([
    #     r"D:\SpecialProjects\Huo\IS_JiJin\20231016\laos\grids\drive-download-20231016T073036Z-001\laos_grids_cal_0.geojson",
    #     r"D:\SpecialProjects\Huo\IS_JiJin\20231016\laos\grids\drive-download-20231016T073036Z-001\laos_grids_cal_1.geojson",
    #     r"D:\SpecialProjects\Huo\IS_JiJin\20231016\laos\grids\drive-download-20231016T073036Z-001\laos_grids_cal_2.geojson",
    #     r"D:\SpecialProjects\Huo\IS_JiJin\20231016\laos\grids\drive-download-20231016T073036Z-001\laos_grids_cal_3.geojson",
    #     r"D:\SpecialProjects\Huo\IS_JiJin\20231016\laos\grids\drive-download-20231016T073036Z-001\laos_grids_cal_4.geojson",
    #     r"D:\SpecialProjects\Huo\IS_JiJin\20231016\laos\grids\drive-download-20231016T073036Z-001\laos_grids_cal_5.geojson"
    # ],
    #     r"D:\SpecialProjects\Huo\IS_JiJin\20231016\laos\grids\drive-download-20231016T073036Z-001\laos_grids_cal.geojson")
    # to_trans1(r"D:\SpecialProjects\Huo\IS_JiJin\20231016\laos\grids\laos_grids_cal.geojson",
    #           r"D:\SpecialProjects\Huo\IS_JiJin\20231016\laos\laos_grids.xlsx",
    #           r"D:\SpecialProjects\Huo\IS_JiJin\20231016\laos\grids\laos_grids_cal2.geojson",
    #           sheet_name="Sheet1")
    # merge_geojson(
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T115816Z-001\thai_region1.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T115816Z-001\Thai_result.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T115816Z-001\Thai_result2.geojson",
    #     merge_key="NAME_1"
    # )
    # show_properties_names(
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T115816Z-001\Thai_result2.geojson")
    # change_properties_names(
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T115816Z-001\Thai_result2.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T115816Z-001\Thai_result2_1.geojson",
    #     map_dict={"省份中": "CNAME", "中文名": "CNAME_2", "中文名_": "CNAME_3"})
    # show_properties_names(
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T115816Z-001\Thai_result2_1.geojson")
    # to_trans1(
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T115816Z-001\Thai_result2_3.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\thai_is.xlsx",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T115816Z-001\Thai_result2_4.geojson",
    #     sheet_name="end")
    # concat_geojson([
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_0.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_1.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_2.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_3.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_4.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_5.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_6.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_7.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_8.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_9.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_10.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_11.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_12.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_13.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_14.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_15.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_16.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_17.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_18.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_19.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_20.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_21.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_22.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_23.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_24.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_25.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_26.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_27.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_28.geojson",
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal_29.geojson",
    # ], r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\drive-download-20231016T132354Z-001\thai_grids_cal.geojson")
    # to_trans1(r"F:\ProjectSet\Huo\zhondianyanfa\20231016\laos\2\laos_is.geojson",
    #           r"F:\ProjectSet\Huo\zhondianyanfa\20231016\laos\2\laos_is.xlsx",
    #           r"F:\ProjectSet\Huo\zhondianyanfa\20231016\laos\2\laos_is_2.geojson",
    #           sheet_name="end")
    # to_trans1(r"F:\ProjectSet\Huo\zhondianyanfa\20231016\laos\grids\2\laos_grids.geojson",
    #           r"F:\ProjectSet\Huo\zhondianyanfa\20231016\laos\grids\2\laos_grids.xlsx",
    #           r"F:\ProjectSet\Huo\zhondianyanfa\20231016\laos\grids\2\laos_grids_2.geojson",
    #           sheet_name="end")
    # to_trans1(r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\2\thai_is.geojson",
    #           r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\2\thai_is.xlsx",
    #           r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\2\thai_is_2.geojson",
    #           sheet_name="end")
    # to_trans1(r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\grids\2\thai_grids.geojson",
    #           r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\grids\2\thai_grids.xlsx",
    #           r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\grids\2\thai_grids_2.geojson",
    #           sheet_name="end")
    # concat_geojson([
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\drive-download-20231019T090012Z-001\GMS_result0.geojson"
    #     ,r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\drive-download-20231019T090012Z-001\GMS_result1.geojson"
    #     ,r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\drive-download-20231019T090012Z-001\GMS_result2.geojson"
    #     ,r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\drive-download-20231019T090012Z-001\GMS_result3.geojson"
    #     ,r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\drive-download-20231019T090012Z-001\GMS_result4.geojson"
    #     ,r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\drive-download-20231019T090012Z-001\GMS_result5.geojson"
    #     ,r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\drive-download-20231019T090012Z-001\GMS_result6.geojson"
    #     ,r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\drive-download-20231019T090012Z-001\GMS_result7.geojson"
    #     ,r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\drive-download-20231019T090012Z-001\GMS_result8.geojson"
    #     ,r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\drive-download-20231019T090012Z-001\GMS_result9.geojson"
    #     ,r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\drive-download-20231019T090012Z-001\GMS_result10.geojson"
    #     ,r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\drive-download-20231019T090012Z-001\GMS_result11.geojson"
    # ], r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\drive-download-20231019T090012Z-001\GMS_result.geojson")
    # calLanMei1(
    #     r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\gms_is_1.csv",
    #     [
    #         r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\grid\2\gms_miandian_grid_2.csv",
    #         r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\grid\2\gms_yuenan_grid_2.csv",
    #         r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\grid\2\gms_laowo_grid_2.csv",
    #         r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\grid\2\gms_taiguo_grid_2.csv",
    #         r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\grid\2\gms_jianpuzhai_grid_2.csv",
    #         r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\grid\2\gms_yunnan_grid_2.csv",
    #         r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\grid\2\gms_guangxi_grid_2.csv",
    #     ],
    #     {
    #         'FID_GMS': ['0', '5', '13', '137', '393', '0', '0'],
    #         'AREA_1': ['8.00162000000', '0.71096500000', '1.30314000000', '0.98138800000', '0.94707200000',
    #                    '0.00000000000', '0.00000000000'],
    #         'PERIMETE_1': ['18.33000000000', '4.99270000000', '6.39395000000', '6.12690000000', '6.06070000000',
    #                        '0.00000000000', '0.00000000000'],
    #         'ESABOU_': ['2', '7', '15', '145', '446', '0', '0'],
    #         'ESABOU_ID': ['1768', '2620', '2808', '3328', '3890', '0', '0'],
    #         'NAME_1': ['KACHIN STATE', 'HA GIANG', 'PHONGSALI', 'CHIANG RAI', 'ROTANOKIRI', '', ''],
    #         'FENAME': ['Kachin State', 'Ha Giang', 'Phongsali', 'Chiang Rai', 'Ratanakiri', '', ''],
    #         'FCNAME': ['Myanmar', 'Vietnam', 'Laos', 'Thailand', 'Cambodia', 'Yunnan,China', 'Guangxi,China'],
    #         'SOC': ['MMR', 'VNM', 'LAO', 'THA', 'KHM', 'CHN', 'GXS'],
    #         'OWNER': ['克钦邦', '河江省', '丰沙里省', '清莱府', '腊塔纳基里省', '云南', '广西'],
    #         '中文名': ['缅甸', '越南', '老挝', '泰国', '柬埔寨', '中国云南', '中国广西']
    #     })
    # to_trans1(r"F:\ProjectSet\Huo\zhondianyanfa\澜湄流域_泰国_老挝\澜湄流域_泰国_老挝\lammeiliyu_region.geojson",
    #           r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\gms_is_1.xlsx",
    #           r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\gms_is_2.geojson",
    #           sheet_name="end")
    # to_trans1(r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\drive-download-20231019T090012Z-001\GMS_result.geojson",
    #           r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\gms_grids_1.xlsx",
    #           r"F:\ProjectSet\Huo\zhondianyanfa\20231016\gms\gms_grids_1.geojson",
    #           sheet_name="end")
    test1()


def calZhongXin():
    json_fn = r"F:\ProjectSet\Huo\zhondianyanfa\20231016\thai\grids\2\thai_grids_2.geojson"
    d = readJson(json_fn)
    coors = []
    x, y = 0, 0
    for feat in d["features"]:
        coor = np.mean(np.array(feat["geometry"]["coordinates"][0][:6]), axis=0)
        coors.append([coor[0], coor[1], feat["properties"]["v"]])
    coors = np.array(coors)
    # coors[:, 2] = (coors[:, 2] - np.min(coors[:, 2]))/(np.max(coors[:, 2]) - np.min(coors[:, 2]))
    x = np.sum(coors[:, 0] * coors[:, 2]) / np.sum(coors[:, 2])
    y = np.sum(coors[:, 1] * coors[:, 2]) / np.sum(coors[:, 2])
    plt.scatter(coors[:, 0], coors[:, 1])
    plt.scatter(x, y)
    plt.show()


if __name__ == "__main__":
    main()
