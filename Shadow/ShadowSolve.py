# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowSolve.py
@Time    : 2023/7/16 10:52
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : BaseCodes of ShadowFenXi
-----------------------------------------------------------------------------"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SRTCodes.SRTData import DFFilter_eq
from SRTCodes.Utils import printList, saveJson


def hist(x, legend_):
    n, bin_edges = np.histogram(x, bins=30)
    print(np.sum(n))
    plt.plot(bin_edges[:-1], n, label=legend_)


class ShadowAccuracyGroup:

    def __init__(self, csv_fn=""):
        self.csv_fn = csv_fn
        self.df = pd.read_csv(csv_fn)
        self.clf_name = "RF"
        self.spl_name = "SPL_NOSH"

    def showFeatureHist(self, feat_name, acc_name):
        df = DFFilter_eq(self.df, self.clf_name, 1)
        df = DFFilter_eq(df, self.spl_name, 1)
        df0 = DFFilter_eq(df, feat_name, 0)
        hist(df0[acc_name], "Not have " + feat_name)
        df1 = DFFilter_eq(df, feat_name, 1)
        hist(df1[acc_name], feat_name)
        plt.legend()

    def print(self):
        printList("", list(self.df.keys()))


def accuracyChange(csv_fn):
    df: pd.DataFrame = pd.read_csv(csv_fn)
    clf_name = "RF"
    df = DFFilter_eq(df, clf_name, 1)
    spl_name = "SPL_NOSH"
    df = DFFilter_eq(df, spl_name, 1)
    acc = {"AS_SIGMA": [], "AS_C2": [], "AS_LAMD": [], "DE_SIGMA": [], "DE_C2": [], "DE_LAMD": []}
    feats = ["AS_SIGMA", "AS_C2", "AS_LAMD", "DE_SIGMA", "DE_C2", "DE_LAMD"]
    accs = ["OATrain", "KappaTrain", "IS UATrain", "IS PATrain", "VEG UATrain", "VEG PATrain", "SOIL UATrain",
            "SOIL PATrain", "WAT UATrain", "WAT PATrain",
            "OATest", "KappaTest", "IS UATest", "IS PATest", "VEG UATest", "VEG PATest", "SOIL UATest",
            "SOIL PATest", "WAT UATest", "WAT PATest"]
    for feat in feats:
        print(feat)
        for i in range(len(df)):
            if df.loc[i][feat] == 1:
                df_find = df.loc[i][feats]
                df_find[feat] = 0
                df_acc = df.loc[i][accs]
                is_find = True
                for j in range(len(df)):
                    if df_find.equals(df.loc[j][feats]):
                        print(j, end=" ")
                        if is_find:
                            df_delta_acc = df_acc - df.loc[j][accs]
                            acc[feat].append(df_delta_acc.to_dict())
                            acc[feat][-1]["ModelName 1"] = df.loc[i]["ModelName"]
                            acc[feat][-1]["ModelName 0"] = df.loc[j]["ModelName"]
                            is_find = False
                        else:
                            print("not unique")
        print()
    save_list = []
    for feat in acc:
        df_feat = pd.DataFrame(acc[feat])
        df_feat_des = df_feat.describe()

        line = method_name(df_feat_des, feat, "max")
        save_list.append(line)

        line = method_name(df_feat_des, feat, "mean")
        save_list.append(line)

        line = method_name(df_feat_des, feat, "min")
        save_list.append(line)

    save_list = pd.DataFrame(save_list)
    save_list.to_csv(r"F:\ProjectSet\Shadow\QingDao\Dissertation\4_Analyze\accuracyChange\t1.csv")
    saveJson(acc, r"F:\ProjectSet\Shadow\QingDao\Dissertation\4_Analyze\acc_change1.json")


def method_name(df_feat_des, feat, line_type):
    mean_line = df_feat_des.loc[line_type].to_dict()
    mean_line["Line Type"] = line_type + " " + feat
    return mean_line


def main():
    csv_fn = r"F:\ProjectSet\Shadow\QingDao\Mods\20230707H200910\train_save_20230707H200910.csv"
    # "ModelName",
    # "OATrain", "KappaTrain", "IS UATrain","IS PATrain", "VEG UATrain", "VEG PATrain", "SOIL UATrain",
    # "SOIL PATrain", "WAT UATrain", "WAT PATrain",
    # "OATest", "KappaTest", "IS UATest", "IS PATest", "VEG UATest", "VEG PATest", "SOIL UATest",
    # "SOIL PATest", "WAT UATest", "WAT PATest",
    # "MOD_TYPE", "SPL_TYPE", "FEAT_TYPE", "TAG_TYPE",
    # "TRAIN_CM", "TEST_CM", "NUMBER",
    # "OPTICS",
    # "RF", "SVM", "SPL_NOSH", "SPL_SH",
    # "AS_SIGMA", "AS_C2", "AS_LAMD", "DE_SIGMA", "DE_C2", "DE_LAMD",
    # "TAG",

    # sag = ShadowAccuracyGroup(csv_fn)
    # sag.print()
    # sag.clf_name = "SVM"
    # sag.spl_name = "SPL_NOSH"
    # sag.showFeatureHist("DE_SIGMA", "OATest")
    # plt.show()

    accuracyChange(csv_fn)

    pass


if __name__ == "__main__":
    main()
