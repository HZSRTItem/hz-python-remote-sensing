# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2MLModel.py
@Time    : 2024/7/4 14:19
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2MLModel
-----------------------------------------------------------------------------"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from Shadow.Hierarchical.SHH2Model import mapDict


class SHH2MLTraining:

    def __init__(self):
        self.category_names = None
        self.df = None
        self.models = {}
        self.categorys = {}
        self.acc_dict = {}
        self.clf = None
        self.map_dict = None
        self.test_filters = {0: [], 1: []}

    def train(self, name, x_keys=None, c_fn="CATEGORY", map_dict=None, clf=None, *args, **kwargs):
        if x_keys is None:
            x_keys = []
        if map_dict is None:
            map_dict = self.map_dict

        x_train, y_train, category_names, df_train = self.train_test(1, x_keys, c_fn, map_dict, )
        x_test, y_test, category_names, df_test = self.train_test(0, x_keys, c_fn, map_dict, )

        self.category_names = category_names

        if clf is None:
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=1, min_samples_split=2)
            # clf = SVC(kernel="rbf", C=4.742, gamma=0.42813)

        clf.fit(x_train, y_train)
        train_acc, test_acc = clf.score(x_train, y_train) * 100, clf.score(x_test, y_test) * 100
        self.clf = clf

        to_dict = self.addAccuracy(df_test, x_test, y_test)

        self.categorys[name] = to_dict
        self.acc_dict[name] = {}

        self.models[name] = clf
        return train_acc, test_acc

    def addAccuracy(self, df_test, x_test, y_test):
        y2 = self.clf.predict(x_test)
        to_dict = {"y1": y_test, "y2": y2.tolist(), }
        if "X" in df_test:
            to_dict["X"] = df_test["X"].tolist()
        if "Y" in df_test:
            to_dict["Y"] = df_test["Y"].tolist()
        if "SRT" in df_test:
            to_dict["SRT"] = df_test["SRT"].tolist()
        if "CNAME" in df_test:
            to_dict["CNAME"] = df_test["CNAME"].tolist()
        return to_dict

    def readCSV(self, csv_fn):
        self.df = pd.read_csv(csv_fn)

    def toCSV(self, csv_fn, **kwargs):
        self.df.to_csv(csv_fn, index=False, **kwargs)

    def train_test(self, n, x_keys, c_fn=None, map_dict=None):
        _df = self.df[self.df["TEST"] == n]

        data_filter = self.test_filters[n]
        if len(data_filter) != 0:
            df_list = []
            for k, data in data_filter:
                df_list.extend(_df[_df[k] == data].to_dict("records"))
            _df = pd.DataFrame(df_list)

        y, data_select = mapDict(_df[c_fn].tolist(), map_dict, return_select=True)
        _df = _df[data_select]
        x = _df[x_keys].values
        return x, y, _df["CNAME"].tolist(), _df


def main():
    pass


if __name__ == "__main__":
    main()
