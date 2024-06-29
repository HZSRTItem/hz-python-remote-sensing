# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2MLThree.py
@Time    : 2024/6/19 22:28
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2MLThree
-----------------------------------------------------------------------------"""

from sklearn.svm import SVC

from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.SRTFeature import SRTFeaturesMemory
from Shadow.Hierarchical.SHH2Accuracy import accuracyY12


def mapDict(data, map_dict):
    if map_dict is None:
        return data
    to_list = []
    for d in data:
        if d in map_dict:
            to_list.append(map_dict[d])
    return to_list


class SHH2MLTrainSamplesTiao:

    def __init__(self):
        self.map_dict = None
        self.clf = None
        self.df = None
        self.categorys = {}
        self.acc_dict = {}
        self.category_names = []

    def train(self, name, x_keys=None, c_fn="CATEGORY", map_dict=None, clf=None):
        if x_keys is None:
            x_keys = []

        def train_test(n):
            _df = self.df[self.df["TEST"] == n]
            x = _df[x_keys].values
            y = mapDict(_df[c_fn].tolist(), map_dict)
            return x, y, _df["CNAME"].tolist(), _df

        x_train, y_train, category_names, df_train = train_test(1)
        x_test, y_test, category_names, df_test = train_test(0)
        self.category_names = category_names

        if clf is None:
            # clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=1, min_samples_split=2)
            clf = SVC(kernel="rbf", C=4.742, gamma=0.42813)

        clf.fit(x_train, y_train)

        train_acc, test_acc = clf.score(x_train, y_train) * 100, clf.score(x_test, y_test) * 100

        y2 = clf.predict(x_test)
        to_dict = {"y1": y_test, "y2": y2.tolist(), }
        if "X" in df_test:
            to_dict["X"] = df_test["X"].tolist()
        if "Y" in df_test:
            to_dict["Y"] = df_test["Y"].tolist()
        if "SRT" in df_test:
            to_dict["SRT"] = df_test["SRT"].tolist()

        self.categorys[name] = to_dict
        self.acc_dict[name] = {}
        self.clf = clf

        return train_acc, test_acc

    def accuracyOAKappa(self, cm_name, cnames, y1_map_dict=None, y2_map_dict=None, fc_category=None, is_eq_number=True):
        cm_str = ""
        for name, line in self.categorys.items():
            # y1 = mapDict(line["y1"], y1_map_dict)
            # y2 = mapDict(line["y2"], y2_map_dict)
            # cm = ConfusionMatrix(class_names=cnames)
            # cm.addData(y1, y2)

            cm = accuracyY12(
                self.category_names, line["y2"],
                self.map_dict, y2_map_dict,
                cnames=["IS", "VEG", "SOIL", "WAT"],
                fc_category=fc_category,
                is_eq_number=is_eq_number,
            )

            cm = cm.accuracyCategory("IS")
            cm_str += "> {} IS\n".format(name)
            cm_str += "{}\n\n".format(cm.fmtCM())
            self.acc_dict[name]["{}_OA".format(cm_name)] = cm.OA()
            self.acc_dict[name]["{}_Kappa".format(cm_name)] = cm.getKappa()

            cm2 = ConfusionMatrix(class_names=["IS", "VEG", "SOIL", "WAT"])
            cm2.addData(line["y1"], line["y2"])
            cm_str += "> {} {}\n".format(name, " ".join(["IS", "VEG", "SOIL", "WAT"]))
            cm_str += "{}\n\n".format(cm2.fmtCM())

            # cm3 = cm2.accuracyCategory("IS")
            # self.acc_dict[name]["{}_OA2".format(cm_name)] = cm3.OA()
            # self.acc_dict[name]["{}_Kappa2".format(cm_name)] = cm3.getKappa()

        return cm_str


class SHH2ML_TST_Main:

    def __init__(self, city_name=None, is_save_model=False):
        self.city_name = city_name
        self.y1_map_dict = None
        self.y2_map_dict = None
        self.fc_category = None
        self.json_dict = None
        self.train_test_df = None
        self.sfm = SRTFeaturesMemory()
        self.csv_fn = None
        self.td = None
        self.map_dict = {}
        self.category_cnames = None
        self.acc_dict_list = []
        self.mod_list = []
        self.color_table = None
        self.raster_fn = None
        self.n_run = 5
        self.dfn = None
        self.json_fn = None
        self.is_save_model = is_save_model
        self.is_eq_number = True




class SHH2MLThreeMain:

    def __init__(self):
        super(SHH2MLThreeMain, self).__init__()




def main():
    pass


if __name__ == "__main__":
    main()
