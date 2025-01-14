import time
import random
from typing import Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sys
import datetime

_LOG_FN = r"F:\ChinaNorthIS\Run\Models\20240928H202156\hbmodeld.log"

def _this_print(*texts, sep=" ",  end='\n', file=None, flush=False):
    print(*texts, sep=sep, end=end, file=file, flush=flush)
    with open(_LOG_FN, "a", encoding="utf-8") as fw:
        print(*texts, sep=sep, end=end, file=fw, flush=True)

def _show_time():
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
    _this_print("\n[{}]\n".format(formatted_time))


S2_SELECT_NAMES = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12",
    "ND1", "NDVI", "NDWI", "ND4", "ND5", "ND6", "ND7", "ND8", "ND9", "ND10", "ND11", "ND12", "ND13", "ND14", "ND15",
    "Gray_asm", "Gray_contrast", "Gray_corr", "Gray_var", "Gray_idm", "Gray_ent", "Gray_diss",
    "NDWI_stdDev", "NDVI_stdDev", "MNDWI_stdDev", "NDBI_stdDev", "NDVI_max"
]
S1_AS_SELECT_NAMES = [
    "AS_VV", "AS_VH", "AS_VH_VV",
    "AS_VV_stdDev", "AS_VH_stdDev",
    "AS_VV_asm", "AS_VV_contrast", "AS_VV_corr", "AS_VV_var", "AS_VV_idm", "AS_VV_ent", "AS_VV_diss",
    "AS_VH_asm", "AS_VH_contrast", "AS_VH_corr", "AS_VH_var", "AS_VH_idm", "AS_VH_ent", "AS_VH_diss",
]
S1_DE_SELECT_NAMES = [
    "DE_VV", "DE_VH", "DE_VH_VV",
    "DE_VV_stdDev", "DE_VH_stdDev",
    "DE_VV_asm", "DE_VV_contrast", "DE_VV_corr", "DE_VV_var", "DE_VV_idm", "DE_VV_ent", "DE_VV_diss",
    "DE_VH_asm", "DE_VH_contrast", "DE_VH_corr", "DE_VH_var", "DE_VH_idm", "DE_VH_ent", "DE_VH_diss",
]


class Jdt:
    """
    进度条
    """

    def __init__(self, total=100, desc=None, iterable=None, n_cols=20):
        """ 初始化一个进度条对象

        :param iterable: 可迭代的对象, 在手动更新时不需要进行设置
        :param desc: 字符串, 左边进度条描述文字
        :param total: 总的项目数
        :param n_cols: 调整进度条宽度, 默认是根据环境自动调节长度, 如果设置为0, 就没有进度条, 只有输出的信息
        """
        self.total = total
        self.iterable = iterable
        self.n_cols = n_cols
        self.desc = desc if desc is not None else ""

        self.n_split = float(total) / float(n_cols)
        self.n_current = 0
        self.n_print = 0
        self.is_run = False

        self.current_time = time.time()
        self.init_time = time.time()

    def start(self, is_jdt=True):
        """ 开始进度条 """
        if not is_jdt:
            return self
        self.is_run = True
        self._print()
        self.current_time = time.time()
        self.init_time = time.time()
        return self

    def add(self, n=1, is_jdt=True):
        """ 添加n个进度

        :param n: 进度的个数
        :return:
        """
        if not is_jdt:
            return
        if self.is_run:
            self.n_current += n
            self.current_time = time.time()
            if self.n_current > self.n_print * self.n_split:
                self.n_print += 1
                if self.n_print > self.n_cols:
                    self.n_print = self.n_cols
            self._print()

    def setDesc(self, desc):
        """ 添加打印信息 """
        self.desc = desc

    def fmttime(self):
        d_time = (self.current_time - self.init_time) / (self.n_current + 0.000001)

        def timestr(_time):

            tian = int(_time // (60 * 60 * 24))
            _time = _time % (60 * 60 * 24)
            shi = int(_time // (60 * 60))
            _time = _time % (60 * 60)
            fen = int(_time // 60)
            miao = int(_time % 60)

            if tian >= 1:
                return "{}D {:02d}:{:02d}:{:02d}".format(tian, shi, fen, miao)
            if shi >= 1:
                return "{:02d}:{:02d}:{:02d}".format(shi, fen, miao)
            return "{:02d}:{:02d}".format(fen, miao)

        n = 1 / (d_time + 0.000001)
        n_str = "{:.1f}".format(n)
        if len(n_str) >= 6:
            n_str = ">1000".format(n)

        fmt = "[{}<{}, {}it/s]            ".format(timestr(
            self.current_time - self.init_time), timestr(d_time * self.total), n_str)
        return fmt

    def _print(self):
        des_info = "\r{0}: {1:>3d}% |".format(self.desc, int(self.n_current / self.total * 100))
        des_info += "*" * self.n_print + "-" * (self.n_cols - self.n_print)
        des_info += "| {0}/{1}".format(self.n_current, self.total)
        des_info += " {}".format(self.fmttime())
        print(des_info, end="")

    def end(self, is_jdt=True):
        if not is_jdt:
            return
        """ 结束进度条 """
        self.n_split = float(self.total) / float(self.n_split)
        self.n_current = 0
        self.n_print = 0
        self.is_run = False
        print()


class TableLinePrint:

    def __init__(self, widths: Union[int, list] = 20, n_float: int = 6, alignment=">"):
        self.widths = widths if isinstance(widths, list) else [widths]
        self.n_float = n_float
        self.alignment = alignment
        self.float_fmts = None
        self.other_fmts = None

    def _getFmts(self):
        self.float_fmts = ["{:" + self.alignment + str(self.widths[i]) + "." + str(self.n_float) + "f}" for i in
                           range(len(self.widths))]
        self.other_fmts = ["{:" + self.alignment + str(self.widths[i]) + "}" for i in range(len(self.widths))]

    def separationLine(self, _type="-", sep=" ", end="\n", func_print=_this_print):
        texts = ["-" * width for width in self.widths]
        func_print(*texts, sep=sep, end=end, )

    def print(self, *texts, sep=" ", end="\n", func_print=_this_print):
        if len(texts) > len(self.widths):
            for i in range(len(self.widths), len(texts)):
                self.widths.append(self.widths[-1])
            self._getFmts()
        texts = list(texts)
        for i, text in enumerate(texts):
            if isinstance(text, float):
                text = self.float_fmts[i].format(text)
            else:
                text = self.other_fmts[i].format(text)
            texts[i] = text
        func_print(*texts, sep=sep, end=end, )


def df_cat(df_data, df, is_update=False):
    df_list = df_data.sort_values("SRT").to_dict("records")
    df_data_list = df.sort_values("SRT").to_dict("records")
    jdt = Jdt(len(df_list), "func27").start()
    i_spl_tmp, n_tmp = 0, 0
    for spl in df_list:
        for i_spl_tmp in range(n_tmp, len(df_data_list)):
            n_tmp = i_spl_tmp
            if spl["SRT"] == df_data_list[i_spl_tmp]["SRT"]:
                spl_tmp = df_data_list[i_spl_tmp]
                for name in spl_tmp:
                    if not is_update:
                        if name not in spl:
                            spl[name] = spl_tmp[name]
                    else:
                        spl[name] = spl_tmp[name]
                break
        if n_tmp == (len(df_data_list) - 1):
            n_tmp = 0
        jdt.add()
    jdt.end()
    df = pd.DataFrame(df_list)
    return df


def func_get_data1(year, feature_names, category_name=None, data_deal=None,
                       random_select=None, data_coll=None, filter_func=None):
    if category_name is None:
        category_name = "CATEGORY2_{}".format(year)
    df_category = pd.read_csv(
        r"F:\ChinaNorthIS\Run\Samples\hb\update2\hb_spl_random_INIT_release1_select10w_update2_data2.csv")
    df_category = df_category[["SRT", "CATEGORY2_{}".format(year), "IS4_CATE", "GUB"]]
    df_category["CATEGORY{}".format(year)] = df_category["CATEGORY2_{}".format(year)]
    csv_fn_fmt = r"F:\ChinaNorthIS\Run\Samples\hb\update2\hb_update2_{}_spl1.csv"
    csv_fn = csv_fn_fmt.format(year)
    _this_print("csv_fn", csv_fn)
    df = pd.read_csv(csv_fn)
    df = df_cat(df_category, df, is_update=True)

    if filter_func is not None:
        _list = df.to_dict("records")
        _list_2 = []
        for _spl in _list:
            if filter_func(_spl):
                _list_2.append(_spl)
        df = pd.DataFrame(_list_2)

    _this_print(df["CATEGORY{}".format(year)].sum(), df["CATEGORY2_{}".format(year)].sum())
    _this_print(df)

    if data_deal is not None:
        if data_deal == "minmax":
            for feat_name in feature_names:
                df[feat_name] = (df[feat_name] - df[feat_name].min()) / (df[feat_name].max() - df[feat_name].min())
        elif data_deal == "z-score":
            for feat_name in feature_names:
                df[feat_name] = (df[feat_name] - df[feat_name].mean()) / df[feat_name].std()

    def _train_test_data(_test):
        _df = df[df["TEST"] == _test]
        _df = _df[feature_names + [category_name]]
        _df = _df.dropna(axis=0, how="any")
        if random_select is not None:
            if _test == 1:
                _df_list = _df.to_dict("records")
                n = random_select
                if 0 < random_select < 1:
                    n = random_select * len(_df_list)
                _df_list = random.sample(_df_list, int(n))
                _df = pd.DataFrame(_df_list)
        _x = _df[feature_names]
        _y = _df[category_name].tolist()
        return _x, _y

    x_train, y_train = _train_test_data(1)
    x_test, y_test = _train_test_data(0)
    _this_print("train[{}] -> 1:{}, 0:{}".format(len(y_train), np.sum(y_train), np.size(y_train)-np.sum(y_train)), )
    _this_print("test[{}] -> 1:{}, 0:{}".format(len(y_test), np.sum(y_test), np.size(y_test)-np.sum(y_test)), )

    return feature_names, x_test, x_train, y_test, y_train


if __name__ == "__main__":
    def _filter_fun(_spl):
        if _spl["GUB"] == 0:
            if _spl["IS4_CATE"] == 1:
                return False
        return True
    
    feature_names_dict = {
        "AS":S2_SELECT_NAMES + S1_AS_SELECT_NAMES, 
        "ASDE":S2_SELECT_NAMES + S1_AS_SELECT_NAMES + S1_DE_SELECT_NAMES, 
    }

    # year = 16
    # init_feature_names = S2_SELECT_NAMES + S1_AS_SELECT_NAMES
    # to_name = "ASDE"

    def run(year, to_name):
        _show_time()
        _this_print("\n#", "-"*50, year, "-"*50,"#")
        _this_print("#", to_name, "-"*6)

        init_feature_names = feature_names_dict[to_name]
        feature_names, x_test, x_train, y_test, y_train = func_get_data1(
                year, init_feature_names,
                # data_deal="minmax_0.01",
                filter_func=_filter_fun
            )
        
        tlp = TableLinePrint()
        tlp.print("No.", "NAME", "MIN", "MAX")
        tlp.separationLine()
        for i, feat_name in enumerate(feature_names):
            tlp.print(i + 1, feat_name, x_train[feat_name].min(), x_train[feat_name].max(), )

        clf = RandomForestClassifier(
                **{'n_estimators': 120, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2})
        clf.fit(x_train, y_train)
        _this_print("Accuracy:", clf.score(x_test, y_test))

        to_fn = r"F:\ChinaNorthIS\Run\Models\20240928H202156\hb_rf_mod_{}_2_{}.mod".format(to_name, year)
        _this_print("to_fn", to_fn)
        joblib.dump(clf, to_fn)
    
    # run(17, "ASDE")
    # run(17, "AS")

    # run(18, "ASDE")
    # run(18, "AS")

    # run(19, "ASDE")
    # run(19, "AS")

    # run(20, "ASDE")
    # run(20, "AS")

    # run(21, "ASDE")
    # run(21, "AS")

    run(22, "AS")
    run(23, "AS")

    


