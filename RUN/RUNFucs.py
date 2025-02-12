# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : RUNFucs.py
@Time    : 2023/12/4 21:01
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of RUNFucs
-----------------------------------------------------------------------------"""
import csv
import os
import shutil
import time

import pandas as pd
from tabulate import tabulate

from SRTCodes.SRTReadWrite import SRTInfoFileRW
from SRTCodes.Utils import concatCSV


def checkDFKeys(df, k, filename=""):
    if k not in df:
        print("Can not find field name \"{0}\" in DataFrame of {1}".format(k, filename))
    return k in df


CATEGORY_DICT = {
    "shadow": [{"code": 0, "color": (0, 0, 0), "name": "NOT_KNOW"},
               {"code": 11, "color": (255, 0, 0), "name": "IS"},
               {"code": 12, "color": (125, 0, 0), "name": "IS_SH"},
               {"code": 21, "color": (0, 255, 0), "name": "VEG"},
               {"code": 22, "color": (0, 125, 0), "name": "VEG_SH"},
               {"code": 31, "color": (255, 255, 0), "name": "SOIL"},
               {"code": 32, "color": (125, 125, 0), "name": "SOIL_SH"},
               {"code": 41, "color": (0, 0, 255), "name": "WAT"},
               {"code": 42, "color": (0, 0, 125), "name": "WAT_SH"}],
    "shadow2": [{"code": 0, "color": (0, 0, 0), "name": "NOT_KNOW"},
                {"code": 11, "color": (255, 0, 0), "name": "IS"},
                {"code": 12, "color": (125, 0, 0), "name": "IS_SH"},
                {"code": 13, "color": (125, 0, 0), "name": "IS_AS_SH"},
                {"code": 14, "color": (125, 0, 0), "name": "IS_DE_SH"},
                {"code": 21, "color": (0, 255, 0), "name": "VEG"},
                {"code": 22, "color": (0, 125, 0), "name": "VEG_SH"},
                {"code": 23, "color": (0, 125, 0), "name": "VEG_AS_SH"},
                {"code": 24, "color": (0, 125, 0), "name": "VEG_DE_SH"},
                {"code": 31, "color": (255, 255, 0), "name": "SOIL"},
                {"code": 32, "color": (125, 125, 0), "name": "SOIL_SH"},
                {"code": 33, "color": (125, 125, 0), "name": "SOIL_AS_SH"},
                {"code": 34, "color": (125, 125, 0), "name": "SOIL_DE_SH"},
                {"code": 41, "color": (0, 0, 255), "name": "WAT"},
                {"code": 42, "color": (0, 0, 125), "name": "WAT_SH"},
                {"code": 43, "color": (0, 0, 125), "name": "WAT_AS_SH"},
                {"code": 44, "color": (0, 0, 125), "name": "WAT_DE_SH"}, ],
    "is": [{"code": 0, "color": (0, 0, 0), "name": "NOT_KNOW"},
           {"code": 1, "color": (255, 0, 0), "name": "IS"},
           {"code": 2, "color": (0, 255, 0), "name": "NOIS"}],
    # 1: (0, 255, 0), 2: (200, 200, 200), 3: (36, 36, 36),
    "vhl": [{"code": 0, "color": (0, 0, 0), "name": "NOT_KNOW"},
            {"code": 1, "color": (0, 255, 0), "name": "VEG"},
            {"code": 2, "color": (200, 200, 200), "name": "HIGH"},
            {"code": 3, "color": (36, 36, 36), "name": "LOW"}],
    "is_fc": [{"code": 0, "color": (0, 0, 0), "name": "NOT_KNOW"},
              {"code": 1, "color": (255, 255, 0), "name": "SOIL"},
              {"code": 2, "color": (255, 0, 0), "name": "IS"}, ],
}


def df2SplTxt(df, category, to_filename, ):
    if isinstance(category, str):
        category = CATEGORY_DICT[category]

    with open(to_filename, "w", encoding="utf-8") as f:
        f.write("# QGIS SAMPLE (C)Copyright 2023, ZhengHan. All rights reserved.\n"
                "# If this file is generated for the first time, Please modify as follows\n"
                "\n"
                "> CATEGORY\n"
                "\n"
                "# Each category is on one line, with the following format\n"
                "#  - CODE: (R,G,B) | NAME\n"
                "# Example: `0: (  0,   0,   0) | NOT_KNOW`\n"
                "# The available colors are as follows:\n"
                "#   (  0,   0,   0, ) (  0, 255,   0, ) (255,   0,   0, ) \n"
                "#   (255, 255,   0, ) (  0,   0, 255, ) (255, 226, 148, ) \n"
                "#   (132, 150, 176, ) (165, 165, 165, ) (255, 102, 255, ) \n"
                "#   (  0, 102, 255, ) \n"
                "\n")
        category_code = {}
        for cate in category:
            f.write("{0:>3d}: ({1:>3d},{2:>3d},{3:>3d}) | {4}\n".format(
                cate["code"], cate["color"][0], cate["color"][1], cate["color"][2], cate["name"]))
            category_code[cate["code"]] = cate["name"]
        f.write("\n"
                "> FIELDS\n"
                "\n"
                "# Some fields for each sample. \n"
                "# Each field is on one line\n"
                "\n")
        is_write_data = True
        while df is not None:

            if df is None:
                break

            if not checkDFKeys(df, "X", "df"):
                break
            if not checkDFKeys(df, "Y", "df"):
                break
            if not checkDFKeys(df, "CATEGORY", "df"):
                break

            for k in df:
                if k not in ["X", "Y", "CATEGORY"]:
                    f.write(str(k))
                    f.write("\n")

            f.write("\n")
            f.write("> DATA\n"
                    "\n"
                    "# Please do not change\n"
                    "\n")
            is_write_data = False

            for i in range(len(df)):
                f.write(str(i + 1))
                f.write(",")
                f.write(str(category_code[int(df["CATEGORY"][i])]))
                f.write(",False,")
                f.write(str(float(df["X"][i])))
                f.write(",")
                f.write(str(float(df["Y"][i])))
                for k in df:
                    if k not in ["X", "Y", "CATEGORY"]:
                        f.write(",")
                        f.write(str(df[k][i]))
                f.write("\n")
            break

        if is_write_data:
            f.write("> DATA\n"
                    "\n"
                    "# Please do not change\n"
                    "\n")


class QJYTxt_main:

    def __init__(self):
        self.name = "qjy_txt"
        self.description = "Build QJY txt file from excel or csv file"
        self.argv = []

        self.category_dict = CATEGORY_DICT

    def usage(self):
        print("{0} [filename] [-o] [-sheet_name] [-ccn code|color|name]\n"
              "@Des: {1}\n"
              "    filename: excel or csv filename\n"
              "    -o: to file name default:\"spl.txt\"\n"
              "    -sheet_name: excel sheet name\n"
              "    -ccn: category of \"code|color|name\" or shadow|is|vhl\n"
              "    -to_csv spl_txt_fn to_csv_fn"
              "".format(self.name, self.description))

    def run(self, argv):

        if len(argv) == 1:
            self.usage()
            return

        filename = None
        to_filename = "spl.txt"
        sheet_name = None
        category_dict = []

        i = 1
        while i < len(argv):
            if "-sheet_name" == argv[i] and i < len(argv) - 1:
                sheet_name = argv[i + 1]
                i += 1
            elif "-o" == argv[i] and i < len(argv) - 1:
                to_filename = argv[i + 1]
                i += 1
            elif "-to_csv" == argv[i] and i < len(argv) - 2:
                print(argv[i + 1], argv[i + 2])
                splTxt2Csv(argv[i + 1], argv[i + 2])
                return
            elif "-ccn" == argv[i] and i < len(argv) - 1:
                name = argv[i + 1]
                if name in self.category_dict:
                    category_dict += self.category_dict[name]
                else:
                    try:
                        names = name.split("|")
                        category_dict.append({"code": int(names[0]), "color": eval(names[1]), "name": str(names[2])})
                    except:
                        print("Warning: can not format category of \"{0}\"".format(name))
            elif filename is None:
                filename = os.path.abspath(argv[i])
            i += 1

        if not category_dict:
            category_dict = self.category_dict["shadow"]

        to_tishi_line = ""
        ext = os.path.splitext(filename)[1]
        if ext == ".xlsx" or ext == ".xls":
            to_tishi_line = "{0}, sheet_name={1}".format(filename, sheet_name)
        elif ext == ".csv" or ext == ".txt":
            to_tishi_line = "{0}".format(filename)

        ext = os.path.splitext(filename)[1]
        df = None
        if ext == ".xlsx" or ext == ".xls":
            df = pd.read_excel(filename, sheet_name=sheet_name)
        elif ext == ".csv" or ext == ".txt":
            df = pd.read_csv(filename)

        with open(to_filename, "w", encoding="utf-8") as f:
            f.write("# QGIS SAMPLE (C)Copyright 2023, ZhengHan. All rights reserved.\n"
                    "# If this file is generated for the first time, Please modify as follows\n"
                    "# {0}"
                    "\n"
                    "> CATEGORY\n"
                    "\n"
                    "# Each category is on one line, with the following format\n"
                    "#  - CODE: (R,G,B) | NAME\n"
                    "# Example: `0: (  0,   0,   0) | NOT_KNOW`\n"
                    "# The available colors are as follows:\n"
                    "#   (  0,   0,   0, ) (  0, 255,   0, ) (255,   0,   0, ) \n"
                    "#   (255, 255,   0, ) (  0,   0, 255, ) (255, 226, 148, ) \n"
                    "#   (132, 150, 176, ) (165, 165, 165, ) (255, 102, 255, ) \n"
                    "#   (  0, 102, 255, ) \n"
                    "\n".format(to_tishi_line))
            category_code = {}
            for cate in category_dict:
                f.write("{0:>3d}: ({1:>3d},{2:>3d},{3:>3d}) | {4}\n".format(
                    cate["code"], cate["color"][0], cate["color"][1], cate["color"][2], cate["name"]))
                category_code[cate["code"]] = cate["name"]
            f.write("\n"
                    "> FIELDS\n"
                    "\n"
                    "# Some fields for each sample. \n"
                    "# Each field is on one line\n"
                    "\n")
            is_write_data = True
            while filename is not None:

                if df is None:
                    break

                if not checkDFKeys(df, "X", filename):
                    break
                if not checkDFKeys(df, "Y", filename):
                    break
                if not checkDFKeys(df, "CATEGORY", filename):
                    break

                for k in df:
                    if k not in ["X", "Y", "CATEGORY"]:
                        f.write(str(k))
                        f.write("\n")

                f.write("\n")
                f.write("> DATA\n"
                        "\n"
                        "# Please do not change\n"
                        "\n")
                is_write_data = False

                for i in range(len(df)):
                    f.write(str(i + 1))
                    f.write(",")
                    f.write(str(category_code[int(df["CATEGORY"][i])]))
                    f.write(",False,")
                    f.write(str(float(df["X"][i])))
                    f.write(",")
                    f.write(str(float(df["Y"][i])))
                    for k in df:
                        if k not in ["X", "Y", "CATEGORY"]:
                            f.write(",")
                            f.write(str(df[k][i]))
                    f.write("\n")
                break

            if is_write_data:
                f.write("> DATA\n"
                        "\n"
                        "# Please do not change\n"
                        "\n")


def fmtLinesCSV(lines):
    cr = csv.reader(lines)
    return [line for line in cr]


def main():
    # spl_txt_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\13\sh2_spl13_12_spl2.txt"
    # to_csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\13\sh2_spl13_12_spl2_3.csv"
    #
    # splTxt2Csv(spl_txt_fn, to_csv_fn)

    return


def splTxt2Csv(spl_txt_fn, to_csv_fn):
    sif_rw = SRTInfoFileRW(spl_txt_fn)
    to_dict = sif_rw.readAsDict()
    lines = fmtLinesCSV(to_dict["DATA"])
    fields = ["N", "X", "Y", "CATEGORY_NAME", "CATEGORY_CODE", "IS_TAG"] + to_dict["FIELDS"]
    name_to_code = {}
    for category_str in to_dict["CATEGORY"]:
        category_str = category_str.strip()
        clist1 = category_str.split(":", maxsplit=1)
        c_code = int(clist1[0])
        clist2 = clist1[1].split("|", maxsplit=1)
        c_color = eval(clist2[0])
        name = clist2[1].strip()
        name_to_code[name] = c_code
    with open(to_csv_fn, "w", encoding="utf-8", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(fields)
        for i, line in enumerate(lines):
            for j in range(5):
                line[j] = line[j].strip()
            to_line = [i + 1, line[3], line[4], line[1], name_to_code[line[1]], line[2], *line[5:]]
            cw.writerow(to_line)


def splTxt2Dict(spl_txt_fn, is_ret_name_to_code=False):
    sif_rw = SRTInfoFileRW(spl_txt_fn)
    to_dict = sif_rw.readAsDict()
    lines = fmtLinesCSV(to_dict["DATA"])
    fields = ["N", "X", "Y", "CATEGORY_NAME", "CATEGORY_CODE", "IS_TAG"] + to_dict["FIELDS"]
    name_to_code = {}
    for category_str in to_dict["CATEGORY"]:
        category_str = category_str.strip()
        clist1 = category_str.split(":", maxsplit=1)
        c_code = int(clist1[0])
        clist2 = clist1[1].split("|", maxsplit=1)
        c_color = eval(clist2[0])
        name = clist2[1].strip()
        name_to_code[name] = c_code

    to_dict = {k: [] for k in fields}
    for i, line in enumerate(lines):
        for j in range(5):
            line[j] = line[j].strip()
        to_line = [i + 1, line[3], line[4], line[1], name_to_code[line[1]], line[2], *line[5:]]
        for j, k in enumerate(to_dict):
            to_dict[k].append(to_line[j])
    return to_dict


class DFColumnCount_main:

    def __init__(self):
        self.name = "dfcolumncount"
        self.description = "Calculate the number of independent columns"
        self.argv = []

    def usage(self):
        print("{0} [filename] [name] \n"
              "@Des: {1}\n"
              "    filename: excel or csv filename\n"
              "    -sheet_name: excel sheet name default:Sheet1"
              "    -gn: group field name"
              "".format(self.name, self.description))

    def run(self, argv):

        if len(argv) == 1:
            self.usage()
            return

        filename = None
        sheet_name = "Sheet1"
        name = None
        gn = None

        i = 1
        while i < len(argv):
            if "-sheet_name" == argv[i] and i < len(argv) - 1:
                sheet_name = argv[i + 1]
                i += 1
            elif "-gn" == argv[i] and i < len(argv) - 1:
                gn = argv[i + 1]
                i += 1
            elif filename is None:
                filename = os.path.abspath(argv[i])
            elif name is None:
                name = argv[i]
            i += 1

        ext = os.path.splitext(filename)[1]
        if ext == ".xlsx" or ext == ".xls":
            df = pd.read_excel(filename, sheet_name=sheet_name)
        elif ext == ".csv" or ext == ".txt":
            df = pd.read_csv(filename)
        else:
            return

        if gn is None:
            data_list = df[name].tolist()
            counts_dict = {}
            for data in data_list:
                data = str(data)
                if data not in counts_dict:
                    counts_dict[data] = 0
                counts_dict[data] += 1

            len_max = max([len(k) for k in counts_dict])
            if len_max < 5:
                len_max = 5
            n = sum([k for k in counts_dict.values()])
            fmt = "{0:" + str(len_max) + "} {1}"
            print(fmt.format("NAME", "COUTS"))
            print("-" * len_max, "-" * 6)
            for k, data in counts_dict.items():
                print(fmt.format(k, data))
            print("-" * len_max, "-" * 6)
            print(fmt.format("ALL", n))

        else:
            data = df.groupby(gn)[name].value_counts()
            data_dict = {}
            for k in data.index:
                if k[0] not in data_dict:
                    data_dict[k[0]] = {}
                data_dict[k[0]][k[1]] = data[k]
            df_des = pd.DataFrame(data_dict)
            df_des[pd.isna(df_des)] = 0
            df_des["SUM"] = df_des.apply(lambda x: x.sum(), axis=1)
            df_des.loc["SUM"] = df_des.apply(lambda x: x.sum())
            print(tabulate(df_des, headers="keys"))


class ConcatCSV_main:

    def __init__(self):
        self.name = "concatcsv"
        self.description = "Concate csv files"
        self.argv = []

    def usage(self):
        print("{0} to_csv_fn [-filelist filelist.txt] [-dir dirname] [-filter string] [--y]"
              "*csv_fns".format(self.name))
        print("@Des: {0}".format(self.description))
        print("    to_csv_fn: save csv file")
        print("    -filelist: csv filelist")
        print("    -dir: dirname of all csv file")
        print("    -filter: filters of dirname")
        print("    --y: to csv file exist, whether delete")
        print("    *csv_fns: csv files")

    def run(self, argv):
        self.argv = argv
        if len(argv) == 1:
            self.usage()
            return None

        to_csv_fn = None
        csv_fns = []
        ext = ".tif"
        is_to_csv_fn_del = False

        i = 1
        while i < len(argv):
            if (argv[i] == "-filelist") and (i < len(argv) - 1):
                i += 1
                with open(argv[i], "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        line = line.strip("\"")
                        if line == "":
                            continue
                        csv_fns.append(line)

            elif (argv[i] == "-dir") and (i < len(argv) - 1):
                i += 1
                dirname = argv[i]
                for fn in os.listdir(dirname):
                    fn = os.path.join(dirname, fn)
                    if os.path.isfile(fn):
                        if fn.endswith(ext):
                            csv_fns.append(fn)

            elif (argv[i] == "-filter") and (i < len(argv) - 1):
                i += 1
                ext = argv[i]

            elif argv[i] == "--y":
                is_to_csv_fn_del = True

            elif to_csv_fn is None:
                to_csv_fn = argv[i]

            else:
                csv_fns.append(argv[i])

            i += 1

        if to_csv_fn is None:
            print("Not find to csv file")
            return

        if len(csv_fns) == 0:
            print("Not find csv files")
            return

        if not os.path.isfile(to_csv_fn):
            concatCSV(*csv_fns, to_csv_fn=to_csv_fn)
            print("Save to", to_csv_fn)
        else:
            if not is_to_csv_fn_del:
                is_y = input("CSV file existed: \"{}\"\nWhether delete or not? [y/n] ".format(to_csv_fn))
                is_to_csv_fn_del = (is_y[0] == "y")
            if is_to_csv_fn_del:
                concatCSV(*csv_fns, to_csv_fn=to_csv_fn)
                print("Save to", to_csv_fn)
            else:
                print("Can not concat csv.")


class CopyDir:

    def __init__(self):
        self.name = "copydir"
        self.description = "Copy files to dir"
        self.argv = []

    def usage(self):
        print("{0} init_dir to_dir [--help]".format(self.name))
        print("@Des: {0}".format(self.description))

    def run(self, argv):
        self.argv = argv
        if len(argv) == 1:
            self.usage()
            return None

        if "--help" in argv:
            self.usage()
            return

        if len(argv) < 3:
            print("Number of argv < 3")
            self.usage()
            return

        def _is_dir(_dirname):
            if not os.path.isdir(_dirname):
                print("Can not find dirname: \"{}\"".format(_dirname))
                self.usage()
                return
            else:
                return _dirname

        init_dir = _is_dir(argv[1])
        to_dir = _is_dir(argv[2])

        class _DirFileName:

            def __init__(self, _dirname):
                self.dirname = _dirname

            def fn(self, *_names):
                return os.path.join(self.dirname, *_names)

            def walk(self, ):
                return os.walk(self.dirname)

        def _mkdir(dirname):
            if os.path.isdir(dirname):
                return
            else:
                try:
                    os.mkdir(dirname)
                    return
                except:
                    _mkdir(os.path.dirname(dirname))

        init_dfn = _DirFileName(init_dir)
        to_dfn = _DirFileName(to_dir)

        for dirpath, dirnames, filenames in init_dfn.walk():
            init_dfn1 = _DirFileName(dirpath)
            to_dirname = dirpath[len(init_dir)+1:]
            to_dfn1 = _DirFileName(os.path.join(to_dir, to_dirname))

            for fn in filenames:
                if fn.startswith("~$"):
                    continue

                if fn.startswith("~"):
                    continue

                to_fn = to_dfn1.fn(fn)
                fn = init_dfn1.fn(fn)

                dirname = os.path.dirname(to_fn)

                if not os.path.isfile(to_fn):
                    creation_time = os.path.getctime(fn)
                    readable_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(creation_time))
                    # if not readable_time.startswith("2024-01-28"):
                    print("not    |", readable_time, "->", readable_time, "|", fn, "->", to_fn, dirname)
                    if not os.path.isdir(dirname):
                        _mkdir(dirname)
                    shutil.copyfile(fn, to_fn)
                else:
                    time1 = time.localtime(os.path.getmtime(to_fn))
                    time2 = time.localtime(os.path.getmtime(fn))
                    if time2 > time1:
                        modification_time1 = time.strftime('%Y-%m-%d %H:%M:%S', time1)
                        modification_time2 = time.strftime('%Y-%m-%d %H:%M:%S', time2)
                        if modification_time1 != modification_time2:
                            print("change |", modification_time1, "->", modification_time2, "|", fn, "->", to_fn)


if __name__ == "__main__":
    CopyDir().run(["", r"F:\材料", r"M:\F\材料"])
