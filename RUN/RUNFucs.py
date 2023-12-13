# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : RUNFucs.py
@Time    : 2023/12/4 21:01
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of RUNFucs
-----------------------------------------------------------------------------"""
import os

import pandas as pd


def checkDFKeys(df, k, filename=""):
    if k not in df:
        print("Can not find field name \"{0}\" in DataFrame of {1}".format(k, filename))
    return k in df


class QJYTxt_main:

    def __init__(self):
        self.name = "qjy_txt"
        self.description = "Build QJY txt file from excel or csv file"
        self.argv = []

        self.category_dict = {
            "shadow": [{"code": 0, "color": (0, 0, 0), "name": "NOT_KNOW"},
                       {"code": 11, "color": (255, 0, 0), "name": "IS"},
                       {"code": 12, "color": (125, 0, 0), "name": "IS_SH"},
                       {"code": 21, "color": (0, 255, 0), "name": "VEG"},
                       {"code": 22, "color": (0, 125, 0), "name": "VEG_SH"},
                       {"code": 31, "color": (255, 255, 0), "name": "SOIL"},
                       {"code": 32, "color": (125, 125, 0), "name": "SOIL_SH"},
                       {"code": 41, "color": (0, 0, 255), "name": "WAT"},
                       {"code": 42, "color": (0, 0, 125), "name": "WAT_SH"}],
            "is": [{"code": 0, "color": (0, 0, 0), "name": "NOT_KNOW"},
                   {"code": 1, "color": (255, 0, 0), "name": "IS"},
                   {"code": 2, "color": (0, 255, 0), "name": "NOIS"}],
        }

    def usage(self):
        print("{0} [filename] [-o] [-sheet_name] [-ccn code|color|name]\n"
              "@Des: {1}\n"
              "    filename: excel or csv filename\n"
              "    -o: to file name default:\"spl.txt\"\n"
              "    -sheet_name: excel sheet name\n"
              "    -ccn: category of \"code|color|name\" or shadow or is".format(self.name, self.description))

    def run(self, argv):

        if len(argv) == 1:
            self.usage()

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
                ext = os.path.splitext(filename)[1]
                df = None
                if ext == ".xlsx" or ext == ".xls":
                    df = pd.read_excel(filename, sheet_name=sheet_name)
                elif ext == ".csv" or ext == ".txt":
                    df = pd.read_csv(filename)

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
                        f.write(k)
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


def main():
    pass


if __name__ == "__main__":
    main()
