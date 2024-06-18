# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : RUNFileSave.py
@Time    : 2024/6/8 18:35
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of RUNFileSave
-----------------------------------------------------------------------------"""
import json
import os
import sys
from datetime import datetime
from shutil import copyfile


class _SHH2RS:

    def __init__(self, name, init_dirname):
        self.init_dirname = init_dirname
        self.name = name
        self.this_filename = os.path.join(self.init_dirname, name)
        self.this_init_dirname = os.path.join(self.init_dirname, self.name)
        if not os.path.isdir(self.this_init_dirname):
            os.mkdir(self.this_init_dirname)
        self.current_time_str = ""
        self.filelist = []

    def add(self, fn, description="", is_update=False):
        fn = os.path.abspath(fn)
        current_time = datetime.now()
        time_str = current_time.strftime("%Y%m%d%H%M%S")
        self.filelist.append({"filename": fn, "time_str": time_str, "description": description, })
        to_fn = self._tofn(fn, time_str)
        copyfile(fn, to_fn)

    def _tofn(self, fn, time_str):
        to_fn = os.path.split(fn)[1]
        to_fn = "{}_{}".format(time_str, to_fn)
        to_fn = os.path.join(self.this_init_dirname, to_fn)
        return to_fn

    def update(self, time_str):
        file_tors = self.find(time_str)
        to_fn = self._tofn(file_tors["filename"], file_tors["time_str"])

    def find(self, time_str):
        for line in self.filelist:
            if line["time_str"] == time_str:
                return line
        return None


class RUNFileSave_main:

    def __init__(self, init_dirname):
        init_dirname = os.path.abspath(init_dirname)
        self.init_dirname = init_dirname
        if not os.path.isdir(self.init_dirname):
            os.mkdir(self.init_dirname)
        fn = os.path.split(self.init_dirname)[1]
        """
        filename time_str des
        """
        self.data = {}
        self.update_log_fn = os.path.join(self.init_dirname, "{}_fs.txt".format(fn))
        self.update_json_fn = os.path.join(self.init_dirname, "{}_fs.json".format(fn))
        if os.path.isfile(self.update_log_fn):
            copyfile(self.update_log_fn, self.update_log_fn + "-back")
        if os.path.isfile(self.update_json_fn):
            copyfile(self.update_json_fn, self.update_json_fn + "-back")
            with open(self.update_json_fn, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            with open(self.update_json_fn, "w", encoding="utf-8") as f:
                json.dump(self.data, f, )
        self.is_show_full = False

    def main(self, argv: list):

        if len(argv) == 1:
            self.usage()
            return

        filename = None
        des = ""
        is_update = False
        is_show = False
        find_str = None
        is_find_str_all = False
        is_load = False

        i = 1

        while i < len(argv):

            if (argv[i] == "-des") and (i < len(argv) - 1):
                des += argv[i + 1] + " "
                i += 1
            elif (argv[i] == "-fs") and (i < len(argv) - 1):
                find_str = argv[i + 1]
                i += 1
            elif argv[i] == "-fs--all":
                if i < len(argv) - 1:
                    find_str = argv[i + 1]
                    i += 1
                is_find_str_all = True
            elif argv[i] == "--update":
                is_update = True
            elif argv[i] == "--show":
                is_show = True
            elif argv[i] == "--show-full":
                self.is_show_full = True
            elif argv[i] == "--load":
                is_load = True
            elif filename is None:
                filename = argv[i]

            i += 1

        if is_load:
            if filename is None:
                print("Can not get filename.")
                return
            is_find = False
            to_fn = os.path.split(filename)[1]

            def copyfileto(_fn):
                is_delete = True
                if os.path.isfile(to_fn):
                    _line = input("To file exist.\nWhether delete or not? [y/n]")
                    is_delete = False
                    if _line.startswith("y"):
                        is_delete = True
                if not is_delete:
                    return False
                _fn_savefile = "{}_{}".format(self.data[_fn]["time_str"], os.path.split(_fn)[1])
                _fn_savefile = os.path.join(self.init_dirname, _fn_savefile)
                copyfile(_fn_savefile, to_fn)
                print("Copy file \"{}\"".format(_fn_savefile))
                return True

            if filename in self.data:
                self.show(filename)
                copyfileto(filename)
                is_find = True
            else:
                for fn in self.data:
                    if fn.endswith(filename):
                        self.show(fn)
                        copyfileto(fn)
                        is_find = True
            if not is_find:
                print("Can not find \"{}\"".format(filename))
            return

        is_find = False
        if find_str is not None:
            for fn in self.data:
                if find_str in fn:
                    if not is_find_str_all:
                        print("Find \"{}\" in filename".format(find_str))
                        print(self.show(fn))
                        return
                    else:
                        print(self.show(fn))
                        is_find = True

            for fn in self.data:
                line = self.data[fn]
                if find_str in line["time_str"]:
                    if not is_find_str_all:
                        print("Find \"{}\" in time".format(find_str))
                        print(self.show(fn))
                        return
                    else:
                        print(self.show(fn))
                        is_find = True

            for fn in self.data:
                line = self.data[fn]
                for des in line["des"]:
                    if find_str in des["des"]:
                        if not is_find_str_all:
                            print("Find \"{}\" in description".format(find_str))
                            print(self.show(fn))
                            return
                        else:
                            print(self.show(fn))
                            is_find = True
            if not is_find:
                print("Not find \"{}\"".format(find_str))
            return

        if is_show:
            for fn in self.data:
                print(self.show(fn))
            return

        if filename is None:
            print("Can not find filename.")
            self.usage()
            return

        filename = os.path.abspath(filename)
        # print("filename", filename)

        if (not os.path.isfile(filename)) and (filename in self.data):
            print("Warning: original file can not find. \"{}\"".format(filename))
            print(self.show(filename))
            if is_update:
                to_fn = self.saveFN(filename)
                print("Copy file to: {}".format(filename))
                copyfile(to_fn, filename)
            with open(self.update_json_fn, "w", encoding="utf-8") as f:
                json.dump(self.data, f, )
            return

        if (os.path.isfile(filename)) and (filename in self.data):
            if is_update:
                to_fn = self.saveFN(filename)
                print("Copy file to: {}".format(to_fn))
                copyfile(filename, to_fn)
                if des == "":
                    des = input("Update input description: ")
                current_time = datetime.now()
                time_str = current_time.strftime("%Y%m%d%H%M%S")
                print(time_str)
                self.data[filename]["des"].append({"time_str": time_str, "des": des})
            print(self.show(filename))
            with open(self.update_json_fn, "w", encoding="utf-8") as f:
                json.dump(self.data, f, )
            return

        if not is_update:
            line = input("whether update or not? [y/n]")
            if line[0] == "y":
                is_update = True
            else:
                print("Not update file. \"{}\"".format(filename))

        if is_update:
            if not os.path.isfile(filename):
                print("File is not existed, \"{}\"".format(filename))
                return
            current_time = datetime.now()
            time_str = current_time.strftime("%Y%m%d%H%M%S")
            to_fn = self.saveFN(filename, time_str)
            copyfile(filename, to_fn)
            if des == "":
                des = input("Input description: ")
            self.data[filename] = {"filename": filename, "time_str": time_str,
                                   "des": [{"time_str": time_str, "des": des}]}
            print(self.show(filename))

        with open(self.update_json_fn, "w", encoding="utf-8") as f:
            json.dump(self.data, f, )

    def show(self, filename):
        n = list(self.data).index(filename) + 1
        line = self.data[filename]
        to_fn = self.saveFN(filename)
        if filename in self.data:
            fn = os.path.split(filename)[1]
            if not self.is_show_full:
                des = line["des"][0]
                show_str = "[{0}] {3}: {1} \"{2}\"\n".format(n, fn, des["des"], self._timeStr(line["time_str"]))
                return show_str.strip()

            show_str = "[{0}] {3}: {1} \"{2}\"\n".format(n, fn, filename, self._timeStr(line["time_str"]))
            show_str += "  ->\"{}\"\n".format(to_fn)
            if len(line["des"]) == 1:
                des = line["des"][0]
                show_str += "  {}: {}".format(self._timeStr(des["time_str"]), des["des"])
            else:
                for i in range(len(line["des"])):
                    des = line["des"][i]
                    show_str += "  {}. {}: {}\n".format(
                        i + 1, self._timeStr(des["time_str"]), des["des"])
            return show_str.strip()
        return None

    def _timeStr(self, time_str):
        n = [0, 4, 6, 8, 10, 12, 14]
        return "{}-{}-{} {}:{}:{}".format(*(time_str[n[i]:n[i + 1]] for i in range(len(n) - 1)))

    def update(self, filename):
        filename = os.path.abspath(filename)
        if not os.path.isfile(filename):
            return False
        current_time = datetime.now()
        time_str = current_time.strftime("%Y%m%d%H%M%S")
        to_fn = self.saveFN(filename, time_str)
        copyfile(filename, to_fn)
        return time_str

    def saveFN(self, filename, time_str=None):
        if time_str is None:
            if filename in self.data:
                time_str = self.data[filename]["time_str"]
        if time_str is None:
            current_time = datetime.now()
            time_str = current_time.strftime("%Y%m%d%H%M%S")
        to_fn = os.path.split(filename)[1]
        to_fn = "{}_{}".format(time_str, to_fn)
        to_fn = os.path.join(self.init_dirname, to_fn)
        return to_fn

    def wl(self, *texts, sep=" ", end="\n"):
        with open(self.update_log_fn, "a", encoding="utf-8") as f:
            for text in texts:
                f.write(text)
                f.write(sep)
            f.write(end)

    def usage(self):
        print("srt_fs [filename] [--update] [-des] [--show] [-fs] [-fs--all]\n"
              "@Des: save file to directory \"{0}\"\n"
              "    filename: file to save\n"
              "    --update: save file to directory\n"
              "    -des: description of save file\n"
              "    -fs: find string\n"
              "    --show: show file descriptions\n"
              "    --show-full: show file descriptions full"
              "".format(self.init_dirname))


def main():
    if sys.argv[1] == "shh2_samples_release":
        RUNFileSave_main(r"F:\ProjectSet\Shadow\Hierarchical\Samples\SaveSamples").main(sys.argv[1:])

    pass


if __name__ == "__main__":
    main()
