# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : FuHangNiTanDi.py
@Time    : 2024/6/18 21:13
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of FuHangNiTanDi
-----------------------------------------------------------------------------"""
import os.path

from SRTCodes.Utils import readJson, saveJson, filterFileEndWith


def concat_geojson(json_fn_list, to_json_fn):
    json_list = [readJson(fn) for fn in json_fn_list]
    json_out = json_list[0]
    for json0 in json_list[1:]:
        json_out["features"] += json0["features"]
    saveJson(json_out, to_json_fn)


def main():
    dirname = r"F:\ProjectSet\Huo\fh\samples\drive-download-20240618T133850Z-001"
    fns = filterFileEndWith(dirname, ".geojson")
    concat_geojson(fns, os.path.join(dirname, "ntd_spl4.geojson"))
    pass


if __name__ == "__main__":
    main()
