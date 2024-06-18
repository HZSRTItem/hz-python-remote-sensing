# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : GDALRun.py
@Time    : 2023/11/6 16:49
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of GDALRun
-----------------------------------------------------------------------------"""
import os
import sys

import numpy as np
from osgeo_utils.gdal2tiles import main as gdal2tiles_main
from osgeo_utils.gdal_merge import Usage as gdal_merge_Usage
from osgeo_utils.gdal_merge import main as gdal_merge_main
from osgeo_utils.gdal_polygonize import GDALPolygonize
from osgeo_utils.gdal_retile import Usage as gdal_retile_Usage
from osgeo_utils.gdal_retile import main as gdal_retile_main
from osgeo_utils.gdal_sieve import Usage as gdal_sieve_Usage
from osgeo_utils.gdal_sieve import main as gdal_sieve_main
from osgeo_utils.gdalcompare import Usage as gdalcompare_Usage
from osgeo_utils.gdalcompare import main as gdalcompare_main
from osgeo_utils.gdalmove import Usage as gdalmove_Usage
from osgeo_utils.gdalmove import main as gdalmove_main
from osgeo_utils.ogrmerge import Usage as ogrmerge_Usage
from osgeo_utils.ogrmerge import main as ogrmerge_main

from RUN.GLI import GLI_main
from SRTCodes.GDALRasterIO import GDALRaster, GDALRasterChannel
from SRTCodes.Utils import printListTable


class GDAL2Tiles_main:

    def __init__(self):
        self.name = "gdal2tiles"
        self.description = "Convert a raster into TMS (Tile Map Service) tiles in a directory."
        self.argv = []

    def run(self, argv):
        self.argv = argv
        gdal2tiles_main(argv)

    def usage(self):
        gdal2tiles_main(["gdal2tiles", "-h"])


class GDALMerge_main:

    def __init__(self):
        self.name = "gdal_merge"
        self.description = "Geographic raster image merge"
        self.argv = []

    def run(self, argv):
        self.argv = argv
        gdal_merge_main(argv)

    def usage(self):
        gdal_merge_Usage()


class GDALPolygonize_main:

    def __init__(self):
        self.name = "gdal_polygonize"
        self.description = "Application for converting raster data to a vector polygon layer."
        self.argv = []

    def run(self, argv):
        self.argv = argv
        GDALPolygonize().main(argv)

    def usage(self):
        GDALPolygonize().main(["gdal_polygonize", "-h"])


class GDALRetiles_main:

    def __init__(self):
        self.name = "gdal_retile"
        self.description = "Module for retiling (merging) tiles and building tiled pyramids"
        self.argv = []

    def run(self, argv):
        self.argv = argv
        gdal_retile_main(argv)

    def usage(self):
        gdal_retile_Usage()


class GDALSieve_main:

    def __init__(self):
        self.name = "gdal_sieve"
        self.description = "Application for applying sieve filter to raster data."
        self.argv = []

    def run(self, argv):
        self.argv = argv
        gdal_sieve_main(argv)

    def usage(self):
        gdal_sieve_Usage()


class GDALCompare_main:

    def __init__(self):
        self.name = "gdalcompare"
        self.description = "Compare two files for differences and report."
        self.argv = []

    def run(self, argv):
        self.argv = argv
        gdalcompare_main(argv)

    def usage(self):
        gdalcompare_Usage()


class GDALMove_main:

    def __init__(self):
        self.name = "gdalmove"
        self.description = "Application for \"warping\" an image by just updating it's SRS and geotransform."
        self.argv = []

    def run(self, argv):
        self.argv = argv
        gdalmove_main(argv)

    def usage(self):
        gdalmove_Usage()


class OGRMerge_main:

    def __init__(self):
        self.name = "ogrmerge"
        self.description = "Merge the content of several vector datasets into a single one."
        self.argv = []

    def run(self, argv):
        self.argv = argv
        ogrmerge_main(argv)

    def usage(self):
        ogrmerge_Usage()


class GDALListNames_main:

    def __init__(self):
        self.name = "gdallistnames"
        self.description = "List the names of each channel in the GDAL based grid file."
        self.argv = []

    def run(self, argv):
        self.argv = argv
        if len(argv) == 1:
            self.usage()
            return None
        gr = GDALRaster(argv[1])
        for name in gr:
            if "--list" in argv:
                print("\"{}\",".format(name), end=" ")
            else:
                print(name)
        print()

    def usage(self):
        print("{0} raster_fn [--list]".format(self.name))
        print("@Des: {0}".format(self.description))
        print("    raster_fn: raster file name.")


class GDALDataDes_main:

    def __init__(self):
        self.name = "gdaldatades"
        self.description = "Calculate the maximum, minimum, mean, and variance of each band output."
        self.argv = []

    def run(self, argv):
        self.argv = argv
        if len(argv) == 1:
            self.usage()
            return None

        raster_fn = None
        fmt = "table"
        n_xiaoshu = 6

        i = 1
        while i < len(argv):
            if argv[i] == "-fmt":
                fmt = argv[i + 1]
                i += 1
            else:
                raster_fn = argv[i]
            i += 1

        grc = GDALRasterChannel()
        channels = grc.getRasterNames(raster_fn)
        gr = GDALRaster(raster_fn)

        # gr = GDALRaster(raster_fn)
        # data = gr.readAsLine(int(gr.n_rows / 2), int(gr.n_columns / 2))
        len_name = max([len(name) for name in channels])
        len_column = 10
        print(len_column)

        fmt_name = "{:<" + str(len_name) + "}"
        fmt_column_names = "{:>" + str(len_column + n_xiaoshu) + "}"
        fmt_column = "{:>" + str(len_column+ n_xiaoshu) + "." + str(n_xiaoshu) + "f}"

        if fmt == "lines" or fmt == "csv":
            if fmt == "lines":
                print("NAME", "MIN", "MAX", "MEAN", "STD", sep=" ")
            elif fmt == "csv":
                print("NAME", "MIN", "MAX", "MEAN", "STD", sep=",")
            for name in channels:
                name = grc.addGDALData(raster_fn, name)
                d = grc[name]
                if fmt == "lines":
                    print(name, np.min(d), np.max(d), np.mean(d), np.std(d), sep=" ")
                elif fmt == "csv":
                    print(name, np.min(d), np.max(d), np.mean(d), np.std(d), sep=",")
        elif fmt == "table":
            datas = [["NAME", "MIN", "MAX", "MEAN", "STD"]]

            print("{} | {} | {} | {} | {}".format(
                fmt_name.format("NAME"),
                fmt_column_names.format("MIN"),
                fmt_column_names.format("MAX"),
                fmt_column_names.format("MEAN"),
                fmt_column_names.format("STD"),
            ))

            for name in channels:
                d = gr.readGDALBand(name)
                print("{} | {} | {} | {} | {}".format(
                    fmt_name.format(name),
                    fmt_column.format(float(np.min(d))),
                    fmt_column.format(float(np.max(d))),
                    fmt_column.format(float(np.mean(d))),
                    fmt_column.format(float(np.std(d))),
                ))
                # datas.append([name, float(np.min(d)), float(np.max(d)), float(np.mean(d)), float(np.std(d))])
            # printListTable(datas, precision=6, rcl=">")

    def usage(self):
        print("{0} raster_fn [-fmt table|csv|lines]".format(self.name))
        print("@Des: {0}".format(self.description))
        print("    raster_fn: raster file name.")
        print("    -fmt: out format table|csv")


class GDALDesCopy_main:

    def __init__(self):
        self.name = "gdaldescopy"
        self.description = "Copy each channel description information from a grid to multiple raster files."
        self.argv = []

    def run(self, argv):
        self.argv = argv
        if len(argv) == 1:
            self.usage()
            return None

        description_f = None
        to_fns = []
        ext = ".tif"

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
                        to_fns.append(line)

            elif (argv[i] == "-dir") and (i < len(argv) - 1):
                i += 1
                dirname = argv[i]
                for fn in os.listdir(dirname):
                    fn = os.path.join(dirname, fn)
                    if os.path.isfile(fn):
                        if fn.endswith(ext):
                            to_fns.append(fn)
            elif (argv[i] == "-filter") and (i < len(argv) - 1):
                i += 1
                ext = argv[i]

            elif description_f is None:
                description_f = argv[i]

            else:
                to_fns.append(argv[i])

            i += 1

        if description_f is None:
            print("Not find description raster file")
            return

        gr1 = GDALRaster(description_f)

        for i, fn in enumerate(to_fns):
            gr2 = GDALRaster(fn)
            print("{:2d}: {}".format(i, fn))
            for i in range(gr2.n_channels):
                band = gr2.raster_ds.GetRasterBand(i + 1)
                band.SetDescription(gr1.names[i])

    def usage(self):
        print(
            "{0} description_f [-filelist filelist.txt] [-dir dirname] [-filter string] *raster_fns".format(self.name))
        print("@Des: {0}".format(self.description))
        print("    description_f: description raster file name.")
        print("    -filelist: to raster filelist")
        print("    -dir: dirname of all raster file")
        print("    -filter: filters of dirname")
        print("    *raster_fns: raster files")


class SRTGeo:

    def __init__(self):
        self.name = "srt_geo"
        self.description = "Some self-developed exe about GEO. \n(C)Copyright 2022, ZhengHan. All rights reserved."
        self.exes = {}

    def usage(self):
        print("{0} mark/--h [options]\n@Description:\n{1}\n@Args:\n    mark: mark of exe\n"
              "    --h: get help of this\n@Marks:".format(self.name, "    " + self.description.replace("\n", "\n    ")))
        for k in self.exes:
            print("    {0}: {1}".format(k, self.exes[k].description))

    def add(self, exe):
        if exe.name in self.exes:
            raise Exception("mark:{0} has in this".format(exe.name))
        self.exes[exe.name] = exe

    def run(self, mark, argv):
        mark = mark.lower()
        self.exes[mark].run(argv)

        if mark == "help" and self.exes["help"].show_help is not None:
            if self.exes["help"].show_help in self.exes:
                print("`{0}` help information are as follows:\n".format(self.exes["help"].show_help))
                self.exes[self.exes["help"].show_help].usage()
            else:
                print("Show help fault.\nCan not find mark: `{0}`".format(self.exes["help"].show_help))


class Help_main:

    def __init__(self):
        self.show_help = None
        self.name = "help"
        self.description = "Get help information for mark of exe"
        self.argv = []

    def run(self, argv):
        self.argv = argv
        if len(argv) == 1:
            self.usage()
        else:
            self.show_help = argv[1]

    def usage(self):
        print("{0} mark \n @Description: {1}".format(self.name, self.description))


def main_gdalrun(argv):
    srt_geo = SRTGeo()
    srt_geo.add(Help_main())
    srt_geo.add(GDALRetiles_main())
    srt_geo.add(GDALMerge_main())
    srt_geo.add(GDAL2Tiles_main())
    srt_geo.add(GDALPolygonize_main())
    srt_geo.add(GDALSieve_main())
    srt_geo.add(GDALCompare_main())
    srt_geo.add(GDALMove_main())
    srt_geo.add(OGRMerge_main())
    srt_geo.add(GLI_main())
    srt_geo.add(GDALListNames_main())
    srt_geo.add(GDALDataDes_main())
    srt_geo.add(GDALDesCopy_main())
    if len(argv) == 1:
        srt_geo.usage()
    else:
        if argv[1] == "--h":
            srt_geo.usage()
            return
        if argv[1] in srt_geo.exes:
            srt_geo.run(argv[1], argv[1:])
        else:
            print("Can not find mark:`{0}` ".format(argv[1]))
            srt_geo.usage()


if __name__ == '__main__':
    main_gdalrun(sys.argv)
