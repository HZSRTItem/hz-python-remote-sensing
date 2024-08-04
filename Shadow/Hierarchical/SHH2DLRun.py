# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2DLRun.py
@Time    : 2024/7/21 11:08
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2DLRun
-----------------------------------------------------------------------------"""
from DeepLearning.EfficientNet import efficientnet_b7, efficientnet_v2_m
from DeepLearning.MobileNetV2 import MobileNetV2
from DeepLearning.MobileNetV3 import mobilenet_v3_small
from DeepLearning.ResNet import resnext50_32x4d
from DeepLearning.ShuffleNetV2 import shufflenet_v2_x0_5


def getCSVFn(city_name):
    if city_name == "qd":
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\vhl\sh2_spl25_vhl_2_spl2.csv"
    elif city_name == "cd":
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\cd\sh2_spl30_cd6_spl.csv"
    elif city_name == "bj":
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\3\sh2_spl273_5_spl.csv"
    else:
        raise Exception("City name \"{}\"".format(city_name))
    return csv_fn


def run1(city_name):
    from torch import nn

    from SRTCodes.SRTModel import SamplesData, TorchModel
    from Shadow.Hierarchical import SHH2Config
    from DeepLearning.DenseNet import densenet121
    from DeepLearning.GoogLeNet import GoogLeNet
    from DeepLearning.InceptionV3 import Inception3
    from DeepLearning.ResNet import BasicBlock, ResNet
    from DeepLearning.SqueezeNet import SqueezeNet
    from DeepLearning.VisionTransformer import VisionTransformerChannel

    from SRTCodes.SRTTimeDirectory import TimeDirectory
    from SRTCodes.Utils import DirFileName, RumTime

    _DL_SAMPLE_DIRNAME = r"F:\ProjectSet\Shadow\Hierarchical\Samples\DL"

    def x_deal(_x):
        for i in range(0, 6):
            _x[i] = _x[i] / 1600
        for i in range(6, 10):
            _x[i] = (_x[i] + 30) / 35
        for i in range(12, 16):
            _x[i] = (_x[i] + 30) / 35
        _x[11] = _x[11] / 90
        _x[17] = _x[11] / 90
        return _x

    def get_model(_mod_name, _in_ch=None, _n_category=4):
        if _in_ch is None:
            _in_ch = len(get_names)

        if _mod_name == "ResNet18":
            return ResNet(BasicBlock, [2, 2, 2, 2], in_ch=len(get_names), num_classes=4)

        if _mod_name == "VIT":
            return VisionTransformerChannel(
                in_channels=_in_ch, image_size=win_size[0], patch_size=6, num_layers=12,
                num_heads=12, hidden_dim=768, mlp_dim=3072, num_classes=4
            )

        if _mod_name == "SqueezeNet":
            return SqueezeNet(version="1_1", num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "GoogLeNet":
            return GoogLeNet(num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "DenseNet121":
            return densenet121(num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "Inception3":
            return Inception3(num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "ShuffleNetV2X05":
            return shufflenet_v2_x0_5(num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "MobileNetV2":
            return MobileNetV2(num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "MobileNetV3Small":
            return mobilenet_v3_small(num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "ResNeXt5032x4d":
            return resnext50_32x4d(num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "EfficientNetB7":
            return efficientnet_b7(num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "EfficientNetV2M":
            return efficientnet_v2_m(num_classes=_n_category, in_channels=_in_ch)

        return None

    dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods")
    td = TimeDirectory(dfn.fn())
    td.initLog()
    td.kw("DIRNAME", td.time_dirname())

    init_model_name = "DL"
    td.log("\n#", "-" * 50, city_name.upper(), init_model_name, "-" * 50, "#\n")
    csv_fn = getCSVFn(city_name)
    raster_fn = SHH2Config.GET_RASTER_FN(city_name)
    get_names = [
        #  0  1  2  3  4  5
        "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
        #  6  7  8  9 10 11
        "AS_VV", "AS_VH", "AS_C11", "AS_C22", "AS_H", "AS_Alpha",
        # 12 13 14 15 16 17
        "DE_VV", "DE_VH", "DE_C11", "DE_C22", "DE_H", "DE_Alpha",
    ]
    map_dict = {
        "IS": 0, "VEG": 1, "SOIL": 2, "WAT": 3,
        "IS_SH": 0, "VEG_SH": 1, "SOIL_SH": 2, "WAT_SH": 3,
    }
    cm_names = ["IS", "VEG", "SOIL", "WAT"]
    epochs = 100

    td.kw("CITY_NAME", city_name)
    td.kw("INIT_MODEL_NAME", init_model_name)
    td.kw("CSV_FN", csv_fn)
    td.kw("GET_NAMES", get_names)
    td.copyfile(csv_fn)
    td.copyfile(__file__)

    read_size = (25, 25)

    sd = SamplesData(_dl_sample_dirname=_DL_SAMPLE_DIRNAME)
    sd.addDLCSV(
        csv_fn, read_size, get_names, x_deal,
        grs={"qd": SHH2Config.QD_GR(), "bj": SHH2Config.BJ_GR(), "cd": SHH2Config.CD_GR(), }
    )

    win_size_list = [
        ("ResNet18", (7, 7)),
        ("VIT", (24, 24)),
        ("SqueezeNet", (24, 24)),
        ("GoogLeNet", (24, 24)),
        # ("DenseNet121", (24, 24)),  # size small
        # ("Inception3", (24, 24)),  # size small
        ("ShuffleNetV2X05", (7, 7)),
        ("MobileNetV2", (7, 7)),
        ("MobileNetV3Small", (7, 7)),
        ("ResNeXt5032x4d", (7, 7)),
        ("EfficientNetB7", (7, 7)),
        ("EfficientNetV2M", (7, 7)),
    ]
    run_time = RumTime(len(win_size_list)).strat()

    for i_win_size, (init_model_name, win_size) in enumerate(win_size_list):
        model = get_model(init_model_name)
        model_name = "{}_{}-{}".format(init_model_name, win_size[0], win_size[1])
        dfn_tmp = DirFileName(td.fn(model_name))
        dfn_tmp.mkdir()

        td.log("\n#", "-" * 30, i_win_size + 1, model_name, "-" * 30, "#\n")
        torch_mod = TorchModel()
        torch_mod.filename = dfn_tmp.fn(model_name + ".hm")
        torch_mod.map_dict = map_dict
        torch_mod.color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), }
        torch_mod.model = model
        torch_mod.criterion = nn.CrossEntropyLoss()
        torch_mod.win_size = win_size
        torch_mod.read_size = read_size
        torch_mod.epochs = 100
        torch_mod.n_epoch_save = -1
        torch_mod.train_filters.extend([("city", "==", city_name)])
        torch_mod.test_filters.extend([("city", "==", city_name)])
        torch_mod.cm_names = cm_names

        td.kw("TORCH_MOD.FILENAME", torch_mod.filename)
        td.kw("TORCH_MOD.MAP_DICT", torch_mod.map_dict)
        td.kw("TORCH_MOD.COLOR_TABLE", torch_mod.color_table)
        td.kw("TORCH_MOD.MODEL", torch_mod.model.__class__)
        td.kw("TORCH_MOD.CRITERION", torch_mod.color_table)
        td.kw("TORCH_MOD.WIN_SIZE", torch_mod.win_size)
        td.kw("TORCH_MOD.READ_SIZE", torch_mod.read_size)
        td.kw("TORCH_MOD.EPOCHS", torch_mod.epochs)
        td.kw("TORCH_MOD.N_EPOCH_SAVE", torch_mod.n_epoch_save)
        td.kw("TORCH_MOD.CM_NAMES", torch_mod.cm_names)

        model_sw = td.buildWriteText(r"{}\{}.txt".format(model_name, model_name), "a")
        model_sw.write(torch_mod.model)

        torch_mod.sampleData(sd)
        torch_mod.samples.showCounts(td.log)

        line_sw = td.buildWriteText(r"{}\{}_training-log.txt".format(model_name, model_name), "a")
        to_list = []

        def func_field_record_save(field_records):
            line = field_records.line
            to_list.append(line.copy())

            if int(line["Accuracy"]) != -1:
                for k in line:
                    line_sw.write("| {}:{} ".format(k, line[k]), end="")
                line_sw.write("|")
                if line["Batch"] == 0:
                    td.log("+ Epoch:", "{:<6d}".format(line["Epoch"]), end=" ", is_print=False)
                    td.log("Loss:", "{:<12.6f}".format(line["Loss"]), end=" ", is_print=False)
                    td.log("Accuracy:", "{:>6.3f}".format(line["Accuracy"]), end="\n", is_print=False)

        torch_mod.func_field_record_save = func_field_record_save
        torch_mod.save_model_fmt = dfn_tmp.fn(model_name + "_" + city_name + "_{}.pth")
        td.kw("TORCH_MOD.SAVE_MODEL_FMT", torch_mod.save_model_fmt)
        torch_mod.train()

        torch_mod.imdc(
            raster_fn, data_deal=x_deal,
            mod_fn=None, read_size=(500, -1), is_save_tiles=True, fun_print=td.log
        )

        td.saveJson(r"{}\{}_training-log.json".format(model_name, model_name), to_list)
        # except Exception as e:
        #     print(e)

        run_time.add().printInfo()

    return


def run2(city_name, init_model_name, win_size, is_train):
    from torch import nn

    from SRTCodes.SRTModel import SamplesData, TorchModel
    from Shadow.Hierarchical import SHH2Config
    from DeepLearning.DenseNet import densenet121
    from DeepLearning.GoogLeNet import GoogLeNet
    from DeepLearning.InceptionV3 import Inception3
    from DeepLearning.ResNet import BasicBlock, ResNet
    from DeepLearning.SqueezeNet import SqueezeNet
    from DeepLearning.VisionTransformer import VisionTransformerChannel

    from SRTCodes.SRTTimeDirectory import TimeDirectory
    from SRTCodes.Utils import DirFileName

    _DL_SAMPLE_DIRNAME = r"F:\ProjectSet\Shadow\Hierarchical\Samples\DL"

    def x_deal(_x):
        for i in range(0, 6):
            _x[i] = _x[i] / 1600
        for i in range(6, 10):
            _x[i] = (_x[i] + 30) / 35
        for i in range(12, 16):
            _x[i] = (_x[i] + 30) / 35
        _x[11] = _x[11] / 90
        _x[17] = _x[11] / 90
        return _x

    def get_model(_mod_name, _in_ch=None, _n_category=4):
        if _in_ch is None:
            _in_ch = len(get_names)

        if _mod_name == "ResNet18":
            return ResNet(BasicBlock, [2, 2, 2, 2], in_ch=len(get_names), num_classes=4)

        if _mod_name == "VIT":
            return VisionTransformerChannel(
                in_channels=_in_ch, image_size=win_size[0], patch_size=6, num_layers=12,
                num_heads=12, hidden_dim=768, mlp_dim=3072, num_classes=4
            )

        if _mod_name == "SqueezeNet":
            return SqueezeNet(version="1_1", num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "GoogLeNet":
            return GoogLeNet(num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "DenseNet121":
            return densenet121(num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "Inception3":
            return Inception3(num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "ShuffleNetV2X05":
            return shufflenet_v2_x0_5(num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "MobileNetV2":
            return MobileNetV2(num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "MobileNetV3Small":
            return mobilenet_v3_small(num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "ResNeXt5032x4d":
            return resnext50_32x4d(num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "EfficientNetB7":
            return efficientnet_b7(num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "EfficientNetV2M":
            return efficientnet_v2_m(num_classes=_n_category, in_channels=_in_ch)

        return None

    dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods")
    td = TimeDirectory(dfn.fn(), time_dirname=r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240728H135412")
    td.initLog(mode="a")
    td.kw("DIRNAME", td.time_dirname())
    td.log("\n#", "-" * 50, city_name.upper(), init_model_name, "-" * 50, "#\n")
    csv_fn = getCSVFn(city_name)
    raster_fn = SHH2Config.GET_RASTER_FN(city_name)
    get_names = [
        #  0  1  2  3  4  5
        "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
        #  6  7  8  9 10 11
        "AS_VV", "AS_VH", "AS_C11", "AS_C22", "AS_H", "AS_Alpha",
        # 12 13 14 15 16 17
        "DE_VV", "DE_VH", "DE_C11", "DE_C22", "DE_H", "DE_Alpha",
    ]
    map_dict = {
        "IS": 0,
        # "VEG": 1,
        "SOIL": 1,
        # "WAT": 3,
        # "IS_SH": 0, "VEG_SH": 1, "SOIL_SH": 2, "WAT_SH": 3,
    }
    cm_names = ["IS", "VEG", "SOIL", "WAT"]
    epochs = 100

    td.kw("CITY_NAME", city_name)
    td.kw("INIT_MODEL_NAME", init_model_name)
    td.kw("CSV_FN", csv_fn)
    td.kw("GET_NAMES", get_names)
    td.copyfile(csv_fn)
    td.copyfile(__file__)

    model = get_model(init_model_name)
    model_name = "{}_{}_{}-{}".format(city_name, init_model_name, win_size[0], win_size[1])
    dfn_tmp = DirFileName(td.fn(model_name))
    dfn_tmp.mkdir()

    read_size = (25, 25)

    td.log("\n#", "-" * 30, model_name, "-" * 30, "#\n")
    torch_mod = TorchModel()
    torch_mod.filename = dfn_tmp.fn(model_name + ".hm")
    torch_mod.map_dict = map_dict
    torch_mod.color_table = {1: (255, 0, 0), 2: (255, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), }
    torch_mod.model = model
    torch_mod.criterion = nn.CrossEntropyLoss()
    torch_mod.win_size = win_size
    torch_mod.read_size = read_size
    torch_mod.epochs = epochs
    torch_mod.n_epoch_save = -1
    torch_mod.train_filters.extend([("city", "==", city_name)])
    torch_mod.test_filters.extend([("city", "==", city_name)])
    torch_mod.cm_names = cm_names

    td.kw("TORCH_MOD.FILENAME", torch_mod.filename)
    td.kw("TORCH_MOD.MAP_DICT", torch_mod.map_dict)
    td.kw("TORCH_MOD.COLOR_TABLE", torch_mod.color_table)
    td.kw("TORCH_MOD.MODEL", torch_mod.model.__class__)
    td.kw("TORCH_MOD.CRITERION", torch_mod.color_table)
    td.kw("TORCH_MOD.WIN_SIZE", torch_mod.win_size)
    td.kw("TORCH_MOD.READ_SIZE", torch_mod.read_size)
    td.kw("TORCH_MOD.EPOCHS", torch_mod.epochs)
    td.kw("TORCH_MOD.N_EPOCH_SAVE", torch_mod.n_epoch_save)
    td.kw("TORCH_MOD.CM_NAMES", torch_mod.cm_names)

    torch_mod.save_model_fmt = dfn_tmp.fn(model_name + "_" + city_name + "_{}.pth")
    td.kw("TORCH_MOD.SAVE_MODEL_FMT", torch_mod.save_model_fmt)

    def train():
        sd = SamplesData(_dl_sample_dirname=_DL_SAMPLE_DIRNAME)
        sd.addDLCSV(
            csv_fn, read_size, get_names, x_deal,
            grs={"qd": SHH2Config.QD_GR(), "bj": SHH2Config.BJ_GR(), "cd": SHH2Config.CD_GR(), }
        )

        model_sw = td.buildWriteText(r"{}\{}.txt".format(model_name, model_name), "a")
        model_sw.write(torch_mod.model)

        torch_mod.sampleData(sd)
        torch_mod.samples.showCounts(td.log)

        line_sw = td.buildWriteText(r"{}\{}_training-log.txt".format(model_name, model_name), "a")
        to_list = []

        def func_field_record_save(field_records):
            line = field_records.line
            to_list.append(line.copy())

            if int(line["Accuracy"]) != -1:
                for k in line:
                    line_sw.write("| {}:{} ".format(k, line[k]), end="")
                line_sw.write("|")
                if line["Batch"] == 0:
                    td.log("+ Epoch:", "{:<6d}".format(line["Epoch"]), end=" ", is_print=False)
                    td.log("Loss:", "{:<12.6f}".format(line["Loss"]), end=" ", is_print=False)
                    td.log("Accuracy:", "{:>6.3f}".format(line["Accuracy"]), end="\n", is_print=False)

        torch_mod.func_field_record_save = func_field_record_save
        torch_mod.train()
        td.saveJson(r"{}\{}_training-log.json".format(model_name, model_name), to_list)

    def imdc():
        torch_mod.x_keys = get_names
        torch_mod.imdc(
            raster_fn, data_deal=x_deal,
            mod_fn=torch_mod.save_model_fmt.format(epochs), read_size=(500, -1), is_save_tiles=True, fun_print=td.log
        )

    if is_train:
        train()
    else:
        imdc()

    return


def main():
    win_size_list = [
        ("ResNet18", (7, 7)),
        # ("VIT", (24, 24)),
        ("SqueezeNet", (24, 24)),
        ("GoogLeNet", (24, 24)),
        # ("DenseNet121", (24, 24)),  # size small
        # ("Inception3", (24, 24)),  # size small
        ("ShuffleNetV2X05", (7, 7)),
        ("MobileNetV2", (7, 7)),
        ("MobileNetV3Small", (7, 7)),
        ("ResNeXt5032x4d", (7, 7)),
        ("EfficientNetB7", (7, 7)),
        ("EfficientNetV2M", (7, 7)),
    ]

    for city_name in ["qd", "bj", "cd"]:
        for name, win_size in win_size_list:
            print("python -c \""
                  r"import sys; "
                  r"sys.path.append(r'F:\PyCodes'); "
                  r"from Shadow.Hierarchical.SHH2DLRun import run2; "
                  "run2('{}', '{}', {}, True)\"".format(
                city_name, name, win_size
            ))
            print("python -c \""
                  r"import sys; "
                  r"sys.path.append(r'F:\PyCodes'); "
                  r"from Shadow.Hierarchical.SHH2DLRun import run2; "
                  "run2('{}', '{}', {}, False)\"".format(
                city_name, name, win_size
            ))


if __name__ == "__main__":
    main()
    r"""
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2DLRun import main; main()"
    """
