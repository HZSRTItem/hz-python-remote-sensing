# 时间：2024/5/22 14:21
import random

import joblib
import numpy as np
from osgeo import gdal
from sklearn.ensemble import RandomForestClassifier

from SRTCodes.ModelTraining import ConfusionMatrix, dataModelPredict
from Shadow.ShadowTraining import trainRF_RandomizedSearchCV
from SRTCodes.GDALRasterIO import GDALRaster, tiffAddColorTable


def sampling(image_data, label_data, number_dict):
    sample_dict = {1: [], 2: [], 3: [], 4: []}
    while True:
        row, column = random.randint(0, label_data.shape[0] - 1), random.randint(0, label_data.shape[1] - 1)
        category = label_data[row, column]
        number_dict[category] -= 1
        if number_dict[category] >= 0:
            sample_dict[category].append(image_data[:, row, column].tolist())
        is_return = True
        for k in number_dict:
            is_return = is_return and (number_dict[k] <= 0)
        if is_return:
            break
    x, y = [], []
    for k in sample_dict:
        x.extend(sample_dict[k])
        y.extend([k] * len(sample_dict[k]))
    x, y = np.array(x), np.array(y)
    return x, y


def main():
    image_data = GDALRaster(r"E:\data\experiment\pre_image\Part_Huangdao.tif").readAsArray()
    label_data = GDALRaster(r"E:\data\experiment\pre_label\Part_Huangdao.tif").readAsArray()
    print(image_data.shape)
    print(label_data.shape)
    print(np.unique(label_data, return_counts=True))

    # 训练
    def train(filename=None):
        label_data[label_data == 255] = 4
        number_dict = {1: 800, 2: 800, 3: 800, 4: 800}
        x_train, y_train = sampling(image_data, label_data, number_dict)
        _clf = RandomForestClassifier(n_estimators=60)
        _clf.fit(x_train, y_train)
        if filename is not None:
            joblib.dump(_clf, filename)
        return _clf

    def test():
        # 测试

        # label_data[label_data == 255] = 4
        # number_dict = {1: 200, 2: 200, 3: 200, 4: 200}
        # x_test, y_test = sampling(image_data, label_data, number_dict)

        data_list = []
        y_test = []
        for i in range(1, 5):
            row, column = np.where(label_data==i)
            data = image_data[:, row, column].T
            data_list.append(data)
            y_test.extend([i]*len(data))
            continue
        x_test = np.concatenate(data_list)
        y_test = np.array(y_test)
        print(x_test.shape, y_test.shape)

        print(clf.score(x_test, y_test))
        cm = ConfusionMatrix(class_names= ["Qiao", "Guan", "Cao", "Beijing"])
        cm.addData(y_test, clf.predict(x_test))
        print(cm.fmtCM())
        print("Kappa", cm.getKappa())

    def predict():
        imdc = dataModelPredict(image_data, data_deal=None, is_jdt=True, model=clf)
        # imdc = np.random.randint(1, 5, label_data.shape)
        gr = GDALRaster(r"E:\data\experiment\image\Part_Huangdao.tif")
        gr.save(imdc.astype("int8"), r"E:\data\experiment\rf\rf1.tif", fmt="GTiff", dtype=gdal.GDT_Byte)
        tiffAddColorTable(r"E:\data\experiment\rf\rf1.tif", 1,
                          {
                              1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 0), 4: (0, 0, 0)
                          })

    # clf = train(r"E:\data\experiment\rf\rf_mod1.mod")
    clf = joblib.load(r"E:\data\experiment\rf\rf_mod1.mod")
    test()
    # predict()

    # trainRF_RandomizedSearchCV(x, y, n_iter=100)

    # print(image_data.shape)
    # print(label_data.shape)
    # print(np.unique(label_data, return_counts=True))
    # trainRF_RandomizedSearchCV()


if __name__ == "__main__":
    main()
