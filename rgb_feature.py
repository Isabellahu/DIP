# encoding:utf-8

import pandas as pd
import numpy as np
import os
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import os
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold


def get_label(path):
    # 文件个数
    num = []
    files = os.listdir(path)
    for img_file in files:
        paths = os.path.join(path, img_file)
        child_files = os.listdir(paths)
        num.append(len(child_files))
    y = np.full(num[0], 0)
    for _ in range(1, len(num)):
        y = np.hstack((y, np.full(num[_], _)))
    return y


def read_file(path):
    all_path = []
    files = os.listdir(path)
    for img_file in files:
        paths = os.path.join(path, img_file)
        child_files = os.listdir(paths)
        for child_file in child_files:
            all_path.append(paths+'/'+child_file)

    return all_path


def rgb_feature():
    rgb_list = []
    color = ('b', 'g', 'r')

    img_path = read_file(dataset_path)
    for _ in img_path:
        img = cv2.imread(_)
        print(_)
        rgb_m = []
        for i, col in enumerate(color):
            mv = []
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            histr = histr[:-1].reshape((5, -1))
            for j in histr[:-1]:
                # max = np.mean(j)
                mean = np.mean(j)
                var = np.var(j)
                # print(max, mean, var)
                # mv.append(max)
                mv.append(mean)
                mv.append(var)
            rgb_m.append(mv)
        rgb_m = np.array(rgb_m).reshape(1, -1)
        rgb_list.append(rgb_m[0])
    np.save(dataset_path + 'rgb_list.npy', np.array(rgb_list))

    print('##########  end   ############')


if __name__ == '__main__':
    fruit_class = ['大台农芒', '冰糖心苹果', '国产青苹果', '特级红富士', '纽荷脐橙', '砀山梨', '泰国香蕉', '丰水梨', '黄金梨',
                   '澳洲蜜柑','贡梨', '米蕉', '小台农芒', '花牛红苹果', '雪梨', '鸭梨', '台湾青枣', '澳洲大芒', '脆肉瓜',
                   '番石榴', '富士王', '陕西香酥梨', '百香果','冰糖橙', '海南香蕉', '蜜梨']

    dataset_path = '/home/huwenxin/文档/project/DIP/DIP_project2/dataset/'

    # 读取父文件夹下的子文件
    # read_file(dataset_path)
    rgb_feature()

    #
    # # step two
    # # 行  3207
    # x1 = np.load('/home/huwenxin/文档/project/DIP/DIP_project2/rgb_list.npy')
    # x2 = np.load('/home/huwenxin/文档/project/DIP/DIP_project2/hist_list.npy')
    #
    #
    # x = np.hstack((x1,x2))
    # print(x.shape)
    # label = get_label(dataset_path)
    # label = np.array(label)
    # print(label.shape)
    #
    # # clf = SVC()
    # # clf = KNeighborsClassifier(n_neighbors=3)
    #
    # clf = RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=4, oob_score=True)
    #
    # skf = StratifiedKFold(n_splits=10)
    #
    # for train_index, test_index in skf.split(x, label):
    #     X_train, X_test, y_train, y_test = x[train_index], x[test_index], label[train_index], label[test_index]
    #     clf.fit(X_train, y_train)
    #     print('Score = ', clf.score(X_test, y_test))



