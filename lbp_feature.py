# encoding:utf-8

import numpy as np
from skimage import color
from skimage.feature import local_binary_pattern
from skimage import io
from sklearn.ensemble import RandomForestClassifier
import os
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
        # num.append(len(child_files))
        for child_file in child_files:
            # return paths+'/'+child_file
            # print(paths+'/'+child_file)
            all_path.append(paths+'/'+child_file)
        # print(len(all_path))
    return all_path


def lbp_feature():
    hist_list = []
    img_path = read_file(dataset_path)
    for i in img_path:
        print(i)
        img = io.imread(i)
        img = color.rgb2gray(img)
        radius = 2
        n_points = 8 * radius
        METHOD = 'uniform'
        lbp = local_binary_pattern(img, n_points, radius, METHOD)
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
        print('hist = ', len(hist))
        hist_list.append(hist)
        print(len(hist_list[0]))

    # 合并为一个array
    # return np.array(hist_list)
    np.save('/home/huwenxin/文档/project/DIP/DIP_project2/hist_list.npy', np.array(hist_list))

    print('##########  end lbp  ############')


if __name__ == '__main__':

    # fruit_class = ['大台农芒', '冰糖心苹果', '国产青苹果', '特级红富士', '纽荷脐橙', '砀山梨', '泰国香蕉', '丰水梨', '黄金梨',
    #                '澳洲蜜柑','贡梨', '米蕉', '小台农芒', '花牛红苹果', '雪梨', '鸭梨', '台湾青枣', '澳洲大芒', '脆肉瓜',
    #                '番石榴', '富士王', '陕西香酥梨', '百香果','冰糖橙', '海南香蕉', '蜜梨']


    dataset_path = '/home/huwenxin/文档/project/DIP/DIP_project2/dataset/'
    # read_file(dataset_path)
    lbp_feature()






    # step two
    # 行  3207
    # x = np.load('/home/huwenxin/文档/project/DIP/DIP_project2/hist_list.npy')
    # print(x.shape)
    # label = get_label(dataset_path)
    # label = np.array(label)
    # print(label.shape)
    #
    #
    # # svm
    # # X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    # # # ','.join([str(_) for _ in range(0,26)])
    # # y = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
    #
    # clf = RandomForestClassifier(n_estimators=50)
    #
    # skf = StratifiedKFold(n_splits=10)
    #
    # for train_index, test_index in skf.split(x, label):
    #     X_train, X_test, y_train, y_test = x[train_index], x[test_index], label[train_index], label[test_index]
    #
    #     clf.fit(X_train, y_train)
    #     print('Score = ', clf.score(X_test, y_test))

