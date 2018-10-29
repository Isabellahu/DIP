# encoding:utf-8


import math
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import os
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.base import ClassifierMixin


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


def getHogDescriptor(image, binNumber = 16):
   gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
   gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
   mag, ang = cv2.cartToPolar(gx, gy)
   bins = np.int32(binNumber*ang/(2*np.pi))    # quantizing binvalues in (0...16)
   bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
   mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
   hists = [np.bincount(b.ravel(), m.ravel(), binNumber) for b, m in zip(bin_cells, mag_cells)]
   hist = np.hstack(hists)     # hist is a 64 bit vector
   hist = np.array(hist, dtype=np.float32)
   return hist


def gen_color_feature(img):
    h = np.zeros((256, 256, 3))  # 创建用于绘制直方图的全0图像
    bins = np.arange(32).reshape(32, 1)  # 直方图中各bin的顶点位置
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR三种颜色
    originHist_list = []
    for ch, col in enumerate(color):
        originHist = cv2.calcHist([img], [ch], None, [32], [0, 256])
        originHist_re = originHist.reshape(1, 32)
        # print(originHist.shape)
        originHist_list.append(originHist_re)

        cv2.normalize(originHist, originHist, 0, 255 * 0.9, cv2.NORM_MINMAX)
        hist = np.int32(np.around(originHist))
        pts = np.column_stack((bins, hist))
        cv2.polylines(h, [pts], False, col)
    color_feature = np.array(originHist_list).squeeze(1)
    return color_feature


def gen_vec(path='dataset/丰水梨/20170308103301.jpg'):
    gen_list = []
    img_path = read_file(dataset_path)
    for _ in img_path:
        img = cv2.imread(_)
        print(_)
        img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hog = getHogDescriptor(img2)
        vector_to_numpy = np.array(hog).flatten()
        color_feature = gen_color_feature(img).flatten()
        vec = np.concatenate((color_feature, vector_to_numpy), 0)
        # print(vec)
        gen_list.append(vec)
    np.save(dataset_path + 'gen_list.npy', np.array(gen_list))
    print('##########  end   ############')


if __name__ == '__main__':

    dataset_path = '/home/huwenxin/文档/project/DIP/DIP_project2/dataset/'
    # 读取父文件夹下的子文件
    # # read_file(dataset_path)
    gen_vec()

    # # step two
    # # 行  3207
    # x0 = np.load(dataset_path + '/gen_list.npy')
    # x1 = np.load(dataset_path + '/rgb_list.npy')
    # x2 = np.load(dataset_path + '/hist_list.npy')
    #
    # x = np.hstack((x0, x1, x2))
    # print(x.shape)
    # label = get_label(dataset_path)
    # label = np.array(label)
    # print(label.shape)


    # clf = RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=4, oob_score=True, max_depth=200)
    #
    # skf = StratifiedKFold(n_splits=10)
    # #   自动调用
    # print(isinstance(clf, ClassifierMixin))
    # final_score = []
    # for train_index, test_index in skf.split(x, label):
    #     X_train, X_test, y_train, y_test = x[train_index], x[test_index], label[train_index], label[test_index]
    #     clf.fit(X_train, y_train)
    #     score = clf.score(X_test, y_test)
    #     print('Score = ', score)
    #     final_score.append(score)
    #
    # print(np.array(final_score).mean())
    # print(np.array(final_score).var())


