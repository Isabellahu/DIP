# encoding:utf-8


from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib
import os


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


if __name__ == '__main__':

    dataset_path = '/home/huwenxin/文档/project/DIP/DIP_project2/dataset/'
    # 读取父文件夹下的子文件

    # 行  3207
    x0 = np.load('gen_list.npy')
    x1 = np.load('rgb_list.npy')
    x2 = np.load('hist_list.npy')

    x = np.hstack((x0, x1, x2))
    print(x.shape)
    label = get_label(dataset_path)
    label = np.array(label)
    print(label.shape)

    clf = RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=4, oob_score=True, max_depth=150)
    clf.fit(x, label)
    joblib.dump(clf, "svm_model.m")

    # fruit_class = ['大台农芒', '冰糖心苹果', '国产青苹果', '特级红富士', '纽荷脐橙', '砀山梨', '泰国香蕉', '丰水梨', '黄金梨',
    #                '澳洲蜜柑','贡梨', '米蕉', '小台农芒', '花牛红苹果', '雪梨', '鸭梨', '台湾青枣', '澳洲大芒', '脆肉瓜',
    #                '番石榴', '富士王', '陕西香酥梨', '百香果','冰糖橙', '海南香蕉', '蜜梨']


    # 10折分层抽样 验证

    # skf = StratifiedKFold(n_splits=10)
    # # #   自动调用
    # # print(isinstance(clf, ClassifierMixin))
    # final_score = []
    # for train_index, test_index in skf.split(x, label):
    #     X_train, X_test, y_train, y_test = x[train_index], x[test_index], label[train_index], label[test_index]
    #     clf.fit(X_train, y_train)
    #     score = clf.score(X_test, y_test)
    #     print('Score = ', score)
    #     final_score.append(score)
    #
    # print('mean =', np.array(final_score).mean())
    # print('var =', np.array(final_score).var())





