# encoding:utf-8

from sklearn.externals import joblib

if __name__ == '__main__':
    # 通过 get_label()  得到数组从0到25替换表示
    fruit_class = ['大台农芒', '冰糖心苹果', '国产青苹果', '特级红富士', '纽荷脐橙', '砀山梨', '泰国香蕉', '丰水梨', '黄金梨',
                   '澳洲蜜柑','贡梨', '米蕉', '小台农芒', '花牛红苹果', '雪梨', '鸭梨', '台湾青枣', '澳洲大芒', '脆肉瓜',
                   '番石榴', '富士王', '陕西香酥梨', '百香果','冰糖橙', '海南香蕉', '蜜梨']

    # 输入 X, y_label
    clf = joblib.load("svm_model.m")
    X = [[0, 0], [1, 1]]
    y_label = [0, 1]
    y_test = (clf.predict(X))
    # print(y_test)

    # 正确率
    cnt = 0
    for i in range(len(y_label)):
        if fruit_class[y_test[i]] == y_label[i]:
            cnt += 1
    right = cnt / float(len(y_label))
    print('predicting, classification right = ', right)

