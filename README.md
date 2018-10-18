# DIP
clf = RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=4, oob_score=True, max_depth=150)
本模型的选择参数如上。
在tets_model.py文件中加入测试数据之后，直接运行tets_model.py文件即可测试正确率。
