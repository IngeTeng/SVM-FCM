import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

## 读取数据
# 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
path = './iris.data'  # 数据文件路径
data = pd.read_csv(path, header=None)

x, y = data[list(range(4))], data[4]
y = pd.Categorical(y).codes
x = x[[0, 1 , 2 ,3]]
# print(x.shape)
# print(type(x))
## 数据分割
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=28, train_size=0.6)
print(type(x_train))
#svm1 = SVC(C=1, kernel='linear')
svm2 = SVC(C=1, kernel='rbf')
#svm3 = SVC(C=1, kernel='poly')
#svm4 = SVC(C=1, kernel='sigmoid')

## 模型训练
# t0=time.time()
# svm1.fit(x_train, y_train)
# t1=time.time()
svm2.fit(x_train, y_train)
t2=time.time()
# svm3.fit(x_train, y_train)
# t3=time.time()
# svm4.fit(x_train, y_train)
# t4=time.time()


# svm1_score1 = accuracy_score(y_train, svm1.predict(x_train))
# svm1_score2 = accuracy_score(y_test, svm1.predict(x_test))

svm2_score1 = accuracy_score(y_train, svm2.predict(x_train))
print(svm2_score1)


svm2_score2 = accuracy_score(y_test, svm2.predict(x_test))
print(svm2_score2)
# svm3_score1 = accuracy_score(y_train, svm3.predict(x_train))
# svm3_score2 = accuracy_score(y_test, svm3.predict(x_test))
#
# svm4_score1 = accuracy_score(y_train, svm4.predict(x_train))
# svm4_score2 = accuracy_score(y_test, svm4.predict(x_test))