import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


TrainFcmMat = np.load(file='TrainGCN.npy')
TestFcmMat = np.load(file='TestGCN.npy')

x_train = []
y_train = []

x_test = []
y_test = []
#把矩阵中的每一个元素当做一个特征，将每个二维矩阵变成一维

#训练数据
for i in range(TrainFcmMat.shape[0]):

    if (TrainFcmMat[i, 0 , 0] == 1):
        y_train.append(1)
    else:
        y_train.append(2)

    xtmp = []
    for j in range(TrainFcmMat.shape[1]):
        if j > 0:
            for k in range(TrainFcmMat.shape[2]):
                xtmp.append(TrainFcmMat[i,j,k])
    x_train.append(xtmp)


for i in range(TestFcmMat.shape[0]):

    if (TestFcmMat[i, 0 , 0] == 1):
        y_test.append(1)
    else:
        y_test.append(2)

    xtmp = []
    for j in range(TestFcmMat.shape[1]):
        if j > 0:
            for k in range(TestFcmMat.shape[2]):
                xtmp.append(TestFcmMat[i,j,k])
    x_test.append(xtmp)


svm = SVC(C=0.1, kernel='rbf', gamma=0.01)


svm.fit(x_train, y_train)
t2=time.time()

svm_score1 = accuracy_score(y_train, svm.predict(x_train))
print(svm_score1)
#
#
svm_score2 = accuracy_score(y_test, svm.predict(x_test))
print(svm_score2)


grid = GridSearchCV(SVC(kernel='rbf'), param_grid={"C":[0.1, 1, 10], "gamma": [1, 0.1, 0.01]}, cv=4)
grid.fit(x_train, y_train)
print("The best parameters are %s with a score of %0.5f"
      % (grid.best_params_, grid.best_score_))