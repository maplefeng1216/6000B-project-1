#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:17:03 2017

@author: qiufeng
"""

import pandas as pd
from pandas import DataFrame 
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import cross_val_score



X_train =pd.read_csv('traindata.csv')
X_label =pd.read_csv('trainlabel.csv')
X_train.fillna(-99999, inplace=True)
X_train=preprocessing.scale(X_train)
# =============================================================================
# #min_max_scaler = preprocessing.MinMaxScaler()
# #X_normalized = preprocessing.normalize(X_train, norm='l2')
# =============================================================================

X_laterly=pd.read_csv('testdata.csv')
X, y=X_train,X_label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3 )
# =============================================================================
# #from sklearn.model_selection import GridSearchCV
# #grid = GridSearchCV(SVC(), param_grid={"C":[0.1, 1, 10], "gamma": [1, 0.1, 0.01]}, cv=4)
# #grid.fit(X, y)
# #print "The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_)
# =============================================================================
# =============================================================================
# import matplotlib.pyplot as plt 
# k_range = range(1, 40)
# k_scores = []
# for k in k_range:
#     clf = SVC(C=k,gamma=0.01)
#     scores = cross_val_score(logreg, X, y.values.ravel(), cv=10, scoring='accuracy')
#     k_scores.append(scores.mean())
# plt.plot(k_range, k_scores)
# plt.xlabel('Value of C for SVM')
# plt.ylabel('Cross-Validated Accuracy')
# plt.show()
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=7)
# model=knn.fit(X_train, y_train)
# print cross_val_score(model, X, y.values.ravel(), cv=10, scoring='accuracy').mean()
# print cross_val_score(model, X, y.values.ravel(), cv=10, scoring='accuracy').std()
# =============================================================================
clf = SVC(C=28,gamma=0.01)
model=clf.fit(X_train, y_train)
#print cross_val_score(model,X,y.values.ravel(),cv=10,scoring='accuracy').mean()
#print cross_val_score(model,X,y.values.ravel(),cv=10,scoring='accuracy').std()
# =============================================================================
# from sklearn.linear_model import LogisticRegression
# import matplotlib.pyplot as plt 
# k_range = range(1, 40)
# k_scores = []
# for k in k_range:
#     logreg = LogisticRegression(C=k)
#     scores = cross_val_score(logreg, X, y.values.ravel(), cv=10, scoring='accuracy')
#     k_scores.append(scores.mean())
# plt.plot(k_range, k_scores)
# plt.xlabel('Value of C for LR')
# plt.ylabel('Cross-Validated Accuracy')
# plt.show()
# 
# logreg = LogisticRegression(C=18)
# model=logreg.fit(X_train, y_train)
# print cross_val_score(model, X, y.values.ravel(), cv=10, scoring='accuracy').mean()
# print cross_val_score(model, X, y.values.ravel(), cv=10, scoring='accuracy').std()
# =============================================================================
# =============================================================================
# from sklearn.neighbors import KNeighborsClassifier
# import matplotlib.pyplot as plt 
# k_range = range(1, 40)
# k_scores = []
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, X, y.values.ravel(), cv=10, scoring='accuracy')
#     k_scores.append(scores.mean())
# plt.plot(k_range, k_scores)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Cross-Validated Accuracy')
# plt.show()
# 
# knn = KNeighborsClassifier(n_neighbors=7)
# model=knn.fit(X_train, y_train)
# print cross_val_score(model, X, y.values.ravel(), cv=10, scoring='accuracy').mean()
# print cross_val_score(model, X, y.values.ravel(), cv=10, scoring='accuracy').std()
# =============================================================================
accuracy=clf.score(X_test, y_test) 
forecast_set=clf.predict(X_laterly)
df=pd.DataFrame(forecast_set)
df.to_csv('project1_20398324.csv')
print df,accuracy




