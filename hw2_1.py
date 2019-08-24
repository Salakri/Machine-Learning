# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 20:38:46 2019

@author: Xinyun Zhao
         Muyao Wang
"""

import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston


print("=================TRAIN LINER====================")
####################TRAINNING LINEAR###############
boston_data = load_boston()

X = boston_data.data
y = boston_data.target.reshape(X.shape[0], 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# You may use the sklearn method to create a polynomial transformation of degree 2
poly_tranform = sklearn.preprocessing.PolynomialFeatures(degree=2)
Z = poly_tranform.fit_transform(X)

#To learn more about the Boston housing data set type:
#print(boston_data.DESCR)


xtn, xtm = X_train.shape
ytn, ytm = y_train.shape

# fit a linear regression model

'''
# n = 404 row
# m = 13 column
# Calculate mean for each column
List_of_mean = []
for n in range(0, xtn):
    for m in range (0, xtm):
        if n == 0:
            List_of_mean.append(X_train[n][m])
        else:
            List_of_mean[m] += X_train[n][m]
for m in range (0, xtm):        
    List_of_mean[m] = List_of_mean[m]/float(xtn)   

# Calculate l2 norm for each colum
List_of_norm = []
for n in range(0, xtn):
    for m in range (0, xtm):
        if n == 0:
            List_of_norm.append((X_train[n][m])*(X_train[n][m]))
        else:
            List_of_norm[m] += (X_train[n][m])*(X_train[n][m])
for m in range (0, xtm):        
    List_of_norm[m] = math.sqrt(List_of_norm[m]) 
            
# Minus Average
X_train_t = X_train
for n in range(0, xtn):
    for m in range (0, xtm):
       X_train_t[n][m] = (X_train_t[n][m] - List_of_mean[m])/float(List_of_norm[m])

xInsert = np.ones((xtn, 1))
xtNew = np.hstack((xInsert, X_train_t))
xtm += 1

'''
xInsert = np.ones((xtn, 1))
xtNew = np.hstack((xInsert, X_train))
xtm += 1


# n = 404 row
# m = 14 column

# Compute w
pInverseXT = np.linalg.pinv(xtNew)
wt = np.dot(pInverseXT, y_train)
#print("Linear parament matrix: \n", wt)

# Comput hat y
hatyt = np.dot(xtNew, wt)

# RSS
rsst = 0
for n in range(0, xtn):
    rsst = rsst + (y_train[n][0] - hatyt[n][0]) ** 2
#print("Training Linear Regression RSS = ", rsst)

# TSS
yt_mean = 0
tsst = 0
for n in range(0, xtn):
    yt_mean += y_train[n][0]
yt_mean = yt_mean/float(xtn)

for n in range(0, xtn):
    tsst = tsst + (y_train[n][0] - yt_mean) ** 2
print("Training Linear Regression TSS = ", tsst)

# R square
rst = (tsst-rsst)/tsst
print("Training Linear Regression R Square = ", rst)
print("=================TEST LINEAR======================")

#################TESTING LINEAR################################
xn, xm = X_test.shape
yn, ym = y_test.shape


'''
# n = 102 row
# m = 13 column
# Calculate mean for each column
List_of_mean = []
for n in range(0, xn):
    for m in range (0, xm):
        if n == 0:
            List_of_mean.append(X_test[n][m])
        else:
            List_of_mean[m] += X_test[n][m]
for m in range (0, xm):        
    List_of_mean[m] = List_of_mean[m]/float(xtn)   

# Calculate l2 norm for each colum
List_of_norm = []
for n in range(0, xn):
    for m in range (0, xm):
        if n == 0:
            List_of_norm.append((X_test[n][m])*(X_test[n][m]))
        else:
            List_of_norm[m] += (X_test[n][m])*(X_test[n][m])
for m in range (0, xm):        
    List_of_norm[m] = math.sqrt(List_of_norm[m]) 
            
# Minus Average
X_test_t = X_test
for n in range(0, xn):
    for m in range (0, xm):
       X_test_t[n][m] = (X_test_t[n][m] - List_of_mean[m])/float(List_of_norm[m])


xInsert = np.ones((xn, 1))
xNew = np.hstack((xInsert, X_test_t))
xm += 1
'''
xInsert = np.ones((xn, 1))
xNew = np.hstack((xInsert, X_test))
xm += 1

# Compute w
#pInverseX = np.linalg.pinv(xNew)
#w = np.dot(pInverseX, y_test)

# Comput hat y
haty = np.dot(xNew, wt)

# RSS
rss = 0
for n in range(0, xn):
    rss = rss + (y_test[n][0] - haty[n][0]) ** 2
print("Testing Linear Regression RSS = ", rss)

# TSS
y_mean = 0
tss = 0
for n in range(0, xn):
    y_mean += y_test[n][0]
y_mean = y_mean/float(xn)

for n in range(0, xn):
    tss = tss + (y_test[n][0] - y_mean) ** 2
print("Testing Linear Regression TSS = ", tss)

# R square
rs = (tss-rss)/tss
print("Tesing Linear Regression R Square = ", rs)
print("==================TRAIN RIDGE=====================")

##############TRAIN RIDGE######################

# Lambda Set
set_lambda = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16,\
              0.32, 0.64, 1.28, 2.56, 5.12, 10.24]

xtNewTrans = xtNew.transpose()
prot = np.dot(xtNewTrans, xtNew)
Identityt = np.identity(prot.shape[0])

# Recording minimum
result_rss = 0
result_index = 0
result_lambda = 0
min_ridge = 99999999

for l in set_lambda:
    # Compute w
    wtl = np.linalg.inv(prot + l * Identityt)
    wtl = np.dot(wtl, xtNewTrans)
    wtl = np.dot(wtl, y_train)
    hatytl = np.dot(xtNew, wtl)
    rss_ = 0
    sum_wtl = 0
    # Compute rss
    for n in range(0, xtn):
        rss_ = rss_ + (y_train[n][0] - hatytl[n][0]) ** 2
    # Compute sum of w squares
    for i in range(0, wtl.shape[0]):
        sum_wtl += (wtl[i][0]) ** 2
    ridge = (1.0/xtn)*rss_ + l * sum_wtl
    # record min
    if ridge < min_ridge:
        min_ridge = ridge
        result_rss = rss_
        result_lambda = l
        
print("Training Ridge Smallest Lambda = ", result_lambda)
print("Training Ridge RSS = ", result_rss)
print("Training Ridge Regression TSS = ", tsst)

# R square
rst_ = (tsst-result_rss)/tsst
print("Training Ridge Regression R Square = ", rst_)

print("=================TEST RIDGE========================")

##############TEST RIDGE######################

# Lambda Set

xNewTrans = xNew.transpose()
pro = np.dot(xNewTrans, xNew)
Identity = np.identity(pro.shape[0])

# Recording minimum
result_rss = 0
result_index = 0
result_lambda = 0

min_ridge = 99999999

for l in set_lambda:
    # Compute w

    rss_ = 0
    sum_wl = 0
    hatyl = np.dot(xNew, wtl)
    # Compute rss
    for n in range(0, xn):
        rss_ = rss_ + (y_test[n][0] - hatyl[n][0]) ** 2
    # Compute sum of w squares
    for i in range(0, wtl.shape[0]):
        sum_wl += (wtl[i][0]) ** 2
    ridge = (1.0/xn)*rss_ + l * sum_wl
    # record min
    if ridge < min_ridge:
        min_ridge = ridge
        result_rss = rss_
        result_lambda = l
        
print("Testing Ridge Smallest Lambda = ", result_lambda)
print("Testing Ridge RSS = ", result_rss)
print("Testing Ridge Regression TSS = ", tss)

# R square
rst_ = (tss-result_rss)/tss
print("Testing Ridge Regression R Square = ", rst_)

print("================ESTIMATE===========================")




    
    

    







