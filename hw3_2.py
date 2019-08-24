# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:17:03 2019

Xinyun Zhao
Muyao Wang
"""

###################Question 7###############


import sklearn.decomposition as skd
import numpy as np


matrixX = np.array([[5.0, 2.0, 4.0],[9.0, 6.0, 4.0],\
                    [7.0,1.0,0.0],[2.0,5.0,6.0]])
matrixShape = matrixX.shape

# a mean to zero
print("a============================")
meancol1 = 0
meancol2 = 0
meancol3 = 0
for r in range(matrixShape[0]):
    meancol1 += matrixX[r][0]
    meancol2 += matrixX[r][1]
    meancol3 += matrixX[r][2]

meancol1 = meancol1/matrixShape[0]
meancol2 = meancol2/matrixShape[0]
meancol3 = meancol3/matrixShape[0]

#print(meancol1)
#print(meancol2)
#print(meancol3)
matrixB = matrixX
for r in range(matrixShape[0]):
    matrixB[r][0] = matrixB[r][0] - meancol1
    matrixB[r][1] = matrixB[r][1] - meancol2
    matrixB[r][2] = matrixB[r][2] - meancol3
print("Show the matrix B: ")
print(matrixB)

# b covariance of x1 and x2
print("b============================")
covariance = 0
for r in range(matrixShape[0]):
    covariance += (matrixB[r][0] * matrixB[r][2])

covariance = covariance/(matrixShape[0] - 1)
print("Show the covariance of x1 and x3: ")
print(covariance)

# c python command
print("c============================")
covarianceB = np.cov(np.transpose(matrixB))
print("Sample covariance matrix of B: ")
print(covarianceB)

print("The covaraince of x1 and x2: ")
print(covarianceB[0][1])
print("The covaraince of x1 and x3: ")
print(covarianceB[0][2])
print("The covaraince of x2 and x3: ")
print(covarianceB[1][2])

eigenvalues = np.linalg.eig(covarianceB)
print("The largest eigenvalue is: ")
print(max(eigenvalues[0]))

# c python run code
print("d============================")
# .fit computes the principal components (n_components of them)
# The columns of W are the eigenvectors of the covariance matrix of X
pca = skd.PCA(n_components = 3)
skd.PCA.fit(pca,matrixB)
W1 = pca.components_
W = W1.transpose()
Z = pca.transform(matrixB)

first2Z = []
for r in range(Z.shape[0]):
    eachRow = []
    eachRow.append(Z[r][0])
    eachRow.append(Z[r][1])
    first2Z.append(eachRow)

print("Show the first two columns of Z: ")
print(np.array(first2Z))




