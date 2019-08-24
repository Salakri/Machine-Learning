# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:05:03 2019

Xinyun Zhao
Muyao Wang
"""
############Question 6######################

import numpy as np
from sympy import *
from numpy import linalg as la

matrixA = np.array([[0, 14],[6, 9]])

# a. characteristic polynomial
# det(A - lambda*I)
print("a============================")
identityI = np.identity(2)
lam = Symbol('λ')
matrixCalDet = matrixA - identityI*lam
CharaPoly = expand(matrixCalDet[0][0] * matrixCalDet[1][1] - \
matrixCalDet[0][1] * matrixCalDet[1][0])
print("The characteristic polynomial of A is: ")
print(CharaPoly)

# b. eigenvalues
print("b============================")
print("The solve for eigenvalues: ")
eigenvalues = solve(CharaPoly, lam)
for a in eigenvalues:
    print ("λ = ", a)
    
# c. eigenvectors
print("c============================")
x= Symbol('x')
y= Symbol('y')
for a in eigenvalues: 
    v = matrixA - identityI*a
    cal1 = v[0][0]* x + v[0][1] * y
    cal2 = v[1][0]* x + v[1][1] * y
    eigenvectors = solve((cal1, cal2), x, y)
    print(cal1)
    print(cal2)
    print(eigenvectors)

# d. using linalg.eig()
print("d============================")
resValues = la.eig(matrixA)[0]
resVectors = la.eig(matrixA)[1]
print("Eigenvalues are:")
for r in resValues:
    print(r)
print("Eigenvectors are:")
for r in resVectors:
    print(r)




