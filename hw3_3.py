# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:03:21 2019

Xinyun Zhao
Muyao Wang
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.decomposition as skd

from sklearn.datasets import fetch_lfw_people
lfw_people = fetch_lfw_people(min_faces_per_person=70)
n_samples, h, w = lfw_people.images.shape
npix = h*w
fea = lfw_people.data

def plt_face(x):
    global h,w
    plt.imshow(x.reshape((h, w)), cmap=plt.cm.gray)
    plt.xticks([])

plt.figure(figsize=(10,20))
'''
nplt = 4
for i in range(nplt):
    plt.subplot(1,nplt,i+1)
    plt_face(fea[i])
'''

####Fourth face###########
print("a=======================================")
print("Display the fourth face in the first plot below")
example_t = 3
plt.subplot(2,2,1)
plt_face(fea[example_t])

#######Display the mean image####
print("b=======================================")
print("Display the mean face in the second plot below")
meanFace = np.mean(fea, axis = 0)
plt.subplot(2,2,2)
plt_face(meanFace)

#######PCA Python Command####
print("c=======================================")
print("The values of the fourth image:")
pca = skd.PCA(n_components = 5)
skd.PCA.fit(pca,fea)
W1 = pca.components_
W = W1.transpose()
Z = pca.transform(fea)
print(Z[3])

#######PCA Python Command####
print("d=======================================")
print("Display the fourth face with 5 features in the third plot below")
resFace = np.dot(W, Z[3]) + meanFace
plt.subplot(2,2,3)
plt_face(resFace)

print("Display the fourth face with 50 features in the third plot below")
pca = skd.PCA(n_components = 50)
skd.PCA.fit(pca,fea)
W1 = pca.components_
W = W1.transpose()
Z = pca.transform(fea)

resFace50 = np.dot(W, Z[3]) + meanFace
plt.subplot(2,2,4)
plt_face(resFace50)

plt.show()
plt.close()

