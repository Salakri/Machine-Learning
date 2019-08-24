# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 17:19:26 2019

@author: 
Xinyun Zhao
Muyao Wang
"""

import operator
import math
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import numpy.random as r
import matplotlib.pyplot as plt

def cal_log(num1, num2):
    if (num1 == 0) or (num2 == 0):
        return 0
    result = (num1/num2)*math.log2(num1/num2)
    return result

############# function code comes from the class
def f(x):
    return 1 / (1 + np.exp(-x))
def f_deriv(x):
    return f(x) * (1 - f(x))

def setup_and_init_weights(nn_structure):
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))
        b[l] = r.random_sample((nn_structure[l],))
    return W, b

def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_W, tri_b

def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 2))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect

def feed_forward(x, W, b):
    h = {1: x}
    z = {}
    for l in range(1, len(W) + 1):
        # if it is the first layer, then the input into the weights is x, otherwise, 
        # it is the output from the last layer
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        z[l+1] = W[l].dot(node_in) + b[l] # z^(l+1) = W^(l)*h^(l) + b^(l)  
        h[l+1] = f(z[l+1]) # h^(l) = f(z^(l)) 
    return h, z

def calculate_out_layer_delta(y, h_out, z_out):
    # delta^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))
    return -(y-h_out) * f_deriv(z_out)

def calculate_hidden_delta(delta_plus_1, w_l, z_l):
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l)

def train_nn(nn_structure, X, y, iter_num=3000, alpha=0.33):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt%1000 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(len(y)):
            delta = {}
            # perform the feed forward pass and return the stored h and z values, to be used in the
            # gradient descent step
            h, z = feed_forward(X[i, :], W, b)
            # loop from nl-1 to 1 backpropagating the errors
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i,:], h[l], z[l])
                    avg_cost += np.linalg.norm((y[i,:]-h[l]))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
                    # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
                    tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis])) 
                    # trib^(l) = trib^(l) + delta^(l+1)
                    tri_b[l] += delta[l+1]
        # perform the gradient descent step for the weights in each layer
        for l in range(len(nn_structure) - 1, 0, -1):
            W[l] += -alpha * (1.0/m * tri_W[l])
            b[l] += -alpha * (1.0/m * tri_b[l])
        # complete the average cost calculation
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func

def predict_y(W, b, X, n_layers):
    m = X.shape[0]
    y = np.zeros((m,))
    for i in range(m):
        h, z = feed_forward(X[i, :], W, b)
        y[i] = np.argmax(h[n_layers])
    return y
###############################################
    

fb = open("reviewstrain.txt", "r")
fb2 = open("reviewstest.txt", "r")

train_list = []
test_list = []
for line in fb:
    line_list = line.split() 
    train_list.append(line_list)

for line in fb2:
    line_list2 = line.split() 
    test_list.append(line_list2)

###question a###############################   
print("question a===================")
vocabulary = {}
for line in train_list:
    for word in line[1:]:
        if word not in vocabulary:
            vocabulary[word] = 1
        else:
            vocabulary[word] += 1
        

sorted_list = sorted(vocabulary.items(), key=operator.itemgetter(1))

print("5 tokens that occur most frequently in the training set:")
print(sorted_list[-1][0], " ", sorted_list[-2][0], " ", sorted_list[-3][0]\
      , " ", sorted_list[-4][0], " ", sorted_list[-5][0])

###question b###############################
print("question b===================")

# non-duplicate dictionary
attribute = {}
for line in train_list:
    non_duplis = []
    for word in line[1:]:
        if word not in non_duplis:
            non_duplis.append(word)
    for word in non_duplis:
        if word not in attribute:
            attribute[word] = 1
        else:
            attribute[word] += 1

positive_dict = {}
for line in train_list:
    non_duplis = []
    if int(line[0]) == 1:       
        for word in line[1:]:
             if word not in non_duplis:
                 non_duplis.append(word)
        for word in non_duplis:
            if word not in positive_dict:
                positive_dict[word] = 1
            else:
                positive_dict[word] += 1
                
# entropy
count_positive = 0
count_negative = 0
for line in train_list:
    if int(line[0]) == 1:
        count_positive += 1
    else:
        count_negative += 1
length_list = len(train_list)
prob_positive = count_positive / length_list
prob_negative = count_negative / length_list
entropy = -(prob_positive*math.log2(prob_positive) + \
            prob_negative*math.log2(prob_negative))

entropy_dict = attribute
for key, value in attribute.items():
    entropy_1 = value
    entropy_0 = length_list - value
    if key in positive_dict.keys():
        entropy_1_1 = positive_dict[key]
    else:
        entropy_1_1 = 0
    entropy_1_0 = entropy_1 - entropy_1_1
    entropy_0_1 = count_positive - entropy_1_1
    entropy_0_0 = count_negative - entropy_1_0
    
    pro_entropy_1 = -(cal_log(entropy_1_0,entropy_1) + cal_log(entropy_1_1,entropy_1))    
    pro_entropy_0 = -(cal_log(entropy_0_0,entropy_0) + cal_log(entropy_0_1,entropy_0))
    
    gain = entropy - (entropy_1/length_list)*pro_entropy_1 - \
    (entropy_0/length_list)*pro_entropy_0 
    entropy_dict[key] = gain

print("5 attributes with the highest information gain:")
sorted_entropy = sorted(entropy_dict.items(), key=operator.itemgetter(1))
print(sorted_entropy[-1][0], " ", sorted_entropy[-2][0], " ", sorted_entropy[-3][0]\
      , " ", sorted_entropy[-4][0], " ", sorted_entropy[-5][0]) 

###question c###############################
print("question c===================")

def build_y_set(train_list):   
    y_train = []
    for line in train_list:
        if int(line[0]) == 0:           
            temp_lis = [0]
        else:
            temp_lis = [1]
        y_train.append(temp_lis)
    y_train = np.array(y_train)
    return y_train

def build_x_set(num, train_list, sorted_entropy):
    top_attributes = []
    for i in range(1, num+1):
        top_attributes.append(sorted_entropy[-i][0])
    x_train = []
    for line in train_list:
        temp_lis = []
        for attribute in top_attributes:
            if attribute in line[1:]:
                temp_lis.append(1.0)
            else:
                temp_lis.append(0.0)
        x_train.append(temp_lis) 
    x_train = np.array(x_train)
    return x_train

# from my hw2
def confusion_matrix(pred_list, test_list):
    count_TN = 0
    count_FP = 0
    count_TP = 0
    count_FN = 0
    size = len(pred_list)
    size2 = len(test_list)
    if size != size2:
        return[0, 0, 0, 0]
    for i in range(0, size):
        if int(pred_list[i]) == 1:
            if int(pred_list[i]) == int(test_list[i]):
                count_TP += 1
            else:
                count_FP += 1
        else:
            if int(pred_list[i]) == int(test_list[i]):
                count_TN += 1
            else:
                count_FN += 1
            
    return [count_TP, count_TN, count_FP, count_FN]

def zero_R(train_list, test_list):
    num_of_0 = 0
    num_of_1 = 0
    for w in train_list:
        if w[0] == '0':
            num_of_0 += 1
        else:
            num_of_1 +=1
    #print(num_of_0, " ",num_of_1) 
    if num_of_0 > num_of_1:
        pred = '0'
    else: 
        pred = '1'
    #print(pred)
    
    count_TN = 0
    count_FP = 0
    count_TP = 0
    count_FN = 0
    for i in range(0, len(test_list)):
        if pred == '1':
            if pred == test_list[i][0]:
                count_TP += 1
            else:
                count_FP += 1
        else:
            if pred == test_list[i][0]:
                count_TN += 1
            else:
                count_FN += 1   
    return [count_TP, count_TN, count_FP, count_FN]

'''
# k = 50
num = 50

# get dataset
y_train = build_y_set(train_list)
y_v_train = convert_y_to_vect(y_train)
x_train = build_x_set(num, train_list, sorted_entropy)

y_test = build_y_set(test_list)
y_v_test = convert_y_to_vect(y_test)
x_test = build_x_set(num, test_list, sorted_entropy)

# scale
x_scale = StandardScaler()
x_train = x_scale.fit_transform(x_train)

# train
nn_structure = [num, int(num/2), 2]
W, b, avg_cost_func = train_nn(nn_structure, x_train, y_v_train)

# draw graph from class code
plt.plot(avg_cost_func)
plt.ylabel('Average J')
plt.xlabel('Iteration number')
plt.show()

# predict y
y_pred = predict_y(W, b, x_test, 3)
print("Accuracy score: ", accuracy_score(y_test, y_pred)*100)
list_con = confusion_matrix(y_pred, y_test)
print("Confusion Matrix:")
print("TN = ", list_con[1], " FP = ", list_con[2])
print("FN = ", list_con[3], " TP = ", list_con[0])
print("Accuracy for CNN:")
print((list_con[0]+list_con[1])/len(test_list))
'''
###question d###############################
print("question d===================")
list_con = zero_R(train_list, test_list)
print("Accuracy for Zero-R:")
print((list_con[0]+list_con[1])/len(test_list))

###question e###############################
print("question f===================")
print("Choose 10 values of k = 10, 20, 30, 40, 50, 60, 70, 80, 90, 100")

def run_different_k(num, train_list, test_list):
    y_train = build_y_set(train_list)
    y_v_train = convert_y_to_vect(y_train)
    x_train = build_x_set(num, train_list, sorted_entropy)

    y_test = build_y_set(test_list)
    x_test = build_x_set(num, test_list, sorted_entropy)
    
    x_scale = StandardScaler()
    x_train = x_scale.fit_transform(x_train)
    
    nn_structure = [num, int(num/2), 2]
    W, b, avg_cost_func = train_nn(nn_structure, x_train, y_v_train)
    
    y_pred = predict_y(W, b, x_test, 3)
    return accuracy_score(y_test, y_pred)*100

list_accuracy = []
list_k = []
for k in range(1, 11):
    k = k*10
    acc = run_different_k(k, train_list, test_list)
    list_accuracy.append(acc)
    list_k.append(k)

list_accuracy = np.array(list_accuracy)
list_k = np.array(list_k)

# draw graph from class code
plt.plot(list_k, list_accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Value of k')
plt.show()


    
    



        