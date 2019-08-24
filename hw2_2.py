# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 06:00:12 2019

@author: Xinyun Zhao
         Muyao Wang
"""

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

def confusion_matrix(predict, test_list):

    count_TN = 0
    count_FP = 0
    count_TP = 0
    count_FN = 0
    size = len(predict)
    size2 = len(test_list)
    if size != size2:
        return[0, 0, 0, 0]
    for i in range(0, size):
        if predict[i] == '1':
            if predict[i] == test_list[i][0]:
                count_TP += 1
            else:
                count_FP += 1
        else:
            if predict[i] == test_list[i][0]:
                count_TN += 1
            else:
                count_FN += 1
            
    return [count_TP, count_TN, count_FP, count_FN]

def compare_lines(line1, line2):
    d= {}
    counter = 0
    for w in line1[1:]:
        for c in line2[1:]:        
            if w == c:
                if w not in d.keys():
                    d[w] = 1
                    counter += 1
    if counter == 0:
        distance = 2
    else:
        distance = 1/counter
       
    return [line2[0],distance]
    
def k_Nearest(k, test_list, train_list):
    if k <= 0:
        return []
    predict_list = []
    line_list = []
    #print(size_test, size_train)
    # each line in test
    for line in test_list:
        line_list.clear()
        for l in train_list:

            list_c = compare_lines(line, l)
            line_list.append(list_c)

        line_list.sort(key=lambda tup: tup[1])
  
        num_of_0 = 0
        num_of_1 = 0
        counter = 0
        counterk = k
        for counter in range (0, len(line_list)):
            if counter == counterk:
                if line_list[counter][1] == line_list[counter-1][1]:
                    if line_list[counter][0] == '0':
                        num_of_0 += 1  
                    elif line_list[counter][0] == '1':
                        num_of_1 += 1
                    counterk += 1
                    continue
                else:
                    counterk = k
                    break;                   
            elif counter < counterk:
                if line_list[counter][0] == '0':
                    num_of_0 += 1  
                elif line_list[counter][0] == '1':
                    num_of_1 += 1
        #print(num_of_0, " ",num_of_1)    
        if num_of_0 > num_of_1:
            predict_list.append('0')
        else: 
            predict_list.append('1')
              
    return predict_list

def new_compare_lines(line1, line2):

    inter = len(list(set(line1[1:]).intersection(set(line2[1:]))))
    union = len(list(set(line1[1:]).union(set(line2[1:]))))
    #print(inter, " ", union)
    
    distance =1 - inter / union
    #print(distance)
    return [line2[0],distance]

def new_k_Nearest(k, test_list, train_list):
    if k <= 0:
        return []
    predict_list = []
    line_list = []
    #print(size_test, size_train)
    # each line in test
    for line in test_list:
        line_list.clear()
        for l in train_list:

            list_c = new_compare_lines(line, l)
            line_list.append(list_c)

        line_list.sort(key=lambda tup: tup[1])
  
        num_of_0 = 0
        num_of_1 = 0
        counter = 0
        counterk = k
        for counter in range (0, len(line_list)):
            if counter == counterk:
                if line_list[counter][1] == line_list[counter-1][1]:
                    if line_list[counter][0] == '0':
                        num_of_0 += 1  
                    elif line_list[counter][0] == '1':
                        num_of_1 += 1
                    counterk += 1
                    continue
                else:
                    counterk = k
                    break;                   
            elif counter < counterk:
                if line_list[counter][0] == '0':
                    num_of_0 += 1  
                elif line_list[counter][0] == '1':
                    num_of_1 += 1
        #print(num_of_0, " ",num_of_1)    
        if num_of_0 > num_of_1:
            predict_list.append('0')
        else: 
            predict_list.append('1')
              
    return predict_list


##########################MAIN PROGRAM#######################################

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
    

#########################K = 3##################################
print("===============K = 3==================")
print("PROCESSING!")
predict = k_Nearest(1, test_list, train_list)
print("For K = 1")
print("Prediction of Line 18: ", predict[17])

list_con = confusion_matrix(predict, test_list)
print("Confusion Matrix:")
print("TN = ", list_con[1], " FP = ", list_con[2])
print("FN = ", list_con[3], " TP = ", list_con[0])

Trate = (list_con[0] + list_con[1]) / len(predict)
print("Accuracy = ", Trate)

FPR = list_con[2] / (list_con[2] + list_con[1])
print("False Positive Rate= ", FPR)
TPR = list_con[0] / (list_con[0] + list_con[3])
print("True Positive Rate= ", TPR)

#########################K = 5##################################
print("===============K = 5=================")
print("PROCESSING!")
predict = k_Nearest(5, test_list, train_list)
print("For K = 5")
print("Prediction of Line 18: ", predict[17])

list_con = confusion_matrix(predict, test_list)
print("Confusion Matrix:")
print("TN = ", list_con[1], " FP = ", list_con[2])
print("FN = ", list_con[3], " TP = ", list_con[0])

Trate = (list_con[0] + list_con[1]) / len(predict)
print("Accuracy = ", Trate)

FPR = list_con[2] / (list_con[2] + list_con[1])
print("False Positive Rate= ", FPR)
TPR = list_con[0] / (list_con[0] + list_con[3])
print("True Positive Rate= ", TPR)

print("=============ZERO R====================")
print("PROCESSING!")
list_con = zero_R(train_list, test_list)
print("For Zero-R")
print("Confusion Matrix:")
print("TN = ", list_con[1], " FP = ", list_con[2])
print("FN = ", list_con[3], " TP = ", list_con[0])




###################################################################

print("===============3 VALUES===================")
print("===============K = 3===================")
print("PROCESSING!")
sp = int(len(train_list)/5)
train_list1 = train_list[:sp]
train_list2 = train_list[sp:sp*2]
train_list3 = train_list[sp*2:sp*3]
train_list4 = train_list[sp*3:sp*4]
train_list5 = train_list[sp*4:]
# K = 3

predict = k_Nearest(3, train_list1, train_list2 + train_list3 + train_list4+\
                     train_list5)
list_con1 = confusion_matrix(predict, train_list1)
predict = k_Nearest(3, train_list2, train_list1 + train_list3 + train_list4+\
                     train_list5)
list_con2 = confusion_matrix(predict, train_list2)
predict = k_Nearest(3, train_list3, train_list2 + train_list1 + train_list4+\
                     train_list5)
list_con3 = confusion_matrix(predict, train_list3)
predict = k_Nearest(3, train_list4, train_list2 + train_list3 + train_list1+\
                     train_list5)
list_con4 = confusion_matrix(predict, train_list4)
predict = k_Nearest(3, train_list5, train_list2 + train_list3 + train_list4+\
                     train_list1)
list_con5 = confusion_matrix(predict, train_list5)
list_con_k3 = []
for i in range(0, 4):
    list_con_k3.append(list_con1[i] + list_con2[i] + list_con3[i] +list_con4[i]+\
                      list_con5[i])
Trate3 = (list_con_k3[0] + list_con_k3[1]) / (list_con_k3[0] + list_con_k3[1] +\
        list_con_k3[2]+ list_con_k3[3]) 
print("For k = 3")
print("Accuracy = ", Trate3)

# K = 7
print("===============K = 7===================")
print("PROCESSING!")
predict3 = k_Nearest(7, train_list1, train_list2 + train_list3 + train_list4+\
                     train_list5)
list_con1 = confusion_matrix(predict3, train_list1)
predict3 = k_Nearest(7, train_list2, train_list1 + train_list3 + train_list4+\
                     train_list5)
list_con2 = confusion_matrix(predict3, train_list2)
predict3 = k_Nearest(7, train_list3, train_list2 + train_list1 + train_list4+\
                     train_list5)
list_con3 = confusion_matrix(predict3, train_list3)
predict3 = k_Nearest(7, train_list4, train_list2 + train_list3 + train_list1+\
                     train_list5)
list_con4 = confusion_matrix(predict3, train_list4)
predict3 = k_Nearest(7, train_list5, train_list2 + train_list3 + train_list4+\
                     train_list1)
list_con5 = confusion_matrix(predict3, train_list5)
list_con_k7 = []
for i in range(0, 4):
    list_con_k7.append(list_con1[i] + list_con2[i] + list_con3[i] +list_con4[i]+\
                      list_con5[i])
Trate7 = (list_con_k7[0] + list_con_k7[1]) / (list_con_k7[0] + list_con_k7[1] +\
        list_con_k7[2]+ list_con_k7[3]) 
print("For k = 7")
print("Accuracy = ", Trate7)

print("===============K = 99===================")
print("PROCESSING!")
# K = 99

predict3 = k_Nearest(99, train_list1, train_list2 + train_list3 + train_list4+\
                     train_list5)
list_con1 = confusion_matrix(predict3, train_list1)
predict3 = k_Nearest(99, train_list2, train_list1 + train_list3 + train_list4+\
                     train_list5)

list_con2 = confusion_matrix(predict3, train_list2)
predict3 = k_Nearest(99, train_list3, train_list2 + train_list1 + train_list4+\
                     train_list5)
list_con3 = confusion_matrix(predict3, train_list3)
predict3 = k_Nearest(99, train_list4, train_list2 + train_list3 + train_list1+\
                     train_list5)
list_con4 = confusion_matrix(predict3, train_list4)
predict3 = k_Nearest(99, train_list5, train_list2 + train_list3 + train_list4+\
                     train_list1)
list_con5 = confusion_matrix(predict3, train_list5)
list_con_k99 = []
for i in range(0, 4):
    list_con_k99.append(list_con1[i] + list_con2[i] + list_con3[i] +list_con4[i]+\
                      list_con5[i])
Trate99 = (list_con_k99[0] + list_con_k99[1]) / (list_con_k99[0] + list_con_k99[1] +\
        list_con_k99[2]+ list_con_k99[3]) 
print("For k = 99")
print("Accuracy = ", Trate99)

print("===============HIGHEST ONE===================")

max_ = 0
if Trate99 == max(Trate99, Trate7, Trate3):
    max_ = 99
elif Trate7 == max(Trate99, Trate7, Trate3):
    max_ = 7
elif Trate3 == max(Trate99, Trate7, Trate3):
    max_ = 3

print("PROCESSING!")
predict = k_Nearest(max_, test_list, train_list)
list_con = confusion_matrix(predict, test_list)
print("Confusion Matrix:")
print("TN = ", list_con[1], " FP = ", list_con[2])
print("FN = ", list_con[3], " TP = ", list_con[0])

Trate = (list_con[0] + list_con[1]) / len(predict)
print("Accuracy = ", Trate)

########################NEW DISTANCE#######################################
print("=================NEW DISTANCE==================")
print("=================K = 1==================")
print("PROCESSING!")
predict = new_k_Nearest(1, test_list, train_list)
list_con = confusion_matrix(predict, test_list)
print("Confusion Matrix:")
print("TN = ", list_con[1], " FP = ", list_con[2])
print("FN = ", list_con[3], " TP = ", list_con[0])

Trate = (list_con[0] + list_con[1]) / len(predict)
print("Accuracy = ", Trate)

FPR = list_con[2] / (list_con[2] + list_con[1])
print("False Positive Rate= ", FPR)
TPR = list_con[0] / (list_con[0] + list_con[3])
print("True Positive Rate= ", TPR)

print("=================K = 5==================")
print("PROCESSING!")
predict = new_k_Nearest(5, test_list, train_list)
list_con = confusion_matrix(predict, test_list)
print("Confusion Matrix:")
print("TN = ", list_con[1], " FP = ", list_con[2])
print("FN = ", list_con[3], " TP = ", list_con[0])

Trate = (list_con[0] + list_con[1]) / len(predict)
print("Accuracy = ", Trate)

FPR = list_con[2] / (list_con[2] + list_con[1])
print("False Positive Rate= ", FPR)
TPR = list_con[0] / (list_con[0] + list_con[3])
print("True Positive Rate= ", TPR)


    
