# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 14:39:06 2019

@author: 
Xinyun Zhao xz2512 N16849245
Muyao Wang mw4086 N12430279
"""

import csv
import math

#####################TRAINING DATA#############################
with open('spambasetrain.csv', newline = '') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    list_spam = []
    list_not_spam = []
    
    counter_total = 0
    counter_spam = 0
    for row in spamreader:
        counter_total += 1
        if int(row[9]) == 1:
            counter_spam += 1
            #Xspam
            list_spam.append(row)
        else:
            #Xnotspam
            list_not_spam.append(row)

print("========================")
#number of ot spam
counter_not_spam = counter_total - counter_spam
print ("Total data: ",counter_total)
print ("Number of spam: ", counter_spam)
print ("Number of not spam: " , counter_not_spam)
# probability P(Spam) = 
pro_spam = counter_spam / counter_total
print ("P(Spam(C=1))= ", pro_spam)
# probability P(Not_Spam) = 
pro_not_spam = counter_not_spam / counter_total
print ("P(Not Spam(C=0))= ", pro_not_spam)  
print("========================")

##########################SPAM Group###################
# Sum of each column 
sum_of_line_0 = 0.0
sum_of_line_1 = 0.0
sum_of_line_2 = 0.0 
sum_of_line_3 = 0.0
sum_of_line_4 = 0.0 
sum_of_line_5 = 0.0 
sum_of_line_6 = 0.0 
sum_of_line_7 = 0.0 
sum_of_line_8 = 0.0 

for line in list_spam:    
    sum_of_line_0 += float(line[0])
    sum_of_line_1 += float(line[1])
    sum_of_line_2 += float(line[2])
    sum_of_line_3 += float(line[3])
    sum_of_line_4 += float(line[4])
    sum_of_line_5 += float(line[5])
    sum_of_line_6 += float(line[6])
    sum_of_line_7 += float(line[7])
    sum_of_line_8 += float(line[8])

#Average
mean_of_line_0 = sum_of_line_0 / counter_spam
mean_of_line_1 = sum_of_line_1 / counter_spam
mean_of_line_2 = sum_of_line_2 / counter_spam
mean_of_line_3 = sum_of_line_3 / counter_spam
mean_of_line_4 = sum_of_line_4 / counter_spam
mean_of_line_5 = sum_of_line_5 / counter_spam
mean_of_line_6 = sum_of_line_6 / counter_spam
mean_of_line_7 = sum_of_line_7 / counter_spam
mean_of_line_8 = sum_of_line_8 / counter_spam

#Variance
variance_of_line_0 = 0.0
variance_of_line_1 = 0.0
variance_of_line_2 = 0.0
variance_of_line_3 = 0.0
variance_of_line_4 = 0.0
variance_of_line_5 = 0.0
variance_of_line_6 = 0.0
variance_of_line_7 = 0.0
variance_of_line_8 = 0.0
for line in list_spam:    
    variance_of_line_0 += math.pow(float(line[0]) - mean_of_line_0, 2)
    variance_of_line_1 += math.pow(float(line[1]) - mean_of_line_1, 2)
    variance_of_line_2 += math.pow(float(line[2]) - mean_of_line_2, 2)
    variance_of_line_3 += math.pow(float(line[3]) - mean_of_line_3, 2)
    variance_of_line_4 += math.pow(float(line[4]) - mean_of_line_4, 2)
    variance_of_line_5 += math.pow(float(line[5]) - mean_of_line_5, 2)
    variance_of_line_6 += math.pow(float(line[6]) - mean_of_line_6, 2)
    variance_of_line_7 += math.pow(float(line[7]) - mean_of_line_7, 2)
    variance_of_line_8 += math.pow(float(line[8]) - mean_of_line_8, 2)

variance_of_line_0 = variance_of_line_0 / (counter_spam - 1)
variance_of_line_1 = variance_of_line_1 / (counter_spam - 1)
variance_of_line_2 = variance_of_line_2 / (counter_spam - 1)
variance_of_line_3 = variance_of_line_3 / (counter_spam - 1)
variance_of_line_4 = variance_of_line_4 / (counter_spam - 1)
variance_of_line_5 = variance_of_line_5 / (counter_spam - 1)
variance_of_line_6 = variance_of_line_6 / (counter_spam - 1)
variance_of_line_7 = variance_of_line_7 / (counter_spam - 1)
variance_of_line_8 = variance_of_line_8 / (counter_spam - 1)

print("9 Pairs of mean and variance for SPAM group:")
print("char_freq_; mean: ", mean_of_line_0, " varaince: ", variance_of_line_0)
print("char_freq_( mean: ", mean_of_line_1, " varaince: ", variance_of_line_1)
print("char_freq_[ mean: ", mean_of_line_2, " varaince: ", variance_of_line_2)
print("char_freq_! mean: ", mean_of_line_3, " varaince: ", variance_of_line_3)
print("char_freq_$ mean: ", mean_of_line_4, " varaince: ", variance_of_line_4)
print("char_freq_# mean: ", mean_of_line_5, " varaince: ", variance_of_line_5)
print("capital_run_length_average mean: ", mean_of_line_6, " varaince: ", variance_of_line_6)
print("capital_run_length_longest mean: ", mean_of_line_7, " varaince: ", variance_of_line_7)
print("capital_run_length_total mean: ", mean_of_line_8, " varaince: ", variance_of_line_8)
print("========================")

#print ("avearge:", average_line_0)

######################NOT SPAM Group###################
sum_of_line_not0 = 0.0
sum_of_line_not1 = 0.0
sum_of_line_not2 = 0.0 
sum_of_line_not3 = 0.0
sum_of_line_not4 = 0.0 
sum_of_line_not5 = 0.0 
sum_of_line_not6 = 0.0 
sum_of_line_not7 = 0.0 
sum_of_line_not8 = 0.0 

for line in list_not_spam:
    sum_of_line_not0 += float(line[0])
    sum_of_line_not1 += float(line[1])
    sum_of_line_not2 += float(line[2])
    sum_of_line_not3 += float(line[3])
    sum_of_line_not4 += float(line[4])
    sum_of_line_not5 += float(line[5])
    sum_of_line_not6 += float(line[6])
    sum_of_line_not7 += float(line[7])
    sum_of_line_not8 += float(line[8])

#Average
mean_of_line_not0 = sum_of_line_not0 / counter_not_spam
mean_of_line_not1 = sum_of_line_not1 / counter_not_spam
mean_of_line_not2 = sum_of_line_not2 / counter_not_spam
mean_of_line_not3 = sum_of_line_not3 / counter_not_spam
mean_of_line_not4 = sum_of_line_not4 / counter_not_spam
mean_of_line_not5 = sum_of_line_not5 / counter_not_spam
mean_of_line_not6 = sum_of_line_not6 / counter_not_spam
mean_of_line_not7 = sum_of_line_not7 / counter_not_spam
mean_of_line_not8 = sum_of_line_not8 / counter_not_spam

#Variance
variance_of_line_not0 = 0.0
variance_of_line_not1 = 0.0
variance_of_line_not2 = 0.0
variance_of_line_not3 = 0.0
variance_of_line_not4 = 0.0
variance_of_line_not5 = 0.0
variance_of_line_not6 = 0.0
variance_of_line_not7 = 0.0
variance_of_line_not8 = 0.0
for line in list_not_spam:    
    variance_of_line_not0 += math.pow(float(line[0]) - mean_of_line_not0, 2)
    variance_of_line_not1 += math.pow(float(line[1]) - mean_of_line_not1, 2)
    variance_of_line_not2 += math.pow(float(line[2]) - mean_of_line_not2, 2)
    variance_of_line_not3 += math.pow(float(line[3]) - mean_of_line_not3, 2)
    variance_of_line_not4 += math.pow(float(line[4]) - mean_of_line_not4, 2)
    variance_of_line_not5 += math.pow(float(line[5]) - mean_of_line_not5, 2)
    variance_of_line_not6 += math.pow(float(line[6]) - mean_of_line_not6, 2)
    variance_of_line_not7 += math.pow(float(line[7]) - mean_of_line_not7, 2)
    variance_of_line_not8 += math.pow(float(line[8]) - mean_of_line_not8, 2)

variance_of_line_not0 = variance_of_line_not0 / (counter_not_spam - 1)
variance_of_line_not1 = variance_of_line_not1 / (counter_not_spam - 1)
variance_of_line_not2 = variance_of_line_not2 / (counter_not_spam - 1)
variance_of_line_not3 = variance_of_line_not3 / (counter_not_spam - 1)
variance_of_line_not4 = variance_of_line_not4 / (counter_not_spam - 1)
variance_of_line_not5 = variance_of_line_not5 / (counter_not_spam - 1)
variance_of_line_not6 = variance_of_line_not6 / (counter_not_spam - 1)
variance_of_line_not7 = variance_of_line_not7 / (counter_not_spam - 1)
variance_of_line_not8 = variance_of_line_not8 / (counter_not_spam - 1)

print("9 Pairs of mean and variance for NOT SPAM group:")
print("char_freq_; mean: ", mean_of_line_not0, " varaince: ", variance_of_line_not0)
print("char_freq_( mean: ", mean_of_line_not1, " varaince: ", variance_of_line_not1)
print("char_freq_[ mean: ", mean_of_line_not2, " varaince: ", variance_of_line_not2)
print("char_freq_! mean: ", mean_of_line_not3, " varaince: ", variance_of_line_not3)
print("char_freq_$ mean: ", mean_of_line_not4, " varaince: ", variance_of_line_not4)
print("char_freq_# mean: ", mean_of_line_not5, " varaince: ", variance_of_line_not5)
print("capital_run_length_average mean: ", mean_of_line_not6, " varaince: ", variance_of_line_not6)
print("capital_run_length_longest mean: ", mean_of_line_not7, " varaince: ", variance_of_line_not7)
print("capital_run_length_total mean: ", mean_of_line_not8, " varaince: ", variance_of_line_not8)
print("========================")
    

#######################TESTING DATA########################

#count_actualresult_predictresult
count_0_0 = 0
count_0_1 = 0
count_1_0 = 0
count_1_1 = 0

with open('spambasetest.csv', newline = '') as csvfile:
    testreader = csv.reader(csvfile, delimiter=',')
    counter_row = 0
    for row in testreader:
        #P(xi|C)
        counter_row += 1
        result = 0
               
        p_x1_spam = math.log(1.0/(math.sqrt(2.0*math.pi*variance_of_line_0)))-\
        (math.pow(float(row[0])-mean_of_line_0, 2)/(2.0*variance_of_line_0))
        p_x2_spam = math.log(1.0/(math.sqrt(2.0*math.pi*variance_of_line_1)))-\
        (math.pow(float(row[1])-mean_of_line_1, 2)/(2.0*variance_of_line_1))
        p_x3_spam = math.log(1.0/(math.sqrt(2.0*math.pi*variance_of_line_2)))-\
        (math.pow(float(row[2])-mean_of_line_2, 2)/(2.0*variance_of_line_2))
        p_x4_spam = math.log(1.0/(math.sqrt(2.0*math.pi*variance_of_line_3)))-\
        (math.pow(float(row[3])-mean_of_line_3, 2)/(2.0*variance_of_line_3))
        p_x5_spam = math.log(1.0/(math.sqrt(2.0*math.pi*variance_of_line_4)))-\
        (math.pow(float(row[4])-mean_of_line_4, 2)/(2.0*variance_of_line_4))
        p_x6_spam = math.log(1.0/(math.sqrt(2.0*math.pi*variance_of_line_5)))-\
        (math.pow(float(row[5])-mean_of_line_5, 2)/(2.0*variance_of_line_5))
        p_x7_spam = math.log(1.0/(math.sqrt(2.0*math.pi*variance_of_line_6)))-\
        (math.pow(float(row[6])-mean_of_line_6, 2)/(2.0*variance_of_line_6))
        p_x8_spam = math.log(1.0/(math.sqrt(2.0*math.pi*variance_of_line_7)))-\
        (math.pow(float(row[7])-mean_of_line_7, 2)/(2.0*variance_of_line_7))
        p_x9_spam = math.log(1.0/(math.sqrt(2.0*math.pi*variance_of_line_8)))-\
        (math.pow(float(row[8])-mean_of_line_8, 2)/(2.0*variance_of_line_8))


        p_spam = p_x1_spam+p_x2_spam+p_x3_spam+p_x4_spam+p_x5_spam+p_x6_spam+\
        p_x7_spam+p_x8_spam+p_x9_spam+math.log(pro_spam)
       
        
        p_x1_notspam = math.log(1/(math.sqrt(2*math.pi*variance_of_line_not0)))-\
        (math.pow(float(row[0])-mean_of_line_not0, 2)/(2*variance_of_line_not0))
        
        p_x2_notspam = math.log(1/(math.sqrt(2*math.pi*variance_of_line_not1)))-\
        (math.pow(float(row[1])-mean_of_line_not1, 2)/(2*variance_of_line_not1))
        p_x3_notspam = math.log(1/(math.sqrt(2*math.pi*variance_of_line_not2)))-\
        (math.pow(float(row[2])-mean_of_line_not2, 2)/(2*variance_of_line_not2))
        p_x4_notspam = math.log(1/(math.sqrt(2*math.pi*variance_of_line_not3)))-\
        (math.pow(float(row[3])-mean_of_line_not3, 2)/(2*variance_of_line_not3))
        p_x5_notspam = math.log(1/(math.sqrt(2*math.pi*variance_of_line_not4)))-\
        (math.pow(float(row[4])-mean_of_line_not4, 2)/(2*variance_of_line_not4))
        p_x6_notspam = math.log(1/(math.sqrt(2*math.pi*variance_of_line_not5)))-\
        (math.pow(float(row[5])-mean_of_line_not5, 2)/(2*variance_of_line_not5))
        p_x7_notspam = math.log(1/(math.sqrt(2*math.pi*variance_of_line_not6)))-\
        (math.pow(float(row[6])-mean_of_line_not6, 2)/(2*variance_of_line_not6))
        p_x8_notspam = math.log(1/(math.sqrt(2*math.pi*variance_of_line_not7)))-\
        (math.pow(float(row[7])-mean_of_line_not7, 2)/(2*variance_of_line_not7))
        p_x9_notspam = math.log(1/(math.sqrt(2*math.pi*variance_of_line_not8)))-\
        (math.pow(float(row[8])-mean_of_line_not8, 2)/(2*variance_of_line_not8))
        

        
        p_notspam = p_x1_notspam+p_x2_notspam+p_x3_notspam+p_x4_notspam++p_x5_notspam+\
        p_x6_notspam+p_x7_notspam+p_x8_notspam++p_x9_notspam+math.log(pro_not_spam)
         
        if p_spam > p_notspam:
            result = 1
        else:
            result = 0
            
        if int(row[9]) == 1:
            if result == int(row[9]):
                count_1_1 += 1
                print("Actual: 1, Predict: 1, True")
            else:
                count_1_0 += 1
                print("Actual: 1, Predict: 0, False")
        else:
            if result == int(row[9]):
                count_0_0 += 1
                print("Actual: 0, Predict: 0, True")
            else:
                count_0_1 += 1
                print("Actual: 1, Predict: 0, False")
print ("==========================")
print ("Examples classified correctly: ", count_1_1+count_0_0)       
print ("Examples classified incorrectly: ", count_0_1+count_1_0)
print ("Error percentage: ", float((count_0_1+count_1_0)/counter_row))

#######################ZERO R######################


if counter_spam > counter_not_spam:
    predict_result = 1
else:
    predict_result = 0

counter_row = 0
counter_true = 0

with open('spambasetest.csv', newline = '') as csvfile:
    testreader = csv.reader(csvfile, delimiter=',')
    counter_row = 0
    for row in testreader:
        counter_row += 1
        if int(row[9]) == predict_result:
            counter_true += 1
print("==============================") 
print ("Accuracy of Zero-R : ", counter_true/counter_row)
        
    

    
        
