
# coding: utf-8

# In[1]:

import random 
import numpy as np
import math
import sys
import random
import time
from numpy import random


# In[3]:
file_arg = sys.argv[1]

def svm_tree_parse(filepath):
    l = []
    new_l = []

    with open(filepath,'r') as file:
            for line in file:
                row = line.split('\n')
                l.append(row)

    l = [i[0].strip('')for i in l] 

    for i in l:
        new_l.append(i.split(','))

    x = [[int(float(j)) for j in i] for i in new_l]

    for i in range(len(x)):
        if x[i][-1]==0:
            x[i][-1]=-1


    return np.array(x)    


def acc_check_bagged(x_train):
    label = []
    mistake = 0
    for i in x_train:
        count_pos = 0
        count_neg = 0
        for j in i[:len(i)-1]:
            if j ==1:
                count_pos += 1
            else:
                count_neg += 1 
        #print(count_pos)
        #print(count_neg)
        if count_pos>=count_neg:
            label.append(1)
        elif count_pos<count_neg:
            label.append(-1)
        #else:
            #a = random.randint(1,-1)
            #label.append(a)

    for i in range(len(x_train)):
        if label[i]!=x_train[i][-1]:
            mistake += 1
    acc = 1- (mistake/len(x_train))    
    return acc

	
if __name__=="__main__":

	x = svm_tree_parse(file_arg)
	x_train = x[0:2818]
	x_test = x[2818:]
	print("Accuracy on Train: ",acc_check_bagged(x_train)*100,"%")
	print("Accuracy on Test: ",acc_check_bagged(x_test)*100,"%")


	



