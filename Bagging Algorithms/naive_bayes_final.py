
# coding: utf-8

# In[1]:

##Importing Libraries
import numpy as np
import math
import sys
import random
import matplotlib.pyplot as plt
import collections
import time


# In[2]:
train_arg = sys.argv[1]
test_arg = sys.argv[2]
cv1_arg = sys.argv[3]
cv2_arg = sys.argv[4]
cv3_arg = sys.argv[5]
cv4_arg = sys.argv[6]
cv5_arg = sys.argv[7]

def parse_data(filepath):
    target = []
    features = {}
    data =[]
    with open(filepath,'r') as file:
        for line in file:
            #print(line) each line is a string
            row = line.split(' ')
            target.append(int(row[0]))
            if len(row)>2:
                row = row[1:]
                for feature in row:
                    temp = feature.split(':')
                    #print(temp)
                    features[int(temp[0])]=1
            elif len(row)<2:
                features={}            
            data.append(features) 
            features = {}
    return target,data


# In[3]:

#Naive Bayes Algorithm
def naive_bayes(data,target,gamma):
    pos = 0
    neg = 0
    #prior
    for label in target:
        if label == 1:
            pos += 1
        elif label == -1:
            neg += 1
        pos_p = pos / (pos + neg)
        neg_p = neg / (pos + neg)
        
    #Likelihood
    l_y = {}
    l_n = {}
    for col in range(1, max_index+1):
        f_1 = 0
        f_0 = 0
        for row in range(len(data)):
            if (col in data[row].keys()) and target[row] == 1:
                f_1 += 1
            if col in data[row].keys() and target[row] == -1:
                f_0 += 1
        l_y[col] = math.log10((f_1 + gamma)/(pos+(2*gamma)))
        l_n[col] = math.log10((f_0 + gamma)/(neg+(2*gamma)))      
        
    return (pos_p,neg_p),l_y,l_n
    


# In[4]:

def accuracy(data, target, prior, likelihood_yes, likelihood_no):
    labels_predicted = []
    mistakes = 0
    eq=0
    gr=0
    ls=0
    for row in range(len(data)):
        like_yes = 0
        like_no = 0
        for k in range(1,max_index+1):
            if k in data[row].keys():
                if like_yes != 0:
                    like_yes = like_yes + likelihood_yes[k]                    
                else:
                    like_yes = likelihood_yes[k]  
                    
                if like_no != 0:
                    like_no = like_no +  likelihood_no[k]                      
                else:
                    like_no = likelihood_no[k]
            else:
                if like_yes != 0:
                    like_yes = like_yes + math.log10(1 - (10**likelihood_yes[k]))                    
                else:
                    like_yes = likelihood_yes[k]   
                    
                if like_no != 0:
                    like_no = like_no + math.log10(1 - (10**likelihood_no[k]))                     
                else:
                    like_no = likelihood_no[k] 
                
        prob_yes = like_yes + math.log10(prior[0])
        prob_no = like_no + math.log10(prior[1])

        if(prob_yes >= prob_no):
            labels_predicted.append(1)
            label = 1
            gr+=1
        elif prob_yes < prob_no:
            labels_predicted.append(-1)
            label = -1
            ls+=1
        if label != target[row]:
            mistakes += 1
    acc = 100 - ((mistakes/(len(target)))*100)
    return acc,mistakes
        


# In[5]:

def cv_nb(cv1,cv1_t, cv2, cv2_t, cv3, cv3_t, cv4, cv4_t, cv5, cv5_t, gamma):
    cv = [cv1,cv2,cv3,cv4,cv5]
    cv_t = [cv1_t, cv2_t, cv3_t, cv4_t, cv5_t]

    for i in range(5):
        test = cv[i]
        train = cv[(i+1) % 5]
        train = np.append(train,cv[(i+2) % 5],axis=0)
        train = np.append(train,cv[(i+3) % 5],axis=0)
        train = np.append(train,cv[(i+4) % 5],axis=0)

        test_t = cv_t[i]
        train_t = cv_t[(i+1) % 5]
        train_t = np.append(train_t,cv_t[(i+2) % 5],axis=0)
        train_t = np.append(train_t,cv_t[(i+3) % 5],axis=0)
        train_t = np.append(train_t,cv_t[(i+4) % 5],axis=0)         

        accuracy_array = np.array([0.0,0.0,0.0,0.0])        
        for j in range(len(gamma)):
            prior,lhood_y,lhood_n = naive_bayes(train,train_t,gamma[j])
            acc,m = accuracy(test, test_t, prior,lhood_y,lhood_n)
            accuracy_array[j] = acc

        if i == 0:
            avg_acc = accuracy_array
        else:
            avg_acc = avg_acc+accuracy_array
    #Choosing best learning rate
    max_i = np.where(avg_acc == max(avg_acc))[0][0]

    gamma_final = gamma[(max_i)]
    return gamma_final, avg_acc[max_i]/5


# In[6]:

#readt train data
t,d = parse_data(train_arg)
train = np.array(d)
train_target = np.array(t)


#Read test data
t,d = parse_data(test_arg)
test = np.array(d)
test_target = np.array(t)


#Read Cross validation splits
t,d = parse_data(cv1_arg)
cv1 = np.array(d)
cv1_t = np.array(t)


#Read Cross validation splits
t,d = parse_data(cv2_arg)
cv2 = np.array(d)
cv2_t = np.array(t)


#Read Cross validation splits
t,d = parse_data(cv3_arg)
cv3 = np.array(d)
cv3_t = np.array(t)

#Read Cross validation splits
t,d = parse_data(cv4_arg)
cv4 = np.array(d)
cv4_t = np.array(t)


#Read Cross validation splits
t,d = parse_data(cv5_arg)
cv5 = np.array(d)
cv5_t = np.array(t)


# In[7]:

#Find maximum index value in train
#print(train[0][0])
max_findex = []
for feature in train:
    key_list = feature.keys()
    key_list = [int(i) for i in key_list]
    if(len(key_list) > 0):
        max_findex.append(max(key_list))
    elif(len(key_list) == 0):
        max_findex.append(0)
train_max_findex = max(max_findex)
#print(train_max_findex)

max_findex = []
for feature in test:
    key_list = feature.keys()
    key_list = [int(i) for i in key_list]
    if(len(key_list) > 0):
        max_findex.append(max(key_list))
    elif(len(key_list) == 0):
        max_findex.append(0)
test_max_findex = max(max_findex)
#print(test_max_findex)
max_index = max(train_max_findex, test_max_findex)


# In[8]:

gamma = [2.0,1.5,1.0,0.5]
gamma_final, acc = cv_nb(cv1, cv1_t,cv2, cv2_t,cv3, cv3_t,cv4, cv4_t,cv5, cv5_t,gamma)
print("Best gamma: ", gamma_final)
print("Cross validation accuracy: ",acc)


# In[9]:

prior,likelihood_y,likelihood_n = naive_bayes(train,train_target,gamma_final)


# In[10]:

acc_train,m = accuracy(train, train_target, prior,likelihood_y,likelihood_n)
print("Accuracy on train: ",acc_train,"%")
#print(m)

acc_test,m = accuracy(test, test_target, prior,likelihood_y,likelihood_n)
print("Accuracy on test: ",acc_test,"%")
#print(m)

