
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
st = time.time()


# In[2]:
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
        


# In[3]:

##### Support Vector Machines ######
#Predict Function:
def predict_svm(weights, x, label):
    up = 0
    pl = 0
    dp = ((np.dot(weights, x)))
    if dp > 0:
        pl = 1
    else:
        pl = -1
    if dp * label <=1:
        up=1       
    #print(dp)
    return up, pl

#Function to update weight vector
def update_weights_svm(weights, y, r, c, x):
    weights = ((1-r) * weights) + (r * c * y * x)
    return weights

def update_weights_svm1(weights, r):
    weights = (1-r) * weights
    return weights

#Accuracy
def accuracy(final_weight, data):
    mistakes = 0
    data_size = data.shape[1]
    for i in range(len(data)):
        if np.dot(final_weight, data[i][0:(data_size-1)]) >= 0  and data[i][(data_size-1)] != 1:
            mistakes += 1
        elif np.dot(final_weight, data[i][0:(data_size-1)]) < 0  and data[i][(data_size-1)] != -1:
            mistakes += 1
    a = 100 - ((mistakes/len(data)) * 100)
    return a


# In[4]:

def accuracy1(final_weight, data):
    mistakes = 0
    data_size = data.shape[1]
    for i in range(len(data)):
        u, pl = predict_svm(final_weight, data[i][0:(data_size-1)],data[i][(data_size-1)])
        #print("pl ",pl)
        #print("al",data[i][(data_size-1)])
        if pl == 1  and data[i][(data_size-1)] != 1:
            mistakes += 1
        elif pl == -1  and data[i][(data_size-1)] != -1:
            mistakes += 1
    a = 100 - ((mistakes/len(data)) * 100)
    return a


# In[5]:

def cv_svm(cv1, cv2, cv3, cv4, cv5, learn_rate, c_tradeoff, epoch,cv11,cv22,cv33,cv44,cv55):
    cv = [cv1,cv2,cv3,cv4,cv5]
    cvv = [cv11,cv22,cv33,cv44,cv55]
    hyper_combo = []

    for i in range(5):
        #print(i)
        test = cv[i]
        train = cv[(i+1) % 5]
        train = np.append(train,cv[(i+2) % 5],axis=0)
        train = np.append(train,cv[(i+3) % 5],axis=0)
        train = np.append(train,cv[(i+4) % 5],axis=0)
        
        test1 = cvv[i]
        train1 = cvv[(i+1) % 5]
        train1 = np.append(train1,cvv[(i+2) % 5],axis=0)
        train1 = np.append(train1,cvv[(i+3) % 5],axis=0)
        train1 = np.append(train1,cvv[(i+4) % 5],axis=0)
    
        #accuracy = [np.array([0.0 for i in range(len(learn_rate)*len(c))])]
        accuracy = []
        #accuracy = np.array([0,0,0,0,0,0,0,0,0])
        #index = 0
        for l in learn_rate:
            for c in  c_tradeoff:
                final_weights,a_f,a,u,w_f = svm(train1, train,test1,test,l, c, epoch)
                #accuracy[index] = a_f
                accuracy.append(a_f)
                hyper_combo.append((l,c))
                #index = index+1
        accuracy = np.array(accuracy)
        if i == 0:
            avg_acc = accuracy
        else:
            avg_acc = avg_acc+accuracy

    max_i = np.where(avg_acc == max(avg_acc))[0][0]
    c = hyper_combo[max_i]
    #print(avg_acc)
    #print(max_index)
    return c, avg_acc[max_i]/5


# In[6]:

def svm(train, train_wt, test,test_wt, learn_rate, c, epoch):
    updates = 0
    max_index =1000
    #Intialize a weight vector of size number of fearures + 1
    weights = np.array([0.001 for i in range((train.shape[1]))])

    acc = np.array([0.0 for i in range(epoch)])
    weight_epoch = np.array([weights for i in range(epoch)])

    for e in range(epoch):
        #print(e)
        learn_rate_e = learn_rate/(1+e)
        np.random.seed(200+e)
        np.random.shuffle(train_wt)
        for i in range(len(train_wt)):
            label = train_wt[i][max_index+1]
            up_flag,pl = predict_svm(weights, train[i], label)
            #print(y_hat)

            if up_flag == 1:
                
                weights = update_weights_svm(weights,train_wt[i][max_index+1], learn_rate_e, c, train_wt[i][0:max_index+1])
                updates = updates + 1
            
            else:
                weights = update_weights_svm1(weights, learn_rate)
                updates = updates+1
           
        acc[e] = accuracy1(weights, train_wt)
        weight_epoch[e] = weights

    weight_final = weight_epoch[np.where(acc == max(acc))[0][0]]
    accuracy_final = accuracy1(weights, test_wt) 
    return weights, accuracy_final,acc, updates, weight_final


# In[7]:

def cv_splits(data):
    len_split = len(data)/5
    len_split = math.ceil(len_split)
    col = data.shape[1]-1
    #print(len(data))
    #print(len_split)
    #print(col)
    
    cv1 = data[0:len_split][:]
    cv2 = data[len_split:2*len_split][:]
    cv3 = data[2*len_split:3*len_split][:]
    cv4 = data[3*len_split:4*len_split][:]
    cv5 = data[4*len_split:][:]
    
    return cv1,cv2,cv3,cv4,cv5


# In[8]:

#Reading data: with target and without target
x_wt = svm_tree_parse(file_arg)
x = np.delete(x_wt,1000,axis=1)
#inserting bias term
x = np.insert(x,1000,1,axis=1)
x_wt = np.insert(x_wt,1000,1,axis=1)


# In[9]:

#cv1,cv2,cv3,cv4,cv5 = cv_splits(x_wt)
x_train_wt = x_wt[:][0:2818]
x_test_wt = x_wt[:][2818:]
x_train = x[:][0:2818]
x_test = x[:][2818:]

cv1_wt,cv2_wt,cv3_wt,cv4_wt,cv5_wt = cv_splits(x_train_wt)
cv1,cv2,cv3,cv4,cv5 = cv_splits(x_train)


# In[10]:

stime = time.time()
learn_rate = [1,0.1,0.01,0.001,0.0001,0.00001]
c = [10,1,0.1,0.01,0.001,0.0001,0.00001]
c, avg_accuracy = cv_svm(cv1_wt,cv2_wt,cv3_wt,cv4_wt,cv5_wt,learn_rate,c,10,cv1,cv2,cv3,cv4,cv5)
print("Best learning rate: ",c[0]) 
print("Best C: ",c[1])

print("Cross validation accuracy of best combination of hyperparameters: ",avg_accuracy ) 


# In[11]:

#Train the algorithm and find out the epoch having the greatest accuracy
weights,a_f,a,updates,final_weights = svm(x,x_wt,x,x_wt, c[0], c[1], 20)
#weights,a_f,a,updates,final_weights = svm(train_dense,train_dense_wt,train_dense,test_dense_wt, 0.1, 0.1, 20)
print("Updates: ",updates)


# In[12]:

accuracy_on_train = accuracy1(final_weights,x_train_wt)
print("Accuracy on train: ",accuracy_on_train)
accuracy_on_test = accuracy1(final_weights,x_test_wt)
print("Accuracy on  test: ",accuracy_on_test)

