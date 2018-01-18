
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

def predict_logistic(weights, x):
    dp =  ((np.dot(weights, x)))
    #if np.isnan(dp):
        #print(weights)
    
    try:
        sigmoid = 1/(1 + math.exp(-dp))
    except OverflowError:
        #print("except")
        sigmoid = 0
   # print(sigmoid)
    if sigmoid > 0.5:
        return 1
    else:
        return -1

#Function to update weight vector
def update_weights_logistic(weights, r, sigma, y, x):
    wx =  np.dot(weights, x)
    #print(wx)
    try:
#         wx =  np.dot(weights, x)
#         print(wx)
        #weights = (1 - (r/2*sigma)) * weights + ((r * y * math.exp(-(y*yhat)))/(1+math.exp(-(y*yhat)))) * x
        #weights = (1 - (r/2*sigma)) * weights + ((r * y)/(1 + math.exp(y * wx))) * x
        weights = weights - (2*r/sigma) * weights + ((r * y)/(1 + math.exp(y * wx))) * x
    except OverflowError:
        weights = (1 - (2*r/sigma)) * weights
    return weights

#Accuracy
def accuracy(final_weight, data):
    mistakes = 0
    data_size = data.shape[1]
    for i in range(len(data)):
        pl = predict_logistic(final_weight, data[i][0:data_size-1])
        #print(pl)
        if pl == 1  and data[i][(data_size-1)] != 1:
            mistakes = mistakes + 1
        elif predict_logistic(final_weight, data[i][0:data_size-1]) == -1  and data[i][(data_size-1)] != -1:
            mistakes = mistakes + 1
    a = 100 - ((mistakes/len(data)) * 100)
    return a


# In[4]:

def cv_logistic1(cv1, cv2, cv3, cv4, cv5, learn_rate, c_tradeoff, epoch,cv11,cv22,cv33,cv44,cv55):
    cv = [cv1,cv2,cv3,cv4,cv5]
    cvv = [cv11,cv22,cv33,cv44,cv55]
    hyper_combo = []
    for i in range(5):
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
                final_weights,a_f,a,u,w_f = logistic(train1, train,test1,test,l, c, epoch)
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


# In[5]:

def logistic(train, train_wt, test,test_wt, learn_rate, sigma, epoch):
    updates = 0

    #Intialize a weight vector of size number of fearures + 1
    weights = np.array([0.01 for i in range((train.shape[1]))])

    acc = np.array([0.0 for i in range(epoch)])
    weight_epoch = np.array([weights for i in range(epoch)])
    #n=0

    for e in range(epoch):
        #print(e)
        learn_rate_e = learn_rate/(1+e)
        np.random.seed(200+e)
        np.random.shuffle(train_wt)
        for i in range(len(train_wt)):
            y_hat = predict_logistic(weights, train[i])

            weights = update_weights_logistic(weights,learn_rate_e, sigma,train_wt[i][max_index+1], train_wt[i][0:max_index+1])
            updates = updates + 1
        
           
        acc[e] = accuracy(weights, train_wt)
        weight_epoch[e] = weights

    weight_final = weight_epoch[np.where(acc == max(acc))[0][0]]
    accuracy_final = accuracy(weights, test_wt) 
    #return weights,acc, updates, weight_final
    return weights, accuracy_final,acc, updates, weight_final


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
#print(max_index)


# In[8]:

#Adding bias term to train data
for i in range(len(train)):
    train[i][max_index+1] = 1

#Adding bias term to test data
for i in range(len(test)):
    test[i][max_index+1] = 1

#Adding bias term to cv1
for i in range(len(cv1)):
    cv1[i][max_index+1] = 1


#Adding bias term to cv2
for i in range(len(cv2)):
    cv2[i][max_index+1] = 1

#Adding bias term to cv3
for i in range(len(cv3)):
    cv3[i][max_index+1] = 1

#Adding bias term to cv4
for i in range(len(cv4)):
    cv4[i][max_index+1] = 1

#Adding bias term to cv5
for i in range(len(cv5)):
    cv5[i][max_index+1] = 1


# In[9]:

#Convert sparse representaion to dense reresentationimport time
train_dense = ([0 for g in range(len(train))])
for i in range(len(train)):
    x_dense = ([0 for g in range(max_index+1)])
    #print(x_dense)
    for u in range(1,max_index+2):
        if u in train[i].keys():
            x_dense[(u-1)] = 1.0
        else:
            x_dense[(u-1)] = 0.0
    train_dense[i] = x_dense
        

train_dense = np.array(train_dense)
t = np.array([train_target])
train_dense_wt = np.concatenate((train_dense,t.T), axis = 1)


# In[10]:

#Convert sparse representaion to dense reresentationimport time
test_dense = ([0 for g in range(len(test))])
for i in range(len(test)):
    x_dense = ([0 for g in range(max_index+1)])
    #print(x_dense)
    for u in range(1,max_index+2):
        if u in test[i].keys():
            x_dense[(u-1)] = 1.0
        else:
            x_dense[(u-1)] = 0.0
    test_dense[i] = x_dense

test_dense = np.array(test_dense)
t = np.array([test_target])
test_dense_wt = np.concatenate((test_dense, t.T), axis = 1)


# In[11]:

#Convert sparse representaion to dense reresentationimport time
cv1_dense = ([0 for g in range(len(cv1))])
for i in range(len(cv1)):
    x_dense = ([0 for g in range(max_index+1)])
    #print(x_dense)
    for u in range(1,max_index+2):
        if u in cv1[i].keys():
            x_dense[(u-1)] = 1.0
        else:
            x_dense[(u-1)] = 0.0
    cv1_dense[i] = x_dense

cv1_dense = np.array(cv1_dense)
t = np.array([cv1_t])
cv1_dense_wt = np.concatenate((cv1_dense, t.T), axis = 1)


# In[12]:

#Convert sparse representaion to dense reresentationimport time
cv2_dense = ([0 for g in range(len(cv2))])
for i in range(len(cv2)):
    x_dense = ([0 for g in range(max_index+1)])
    #print(x_dense)
    for u in range(1,max_index+2):
        if u in cv2[i].keys():
            x_dense[(u-1)] = 1.0
        else:
            x_dense[(u-1)] = 0.0
    cv2_dense[i] = x_dense

cv2_dense = np.array(cv2_dense)
t = np.array([cv2_t])
cv2_dense_wt = np.concatenate((cv2_dense, t.T), axis = 1)


# In[13]:

#Convert sparse representaion to dense reresentationimport time
cv3_dense = ([0 for g in range(len(cv3))])
for i in range(len(cv3)):
    x_dense = ([0 for g in range(max_index+1)])
    #print(x_dense)
    for u in range(1,max_index+2):
        if u in cv3[i].keys():
            x_dense[(u-1)] = 1.0
        else:
            x_dense[(u-1)] = 0.0
    cv3_dense[i] = x_dense

cv3_dense = np.array(cv3_dense)
t = np.array([cv3_t])
cv3_dense_wt = np.concatenate((cv3_dense, t.T), axis = 1)


# In[14]:

#Convert sparse representaion to dense reresentationimport time
cv4_dense = ([0 for g in range(len(cv4))])
for i in range(len(cv4)):
    x_dense = ([0 for g in range(max_index+1)])
    #print(x_dense)
    for u in range(1,max_index+2):
        if u in cv4[i].keys():
            x_dense[(u-1)] = 1.0
        else:
            x_dense[(u-1)] = 0.0
    cv4_dense[i] = x_dense

cv4_dense = np.array(cv4_dense)
t = np.array([cv4_t])
cv4_dense_wt = np.concatenate((cv4_dense, t.T), axis = 1)


# In[15]:

#Convert sparse representaion to dense reresentationimport time
cv5_dense = ([0 for g in range(len(cv5))])
for i in range(len(cv5)):
    x_dense = ([0 for g in range(max_index+1)])
    #print(x_dense)
    for u in range(1,max_index+2):
        if u in cv5[i].keys():
            x_dense[(u-1)] = 1.0
        else:
            x_dense[(u-1)] = 0.0
    cv5_dense[i] = x_dense

cv5_dense = np.array(cv5_dense)
t = np.array([cv5_t])
cv5_dense_wt = np.concatenate((cv5_dense, t.T), axis = 1)


# In[16]:

learn_rate = [10,0.1,0.01,0.001,0.0001,0.00001]
sigma = [0.1,1,10,100,1000,10000]
c, avg_accuracy = cv_logistic1(cv1_dense_wt,cv2_dense_wt,cv3_dense_wt,cv4_dense_wt,cv5_dense_wt,learn_rate,sigma,10,cv1_dense,cv2_dense,cv3_dense,cv4_dense,cv5_dense)
print("Best learning rate: ",c[0]) 
print("Best Sigma: ",c[1])

print("Cross validation accuracy of best combination of hyperparameters: ",avg_accuracy ) 


# In[17]:

#Train the algorithm and find out the epoch having the greatest accuracy
stime = time.time()
#weights,a_f,a,updates,final_weights = logistic(train_dense,train_dense_wt,train_dense,test_dense_wt, learn_rate_final, c, 20)
weights,a_f,a,updates,final_weights = logistic(train_dense,train_dense_wt,test_dense,test_dense_wt, c[0], c[1], 20)
etime = time.time() - stime


# In[18]:

stime = time.time()
accuracy_on_train = accuracy(final_weights,train_dense_wt)
print("Accuracy on training set: ",accuracy_on_train)
# etime = time.time() - stime
# print("accuracy time: ",etime)

stime = time.time()
accuracy_on_test = accuracy(final_weights,test_dense_wt)
print("Accuracy on test set: ",accuracy_on_test)
# etime = time.time() - stime
# print("accuracy time: ",etime)


# In[19]:

# etime = time.time() - st
# print("Total Time: ",(etime)/60,"Minutes")

