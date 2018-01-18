# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 14:08:43 2017

@author: krish
"""

#Import data
import numpy as np
import math
import sys
import random
import matplotlib.pyplot as plt
#"C:/Users/krish/Desktop/UoU/Fall 2017/Machine learning/HW2/Dataset/phishing.train"
print("Started execution. PLease wait..........")
print("\n")
def generate_data(path):
    train = open(path, 'r')
    lines = train.read().split("\n")
    
    for i in range(len(lines)):
        #Create an empty row of data
        row = [0]
        for n in range(0,69):
            row.append(0)
        #Split each row bt spane to get all the elements
        l = lines[i].split(" ")
        #Assign the value of each feature according to the index
        for k in range(1,len(l)):
            row[(int(l[k].split(":")[0]) - 1)] = float(l[k].split(":")[1])
        row[-1] = l[0][0:2]
        if i == 0:
            data = np.array([row])
        if i > 0:
            data = np.concatenate((data, [row]), axis=0)
    return data


#Predict Function:
def predict(weights, x):
    if np.dot(weights, x) >= 0:
        y = 1
    else: 
        y = -1
    return y

#Function to update weight vector
def update_weights(weights, y, r, x):
    weights = weights + r * y * x
    return weights

#Accuracy
def accuracy(final_weight, data):
    mistakes = 0
    data_size = data.shape[1]
    for i in range(len(data)):
        if np.dot(final_weight, data[i][0:(data_size-1)]) >= 0  and data[i][(data_size-1)] != 1:
            mistakes = mistakes + 1
        elif np.dot(final_weight, data[i][0:(data_size-1)]) < 0  and data[i][(data_size-1)] != -1:
            mistakes = mistakes + 1
    a = 100 - ((mistakes/len(data)) * 100)
    return a
###################################################################################################
def cv_simple_perceptron(cv1, cv2, cv3, cv4, cv5, learn_rate,epoch):
    cv = [cv1,cv2,cv3,cv4,cv5]
    for i in range(5):
        test = cv[i]
        train = cv[(i+1) % 5]
        train = np.append(train,cv[(i+2) % 5],axis=0)
        train = np.append(train,cv[(i+3) % 5],axis=0)
        train = np.append(train,cv[(i+4) % 5],axis=0)
    
        accuracy = np.array([0.0,0.0,0.0])
        for j in range(len(learn_rate)):
            final_weights,a_f,a,u,w_f = simple_perceptron(train,test,learn_rate[j],epoch)
            accuracy[j] = a_f
        #print(accuracy)
        if i == 0:
            avg_acc = accuracy
        else:
            avg_acc = avg_acc+accuracy
        #print(avg_acc)
    index = np.where(avg_acc == max(avg_acc))[0][0]
    return learn_rate[index], avg_acc[index]/5

def cv_dynamic_perceptron(cv1, cv2, cv3, cv4, cv5, learn_rate,epoch):
    cv = [cv1,cv2,cv3,cv4,cv5]
    for i in range(5):
        test = cv[i]
        train = cv[(i+1) % 5]
        train = np.append(train,cv[(i+2) % 5],axis=0)
        train = np.append(train,cv[(i+3) % 5],axis=0)
        train = np.append(train,cv[(i+4) % 5],axis=0)
    
        accuracy = np.array([0.0,0.0,0.0])
        for j in range(len(learn_rate)):
            final_weights,a_f,a,u,w_f = dynamic_perceptron(train,test,learn_rate[j],epoch)
            accuracy[j] = a_f
            
        if i == 0:
            avg_acc = accuracy
        else:
            avg_acc = avg_acc+accuracy
        #print(avg_acc)
    index = np.where(avg_acc == max(avg_acc))[0][0]
    return learn_rate[index], avg_acc[index]/5

def cv_margin_perceptron(cv1, cv2, cv3, cv4, cv5, learn_rate, margain, epoch):
    cv = [cv1,cv2,cv3,cv4,cv5]
    for i in range(5):
        test = cv[i]
        train = cv[(i+1) % 5]
        train = np.append(train,cv[(i+2) % 5],axis=0)
        train = np.append(train,cv[(i+3) % 5],axis=0)
        train = np.append(train,cv[(i+4) % 5],axis=0)
    
        accuracy = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        #accuracy = np.array([0,0,0,0,0,0,0,0,0])
        index = 0
        for j in range(len(learn_rate)):
            for k in range(len(margain)):
                final_weights,a_f,a,u,w_f = margin_perceptron(train,test,learn_rate[j], margain[k], epoch)
                accuracy[index] = a_f
                index = index+1
        if i == 0:
            avg_acc = accuracy
        else:
            avg_acc = avg_acc+accuracy
    #Choosing best learning rate
    max_index = np.where(avg_acc == max(avg_acc))[0][0]
    if max_index < 3:
        learn_rate_final = learn_rate[0]
    elif max_index >= 3 and max_index < 6:
        learn_rate_final = learn_rate[1]
    else:
        learn_rate_final = learn_rate[2]
    #Choosing best margain
    margain_final = margain[(max_index) % 3]
    #print(avg_acc)
    #print(max_index)
    return learn_rate_final, margain_final, avg_acc[max_index]/5

def cv_averaged_perceptron(cv1, cv2, cv3, cv4, cv5, learn_rate,epoch):
    cv = [cv1,cv2,cv3,cv4,cv5]
    for i in range(5):
        test = cv[i]
        train = cv[(i+1) % 5]
        train = np.append(train,cv[(i+2) % 5],axis=0)
        train = np.append(train,cv[(i+3) % 5],axis=0)
        train = np.append(train,cv[(i+4) % 5],axis=0)
    
        accuracy = np.array([0.0,0.0,0.0])
        for j in range(len(learn_rate)):
            final_weights,a_f,a,u,w_f = averaged_perceptron(train,test,learn_rate[j],epoch)
            accuracy[j] = a_f
        #print(accuracy)
        if i == 0:
            avg_acc = accuracy
        else:
            avg_acc = avg_acc+accuracy
        #print(avg_acc)
    index = np.where(avg_acc == max(avg_acc))[0][0]
    return learn_rate[index], avg_acc[index]/5

def cv_aggressive_perceptron(cv1, cv2, cv3, cv4, cv5, margin, epoch):
    cv = [cv1,cv2,cv3,cv4,cv5]
    for i in range(5):
        test = cv[i]
        train = cv[(i+1) % 5]
        train = np.append(train,cv[(i+2) % 5],axis=0)
        train = np.append(train,cv[(i+3) % 5],axis=0)
        train = np.append(train,cv[(i+4) % 5],axis=0)
        
        accuracy = np.array([0.0,0.0,0.0])
        for j in range(len(margin)):
            final_weights,a_f,a,u,w_f  = aggressive_perceptron(train,test,margin[j],epoch)
            accuracy[j] = a_f
        #print(accuracy)
        if i == 0:
            avg_acc = accuracy
        else:
            avg_acc = avg_acc+accuracy
        #print(avg_acc)
    index = np.where(avg_acc == max(avg_acc))[0][0]
    return margin[index], avg_acc[index]/5
##################################################################################################
def simple_perceptron(train, test, learn_rate, epoch):
    train = train.astype(float)
    test = test.astype(float)
    updates = 0
    #Intialize a weight vector of size number of fearures + 1
    weights = np.array([0.01 for i in range((train.shape[1]-1))])
    acc = np.array([0.0 for i in range(epoch)])
    weight_epoch = np.array([weights for i in range(epoch)])
    #Adding extra feature in train and set to account for bias
    #train = np.insert(train, 69, 1, axis = 1)
   # test = np.insert(test, 69, 1, axis = 1)
    for e in range(epoch):
        np.random.seed(200+e)
        np.random.shuffle(train)
        for i in range(len(train)):
            y_hat = predict(weights, train[i][0:70])
            if y_hat != train[i][70]:
                weights = update_weights(weights,train[i][70], learn_rate, train[i][0:70])
                updates = updates + 1
           
        acc[e] = accuracy(weights, test)
        weight_epoch[e] = weights
        #print(weights)

    weight_final = weight_epoch[np.where(acc == max(acc))[0][0]]
    accuracy_final = accuracy(weights, test) 
    return weights, accuracy_final,acc, updates, weight_final

def dynamic_perceptron(train, test, learn_rate, epoch):
    train = train.astype(float)
    test = test.astype(float)
    updates = 0
    #Intialize a weight vector of size number of fearures + 1
    weights = np.array([0.01 for i in range((train.shape[1]-1))])
    acc = np.array([0.0 for i in range(epoch)])
    weight_epoch = np.array([weights for i in range(epoch)])
    #Adding extra feature in train and set to account for bias
    #train = np.insert(train, 69, 1, axis = 1)
    #test = np.insert(test, 69, 1, axis = 1)
    t = 0
    for e in range(epoch):
        learn_rate_dynamic = learn_rate/(1+t)
        t = t+1
        np.random.seed(200+e)
        np.random.shuffle(train)
        for i in range(len(train)):
            #learn_rate_dynamic = learn_rate/(1+t)
            #t = t+1
            y_hat = predict(weights, train[i][0:70])
            if y_hat != train[i][70]:
                weights = update_weights(weights,train[i][70], learn_rate_dynamic, train[i][0:70])
                updates = updates + 1
        acc[e] = accuracy(weights, test)
        weight_epoch[e] = weights
        
    weight_final = weight_epoch[np.where(acc == max(acc))[0][0]]
    accuracy_final = accuracy(weights, test)
    return weights, accuracy_final,acc, updates, weight_final

def margin_perceptron(train, test, learn_rate, margain, epoch):
    train = train.astype(float)
    test = test.astype(float)
    updates = 0
    #Intialize a weight vector of size number of fearures + 1
    weights = np.array([0.01 for i in range((train.shape[1]-1))])
    acc = np.array([0.0 for i in range(epoch)])
    weight_epoch = np.array([weights for i in range(epoch)])
    #Adding extra feature in train and set to account for bias
    #train = np.insert(train, 69, 1, axis = 1)
    #test = np.insert(test, 69, 1, axis = 1)
    for e in range(epoch):
        np.random.seed(200+e)
        np.random.shuffle(train)
        for i in range(len(train)):
            #y_hat = predict(weights, train[i][0:70])
            if train[i][70] * np.dot(weights,train[i][0:70]) < margain:
                weights = update_weights(weights,train[i][70], learn_rate, train[i][0:70])
                updates = updates + 1
        acc[e] = accuracy(weights, test)
        weight_epoch[e] = weights  
        
    weight_final = weight_epoch[np.where(acc == max(acc))[0][0]]        
    accuracy_final = accuracy(weights, test)
    #return weights, acc, updates
    return weights, accuracy_final,acc, updates, weight_final

def averaged_perceptron(train, test, learn_rate, epoch):
    train = train.astype(float)
    test = test.astype(float)
    updates = 0
    #Intialize a weight vector of size number of fearures + 1
    weights = np.array([0.01 for i in range((train.shape[1]-1))])
    acc = np.array([0.0 for i in range(epoch)])
    weight_epoch = np.array([weights for i in range(epoch)])
    #Adding extra feature in train and set to account for bias
    #train = np.insert(train, 69, 1, axis = 1)
    #test = np.insert(test, 69, 1, axis = 1)
    for e in range(epoch):
        np.random.seed(200+e)
        np.random.shuffle(train)
        for i in range(len(train)):
            y_hat = predict(weights, train[i][0:70])
            if y_hat != train[i][70]:
                weights = update_weights(weights,train[i][70], learn_rate, train[i][0:70])
                updates = updates + 1
            if i == 0:
                avg_weights = weights
            else:
                avg_weights = avg_weights + weights
                
        acc[e] = accuracy(avg_weights, test)
        weight_epoch[e] = avg_weights
        
    weight_final = weight_epoch[np.where(acc == max(acc))[0][0]]
    accuracy_final = accuracy(avg_weights, test)
    return avg_weights, accuracy_final,acc, updates, weight_final

def aggressive_perceptron(train, test, margin, epoch):
    train = train.astype(float)
    test = test.astype(float)
    updates = 0
    #Intialize a weight vector of size number of fearures + 1
    weights = np.array([0.01 for i in range((train.shape[1]-1))])
    acc = np.array([0.0 for i in range(epoch)])
    weight_epoch = np.array([weights for i in range(epoch)])   
    #Adding extra feature in train and set to account for bias
    #train = np.insert(train, 69, 1, axis = 1)
    #test = np.insert(test, 69, 1, axis = 1)
    for e in range(epoch):
        np.random.seed(200+e)
        np.random.shuffle(train)
        for i in range(len(train)):
            #y_hat = predict(weights, train[i][0:70])
            if train[i][70] * np.dot(weights,train[i][0:70]) < margin:
                learn_rate = (margin - (train[i][70] * (np.dot(weights,train[i][0:70])))) / (np.dot(train[i][0:70],train[i][0:70]) + 1)
                weights = update_weights(weights,train[i][70], learn_rate, train[i][0:70])
                updates = updates + 1
                
        acc[e] = accuracy(weights, test)
        weight_epoch[e] = weights  
        
    weight_final = weight_epoch[np.where(acc == max(acc))[0][0]]
    accuracy_final = accuracy(weights, test) 
    return weights, accuracy_final,acc, updates, weight_final


###################################################################################################
train = sys.argv[1]
test = sys.argv[2]
dev = sys.argv[3]
cv1 = sys.argv[4]
cv2 = sys.argv[5]
cv3 = sys.argv[6]
cv4 = sys.argv[7]
cv5 = sys.argv[8]



#Reading data
train_data = generate_data(train)
#print(train_data.shape[1])
#Add bias term
train_data = np.insert(train_data, 69, 1, axis = 1)

test_data = generate_data(test)
#print(test_data.shape)
#Add bias term
test_data = np.insert(test_data, 69, 1, axis = 1)

dev_data = generate_data(dev)
#print(dev_data.shape)
#Add bias term
dev_data = np.insert(dev_data, 69, 1, axis = 1)

cv1 = generate_data(cv1)
#print(cv1.shape)
#Add bias term
cv1 = np.insert(cv1, 69, 1, axis = 1)

cv2 = generate_data(cv2)
#print(cv1.shape)
#Add bias term
cv2 = np.insert(cv2, 69, 1, axis = 1)

cv3 = generate_data(cv3)
#print(cv1.shape)
#Add bias term
cv3 = np.insert(cv3, 69, 1, axis = 1)

cv4 = generate_data(cv4)
#print(cv4.shape)
#Add bias term
cv4 = np.insert(cv4, 69, 1, axis = 1)

cv5 = generate_data(cv5)
#print(cv5.shape)
#Add bias term
cv5 = np.insert(cv5, 69, 1, axis = 1)

print("#SIMPLE PERCEPTRON")
###########################################################################################################
#Cross validation to find best learning rate
learn_rate = [1,0.1,0.01]       
learn_rate_final, avg_accuracy = cv_simple_perceptron(cv1,cv2,cv3,cv4,cv5,learn_rate,10)
print("Best learning rate: ",learn_rate_final) 

print("Cross validation accuracy of best hyperparameter: ",avg_accuracy )

#Train the algorithm and find out the epoch having the greatest accuracy
weights,a_f,a,updates,final_weights = simple_perceptron(train_data,dev_data, learn_rate_final, 20)
#print("Accuracy on training set: ",a_f)
print("Updates: ",updates)

print("Development set accuracy: ",list(a))

accuracy_on_train = accuracy(final_weights,train_data.astype(float))
#print("Accuracy on training set: ",accuracy_on_train)

accuracy_on_train = accuracy(final_weights,test_data.astype(float))
print("Accuracy on test set: ",accuracy_on_train)

#Plot
#epoch_index = np.array([i for i in range(20)])
#plt.scatter(epoch_index.astype(int),a)
#fig = plt.plot(epoch_index.astype(int),list(a))
#plt.xlabel("Epochs")
#plt.ylabel("Accuracy")


###########################################################################################################
print("#DYNAMIC LEARNING RATE PERCEPTRON")
###########################################################################################################
#Cross validation to find best learning rate
learn_rate = [1,0.1,0.01]       
learn_rate_final, avg_accuracy = cv_dynamic_perceptron(cv1,cv2,cv3,cv4,cv5,learn_rate,10)
print("Best learning rate: ",learn_rate_final) 

print("Cross validation accuracy of best hyperparameter: ",avg_accuracy )

#Train the algorithm and find out the epoch having the greatest accuracy
weights,a_f,a,updates,final_weights = dynamic_perceptron(train_data,dev_data, learn_rate_final, 20)
#print("Accuracy on training set: ",a_f)
print("Updates: ",updates)

print("Development set accuracy: ",list(a))

accuracy_on_train = accuracy(final_weights,train_data.astype(float))
#print("Accuracy on training set: ",accuracy_on_train)

accuracy_on_train = accuracy(final_weights,test_data.astype(float))
print("Accuracy on test set: ",accuracy_on_train)

#Plot
#epoch_index = np.array([i for i in range(20)])
#plt.scatter(epoch_index.astype(int),a)
#fig = plt.plot(epoch_index.astype(int),list(a))
#plt.xlabel("Epochs")
#plt.ylabel("Accuracy")
###########################################################################################################
print("#MARGAIN PERCEPTRON")
###########################################################################################################
#Cross validation to find best learning rate
learn_rate = [1,0.1,0.01]
margain = [1,0.1,0.01]       
learn_rate_final,margin_final,avg_accuracy = cv_margin_perceptron(cv1,cv2,cv3,cv4,cv5,learn_rate,margain,10) 
print("Best learning rate: ",learn_rate_final)
print("Best margin: ",margin_final) 
print("Cross validation accuracy of best combination of hyperparameters: ",avg_accuracy ) 

#Train the algorithm and find out the epoch having the greatest accuracy
weights,a_f,a,updates,final_weights = margin_perceptron(train_data,dev_data, learn_rate_final, margin_final, 20)
#print("Accuracy on training set: ",a_f)
print("Updates: ",updates)

print("Development set accuracy: ",list(a))

accuracy_on_train = accuracy(final_weights,train_data.astype(float))
#print("Accuracy on training set: ",accuracy_on_train)

accuracy_on_train = accuracy(final_weights,test_data.astype(float))
print("Accuracy on test set: ",accuracy_on_train)

#Plot
#epoch_index = np.array([i for i in range(20)])
#plt.scatter(epoch_index.astype(int),a)
#fig = plt.plot(epoch_index.astype(int),list(a))
#plt.xlabel("Epochs")
#plt.ylabel("Accuracy")

###########################################################################################################
print("#AVERAGED PERCEPTRON")
###########################################################################################################
#Cross validation to find best learning rate
learn_rate = [1,0.1,0.01]       
learn_rate_final, avg_accuracy = cv_averaged_perceptron(cv1,cv2,cv3,cv4,cv5,learn_rate,10)
print("Best learning rate: ",learn_rate_final) 

print("Cross validation accuracy of best hyperparameter: ",avg_accuracy )

#Train the algorithm and find out the epoch having the greatest accuracy
weights,a_f,a,updates,final_weights = averaged_perceptron(train_data,dev_data, learn_rate_final, 20)
#print("Accuracy on training set: ",a_f)
print("Updates: ",updates)

print("Development set accuracy: ",list(a))

accuracy_on_train = accuracy(final_weights,train_data.astype(float))
#print("Accuracy on training set: ",accuracy_on_train)

accuracy_on_train = accuracy(final_weights,test_data.astype(float))
print("Accuracy on test set: ",accuracy_on_train)

#Plot
#epoch_index = np.array([i for i in range(20)])
#plt.scatter(epoch_index.astype(int),a)
#fig = plt.plot(epoch_index.astype(int),list(a))
#plt.xlabel("Epochs")
#plt.ylabel("Accuracy")
###########################################################################################################
print("#AGGRESSIVE PERCEPTRON")
###########################################################################################################
#Cross validation to find best learning rate
margin = [1,0.1,0.01]      
margain_final, avg_accuracy = cv_aggressive_perceptron(cv1,cv2,cv3,cv4,cv5,margin,10) 
print("Best margin: ",margain_final) 

print("Cross validation accuracy of best hyperparameter: ",avg_accuracy )

#Train the algorithm and find out the epoch having the greatest accuracy
weights,a_f,a,updates,final_weights = aggressive_perceptron(train_data,train_data, margain_final, 20)
#print("Accuracy on training set: ",a_f)
print("Updates: ",updates)

print("Development set accuracy: ",list(a))

accuracy_on_train = accuracy(final_weights,train_data.astype(float))
#print("Accuracy on training set: ",accuracy_on_train)

accuracy_on_train = accuracy(final_weights,test_data.astype(float))
print("Accuracy on test set: ",accuracy_on_train)

#Plot
#epoch_index = np.array([i for i in range(20)])
#plt.scatter(epoch_index.astype(int),a)
#fig = plt.plot(epoch_index.astype(int),list(a))
#plt.xlabel("Epochs")
#plt.ylabel("Accuracy")
###########################################################################################################
