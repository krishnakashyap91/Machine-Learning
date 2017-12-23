
# coding: utf-8

# In[2]:

##Importing Libraries
import numpy as np
import math
import sys
import random
import matplotlib.pyplot as plt
import collections
import time
def classification(datapath_train,datapath_test,cvfolds,alg):
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
                        features[int(temp[0])]=float(temp[1])
                elif len(row)<2:
                    features={}            
                data.append(features) 
                features = {}
        return target,data
    
    ##### Support Vector Machines ######
    #Predict Function:
    def predict_svm(weights, x, label):
        up = 0
        pl = 0
        dp = (np.dot(weights, x))
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
    
    #Predict Function:
    def predict_eval(weights, x):
        up = 0
        pl = 0
        dp = ((np.dot(weights, x)))
        if dp > 0:
            pl = 1
        else:
            pl = 0
        return pl

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
        #print(mistakes)
        return a
    
    
    def predict_p(weights, x):
        if np.dot(weights, x) >= 0:
            y = 1
        else: 
            y = -1
        return y    
    
    #Function to update weight vector
    def update_weights_p(weights, y, r, x):
        weights = weights + r * y * x
        return weights

    #Accuracy
    def accuracy_p(final_weight, data):
        mistakes = 0
        data_size = data.shape[1]
        for i in range(len(data)):
            if np.dot(final_weight, data[i][0:(data_size-1)]) >= 0  and data[i][(data_size-1)] != 1:
                mistakes = mistakes + 1
            elif np.dot(final_weight, data[i][0:(data_size-1)]) < 0  and data[i][(data_size-1)] != -1:
                mistakes = mistakes + 1
        a = 100 - ((mistakes/len(data)) * 100)
        return a

    def averaged_perceptron(train, test, learn_rate, epoch):
        #train = train.astype(float)
        #test = test.astype(float)
        updates = 0
        all_features = train.shape[1] -1
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
                y_hat = predict_p(weights, train[i][0:all_features])
                if y_hat != train[i][all_features]:
                    weights = update_weights_p(weights,train[i][all_features], learn_rate, train[i][0:all_features])
                    updates = updates + 1
                if i == 0:
                    avg_weights = weights
                else:
                    avg_weights = avg_weights + weights

            acc[e] = accuracy_p(avg_weights, test)
            weight_epoch[e] = avg_weights

        weight_final = weight_epoch[np.where(acc == max(acc))[0][0]]
        accuracy_final = accuracy_p(avg_weights, test)
        return avg_weights, accuracy_final,acc, updates, weight_final    
    
    def cv_averaged_perceptron(cv_array, learn_rate,epoch,folds):
        #cv = [cv1,cv2,cv3,cv4,cv5]
        for i in range(folds):
            #print(i)
            test = cv_array[i]
            train = cv_array[(i+1) % folds]
            for k in range(2,folds):
                train = np.append(train,cv_array[(i+k) % folds],axis=0)

            accuracy = np.array([0.0 for i in range(len(learn_rate))])
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
        return learn_rate[index], avg_acc[index]/folds    
    
    
    def cv_svm(cv_array, learn_rate, c_tradeoff, epoch,folds):
        #cv = [cv1,cv2,cv3,cv4,cv5]
    #     cvv = [cv11,cv22,cv33,cv44,cv55]
        hyper_combo = []
        for i in range(folds):
            #print(i)
            test = cv_array[i]
            train = cv_array[(i+1) % folds]
            for k in range(2,folds):
                train = np.append(train,cv_array[(i+k) % folds],axis=0)

            #accuracy = [np.array([0.0 for i in range(len(learn_rate)*len(c))])]
            accuracy = []
            #accuracy = np.array([0,0,0,0,0,0,0,0,0])
            #index = 0
            for l in learn_rate:
                for c in  c_tradeoff:
                    final_weights,a_f,a,u,w_f = svm(train, test, l, c, epoch)

                    #accuracy[index] = a_f
                    accuracy.append(a_f)
                    hyper_combo.append((l,c))
                    #index = index+1
            accuracy = np.array(accuracy)
            if i == 0:
                avg_acc = accuracy
            else:
                avg_acc = avg_acc+accuracy
        #Choosing best learning rate
        max_i = np.where(avg_acc == max(avg_acc))[0][0]
        c = hyper_combo[max_i]
        #print(avg_acc)
        #print(max_index)
        return c, avg_acc[max_i]/folds
    

    def svm(train_data, test_data, learn_rate, c, epoch):
        updates = 0
        #max_index = 20
        all_features = train_data.shape[1] -1
        #Intialize a weight vector of size number of fearures + 1
        weights = np.array([0.01 for i in range((train_data.shape[1])-1)])
        #print(weights.shape)

        acc = np.array([0.0 for i in range(epoch)])
        weight_epoch = np.array([weights for i in range(epoch)])

        for e in range(epoch):
            np.random.seed(seed=200+e)
            np.random.shuffle(train_data)
            #print(train_data[0][-1])
            learn_rate_e = learn_rate
            for i in range(len(train_data)):
                #label = train[i][max_index+1]
                up_flag,pl = predict_svm(weights, train_data[i][0:all_features], train_data[i][all_features])
                #print(y_hat)

                if up_flag == 1:

                    weights = update_weights_svm(weights,train_data[i][all_features], learn_rate_e, c, train_data[i][0:all_features])
                    updates = updates + 1

                else:
                    weights = update_weights_svm1(weights, learn_rate_e)
                    updates = updates+1

            acc[e] = accuracy1(weights, train_data)
            weight_epoch[e] = weights

        weight_final = weight_epoch[np.where(acc == max(acc))[0][0]]
        accuracy_final = accuracy1(weights, test_data) 
        return weights, accuracy_final,acc, updates, weight_final
    
    
    def cv_splits1(data,splits):
        len_split = len(data)/splits
        len_split = math.ceil(len_split)
        col = data.shape[1]-1

        cv = []
        for i in range(splits):
            cv.append(data[len_split*i:len_split*(i+1)])

        return cv   

    #Convert sparse representaion to dense reresentationimport time
    def sparse_dense(data, data_target):
        data_dense = ([0 for g in range(len(data))])
        for i in range(len(data)):
            x_dense = ([0 for g in range(max_index+1)])
            #print(x_dense)
            for u in range(1,max_index+2):
                if u in data[i].keys():
                    x_dense[(u-1)] = data[i][u]
                else:
                    x_dense[(u-1)] = 0
            data_dense[i] = x_dense


        data_dense = np.array(data_dense)
        for i in range(len(data_target)):
            if data_target[i] == 0:
                data_target[i] =-1
        t = np.array([data_target])
        data_dense_wt = np.concatenate((data_dense,t.T), axis = 1)
        return data_dense_wt
    
    #readt train data
    #t,d = parse_data("C:/Users/krish/Desktop/UoU/Fall 2017/Machine learning/HW5/data/speeches.train.liblinear")
    t,d = parse_data(datapath_train)
    train_data = np.array(d)
    train = np.array(d)
    train_target = np.array(t)

    #Read test data
    #t,d = parse_data("C:/Users/krish/Desktop/UoU/Fall 2017/Machine learning/HW5/data/speeches.test.liblinear")
    t,d = parse_data(datapath_test)
    test = np.array(d)
    test_target = np.array(t)
    
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

    #Adding bias term to train data
    for i in range(len(train)):
        train[i][max_index+1] = 1

    #Adding bias term to test data
    for i in range(len(test)):
        test[i][max_index+1] = 1

    #Adding bias term to test data
    #for i in range(len(eval_data)):
        #eval_data[i][max_index+1] = 1
        
    train_dense_wt = sparse_dense(train, train_target)
    test_dense_wt = sparse_dense(test, test_target)
    #eval_dense_wt = sparse_dense(eval_data,eval_t)
    
    cv5 = cv_splits1(train_dense_wt,cvfolds)

    
    # stime = time.time()
    if alg == "svm":
        print("SVM")
        learn_rate = [10,1,0.1,0.01,0.001,0.0001]
        c_tradeoff = [10,1,0.1,0.01,0.001,0.0001]
        #c, avg_accuracy = cv_svm1(cv5[0],cv5[1],cv5[2],cv5[3],cv5[4],learn_rate,c_tradeoff,10)
        c, avg_accuracy = cv_svm(cv5,learn_rate,c_tradeoff,10,cvfolds)
        print("Best learning rate: ",c[0]) 
        print("Best C: ",c[1])
        print("Cross validation accuracy of best combination of hyperparameters: ",avg_accuracy ) 
        #Train the algorithm and find out the epoch having the greatest accuracy

        weights,a_f,a,updates,final_weights = svm(train_dense_wt,test_dense_wt, c[0], c[1], 20)
        #weights,a_f,a,updates,final_weights = svm(train_dense_wt,test_dense_wt, 0.1, 10, 1)

        print("Updates: ",updates) 

        accuracy_on_train_allfeatures = accuracy1(final_weights,train_dense_wt)
        print("Accuracy on training set: ",accuracy_on_train_allfeatures)

        accuracy_on_test_allfeatures = accuracy1(final_weights,test_dense_wt)
        print("Accuracy on test set: ",accuracy_on_test_allfeatures)
  
    elif alg == "perceptron":
        print("PERCEPTRON")
        learn_rate = [1,0.1,0.01]
        lean_rate_final, avg_accuracy = cv_averaged_perceptron(cv5,learn_rate,10,cvfolds)
        print("Best learning rate: ",lean_rate_final) 
        print("Cross validation accuracy of hyperparameters: ",avg_accuracy )

        
        #Train the algorithm and find out the epoch having the greatest accuracy

        # weights,a_f,a,updates,final_weights = averaged_perceptron(train_dense_wt,test_dense_wt, lean_rate_final, 20)
        weights,a_f,a,updates,final_weights = averaged_perceptron(train_dense_wt,test_dense_wt, lean_rate_final, 20)

        print("Updates: ",updates)
        accuracy_on_train_allfeatures = accuracy_p(final_weights,train_dense_wt)
        print("Accuracy on training set: ",accuracy_on_train_allfeatures)

        accuracy_on_test_allfeatures = accuracy_p(final_weights,test_dense_wt)
        print("Accuracy on test set: ",accuracy_on_test_allfeatures)
