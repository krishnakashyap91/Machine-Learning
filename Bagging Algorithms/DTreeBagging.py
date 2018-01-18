
# coding: utf-8

# In[ ]:

##Importing Libraries
import numpy as np
import math
import sys
import random
import matplotlib.pyplot as plt
import collections
import time
from numpy import random
st = time.time()

def parse_data(filepath):
    target = []
    features = {}
    data =[]
    with open(filepath,'r') as file:
        for line in file:
            #print(line) each line is a string
            row = line.split(' ')
            target.append(float(row[0]))
            if len(row)>2:
                row = row[1:]
                for feature in row:
                    temp = feature.split(':')
                    #print(temp)
                    features[float(temp[0])]=1
            elif len(row)<2:
                features={}            
            data.append(features) 
            features = {}
    return target,data

def featurize(fname):
    t,d = parse_data(fname)
    train = np.array(d)
    train_target = np.array(t)
    max_index = 67692
    train_dense = ([0 for g in range(len(train))])
    for i in range(len(train)):
        x_dense = ([0 for g in range(max_index)])
        #print(x_dense)
        for u in range(1,max_index+1):
            if u in train[i].keys():
                x_dense[(u-1)] = 1.0
            else:
                x_dense[(u-1)] = 0.0
        train_dense[i] = x_dense

    for i in range(len(train_target)):
        if train_target[i] == -1:
            train_target[i] = 0
    train_d = zip(train_dense,train_target)
    train_ = list(train_d)
    train_ = [list(i)for i in train_]
    return train_




# In[ ]:

import math
import numpy as np

f_train = sys.argv[1]
f_test = sys.argv[2]
feature_text = sys.argv[3]



def main():

    #The file contains information about the features
    #Format -> Feature name:Values it can take (seperated by commas)
    with open(feature_text) as f:
        data_info = f.readlines()

    #Transform the data
    dat_train = np.array(featurize(f_train))
    print('feat train')
    
    #data_train = data_train[0:100]
    #print(data_train)
    data_test = np.array(featurize(f_test))
    data_test = np.vstack((dat_train, data_test))
    print('feat test')
    
    #data_test = data_test[0:50]

    #Create feature nodes
    features = feature_info(data_info)
    
    
    out = list()
    for i in range(1000):
        data_train = dat_train[random.choice(range(len(dat_train)) ,100, replace=False)]
        out.append(test(data_train, data_test, features, 3))
        print(i)
        
    final = list(map(list, zip(*out)))
    final_out = list()
    
    for obs in final:
        final_out.append(max(set(obs), key=obs.count))
            
    print(len(final_out))
    labels = [dat[-1] for dat in data_test]
    
    
    print('Accuracy:', accuracy(final_out, labels))
    file_out = list()
    
    for i, row in enumerate(final):
        file_out.append([list(row), labels[i]])

    thefile = open('dtreedata_out.txt', 'w')
    for item in file_out:
        thefile.write("%s\n" % item)
    

def accuracy(out1, out2):
    out1 = np.array(out1)
    out2 = np.array(out2)
    return np.mean(out1 == out2)

def test(data_train, data_test, features, depth):
    r = ID3(data_train, features, 0, depth)
    out = list()
    for d in data_test:
        out.append(walk_down(r, d[0]))
    return out

def walk_down(node, point):
    if node.name == "leaf":
        return node.value
    
    if node.branches:
        for b in node.branches:
            if b.value == point[node.index]:
                return walk_down(b.child, point)
    #return 0

def ID3(data_samples, attributes, depth, depth_limit):

    if not attributes or depth == depth_limit:
        leaf = Node()
        leaf.set_is_leaf(most_common(data_samples))
        return leaf
    
    if(all_same(data_samples)):
        label = data_samples[0][1]
        root = Node()
        root.set_is_leaf(label)
        return root

    base_entropy = calculate_base_entropy(data_samples)
    root = best_attribute(data_samples, base_entropy, attributes)
    root = Node(root.name, root.possible_vals, root.index)
    depth += 1
    
    for val in root.possible_vals:
        b = Branch(val)
        root.add_branch(b)
        subset = subset_val(data_samples, root.index, val)
        if not subset:
            leaf = Node()
            leaf.set_is_leaf(most_common(data_samples))
            b.set_child(leaf)
        else:
            attributes = remove_attribute(attributes, root)
            b.set_child(ID3(subset, attributes, depth, depth_limit))
    return root


def best_attribute(data, base_entropy, attributes):
    max_ig = 0
    max_a = None
    for a in attributes:
        tmp_ig = base_entropy - expected_entropy(data, a)
        tmp_a = a
        if tmp_ig >= max_ig:
            max_ig = tmp_ig
            max_a = a
    return max_a

# Returns the most common label
def most_common(data_samples):
    p = sum(d[1] for d in data_samples)
    if p >= len(data_samples)/2:
        return 1
    else:
        return 0

def expected_entropy(data, attribute):
    data_total = float(len(data))
    e_entropy = 0.0
    for val in attribute.possible_vals:
        entropy, total = calculate_entropy(data, attribute, val)
        e_entropy += (total/data_total) * entropy
    return e_entropy
    
def calculate_entropy(data, attribute, value):
    subset = subset_val(data, attribute.index, value)
    if not subset:
        return [0, 0]

    return [calculate_base_entropy(subset), len(subset)]

def calculate_base_entropy(data):
    l = len(data)
    p = sum(d[1] for d in data)

    if not p or l == p:
        return 0

    n = l - p

    probP = p/l
    probN = n/l

    return (-probP*math.log(probP)) - (probN*math.log(probN))

# Returns a subset of the data where the given feature has the given value
def subset_val(data, feature_index, val):
    return [d for d in data if d[0][feature_index] == val]

# Returns true if all the labels are the same in the sample data
def all_same(data_samples):
    label = data_samples[0][1]
    for s in data_samples:
        if s[1] != label:
            return False
    return True

 
def remove_attribute(attributes, attribute):
    new_attributes = []
    for a in attributes:
        if a.name != attribute.name:
            new_attributes.append(a)
    return new_attributes

def feature_info(data):
    data_inf = []
    for i, d in enumerate(data):
        d = d.split(":")
        r = list(map(int, d[1].rstrip().split(",")))
        a = Node(d[0], r, i)
        data_inf.append(a)

    return data_inf

class Node:
    
    def __init__(self, name ="leaf", vals=None, index=-1):
        self.name = name
        self.possible_vals = vals
        self.index = index
        self.branches = []

    def set_is_leaf(self, value):
        self.leaf = True
        self.value = value

    def add_branch(self, b):
        self.branches.append(b)

class Branch:
    
    def __init__(self, value):
        self.value = value

    def set_child(self, child):
        self.child = child

if __name__ == '__main__':
    main()

