"""
Created on Sun Sep 10 20:53:00 2017

@author: krish
"""

#Assignment 1

#Import data as a 2d array
import numpy as np
import math
import sys
def generate_data(path):
    test = open(path, "r",encoding="utf-8-sig")
    lines = test.read().split("\n")
    n =1

    for line_test in lines:
        label = [line_test[0]]
        #print(label)
        name = [line_test[2:]]
        #print(name)
        l = name + label
        data = np.array([l])
        #print(data)    
        if n ==1:
            testd = np.array([l])
            n = n+1
            #print(traind)
        if n>1:
            testd = np.concatenate((testd, data), axis=0)
            #print(traind)
    len(testd)


    #Extracting Features
    #Add columns to the array to store features
    testd = np.insert(testd, 0, 0, axis = 1)
    testd = np.insert(testd, 1, 0, axis = 1)
    testd = np.insert(testd, 2, 0, axis = 1)
    testd = np.insert(testd, 3, 0, axis = 1)
    testd = np.insert(testd, 4, 0, axis = 1)
    testd = np.insert(testd, 5, 0, axis = 1)
    #testd = np.insert(testd, 6, 0, axis = 1)
    #testd = np.insert(testd, 7, 0, axis = 1)
    #testd = np.insert(testd, 8, 0, axis = 1)
    #testd = np.insert(testd, 9, 0, axis = 1)


    testd
#1. Is their first name longer than their last name?
#2. Do they have a middle name?
#3. Does their first name start and end with the same letter? (ie "Ada")
#4. Does their first name come alphabetically before their last name? (ie "Dan Klein" because "d" comes before "k")
#5. Is the second letter of their first name a vowel (a,e,i,o,u)?
#6. Is the number of letters in their last name even?
#7) Is the first letter of the first name and last name the same?
#8) Is the name longer than 15 letters?
#9) Is the length of the first name equa to the lebght of the last name?
#10)Does the name have more than 3 vowels in it? 

    alphabet = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","é"] 
    vowel = ["a","e","i","o","u"]    
    
    for i in range(len(testd)):
    #1. Is their first name longer than their last name?
        lenght_first = 0
        lenght_last = 0
        for character in testd[i][1]:
            lenght_first = lenght_first + 1
            if " " in character:
                break
    #Reversing the name to check lenght of last name
        for character in testd[i][1][::-1]:
            lenght_last = lenght_last + 1
            if " " in character:
                break
        if lenght_first > lenght_last:      
            testd[i][0] = 1
        else:
            testd[i][0] = 0
        
    #2. Do they have a middle name?        
        no_of_spaces = 0
        for character in testd[i][6]:
            if " " in character:
                no_of_spaces = no_of_spaces + 1
        if no_of_spaces == 1:
            testd[i][1] = 0
        elif no_of_spaces > 1:
            testd[i][1] = 1
        
    #8) Is the name longer than 15 letters?
       # no_of_spaces = 0
        #for character in testd[i][1]:
         #   if " " in character:
          #      no_of_spaces = no_of_spaces + 1
          #  name_lenght = len(testd[i][1]) - no_of_spaces
        #if name_lenght > 15 :
         #   testd[i][7] = 1
       # else:
           # testd[i][7] = 0
        
        #9) Is the length of the first name equal to the lebght of the last name?
        #lenght_first = 0
        #lenght_last = 0
        #for character in testd[i][1]:
          #  lenght_first = lenght_first + 1
           # if " " in character:
              # break
            #Reversing the name to check lenght of last name
        #for character in testd[i][1][::-1]:
             # lenght_last = lenght_last + 1
             # if " " in character:
                #     break
        #if lenght_first == lenght_last:
           # testd[i][8] = 1
        #else:
            #testd[i][8] = 0
        
        #10)Does the name have more than 3 vowels in it? 
        #no_of_vowels = 0
        #for character in testd[i][1]:
           # if character in vowel:
           #    no_of_vowels = no_of_vowels + 1
        #if no_of_vowels > 3 :
         #   testd[i][9] = 1
       # else:
          #  testd[i][9] = 0
        

    i = 0  
    for n in testd:
#3. Does their first name start and end with the same letter? (ie "Ada")
        names_test = n[6].split()
        first_letter = names_test[0][0] #First letter of first name
        last_letter = names_test[0][-1] #Last letter of first name
        if first_letter.upper() == last_letter.upper():
            testd[i][2] = 1
        else:
            testd[i][2] = 0
        
#4. Does their first name come alphabetically before their last name? (ie "Dan Klein" because "d" comes before "k")
        names_test = n[6].split()
        first_letter_f = names_test[0][0] #First letter of first name
        first_letter_l = names_test[::-1][0][0] #First letter of last name
        if alphabet.index(first_letter_f.lower()) < alphabet.index(first_letter_l.lower()):
            testd[i][3] = 1
        else:
            testd[i][3] = 0           

#5. Is the second letter of their first name a vowel (a,e,i,o,u)?
        names_test = n[6].split()
        if len(names_test[0]) > 1:
            second_letter = names_test[0][1] #Second letter of first name
            if second_letter in  vowel:
                testd[i][4] = 1
            else:
                testd[i][4] = 0
        else:
            testd[i][4] = 0

#6. Is the number of letters in their last name even?
        names_test = n[6].split()
        no_of_letters = len(names_test[::-1][0])
        #print(no_of_letters)
        if no_of_letters % 2 == 0:
            testd[i][5] = 1
        else:
            testd[i][5] = 0

    #7) Is the first letter of the first name and last name the same? 
        #names_test = n[10].split()
        #first_letter_f = names_test[0][0] #First letter of first name
        #first_letter_l = names_test[::-1][0][0] #First letter of last name
        #if first_letter_f == first_letter_l :
         #      testd[i][6] = 1
        #else:
            #testd[i][6] = 0
        
        i = i+1
    testd = np.delete(testd,0, axis = 0)
    testd = np.delete(testd,110, axis = 0)
    testd = np.delete(testd,109, axis = 0)

#Removing name  from data
    testd = np.delete(testd,6, axis = 1)

    len(testd)
    testd = testd.tolist()
    return testd
##################################################


############################################################################
# Class used for learning and building the Decision Tree using the given Training Set
class DecisionTree():
    tree = {}

    def learn(self, training_set, attributes, target):
        self.tree = build_tree(training_set, attributes, target)


# Class Node which will be used while classify a test-instance using the tree which was built earlier
class Node():
    value = ""
    children = []

    def __init__(self, val, dictionary):
        self.value = val
        if (isinstance(dictionary, dict)):
            self.children = dictionary.keys()


# Majority Function which tells which class has more entries in given data-set
def majorClass(attributes, data, target):

    freq = {}
    index = attributes.index(target)

    for tuple in data:
        if not tuple[index] in freq:
            freq[tuple[index]] = 1 
        else:
            freq[tuple[index]] += 1

    max = 0
    major = ""

    for key in freq.keys():
        if freq[key]>max:
            max = freq[key]
            major = key

    return major


# Calculates the entropy of the data given the target attribute
def entropy(attributes, data, targetAttr):

    freq = {}
    dataEntropy = 0.0

    i = 0
    for entry in attributes:
        if (targetAttr == entry):
            break
        i = i + 1

    i = i - 1

    for entry in data:
        if not entry[i] in freq:
            freq[entry[i]] = 1.0
        else:
            freq[entry[i]]  += 1.0

    for freq in freq.values():
        dataEntropy += (-freq/len(data)) * math.log(freq/len(data), 2) 
        
    return dataEntropy


# Calculates the information gain (reduction in entropy) in the data when a particular attribute is chosen for splitting the data.
def info_gain(attributes, data, attr, targetAttr):

    freq = {}
    subsetEntropy = 0.0
    i = attributes.index(attr)

    for entry in data:
        if not entry[i] in freq:
            freq[entry[i]] = 1.0
        else:
            freq[entry[i]]  += 1.0

    for val in freq.keys():
        valProb        = freq[val] / sum(freq.values())
        dataSubset     = [entry for entry in data if entry[i] == val]
        subsetEntropy += valProb * entropy(attributes, dataSubset, targetAttr)

    return (entropy(attributes, data, targetAttr) - subsetEntropy)


# This function chooses the attribute among the remaining attributes which has the maximum information gain.
def attr_choose(data, attributes, target):

    best = attributes[0]
    maxGain = 0;

    for attr in attributes:
        newGain = info_gain(attributes, data, attr, target) 
        if newGain>maxGain:
            maxGain = newGain
            best = attr

    return best


# This function will get unique values for that particular attribute from the given data
def get_values(data, attributes, attr):

    index = attributes.index(attr)
    values = []

    for entry in data:
        if entry[index] not in values:
            values.append(entry[index])

    return values

# This function will get all the rows of the data where the chosen "best" attribute has a value "val"
def get_data(data, attributes, best, val):

    new_data = [[]]
    index = attributes.index(best)

    for entry in data:
        if (entry[index] == val):
            newEntry = []
            for i in range(0,len(entry)):
                if(i != index):
                    newEntry.append(entry[i])
            new_data.append(newEntry)

    new_data.remove([])    
    return new_data


# This function is used to build the decision tree using the given data, attributes and the target attributes. It returns the decision tree in the end.
def build_tree(data, attributes, target):

    data = data[:]
    vals = [record[-1] for record in data]
    default = majorClass(attributes, data, target)

    if not data or (len(attributes) - 1) <= 0:
        return default
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        best = attr_choose(data, attributes, target)
        tree = {best:{}}
    
        for val in get_values(data, attributes, best):
            new_data = get_data(data, attributes, best, val)
            newAttr = attributes[:]
            newAttr.remove(best)
            subtree = build_tree(new_data, newAttr, target)
            tree[best][val] = subtree
    
    return tree

def run_decision_tree(train_set):

    data = [tuple(t) for t in train_set] 
    
    attributes = ['A1','A2','A3','A4', 'A5','A6','A7']
    target = attributes[-1]

    #print("Number of records: %d" % len(data))
    
    tree = DecisionTree()
    tree.learn( data, attributes, target )
    #print(tree.tree)
    return tree.tree
    
    
def accuracy(predicted,actual):
    acc_sum = 0
    for i in range(len(predicted)):
        if predicted[i] == actual[i]:
            acc_sum +=1
    return(acc_sum/len(predicted))


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    #print("fistStr : "+firstStr)
    secondDict = inputTree[firstStr]
    #print("secondDict : " + str(secondDict))
    featIndex = featLabels.index(firstStr)
    #print("featIndex : " + str(featIndex))
    key = testVec[featIndex]
    #print("key : " + str(key))
    valueOfFeat = secondDict[key]
    #print("valueOfFeat : " + str(valueOfFeat))
    if isinstance(valueOfFeat, dict):
        #print("is instance: "+str(valueOfFeat))
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        #print("is Not instance: " + valueOfFeat)
        classLabel = valueOfFeat
    #print(classLabel)
    return classLabel



train = sys.argv[1]
test = sys.argv[2]

def main():
    
    traind = generate_data(train)
    testd = generate_data(test)
    
    tree = run_decision_tree(traind)
     

    output = []
    for dat in testd:
        output.append(classify(tree, ['A1','A2','A3','A4','A5','A6'], dat[0:6]))
        #print(len(output))
    print("Accuracy: " + str(accuracy(output,[dat[-1] for dat in testd])))
 
      
if __name__ == "__main__":
    main()
