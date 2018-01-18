#!/bin/bash
# Compile the code
#javac DecisionTrees/*.java
# Run for each question
echo "========================"
echo "Question 3(3)"
echo "========================"
#java -cp DecisionTrees MainClass Dataset/updated_train.txt Dataset/updated_train.txt 
# python DecisionTrees.py Dataset/updated_train.txt Dataset/updated_train.txt
# python3.5 DecisionTrees.py Dataset/updated_train.txt Dataset/updated_train.txt
python3.6 Perceptron/hw2.py Dataset/phishing.train Dataset/phishing.test Dataset/phishing.dev Dataset/CV/training00.data Dataset/CV/training01.data Dataset/CV/training02.data Dataset/CV/training03.data Dataset/CV/training04.data
echo "========================"
echo "========================"
