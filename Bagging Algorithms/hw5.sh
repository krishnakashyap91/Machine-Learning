#!/bin/bash
# Compile the code
#javac DecisionTrees/*.java
# Run for each question
echo "========================"
echo "Question 2.3(1) SVM Results:"
echo "========================"
python -W ignore svm_final.py Dataset/speeches.train.liblinear Dataset/speeches.test.liblinear Dataset/CVSplits/training00.data Dataset/CVSplits/training01.data Dataset/CVSplits/training02.data Dataset/CVSplits/training03.data Dataset/CVSplits/training04.data


echo "========================"
echo
echo

echo "========================"
echo "Question 2.3(2) Logistic Regression Results: "
echo "========================"
python -W ignore logistic_regression_final.py Dataset/speeches.train.liblinear Dataset/speeches.test.liblinear Dataset/CVSplits/training00.data Dataset/CVSplits/training01.data Dataset/CVSplits/training02.data Dataset/CVSplits/training03.data Dataset/CVSplits/training04.data


echo "========================"
echo
echo

echo "========================"
echo "Question 2.3(3) Naive Bayes Results:"
echo "========================"
python -W ignore naive_bayes_final.py Dataset/speeches.train.liblinear Dataset/speeches.test.liblinear Dataset/CVSplits/training00.data Dataset/CVSplits/training01.data Dataset/CVSplits/training02.data Dataset/CVSplits/training03.data Dataset/CVSplits/training04.data


echo "========================"
echo
echo

echo "========================"
echo "Question 2.3(4) Bagged Forest Results:"
echo "========================"
python -W ignore DTreeBagging.py Dataset/speeches.train.liblinear Dataset/speeches.test.liblinear features.txt
 
python -W ignore Decisiontreeaccuracy.py bagTree_out.txt


echo "========================"
echo
echo

echo "========================"
echo "Question 2.3(5) SVM over Tress"
echo "========================"
python -W ignore svm_over_trees_final.py bagTree_out.txt


echo "========================"
echo
echo

echo "========================"
echo "Question 2.3(6) Logistic Regression over Tress"
echo "========================"
python -W ignore logistic_over_trees_final.py bagTree_out.txt


echo "========================"
echo
echo

