#!/bin/bash
# Run for each question
echo "========================"
echo "Question 1(c)"
echo "========================"

python DecisionTrees.py Dataset/updated_train.txt Dataset/updated_train.txt

echo "========================"
echo
echo

echo "========================"
echo "Question 1(d)"
echo "========================"
python DecisionTrees.py Dataset/updated_train.txt Dataset/updated_test.txt 
echo "========================"
echo
echo

echo "========================"
echo "Question 2"
echo "========================"
python DecisionTrees.py Dataset/updated_train.txt Dataset/updated_test.txt Dataset/Updated_CVSplits 1 2 3 4 5 10 15 echo "========================"
echo
echo