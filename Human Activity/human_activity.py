import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
import pandas as pd

# sets the training and testing sets and their class
train_set = pd.read_csv('train.csv')
train_set_actual = train_set["Activity"]
train_set = train_set.drop(columns="Activity")

test_set = pd.read_csv('test.csv')
test_set_actual = test_set["Activity"]
test_set = test_set.drop(columns="Activity")

# Machine learning algorithm
clf = OneVsRestClassifier(LinearSVC()).fit(train_set, train_set_actual)

print("Accuracy of algorithm with test set: ")
print(str(clf.score(test_set, test_set_actual)*100) + "%")

