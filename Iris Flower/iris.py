import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression

#Reads the data
data = pd.read_csv('iris.data', sep=',', names=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Class'])
print('Our Data looks like this:')
print(data)

input("Press Enter to continue...")

#Creates plots to visualize the values
print('Visualizing data...')

setosa = (data['Class'] == 'Iris-setosa')
versicolor = (data['Class'] == 'Iris-versicolor')
virginica = (data['Class'] == 'Iris-virginica')

fig, ax = plt.subplots(2,2)

sepal_length_values = list([data.loc[setosa, 'Sepal Length'],data.loc[versicolor, 'Sepal Length'],data.loc[virginica, 'Sepal Length']])
sepal_width_values = list([data.loc[setosa, 'Sepal Width'],data.loc[versicolor, 'Sepal Width'],data.loc[virginica, 'Sepal Width']])
petal_length_values = list([data.loc[setosa, 'Petal Length'],data.loc[versicolor, 'Petal Length'],data.loc[virginica, 'Petal Length']])
petal_width_values = list([data.loc[setosa, 'Petal Width'],data.loc[versicolor, 'Petal Width'],data.loc[virginica, 'Petal Width']])

ax[0,0].boxplot(sepal_length_values)
ax[0,1].boxplot(sepal_width_values)
ax[1,0].boxplot(petal_length_values)
ax[1,1].boxplot(petal_width_values)
#Sets the labels of the axis
for i in [0,1]:
    for j in [0,1]:
        ax[i,j].set_xlabel('Flower')
        xticklabels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        ax[i,j].set_xticklabels(xticklabels)
        ax[i,j].yaxis.grid(True)

ax[0,0].set_ylabel('Sepal Length')
ax[0,1].set_ylabel('Sepal Width')
ax[1,0].set_ylabel('Petal Length')
ax[1,1].set_ylabel('Petal Width')

plt.tight_layout()
plt.show()

input("Press Enter to continue...")
#Adjusting data for training

#Shuffles data to ensure training is not biased
data = data.sample(frac=1).reset_index(drop=True)

#Creates training and testing set and converts to NumPy array
val = 50
training_set = data.iloc[0:val, 0:4].to_numpy()
training_set_class = data.iloc[0:val, 4].to_numpy()
testing_set = data.iloc[val:150, 0:4].to_numpy()
testing_set_class = data.iloc[val:150, 4].to_numpy()

#Training data for machine learning algorithm
#Machine Learning Algorithm
clf = LogisticRegression(random_state=0).fit(training_set, training_set_class)
hypothesis = clf.predict(testing_set)

#Visually compare hypothesis to actual values
print('HYPOTHESIS:')
print(hypothesis)
print('-----------')
print('ACTUAL:')
print(testing_set_class)

#Calculate and display testing accuracy
def cal_accuracy(test, hypo):
    num = len(test)
    print('Number of tests: ' + str(num))
    
    correct = 0
    for i in range(0, num):
        if test[i] == hypo[i]:
            correct = correct + 1;
    
    return (correct / num) * 100

accuracy_percent = cal_accuracy(testing_set_class, hypothesis)
print('Accuracy of machine learning algorithm: ' + str(accuracy_percent))

input("Press Enter to continue...")

#Showing how number of training set affects accuracy
print('Changing number of training sets (2 to 125) and comparing accuracy of our hypothesis')

#Calculate all the accuracies and place into list
accuracies = []
for i in range(2, 126):
    training_set = data.iloc[0:i, 0:4].to_numpy()
    training_set_class = data.iloc[0:i, 4].to_numpy()
    testing_set = data.iloc[i:150, 0:4].to_numpy()
    testing_set_class = data.iloc[i:150, 4].to_numpy()

    clf = LogisticRegression(random_state=0, max_iter = 500).fit(training_set, training_set_class)
    hypothesis = clf.predict(testing_set)
    accuracies.append(cal_accuracy(testing_set_class, hypothesis))

print(accuracies)

print("Displaying accuracies")

#Display all the accuracies
plt.plot(range(2,126), accuracies)
plt.ylabel(r'Accuracy (%)')
plt.xlabel(r'Num. of training sets')
plt.show()