from DecisionTree import DecisionTree
import numpy as np
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
## Dividing the dataset into train and test

n = len(X)
n_train = int(0.80 * n)

X_train = X[0:n_train,:]
y_train  = y[0:n_train]

X_test = X[n_train:,:]
y_test =y[n_train:]


myDecisionTree = DecisionTree(40)
myDecisionTree.fit(X_train,y_train)
predictions  = myDecisionTree.predict(X_test)



def accuracy(y,y_hat):
    return np.sum(y==y_hat)/len(y)

print(accuracy(y_test,predictions))


features = ['sepal length','sepal width', 'petal length', 'petal width']
labels= ['Iris setosa', 'Iris versicolor', 'Iris virginica']
myDecisionTree.printDecsionTree(features,labels)