from KNN import KNeighborsClassifier
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





def accuracy(y,y_hat):
    return np.sum(y==y_hat)/len(y)


print("Non Weighted KNN")
k_values = np.arange(1,20,1)
for k in k_values:
    my_knn = KNeighborsClassifier(k,2)
    my_knn.fit(X_train,y_train)
    y_pred = my_knn.predict(X_test)
    print(f'K :{k} , Accuracy : {accuracy(y_test,y_pred)}')



print("Weighted KNN")
k_values = np.arange(1,20,1)
for k in k_values:
    my_knn = KNeighborsClassifier(k,2,isWeighted=True)
    my_knn.fit(X_train,y_train)
    y_pred = my_knn.predict(X_test)
    print(f'K :{k} , Accuracy : {accuracy(y_test,y_pred)}')



print("Observations")

print("1. Max Accuracy of 83.33 is obtained in case of distance metric as Euclidian in both case of Weighted as well non Weighted KNN (Most Surprisingly at K=1")
print("2. For any given value of K , Weighted KNN performes better than NoN_Weighted KNN")