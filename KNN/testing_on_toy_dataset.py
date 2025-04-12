import numpy as np
from KNN import KNeighborsClassifier




data = np.array([
    [150, 7.0, 1, 'Apple'],
    [120, 6.5, 0, 'Banana'],
    [180, 7.5, 2, 'Orange'],
    [155, 7.2, 1, 'Apple'],
    [110, 6.0, 0, 'Banana'],
    [190, 7.8, 2, 'Orange'],
    [145, 7.1, 1, 'Apple'],
    [115, 6.3, 0, 'Banana']
])


### Encoding the Data

X= data[:,:-1].astype(np.float32)
y = data[:,-1]
label_encoding = {'Apple':0,'Banana':1,'Orange':2}
for i  in range(len(y)):
    y[i] = label_encoding[y[i]]

y = y.astype(np.int32)




X_test = np.array([
    [118, 6.2, 0],  # Expected: Banana
    [160, 7.3, 1],  # Expected: Apple
    [185, 7.7, 2]   # Expected: Orange
])

my_knn = KNeighborsClassifier(3,2)
my_knn.fit(X,y)
predictions = my_knn.predict(X_test)

labels = ['Apple','Banana','Orange']
for pred in predictions:
    print(labels[pred])

