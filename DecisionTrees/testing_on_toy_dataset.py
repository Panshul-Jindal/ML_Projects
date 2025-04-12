from DecisionTree import DecisionTree
import numpy as np

data = np.array([
    [12.0, 1.5, 1, 'Wine'],
    [5.0, 2.0, 0, 'Beer'],
    [40.0, 0.0, 1, 'Whiskey'],
    [13.5, 1.2, 1, 'Wine'],
    [4.5, 1.8, 0, 'Beer'],
    [38.0, 0.1, 1, 'Whiskey'],
    [11.5, 1.7, 1, 'Wine'],
    [5.5, 2.3, 0, 'Beer']
])


### Encoding
X= data[:,:-1].astype(np.float32)
y = data[:,-1]

label_encoding = {'Beer':0,'Wine':1,'Whiskey':2}
label_decoding = {0:'Beer',1:'Wine',2:'Whiskey'}
for i  in range(len(y)):
    y[i] = label_encoding[y[i]]

y = y.astype(np.int32)


test_data = np.array([
    [6.0, 2.1, 0],   # Expected: Beer
    [39.0, 0.05, 1], # Expected: Whiskey
    [13.0, 1.3, 1]   # Expected: Wine
])

myDecisionTree = DecisionTree(40)
myDecisionTree.fit(X,y)
predictions  = myDecisionTree.predict(test_data)

labels= ["Beer","Wine","Whiskey"]


print("####### Predictions ###########")
for prediction in predictions:
    print(labels[prediction])


features = ['Alchohol Content','Sugar','Color']





myDecisionTree.printDecsionTree(features,labels)