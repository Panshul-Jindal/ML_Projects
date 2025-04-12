# On Toy Dataset


####### Predictions ###########
Beer
Whiskey
Wine
```
if Alchohol Content > 8.50:
  if Alchohol Content > 25.75:
    Predict: Whiskey 
  else:  # Alchohol Content <= 25.75
    Predict: Wine 
else:  # Alchohol Content <= 8.50
  Predict: Beer 
```
# On Iris Dataset (max_depth =40)

Accuracy : = 0.7333333333333333

if petal length > 2.45:
  if petal length > 4.95:
    if petal width > 1.75:
      Predict: Iris virginica 
    else:  # petal width <= 1.75
      if sepal width > 2.45:
        Predict: Iris versicolor 
      else:  # sepal width <= 2.45
        Predict: Iris virginica 
  else:  # petal length <= 4.95
    if sepal length > 4.95:
      Predict: Iris versicolor 
    else:  # sepal length <= 4.95
      if sepal width > 2.45:
        Predict: Iris virginica 
      else:  # sepal width <= 2.45
        Predict: Iris versicolor 
else:  # petal length <= 2.45
  Predict: Iris setosa 
