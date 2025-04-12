import numpy as np
class KNeighborsClassifier:


    def __init__(self,k=3,distance_metric =2,isWeighted=False):
        self.n_neighbors = k
        self.X_train = None
        self.y_train = None
        self.distance_metric = distance_metric
        self.inverse_distance_weighted_classification = isWeighted
    
    def _normDistance(self,x,y,l):
        n = len(x)
        sum =0
    
        for i in range(n):
            sum += np.abs((x[i] - y[i]))**l

        return np.sqrt(sum)


    def _k_smallest_indices(self,arr,k):


        arr_index_map = []

        n = len(arr)
        for i in range(n):
            arr_index_map.append([i,arr[i]])

        smallest_indices = []

        for i in range(k):
            for j in range(0,n-i-1):
                if arr_index_map[j][1]< arr_index_map[j+1][1]:
                    arr_index_map[j] ,arr_index_map[j+1] = arr_index_map[j+1] , arr_index_map[j]

            smallest_indices.append(arr_index_map[n-i-1][0])
        return smallest_indices


    def fit(self,X,y):
        self.X_train = X
        self.y_train =  y

    def predict(self,X_test):
        predict_labels = []
        for x_test in X_test: 
            predict_labels.append(self.predict_one(x_test))
        return np.array(predict_labels)
    
    def predict_one(self, x):
         
    
        distances = np.array([self._normDistance(x,x_train,self.distance_metric) for x_train in self.X_train])
        xnn = self._k_smallest_indices(distances,self.n_neighbors)
        ynn = self.y_train[xnn]  
        if self.inverse_distance_weighted_classification ==False:
           labels,counts = np.unique(ynn,return_counts=True)
           predict_label = labels[np.argmax(counts)] 
           return predict_label
        else:
            k_smallest_distances =distances[xnn]

            # Each ynn is given weightage inversely corresponding to it's distance
            
            weights = np.array([1/(distance + 1e-6) for distance in k_smallest_distances])
        

            classes = np.unique(self.y_train)
            contributions = np.zeros(np.size(classes))
      

            for i in range(self.n_neighbors):
                contributions[ynn[i]] += weights[i]


            
            #The class with most contribution
            return np.argmax(contributions)
    

    

    
        

        







