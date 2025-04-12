import numpy as np

import numpy as np

class DecisionTree:





    def __init__(self,max_depth=5):
        self.max_depth = max_depth
        self.root = None
    

    class Node:
        def __init__(self,feature_index=None,threshold=None,left=None,right=None,value=None):
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value =value
    
        def is_leaf(self):
            return self.left == None and self.right == None
    

    def fit(self,X,y):
        self.root= self._build(X,y,0,self.max_depth)
    

    def _giniImpurity(self,labels):
        n = np.size(labels)
        _,counts = np.unique(labels,return_counts=True)
        probabilites = counts/n

        sum = 0
        for p_i in probabilites:
            sum += p_i **2

        return 1 -sum

    def _bestSplitFinder(self,X,y):
        features_min_impurities = []
        features_corresponding_thresholds = []


        for col in range(X.shape[1]):
           vals = X[:,col]
           unique_vals = np.unique(vals)
           n = np.size(unique_vals)
           thresholds=[]
           if(np.size(unique_vals)>1):
              thresholds = [(unique_vals[i] + unique_vals[i+1])/2 for i in range(n-1)]
           else:
              thresholds  = unique_vals



           res ={}
           for threshold in thresholds:

              yes =[]
              no =[]
              for index,val in enumerate(vals):
                 if val > threshold:
                    yes.append(index)
                 else:
                    no.append(index)



              yes_len = np.size(yes)
              labels_yes = y[yes]
              impurity_yes =   self._giniImpurity(labels_yes)




              no_len = np.size(no)
              labels_no = y[no]
              impurity_no = self._giniImpurity(labels_no)




              avg_impurity = (yes_len * impurity_yes  + no_len * impurity_no)/(yes_len + no_len)


              res[threshold] = avg_impurity

           res_thresh_with_min_impurity = min(res,key=res.get)
           res_min_impurity = res[res_thresh_with_min_impurity]

           features_min_impurities.append(res_min_impurity)
           features_corresponding_thresholds.append(res_thresh_with_min_impurity)

        res_feature = np.argmin(features_min_impurities)
        res_threshold = features_corresponding_thresholds[res_feature]

        return res_feature,res_threshold,features_min_impurities[res_feature]

    def _build(self,features,labels,cur_depth,max_depth =100):
        if len(np.unique(labels) )==1  or cur_depth == max_depth :
            leaf_value = np.bincount(labels).argmax()
            return self.Node(value = leaf_value)


        best_feature, best_threshold,_ = self._bestSplitFinder(features,labels)

    


        vals = features[:,best_feature]
        yes =[]
        no =[]
        for index,val in enumerate(vals):

           if val > best_threshold:
              yes.append(index)
           else:
              no.append(index)


        features_left = np.array(features[yes])
        features_right = np.array(features[no])


        labels_left = np.array(labels[yes])
        labels_right = np.array(labels[no])







        left_tree = self._build(features_left,labels_left,cur_depth+1,max_depth)
        right_tree = self._build(features_right,labels_right,cur_depth+1,max_depth)

        return self.Node(best_feature,best_threshold,left_tree,right_tree)
    

    def predict(self,X):
        return [self.predict_one(x,self.root) for x in X]


    def predict_one(self,x,root):
       
        
        # depth =0
        while not root.is_leaf():
            # print(f'depth {depth}  Feature {root.feature_index}   Threshold {root.threshold}')
            # depth = depth +1
            if x[root.feature_index] > root.threshold:
                root = root.left
            else:
                root = root.right

        return root.value
    
    def printDecsionTree(self,features,labels):
        self._printDecisionTree(self.root,features,labels,0)
        
    def _printDecisionTree(self,node, features,labels,depth=0):
        indent = "  " * depth  # two spaces per depth level for clarity

        if node.is_leaf():
            print(f"{indent}Predict: {labels[node.value]} ")
            return

        feature = features[node.feature_index]
        threshold = node.threshold

        print(f"{indent}if {feature} > {threshold:.2f}:")
        self._printDecisionTree(node.left,features,labels, depth + 1)

        print(f"{indent}else:  # {feature} <= {threshold:.2f}")
        self._printDecisionTree(node.right,features,labels, depth + 1)

  

    
    

    
    