import numpy as np 
import pandas as pd 


def divide_data(x_data, y_data, fkey, fval):
    x_right = pd.DataFrame([], columns=x_data.columns)
    x_left = pd.DataFrame([], columns=x_data.columns)
    y_right = np.array([])
    y_left = np.array([])

    for ix in range(x_data.shape[0]):
        # Retrieve the current value for the fkey column
        try:
            val = x_data[fkey].loc[ix]
        except:
            print (x_data[fkey])
            val = x_data[fkey].loc[ix]
        if val > fval:
            # pass the row to right
            x_right = x_right.append(x_data.loc[ix])
            y_right = np.append(y_right, y_data[ix])
        else:
            # pass the row to left
            x_left = x_left.append(x_data.loc[ix])
            y_left = np.append(y_left, y_data[ix])
    
    # return the divided datasets
    return x_left, x_right, y_left, y_right


def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])

    return entropy


def gini(target_col):
    elements, counts = np.unique(target_col,return_counts=True)
    impurity = 1

    for i in range(len(elements)):
        prob_of_lbl = counts[i] / np.sum(counts)
        impurity -= prob_of_lbl**2

    return impurity


def information_gain(xdata, ydata, fkey, fval,split_node_criterion):
    left, right, y_left, y_right = divide_data(xdata, ydata, fkey, fval)

    if left.shape[0] == 0 or right.shape[0] == 0:
        return -10000
    if split_node_criterion == 'gini':
        return gini(ydata) - (gini(y_left)*float(y_left.shape[0]/float(y_left.shape[0]+y_right.shape[0])) + gini(y_right)*float(y_left.shape[0]/float(y_left.shape[0]+y_right.shape[0])))
    else:
        return entropy(ydata) - (entropy(y_left)*float(y_left.shape[0]/float(y_left.shape[0]+y_right.shape[0])) + entropy(y_right)*float(y_left.shape[0]/float(y_left.shape[0]+y_right.shape[0])))


class DecisionTree:
    def __init__(self, depth=0, max_depth=5, split_val_metric='mean', min_info_gain=-200, split_node_criterion='gini'):
        self.left = None
        self.right = None
        self.fkey = None
        self.fval = None
        self.max_depth = max_depth
        self.depth = depth
        self.target = None
        self.split_val_metric = split_val_metric
        self.min_info_gain = min_info_gain
        self.split_node_criterion = split_node_criterion
    
    def train(self, X_train, Y_train):
        print (self.depth, '-'*10)
        # Get the best possible feature and division value
        features = X_train.columns
        gains = []
        for fx in features:
            if self.split_val_metric=='mean':
                gains.append(information_gain(X_train, Y_train, fx, X_train[fx].mean(), self.split_node_criterion))
            else :
                gains.append(information_gain(X_train, Y_train, fx, X_train[fx].median(), self.split_node_criterion))

        # store the best feature (using min information gain)
        self.fkey = features[np.argmax(gains)]
        if self.split_val_metric=='mean':
            self.fval = X_train[self.fkey].mean()
        else :
            self.fval = X_train[self.fkey].median() 

        Y_train = Y_train.astype(int)
        if gains[np.argmax(gains)] < self.min_info_gain:
            counts = np.bincount(Y_train)
            self.target = np.argmax(counts)
            return self
        # divide the dataset
        data_left, data_right, y_left, y_right = divide_data(X_train, Y_train, self.fkey, self.fval)
        data_left = data_left.reset_index(drop=True)
        data_right = data_right.reset_index(drop=True)

        # Check the shapes
        if data_left.shape[0] == 0 or data_right.shape[0] == 0:
            counts = np.bincount(Y_train)
            self.target = np.argmax(counts)
            return self
        
        if self.depth >= self.max_depth:
            counts = np.bincount(Y_train)
            self.target = np.argmax(counts)
            return self
        
        # branch to right
        self.right = DecisionTree(depth=self.depth+1, max_depth=self.max_depth)
        self.right.train(data_right, y_right)
        # branch to left
        self.left = DecisionTree(depth=self.depth+1, max_depth=self.max_depth)
        self.left.train(data_left, y_left)

        counts = np.bincount(Y_train)
        self.target = np.argmax(counts)
        
        return self
    
    def predict_single(self, test):
        if test[self.fkey] > self.fval:
            # go right
            if self.right is None:
                return self.target

            return self.right.predict_single(test)
        else:
            # go left
            if self.left is None:
                return self.target

            return self.left.predict_single(test)

    def predict(self, X_test):
        pred = []

        for i in range(X_test.shape[0]):
            pred_i = self.predict_single(X_test.loc[i])
            pred = np.append(pred,pred_i)

        return pred
