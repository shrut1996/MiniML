import numpy as np 
import pandas as pd 


def divide_data(x_data, y_data, fkey, fval, sample_weights):
    x_right = pd.DataFrame([], columns=x_data.columns)
    x_left = pd.DataFrame([], columns=x_data.columns)
    y_right = np.array([])
    y_left = np.array([])
    weights_left = np.array([])
    weights_right = np.array([])
    
    for ix in range(x_data.shape[0]):
        # Retrieve the current value for the fkey column
        try:
            val = x_data[fkey].loc[ix]
        except:
            print (x_data[fkey])
            val = x_data[fkey].loc[ix]
        # Check where the row needs to go
        if val > fval:
            # pass the row to right
            x_right = x_right.append(x_data.loc[ix])
            y_right = np.append(y_right, y_data[ix])
            weights_right = np.append(weights_right,sample_weights[ix])
        else:
            # pass the row to left
            x_left = x_left.append(x_data.loc[ix])
            y_left = np.append(y_left, y_data[ix])
            weights_left = np.append(weights_left,sample_weights[ix])
    
    # return the divided datasets
    return x_left, x_right, y_left, y_right, weights_left, weights_right


def entropy(target_col, sample_weights):
    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-target_col[i]*sample_weights[i] / (np.sum(counts) * np.sum(sample_weights)))*np.log2(target_col[i]*sample_weights[i] / (np.sum(counts) * np.sum(sample_weights))) for i in range(len(target_col))])
    return entropy


def gini(target_col, sample_weights):
    elements,counts = np.unique(target_col,return_counts = True)
    impurity = 1
    for i in range(len(target_col)):
        prob_of_lbl = target_col[i]*sample_weights[i] / (np.sum(counts) * np.sum(sample_weights))
        impurity -= prob_of_lbl**2
    return impurity


def information_gain(xdata, ydata, fkey, fval,split_node_criterion,sample_weights):
    left, right, y_left, y_right, weights_left, weights_right = divide_data(xdata, ydata, fkey, fval, sample_weights)
    
    if left.shape[0] == 0 or right.shape[0] == 0:
        return -10000
    if split_node_criterion == 'gini':
        return gini(ydata, sample_weights) - (gini(y_left, weights_left)*float(y_left.shape[0]/float(y_left.shape[0]+y_right.shape[0])) + gini(y_right, weights_right)*float(y_left.shape[0]/float(y_left.shape[0]+y_right.shape[0])))
    else :
        return entropy(ydata, sample_weights) - (entropy(y_left, weights_left)*float(y_left.shape[0]/float(y_left.shape[0]+y_right.shape[0])) + entropy(y_right, weights_right)*float(y_left.shape[0]/float(y_left.shape[0]+y_right.shape[0])))


class BoostingDecisionTree:
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
        self.probability = None
    
    def proba_single(self, test):
        if test[self.fkey] > self.fval:
            # go right
            if self.right is None:
                return self.probability
            return self.right.proba_single(test)
        else:
            # go left
            if self.left is None:
                return self.probability
            return self.left.proba_single(test)

    def predict_proba(self, X):
        pred = []
        for i in range(X.shape[0]):
            pred_i = self.proba_single(X.loc[i])
            pred.append(pred_i)
        pred=np.array(pred)
        print(pred.shape)
        return pred

    def train(self, X_train, Y_train, sample_weights = None,nclasses=None):
        print (self.depth, '-'*10)
        if nclasses is None:
            self.nclasses = np.unique(Y_train)
        else :
            self.nclasses = nclasses
        # Get the best possible feature and division value
        features = X_train.columns
        gains = []
        for fx in features:
            if self.split_val_metric=='mean':
                gains.append(information_gain(X_train, Y_train, fx, X_train[fx].mean(), self.split_node_criterion,sample_weights))
            else :
                gains.append(information_gain(X_train, Y_train, fx, X_train[fx].median(), self.split_node_criterion,sample_weights))

        # store the best feature (using min information gain)
        self.fkey = features[np.argmax(gains)]
        # print(self.fkey)
        if self.split_val_metric=='mean':
            self.fval = X_train[self.fkey].mean()
        else :
            self.fval = X_train[self.fkey].median() 

        Y_train = Y_train.astype(int)
        if gains[np.argmax(gains)] < self.min_info_gain:
            counts = np.bincount(Y_train)
            self.target = np.argmax(counts)
            prob = []
            for i in range(self.nclasses.shape[0]):
                cz=0
                for j in range(Y_train.shape[0]):
                    if Y_train[j]==self.nclasses[i]:
                        cz+=1
                prob.append(cz/Y_train.shape[0])        
            self.probability = np.array(prob)
            return
        # divide the dataset
        data_left, data_right, y_left, y_right, weights_left, weights_right = divide_data(X_train, Y_train, self.fkey, self.fval,sample_weights)
        data_left = data_left.reset_index(drop=True)
        data_right = data_right.reset_index(drop=True)

        # Check the shapes
        if data_left.shape[0] == 0 or data_right.shape[0] == 0:
            counts = np.bincount(Y_train)
            self.target = np.argmax(counts)
            prob = []
            for i in range(self.nclasses.shape[0]):
                cz=0
                for j in range(Y_train.shape[0]):
                    if Y_train[j] == self.nclasses[i]:
                        cz += 1
                prob.append(cz/Y_train.shape[0])        
            self.probability = np.array(prob)
            return
        
        if self.depth >= self.max_depth:
            counts = np.bincount(Y_train)
            self.target = np.argmax(counts)
            prob = []
            for i in range(self.nclasses.shape[0]):
                cz=0
                for j in range(Y_train.shape[0]):
                    if Y_train[j] == self.nclasses[i]:
                        cz += 1
                prob.append(cz/Y_train.shape[0])        
            self.probability = np.array(prob)
            return
        
        # branch to right
        self.right = BoostingDecisionTree(depth=self.depth+1, max_depth=self.max_depth)
        self.right.train(data_right, y_right,weights_right,self.nclasses)
        # branch to left
        self.left = BoostingDecisionTree(depth=self.depth+1, max_depth=self.max_depth)
        self.left.train(data_left, y_left, weights_left,self.nclasses)

        counts = np.bincount(Y_train)
        self.target = np.argmax(counts)
        prob = []
        for i in range(self.nclasses.shape[0]):
            cz=0
            for j in range(Y_train.shape[0]):
                if Y_train[j]==self.nclasses[i]:
                    cz+=1
            prob.append(cz/Y_train.shape[0])        
        self.probability = np.array(prob)
        return
    
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
