import pandas as pd
import numpy as np
import random

from random import randrange
from decision_tree import DecisionTree


class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, split_val_metric='mean', min_info_gain_split=-200,
                 split_node_criterion='gini', max_features=3, bootstrap=True, n_cores=1, sample_size=0.8):
        self.trees = list()
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.max_depth = max_depth
        self.split_val_metric = split_val_metric
        self.min_info_gain = min_info_gain_split
        self.split_node_criterion = split_node_criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_cores = n_cores
        self.feature_indexes = list()

    def bagging_predict(self, trees, row):
        predictions = [tree.predict_single(row[index]) for tree, index in zip(trees, self.feature_indexes)]
        return max(set(predictions), key=predictions.count)

    def predict(self, X):
        predictions = [self.bagging_predict(self.trees, X.loc[i]) for i in range(X.shape[0])]
        return predictions

    def train(self, X, y):
        for i in range(self.n_trees):
            X_train, y_train = self.subsample(X, y, self.sample_size)
            X_train = self.drop_features(X_train, self.max_features)
            tree = DecisionTree(max_depth=self.max_depth, split_val_metric=self.split_val_metric,
                                split_node_criterion=self.split_node_criterion, min_info_gain=self.min_info_gain)
            tree.train(X_train, y_train)
            self.trees.append(tree)
        return self

    def drop_features(self, X, max_features):
        indices = random.sample(range(len(X.columns)), max_features)
        self.feature_indexes.append(indices)
        return X[X.columns[indices]]

    def subsample(self, X, y, ratio):
        X_train, y_train = list(), list()
        n_sample = round(len(X) * ratio)

        while len(X_train) < n_sample:
            index = randrange(len(X))
            X_train.append(X.iloc[index])
            y_train.append(y.iloc[index])

        X_train = pd.DataFrame(X_train)
        X_train = X_train.reset_index(drop=True)

        return pd.DataFrame(X_train), np.asarray(y_train)

