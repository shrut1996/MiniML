import numpy as np
from copy import deepcopy

from boosting_decision_tree import BoostingDecisionTree


class AdaBoostClassifier(object):
    def __init__(self, n_trees, learning_rate):
        self.base_estimator = BoostingDecisionTree(max_depth=3)
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.estimators_ = list()
        self.estimator_weights_ = np.zeros(self.n_trees)
        self.estimator_errors_ = np.ones(self.n_trees)

    def train(self, X, y):
        self.n_samples = X.shape[0]
        # There is hidden trouble for classes, here the classes will be sorted.
        # So in boost we have to ensure that the predict results have the same classes sort
        self.classes_ = np.array(sorted(list(set(y))))
        self.n_classes_ = len(self.classes_)
        for iboost in range(self.n_trees):
            if iboost == 0:
                sample_weight = np.ones(self.n_samples) / self.n_samples

            sample_weight, estimator_weight, estimator_error = self.boost(X, y, sample_weight)

            # early stop
            if estimator_error == None:
                break

            # append error and weight
            self.estimator_errors_[iboost] = estimator_error
            self.estimator_weights_[iboost] = estimator_weight

            if estimator_error <= 0:
                break

        return self

    def boost(self, X, y, sample_weight):
        estimator = deepcopy(self.base_estimator)

        estimator.train(X, y, sample_weights=sample_weight)

        y_pred = estimator.predict(X)
        incorrect = y_pred != y
        estimator_error = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)

        # if worse than random guess, stop boosting
        if estimator_error >= 1 - 1 / self.n_classes_:
            return None, None, None

        # update estimator_weight
        estimator_weight = self.learning_rate * np.log((1 - estimator_error) / estimator_error) + np.log(
            self.n_classes_ - 1)

        if estimator_weight <= 0:
            return None, None, None

        # update sample weight
        sample_weight *= np.exp(estimator_weight * incorrect)

        sample_weight_sum = np.sum(sample_weight, axis=0)
        if sample_weight_sum <= 0:
            return None, None, None

        # normalize sample weight
        sample_weight /= sample_weight_sum

        # append the estimator
        self.estimators_.append(estimator)

        return sample_weight, estimator_weight, estimator_error

    def predict(self, X):
        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]

        pred = sum((estimator.predict(X) == classes).T * w for estimator, w in zip(self.estimators_, self.estimator_weights_))

        pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            pred = pred.sum(axis=1)
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def predict_proba(self, X):
        proba = sum(estimator.predict_proba(X) * w for estimator, w in zip(self.estimators_, self.estimator_weights_))
        proba /= self.estimator_weights_.sum()
        proba = np.exp((1. / (self.n_classes_ - 1)) * proba)

        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        return proba
