import numpy as np
import pandas as pd


class NaiveBayesClassifier:
    def __init__(self, type ='Gaussian', class_priors=False, priors_=None):
        self.type = type
        self.class_priors = class_priors
        self.priors_= priors_
        return

    def train(self, X, y):
        if self.type == 'Gaussian':
            self.labels_ = np.unique(y)
            self.mean_ = pd.DataFrame(X).groupby(y).agg(np.mean).values
            self.stds_ = (pd.DataFrame(X).groupby(y).agg(np.std) + 10 ** -5).values

            if self.class_priors == False:
                self.priors_ = (pd.DataFrame(X).groupby(y).agg(lambda x: len(x)) / len(X)).iloc[:,1]

            return self

        elif self.type == 'Multinomial':
            self.labels_ = np.unique(y)

            if self.class_priors == False:
                self.priors_ = (pd.DataFrame(X).groupby(y).agg(lambda x: len(x)) / len(X)).iloc[:, 0]

            params = []
            for i in range(X.shape[1]):
                params.append(pd.crosstab(y, X.iloc[:, i], normalize='index'))

            self.params_ = params
            return self

    def predict(self, X):
        if self.type == 'Gaussian':
            if X.ndim == 1:
                X = X.reshape(1, -1)
            nlab = len(self.labels_)
            score = np.ones((nlab, X.shape[0]))

            # cycle for labels
            for i in range(nlab):
                stds = np.repeat(self.stds_[i, :].reshape(1, -1), X.shape[0], axis=0)
                means = np.repeat(self.mean_[i, :].reshape(1, -1), X.shape[0], axis=0)
                loglik = -0.5 * np.log(2 * np.pi) - np.log(stds) - (X - means) ** 2 / (2 * stds ** 2)
                score[i, :] = loglik.sum(axis=1) + np.log(self.priors_.iloc[i])
            pred = np.argmax(score, axis=0)

            return self.labels_.take(pred)

        elif self.type == 'Multinomial':
            if X.ndim == 1:
                X = X.reshape(1, -1)

            # score is an array nr.labels * nr.obs
            score = np.ones((len(self.labels_), X.shape[0]))

            for i in range(len(self.labels_)):
                loglik = X.copy()
                for j in range(X.shape[1]):
                    loglik.iloc[:, j] = loglik.iloc[:, j].replace(self.params_[j].iloc[i, :].index, self.params_[j].iloc[i, :].values)
                score[i, :] = loglik.sum(axis=1) + np.log(self.priors_.iloc[i])

            pred = np.argmax(score, axis=0)
            return self.labels_.take(pred)
