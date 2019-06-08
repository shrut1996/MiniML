import numpy as np


class LogisticRegression:
    def __init__(self, regularisation='L2', num_steps=300, learning_rate=0.1, lamb=0.1, initial_weights=None):
        self.regularisation = regularisation
        self.numsteps = num_steps
        self.learning_rate = learning_rate
        self.lamb = lamb
        self.initial_weights = None
        self.weights=initial_weights
        self.epsilon = 1e-7

    def train(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w = []
        m = X.shape[0]

        for i in np.unique(y):
            y_copy = np.where(y == i, 1, 0)
            w = np.ones(X.shape[1])

            for _ in range(self.numsteps):
                output = X.dot(w)
                errors = y_copy - self._sigmoid(output)
                grad = errors.dot(X) / m

                if self.regularisation == 'L2':
                    grad = grad + self.lamb*w / m
                elif self.regularisation == 'L1':
                    a = []
                    for j in range(len(w)):
                        if w[j] > 0:
                            a = np.append(a, [1])
                        else:
                            a = np.append(a, [-1])

                    grad = grad + self.lamb*a
                w += self.learning_rate * grad
            self.w.append((w, i))

        return self

    def predict(self, X):
        return [self._predict_one(i) for i in np.insert(X, 0, 1, axis=1)]

    def _predict_one(self, x):
        return max((x.dot(w), c) for w, c in self.w)[1]

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
