import numpy as np

from logistic_regression import LogisticRegression


class Stacking:
    def __init__(self, *arg):
        self.models = [i[0] for i in arg[0]]
        self.freq = [i[1] for i in arg[0]]
        self.trained_models = list()
        self.stacked_model = None

    def stacking_predict(self, models, row):
        stacked_row = list()
        for i in models:
            prediction = i.predict(models[i], row)
            stacked_row.append(prediction)

        stacked_row.append(row[-1])

        return row[0:len(row) - 1] + stacked_row

    def train(self, X, y):
        for clf, freq in zip(self.models, self.freq):
            for i in range(freq):
                self.trained_models.append(clf.train(X, y))

        stacked_preds = []
        for model in self.trained_models:
            preds = model.predict(X)
            stacked_preds.append(preds)

        stacked_preds = list(map(list, zip(*stacked_preds)))
        stacked_preds = np.asarray(stacked_preds)

        self.stacked_model = LogisticRegression()
        self.stacked_model.train(stacked_preds, y)

        return self

    def predict(self, X):
        stacked_preds = []

        for model in self.trained_models:
            preds = model.predict(X)
            stacked_preds.append(preds)

        stacked_preds = list(map(list, zip(*stacked_preds)))
        stacked_preds = np.asarray(stacked_preds)

        predictions = self.stacked_model.predict(stacked_preds)
        return predictions
