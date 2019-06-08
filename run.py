import csv
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from naive_bayes import NaiveBayesClassifier
from logistic_regression import LogisticRegression
from decision_tree import DecisionTree

from random_forest import RandomForest
from stacking import Stacking

from boosting_decision_tree import BoostingDecisionTree
from adaboost import AdaBoostClassifier


############## IRIS ###################

iris_data = csv.reader(open('data/iris.data', 'r'))

iris_data = list(iris_data)
for i in iris_data:
    if i[4] == 'Iris-setosa':
        i[4] = '0'
    elif i[4] == 'Iris-versicolor':
        i[4] = '1'
    else:
        i[4] = '2'

iris_data = pd.DataFrame(iris_data)
for i in iris_data.columns:
    iris_data[i] = iris_data[i].astype(float)

X1 = iris_data.iloc[:, :4]
y1 = iris_data.iloc[:, 4]

############## TITANIC ###################

titanic_data = csv.reader(open('data/titanic.csv', 'r'))
titanic_data = pd.DataFrame(titanic_data)

new_header = titanic_data.iloc[0]
titanic_data = titanic_data[1:]
titanic_data.columns = new_header
titanic_data.drop(['PassengerId', 'Name', 'Embarked', 'Ticket', 'Cabin'], axis=1, inplace=True)
titanic_data['Sex'] = titanic_data['Sex'].apply(lambda x: 1 if x=='female' else 0)
titanic_data['Age'].replace('', 0, inplace=True)
titanic_data['Age'] = titanic_data['Age'].astype(float)

for i in titanic_data.columns:
    titanic_data[i] = titanic_data[i].astype(float)

X2 = titanic_data.iloc[:, 1:]
y2 = titanic_data.iloc[:, 0]

########################## Classification ##################################


X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


clf = NaiveBayesClassifier(type='Gaussian')

####### Convert X_train and X_test into an np array for Logistic Regression ######
# clf = LogisticRegression(num_steps=5000, regularisation='L2')

# clf1 = DecisionTree(max_depth=5, split_val_metric='mean', split_node_criterion='gini')
# clf = RandomForest(n_trees=10, sample_size=0.8, max_features=6,
#                    max_depth=5, split_val_metric='mean', split_node_criterion='gini')

##### Using two decision trees and a single naive bayes here while logistic regression is by default the meta-learner
# clf = Stacking([(clf, 1), (clf1, 2)])

# clf1 = BoostingDecisionTree(max_depth=5, split_val_metric='mean', split_node_criterion='gini')
# clf = AdaBoostClassifier(n_trees=100, learning_rate=1)

#### For Logistic Regression
# clf.train(np.asarray(X_train.values), y_train)
# y_pred = clf.predict(np.asarray(X_test.values))

clf.train(X_train, y_train)
y_pred = clf.predict(X_test)


print((y_test == y_pred).mean())
