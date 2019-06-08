# MiniML

This project was an aim to create a minimalistic machine learning library solely for the purpose of 
classification tasks involving classical algorithms such as logistic regression along with 
ensemble methods like stacking.

We've performed the general data pre-processing followed by the classification
in ‘run.py’. Furthermore, we decided to test the classifiers using the iris and titanic 
datasets and the accuracies achieved were:



Classifier | Titanic  | Iris  |
------- |:----:| ---:|
Naive Bayes |77.65|100|
Logistic Regression |79.32|100
Decision Tree|79.32|100
Random Forest|82.68|100
Stacking|79.32|100
AdaBoost|73.18|73.33


##### Hyperparameters used:


Classifier | Hyperparameters  |
-------- |:--------------:|
Naive Bayes|type=Gaussian, prior=None
Logistic Regression|num_steps=5000, regularisation=’L2’, learning_rate=0.1, lambda=0.1
Decision Tree|max_depth=5, split_val_metric='mean', min_info_gain=-200, split_node_criterion='gini'
Random Forest|n_trees=10, sample_size=0.8, max_features=6, max_depth=5, split_val_metric='mean', split_node_criterion='gini', bootstrap=True
Stacking|Two decision trees and a single naive bayes, while a logistic regression model for the meta learner (All with above params)
AdaBoost|n_trees=100, learning_rate=1, max_depth=3


