# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

#read data

#data slicing
#standardize values
X = Y = 0
#spilt int train and test set
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

#decision tree training
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


#test
y_pred = clf_gini.predict(X_test)
y_pred

#accuracy
print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)

#predict new data






