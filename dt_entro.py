#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 12:33:22 2018

@author: negin
"""

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
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)


#test
y_pred_en = clf_entropy.predict(X_test)
y_pred_en

#accuracy
print ("Accuracy is ", accuracy_score(y_test,y_pred_en)*100)

#predict new data






