# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 19:20:54 2018

@author: kathi
"""
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import neighbors, datasets
from sklearn import preprocessing
import pickle
import collections


def svm(X_train, X_test, y_train):
    
    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)
    
    y_svm = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test,y_svm)*100
    
    #print ("Accuracy is ",accuracy)
    filename = 'svm.sav'
    pickle.dump(clf, open(filename, 'wb'))
    
    return accuracy


def knn(X_train, X_test, y_train):
    h=.02
    n_neighbors = 15
    X_scaled = preprocessing.scale(X)
    Z=[]
    
    for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X_train, y_train)
        
        Z_new=clf.predict(X_test)
        Z.append(Z_new )
    
        print(accuracy_score(y_test, Z))
        
        Z_max=np.amax(Z)
        filename = 'knn'+weights+'.sav'
        pickle.dump(clf, open(filename, 'wb'))
        return Z_max
    
def decTree(X_train, X_test, y_train):
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=15)
    clf_entropy.fit(X_train, y_train)


    #test
    y_pred_en = clf_entropy.predict(X_test)
    y_pred_en

    #accuracy
    #print ("Accuracy is ", accuracy_score(y_test,y_pred_en)*100)
    accuracy=accuracy_score(y_test,y_pred_en)
    
    filename = 'decTree.sav'
    pickle.dump(clf_entropy, open(filename, 'wb'))
    
    return accuracy

def randForest(X_train, X_test, y_train):
    ran_forest = RandomForestClassifier(n_estimators=10, criterion = "gini")

    ran_forest.fit(X_train, y_train)

    y_pred_ran = ran_forest.predict(X_test)
    accuracy =  accuracy_score(y_test, y_pred_ran)
    filename = 'randForest.sav'
    pickle.dump(ran_forest, open(filename, 'wb'))
    #print("Accuracy is ", accuracy_score(y_test, y_pred_ran)*100)
    return accuracy
                           
features = np.load("features.txt")
labels = np.load("label.txt")
print(features.shape)
print(labels.shape)

def load_model_and_predict_class(model,test_input):
    
    loaded_model = pickle.load(open(filename, 'rb'))
    y=loaded_model.predict(test_input)
    return y 
    
    
    
labels_not_hot = []
for element in labels:
    for i in range(0, len(element)):
        if element[i] == 1: 
            labels_not_hot.append(i)

labels_array = np.array(labels_not_hot)
print(labels_array)
print(labels_array.shape)

X = features
y = labels_array
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, random_state = 100)

acc_svm=svm(X_train, X_test, y_train)
print (acc_svm)
acc_knn=knn(X_train, X_test, y_train)
print (acc_knn)
acc_decTree=decTree(X_train, X_test, y_train)
print (acc_decTree)
acc_randForest=randForest(X_train, X_test, y_train)

votes=[]
model_files=['svm.sav','knnuniform.sav','knndistance.sav','decTree.sav','randForest.sav']

for model in model_files:
    new_vote=load_model_and_predict_class(model,test_input)
    votes.append(new_vote)

votes_array=np.array(votes)
unique, counts = numpy.unique(a, return_counts=True) # in counts steht anzahl wie oft diese zahl vorkommt(=index )

max_class= np.argmax(counts)