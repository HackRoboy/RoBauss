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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import collections
import librosa
import AudioSetLoad
from letRoboySpeak import roboy_talk
import scipy.io.wavfile as sciwav
import sounddevice as sd


def svm(X_train, X_test, y_train, y_test):

    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)

    y_svm = clf.predict(X_test)

    accuracy = accuracy_score(y_test,y_svm)

    #print ("Accuracy is ",accuracy)
    filename = 'svm.sav'
    pickle.dump(clf, open(filename, 'wb'))

    return accuracy


def knn(X_train, X_test, y_train, y_test):
    h=.02
    n_neighbors = 15
    X_scaled = preprocessing.scale(X)
    Z=[]

    for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X_train, y_train)

        Z_new=clf.predict(X_test)
        Z.append(Z_new)

        print(accuracy_score(y_test, Z_new))

        Z_max=np.amax(Z)
        filename = 'knn'+weights+'.sav'
        print filename
        pickle.dump(clf, open(filename, 'wb'))
    return Z_max

def decTree(X_train, X_test, y_train, y_test):
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

def randForest(X_train, X_test, y_train, y_test):
    ran_forest = RandomForestClassifier(n_estimators=10, criterion = "gini")

    ran_forest.fit(X_train, y_train)

    y_pred_ran = ran_forest.predict(X_test)
    accuracy =  accuracy_score(y_test, y_pred_ran)
    filename = 'randForest.sav'
    pickle.dump(ran_forest, open(filename, 'wb'))
    #print("Accuracy is ", accuracy_score(y_test, y_pred_ran)*100)
    return accuracy

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def load_model_and_predict_class(model, test_input):

    loaded_model = pickle.load(open(model, 'rb'))
    y = loaded_model.predict([test_input])
    return y

def test_audiofile(model, filename):
    mfccs, chroma, mel, contrast,tonnetz = extract_feature(filename)
    ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
    features = ext_features
    return load_model_and_predict_class(model, features)

if __name__ == "__main__":
    trained = True
    if trained == False:
        features = np.load("feature_all.npy")
        labels = np.load("hot_labels_all.npy")
        print(features.shape)
        print(labels.shape)

        labels_not_hot = np.load("labels_all.npy")
        """for element in labels:
            for i in range(0, len(element)):
                if element[i] == 1:
                    labels_not_hot.append(i)"""

        labels_array = labels_not_hot
        print(labels_array)
        print(labels_array.shape)

        X = features
        y = labels_array
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, random_state = 100)

        #acc_svm=svm(X_train, X_test, y_train, y_test)
        #print (acc_svm)
        acc_knn=knn(X_train, X_test, y_train, y_test)
        print (acc_knn)
        acc_decTree=decTree(X_train, X_test, y_train, y_test)
        print (acc_decTree)
        acc_randForest=randForest(X_train, X_test, y_train, y_test)

    votes=[]
    #model_files=['svm.sav','knnuniform.sav', 'knndistance.sav', 'decTree.sav', 'randForest.sav']
    model_files = ['knnuniform.sav', 'knndistance.sav', 'decTree.sav', 'randForest.sav']

    #fn, ytid, classes = AudioSetLoad.dl_random_file()

    fn = "recorded.wav"
    classes = "recorded stuff"
    for model in model_files:
        new_vote=test_audiofile(model,fn)
        votes.append(new_vote)

    votes_array=np.array(votes)
    counts= np.bincount(votes_array.flatten()) # in counts steht anzahl wie oft diese zahl vorkommt(=index )
    

    max_class= np.argmax(counts)
    print max_class
    print classes, "(", fn, ")"
    roboy_talk(max_class)
    rate, data = sciwav.read(fn)
    sd.play(data, rate, blocking=True)
