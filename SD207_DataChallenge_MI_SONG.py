
# coding: utf-8

# # Challenge SD207 - 2017
# *<p>Author: Pengfei MI, Rui SONG</p>*
# *<p>Date: 06/06/2017</p>*

# In[ ]:

# Basic libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import platform
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
from time import time
from scipy.stats import mode

# Librosa related: audio feature extraction
import librosa
import librosa.display

# Sklearn related: data preprocessing and classifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.base import clone


# In[ ]:

# Define some usefull functions
def load_sound_file(file_name):
    X, sr = librosa.load(os.path.join(FILEROOT, file_name), sr=None)
    return X

def extract_feature(file_name, param): # Late fusion
    X, sample_rate = librosa.load(os.path.join(FILEROOT, file_name), sr=None)
    mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_fft=param['n_fft'], hop_length=param['hop_length'], n_mfcc=param['n_mfcc']).T
    #delta_mfcc = librosa.feature.delta(mfcc, width=5, order=1, trim=True)
    return mfcc

def parse_audio_files(file_names, file_labels, param):
    features, labels = np.empty((0,param['n_mfcc'])), np.empty(0)
    for fn, fl in zip(file_names, file_labels):
        ff = extract_feature(fn, param)
        features = np.vstack([features, ff])
        labels = np.append(labels, fl*np.ones(ff.shape[0]))
    return np.array(features), np.array(labels, dtype = np.int)

def cross_validation(clf, X, y, test_fold, param):
    y_pred, y_pred_sum, y_pred_prod = np.empty_like(y), np.empty_like(y), np.empty_like(y)
    n_folds = len(np.unique(test_fold))
    for i in range(n_folds):
        t0 = time()
        new_clf = clone(clf, safe=True)
        X_train = X[test_fold != i]
        X_test = X[test_fold == i]
        y_train = y[test_fold != i]
        y_test = y[test_fold == i]
        print "Launching fold #%d/%d, train set size: %d, test set size: %d" % (i+1, n_folds, len(X_train), len(X_test))
        clf_train(new_clf, X_train, y_train, param)
        test_pred, test_pred_sum, test_pred_prod = clf_predict(new_clf, X_test, param)
        y_pred[test_fold == i] = test_pred
        y_pred_sum[test_fold == i] = test_pred_sum
        y_pred_prod[test_fold == i] = test_pred_prod
        print "fold#%d done in %0.3fs, score: %0.3f." % (i+1, time()-t0, accuracy_score(y_test, test_pred))
    t0 = time()
    print "Retraining classifier with whole train set..."
    clf_train(clf, X, y, param)
    print "Done in %0.3fs." % (time() - t0)
    return y_pred, y_pred_sum, y_pred_prod

def clf_train(clf, files, file_labels, param):
    X_train, y_train= parse_audio_files(files, file_labels, param)
    clf.fit(X_train, y_train)
        
def predict_maj(clf, X_test):
    y_pred = np.empty(0)
    for x in X_test:
        x_mfccs = extract_feature(x)
        y_predicts = clf.predict(x_mfccs)
        y_pred = np.append(y_pred, mode(y_predicts).mode[0])
    return np.array(y_pred, dtype = np.int)

def predict_sum(clf, X_test):
    y_pred = np.empty(0)
    for x in X_test:
        x_mfccs = extract_feature(x)
        y_predicts = np.sum(clf.predict_proba(x_mfccs), axis=0)
        y_pred = np.append(y_pred, np.argmax(y_predicts))
    return np.array(y_pred, dtype = np.int)

def predict_prod(clf, X_test):
    y_pred = np.empty(0)
    for x in X_test:
        x_mfccs = extract_feature(x)
        y_predicts = np.prod(clf.predict_proba(x_mfccs), axis=0)
        y_pred = np.append(y_pred, np.argmax(y_predicts))
    return np.array(y_pred, dtype = np.int)

def clf_predict(clf, X_test, param):
    y_pred = np.empty(0)
    y_pred_sum = np.empty(0)
    y_pred_prod = np.empty(0)
    for x in X_test:
        x_mfccs = extract_feature(x, param)
        y_predicts = clf.predict(x_mfccs)
        y_predict_probas = clf.predict_proba(x_mfccs)
        y_pred = np.append(y_pred, mode(y_predicts).mode[0])
        y_pred_sum = np.append(y_pred_sum, np.argmax(np.sum(y_predict_probas, axis=0)))
        y_pred_prod = np.append(y_pred_prod, np.argmax(np.prod(y_predict_probas, axis=0)))
    return np.array(y_pred, dtype=np.int), np.array(y_pred_sum, dtype=np.int), np.array(y_pred_prod, dtype=np.int)


# In[ ]:

# Read data and preprocessing
print "Loading files..."
t0 = time()

# Define FILEROOT according to the platform
# My personal computer
if sys.platform == "darwin":
    FILEROOT = './'
# Node of Telecom
elif platform.node()[:4] == 'lame':
    FILEROOT = '/tmp/rsong/'
# The machines of Telecom
else:
    FILEROOT = '/tsi/plato/sons/sd207/'

# Load the cross validation folds
N_FOLDS = 3
train_files, train_scenes, test_fold = np.empty(0, dtype=str), np.empty(0), np.empty(0)
for i in range(N_FOLDS):
    files = pd.read_csv('train%s.txt' % str(i), sep='\s+', header=None)[0].values
    scenes = pd.read_csv('train%s.txt' % str(i), sep='\s+', header=None)[1].values
    print "Fold #%d: %d files from %d sources" % (i+1, len(files), len(np.unique([f.split('_')[0] for f in files])))
    train_files = np.append(train_files, files)
    train_scenes = np.append(train_scenes, scenes)
    test_fold = np.append(test_fold, i*np.ones_like(scenes))

scenes = np.unique(train_scenes)
n_scenes = len(scenes)
labels = pd.factorize(scenes, sort=True)[0]
n_labels = len(labels)
train_labels = pd.factorize(train_scenes, sort=True)[0]
test_files = pd.read_csv('test_files.txt', header=None)[0].values
test_labels = pd.read_csv('meta.txt', header=None)[0].values

print "%d scenes:" % n_scenes, scenes
print "Training set size: %d" % len(train_files)
print "Test set size: %d" % len(test_files)
print "Done in %0.3fs." % (time()-t0)


# In[ ]:

# Train classifier
print "Doing cross validation..."
t0 = time()

np.random.seed(42)

"""n_mfcc = 20
n_fft = 1024
hop_length = 512
n_features = n_mfcc
file_features = {}

clf = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.4)
y_pred, y_pred_sum, y_pred_prod = cross_validation(clf, train_files, train_labels, test_fold)
print "Done in %0.3fs." % (time()-t0)"""

def one_fit(param):
    n_mfcc = param['n_mfcc']
    n_fft = param['n_fft']
    hop_length = param['hop_length']
    sizes = param['hidden_layer_sizes']
    alpha = param['alpha']

    print "Launch fit #%d/%d with: n_mfcc=%d, n_fft=%d, hop_length=%d, nn_sizes=%s, alpha=%f" %                    (i, total_fits, n_mfcc, n_fft, hop_length, sizes, alpha)
    clf = MLPClassifier(hidden_layer_sizes=sizes, alpha=alpha)
    y_pred, y_pred_sum, y_pred_prod = cross_validation(clf, train_files, train_labels, test_fold, param)
    if (accuracy_score(train_labels,y_pred) > best_score):
        best_score = accuracy_score(train_labels,y_pred)
        best_param = p.copy()
    if (accuracy_score(train_labels,y_pred_sum) > best_score_sum):
        best_score_sum = accuracy_score(train_labels,y_pred_sum)
        best_param_sum = p.copy()
    if (accuracy_score(train_labels,y_pred_prod) > best_score_prod):
        best_score_prod = accuracy_score(train_labels,y_pred_prod)
        best_param_prod = p.copy()

param_grid = {'n_mfcc': [17],
              'n_fft': [512, 1024],
              'hop_length': [512],
              'hidden_layer_sizes': [(40), (40,80), (40,20), (64,64,64)],
              'alpha': [0.001, 0.01, 0.05, 0.1]}
params = list(ParameterGrid(param_grid))

best_score, best_score_sum, best_score_prod = 0, 0, 0
best_param, best_param_sum, best_param_prod = None, None, None

total_fits = len(params)
i = 1
pool = ThreadPool(multiprocessing.cpu_count()-1)
pool.map(one_fit, params)
pool.close()
pool.join()

print best_score, best_param
print best_score_sum, best_param_sum
print best_score_prod, best_param_prod
print "Done in %0.3fs." % (time()-t0)


# In[ ]:

# Print cross validation results
t0 = time()
print '-'*60
print "Score on validation test (vote by majority): %f" % accuracy_score(train_labels, y_pred)
print classification_report(train_labels, y_pred, target_names=scenes)
print "Confusion matrix:"
print confusion_matrix(train_labels, y_pred)

print '-'*60
print "Score on validation test (vote by proba sum): %f" % accuracy_score(train_labels, y_pred_sum )
print classification_report(train_labels, y_pred_sum, target_names=scenes)
print "Confusion matrix:"
print confusion_matrix(train_labels, y_pred_sum)

print '-'*60
print "Score on validation test (vote by proba product): %f" % accuracy_score(train_labels, y_pred_prod)
print classification_report(train_labels, y_pred_prod, target_names=scenes)
print "Confusion matrix:"
print confusion_matrix(train_labels, y_pred_prod)
print "Done in %0.3fs." % (time()-t0)


# In[ ]:

y_test_pred, y_test_pred_sum, y_test_pred_prod = clf_predict(clf, test_files)
np.savetxt('y_test_pred_mfcc_mlp.txt', y_test_pred, fmt='%d')
np.savetxt('y_test_pred_mfcc_mlp_sum.txt', y_test_pred_sum, fmt='%d')
np.savetxt('y_test_pred_mfcc_mlp_prod.txt', y_test_pred_prod, fmt='%d')

print "Score by maj: %f" % accuracy_score(test_labels, y_test_pred)
print "Score by sum: %f" % accuracy_score(test_labels, y_test_pred_sum)
print "Score by prod: %f" % accuracy_score(test_labels, y_test_pred_prod)

