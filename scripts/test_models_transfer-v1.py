import json
from cmv_object import cmv_object
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import torch
import matplotlib.pyplot as plt
import random
from ml_model_tester import testmodel
from corps_ingestor import corps_ingest_subsampling

#Import Annotators
from title_tokenizer import title_tokenizer
from stemmer_annotator import stemmer_annotator
from pos_tag_annotator import pos_tag_annotator
from quantity_count_annotator import quantity_count_annotator
from stopword_annotator import stopword_annotator
from word_vector_annotator import word_vector_annotator

from nltk.classify import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import _pickle as cPickle
import nltk

#Import custom models
from CorrelationModels import CorrelationModels
from ClassificationModels import ClassificationModels

from nltk.corpus import stopwords
stopset = list(set(stopwords.words('english')))


freq_limit = 5
top_num = 10
read_limit = 1000000
chunk_size = 100
TRAIN_PERC = 0.8
VALID_PERC = 0.1
TEST_PERC = 1 - TRAIN_PERC - VALID_PERC
START_SELECT = 0
NUM_SELECT = 3746
NUM_CYCLES = 1
NUM_CYCLES_TEST = 1
CLASSIFIERS = [
   # svm.SVC(C=1.0, kernel='linear', cache_size=2000),
    tree.DecisionTreeClassifier(),
    neighbors.KNeighborsClassifier(n_neighbors=5),
    MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
    RandomForestClassifier(n_estimators=10),
    # svm.SVC(C=1.0, kernel='rbf', cache_size=2000),
    'NaiveBayes()'
]



np.random.seed(0)

accuracy0 = 0

all_word_set = set()

progress_count = 0

def remove_word_from_feats(testfeats, rem_word):
    _test_feats = []
    for feat, label in testfeats:
        _test_feats.append((dict([(word, True) for word in feat if word is not rem_word]),label))
    return

def select_words_from_feats(testfeats, selected_words):
    _test_feats = []
    for feat, label in testfeats:
        _test_feats.append((dict([(word, True) for word in feat if word in selected_words]),label))
    return _test_feats

def get_all_words_from_feats(testfeats):
    word_set = set()
    for feat, label in testfeats:
        for word in feat:
            word_set.add(word)
    return word_set

for _classifier in CLASSIFIERS:
    accuracies = []
    # Read the constraint file
    # constraintfile = open('/home/nimadaan/cmv/pythonwksp/src_v2/marginals/CORPS_'+str(_classifier).split('(')[0]+'.csv')
    constraintfile = open('/home/nimadaan/cmv/pythonwksp/src_v2/marginals/PDNUMPY_KMeans_silhoutte_clean_v2.csv')
    constraintreader = csv.reader(constraintfile)
    constraintset = set()
    rowsread = 0
    for row in constraintreader:
        if rowsread < START_SELECT:
            continue
        if rowsread >= NUM_SELECT:
            break
        constraintset.add(row[0].strip())
        rowsread += 1

    for i in range(NUM_CYCLES):
        print('CLASSIFIER', str(_classifier).split('(')[0], 'CYCLE', i)

        classifier = None
        with open('/home/nimadaan/cmv/pythonwksp/src_v2/models/CORPS3_TrainingData_' + str(i) +'_' + str(_classifier).split('(')[0]+ '.pkl', 'rb') as fid:
            classifier = cPickle.load(fid)

        full_testfeats = []
        for j in range(NUM_CYCLES_TEST):
            testfeats = None
            with open('/home/nimadaan/cmv/pythonwksp/src_v2/data/PD_TestData_'+str(j)+'.pkl', 'rb') as fid:
                testfeats = cPickle.load(fid)
                full_testfeats.extend(select_words_from_feats(testfeats, constraintset))

        #Raw accuracy
        accuracy0 = nltk.classify.accuracy(classifier, full_testfeats)

        accuracies.append(accuracy0)
        print('CLASSIFIER', str(_classifier).split('(')[0], 'CYCLE', i, 'RAW_ACCURACY', accuracy0, 'MEAN_ACCURACY',np.mean(np.asarray(accuracies)))
