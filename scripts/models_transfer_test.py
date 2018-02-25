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
import codecs

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
NUM_SELECT = 7000
NUM_CYCLES = 1
NUM_CYCLES_TEST = 1
CLASSIFIERS = [
   # svm.SVC(C=1.0, kernel='linear', cache_size=2000),
    tree.DecisionTreeClassifier(),
    # neighbors.KNeighborsClassifier(n_neighbors=5),
    MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
   # RandomForestClassifier(n_estimators=10),
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

print('NUM_SELECT',NUM_SELECT)
for _classifier in CLASSIFIERS:
    accuracies = []
    # Read the constraint file
  #For Silhoutte
   # constraintfile = codecs.open('/home/nimadaan/cmv/pythonwksp/src_v2/marginals/PDNUMPY_K50_KMeans_silhoutte_v2.csv',encoding='utf-8',errors='ignore')

    #Modified Specificness with soft marginals
    # constraintfile = codecs.open('/home/nimadaan/cmv/pythonwksp/src_v2/marginalspecificscores/CORPS3_'+str(_classifier).split('(')[0]+'.csv',encoding='utf-8',errors='ignore')


    #Modified Silhoutte with soft marginals
    # constraintfile = codecs.open('/home/nimadaan/cmv/pythonwksp/src_v2/marginalssilhouttescores/CORPS3_'+str(_classifier).split('(')[0]+'.csv',encoding='utf-8',errors='ignore')
    # constraintfile = codecs.open('/home/nimadaan/cmv/pythonwksp/src_v2/marginalssilhouttescores/CORPS3_NaiveBayes.csv', encoding='utf-8', errors='ignore')

    #Modified Silhoutte Normalized with soft marginals
    constraintfile = codecs.open('/home/nimadaan/cmv/pythonwksp/src_v2/marginalssilhouettenormalizedscores/CORPS3_'+str(_classifier).split('(')[0]+'.csv',encoding='utf-8',errors='ignore')

    #For Marginals
    #constraintfile = codecs.open('/home/nimadaan/cmv/pythonwksp/src_v2/marginalssoft/CORPS3_'+str(_classifier).split('(')[0]+'.csv',encoding='utf-8',errors='ignore')

    #For soft marginals
    # constraintfile = codecs.open('/home/nimadaan/cmv/pythonwksp/src_v2/marginalssoft/CORPS3_'+str(_classifier).split('(')[0]+'.csv',encoding='utf-8',errors='ignore')

    #Get word set of target corpus. We take the intersection of the words in source
    #and target corpus because words that are not in the intersection were either not
    #involved in training or would no be involved in testing hence rendering them useless.
    target_word_set, target_word_freq_dict = None,None
    with open('/home/nimadaan/cmv/pythonwksp/src_v2/wordsets/PD_words.pkl', 'rb') as fid: #REMEMBER to change target corpus
        target_word_set, target_word_freq_dict = cPickle.load(fid)

    constraintreader = csv.reader(constraintfile)
    constraintset = set()
    rowsread = 0
    for row in constraintreader:
        if rowsread < START_SELECT:
            continue
        if rowsread >= NUM_SELECT:
            break
        if row[0].strip() not in target_word_set:
            continue
        constraintset.add(row[0].strip())
        rowsread += 1
    print('SIZE OF CONSTRAINT SET',len(constraintset))

    for i in range(NUM_CYCLES):
        print('CLASSIFIER', str(_classifier).split('(')[0], 'CYCLE', i)

        classifier = None
        with open('/home/nimadaan/cmv/pythonwksp/src_v2/models/CORPS3_TrainingData_' + str(i) +'_' + str(_classifier).split('(')[0]+ '.pkl', 'rb') as fid:
            classifier = cPickle.load(fid)

        full_testfeats = []
        for j in range(NUM_CYCLES_TEST):
            testfeats = None
            #Comment below if you need to test on a different subset of target data set.
            #Testing Set
            with open('/home/nimadaan/cmv/pythonwksp/src_v2/data/PD_TestData_'+str(j)+'.pkl', 'rb') as fid:
                testfeats = cPickle.load(fid)
                full_testfeats.extend(select_words_from_feats(testfeats, constraintset))
            #Validation set
            with open('/home/nimadaan/cmv/pythonwksp/src_v2/data/PD_ValidData_'+str(j)+'.pkl', 'rb') as fid:
                testfeats = cPickle.load(fid)
                full_testfeats.extend(select_words_from_feats(testfeats, constraintset))
            #Training set
            with open('/home/nimadaan/cmv/pythonwksp/src_v2/data/PD_TrainingData_'+str(j)+'.pkl', 'rb') as fid:
                testfeats = cPickle.load(fid)
                full_testfeats.extend(select_words_from_feats(testfeats, constraintset))

        #Raw accuracy
        accuracy0 = nltk.classify.accuracy(classifier, full_testfeats)

        accuracies.append(accuracy0)
        print('CLASSIFIER', str(_classifier).split('(')[0], 'CYCLE', i, 'ITERATION_ACCURACY', accuracy0, 'MEAN_ACCURACY',np.mean(np.asarray(accuracies)))
