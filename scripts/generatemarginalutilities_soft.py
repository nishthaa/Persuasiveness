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
from sklearn.metrics import brier_score_loss

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
# NUM_SELECT = 6000
NUM_CYCLES = 1
CLASSIFIERS = [
   # svm.SVC(C=1.0, kernel='linear', cache_size=2000),
     tree.DecisionTreeClassifier(),
    # neighbors.KNeighborsClassifier(n_neighbors=5),
    # MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
    # RandomForestClassifier(n_estimators=10),
    # svm.SVC(C=1.0, kernel='rbf', cache_size=2000),
   # 'NaiveBayes()',
     # MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
     # RandomForestClassifier(n_estimators=10)
]

def word_feats(words):
    return dict([(word, True) for word in words.split() if word not in stopset])

def word_feats_selected(words, selected_set):
    return dict([(word, True) for word in words.split() if (word not in stopset) and (word not in selected_set)])

def word_feats_except_one_word(words, exception):
    return dict([(word, True) for word in words.split() if (word not in stopset) and (word != exception)])


np.random.seed(0)

accuracy0 = 0

all_word_set = set()

progress_count = 0

def remove_word_from_feats(testfeats, rem_word):
    _test_feats = []
    for feat, label in testfeats:
        _test_feats.append((dict([(word, True) for word in feat if word is not rem_word]),label))
    return _test_feats

def get_all_words_from_feats(testfeats):
    word_set = set()
    for feat, label in testfeats:
        for word in feat:
            word_set.add(word)
    return word_set

for _classifier in CLASSIFIERS:
    marginalutilities_list = {}
    for i in range(NUM_CYCLES):
        print('CLASSIFIER', str(_classifier).split('(')[0], 'CYCLE', i)

        classifier = None
        with open('/home/nimadaan/cmv/pythonwksp/src_v2/models/CORPS3_TrainingData_' + str(i) +'_' + str(_classifier).split('(')[0]+ '.pkl', 'rb') as fid:
            classifier = cPickle.load(fid)

        # all_word_set, word_freq_dict = None,None
        # with open('/home/nimadaan/cmv/pythonwksp/src_v2/wordsets/CORPS3_words.pkl', 'rb') as fid:
        #     all_word_set, word_freq_dict = cPickle.load(fid)

        testfeats = None
        with open('/home/nimadaan/cmv/pythonwksp/src_v2/data/CORPS3_TestData_'+str(i)+'.pkl', 'rb') as fid:
            testfeats = cPickle.load(fid)

        #Raw accuracy
        accuracy0 = brier_score_loss(np.asarray([int(label) for _,label in testfeats]), np.asarray([classifier.prob_classify(feat).prob('1') for feat, _ in testfeats]))
        print('CLASSIFIER', str(_classifier).split('(')[0], 'CYCLE', i, 'RAW_ACCURACY', accuracy0)

        for word in get_all_words_from_feats(testfeats):
            _testfeats = remove_word_from_feats(testfeats,word)
            _accuracy = brier_score_loss(np.asarray([int(label) for _,label in _testfeats]), np.asarray([classifier.prob_classify(feat).prob('1') for feat, _ in _testfeats]))
            _mu = - accuracy0 + _accuracy
            print('CLASSIFIER', str(_classifier).split('(')[0], 'CYCLE', i, 'WORD', word, 'MU',_mu)
            _list = []
            if word in marginalutilities_list:
                _list = marginalutilities_list[word]
            _list.append(_mu)
            marginalutilities_list[word] = _list

        #Write right away
        mu_dict = {}
        for word in marginalutilities_list:
            mu = np.mean(np.asarray(marginalutilities_list[word]))
            mu_dict[word] = mu

        best_words = sorted(mu_dict.items(), key=lambda x:x[1], reverse=True)
        outfile = open('/home/nimadaan/cmv/pythonwksp/src_v2/marginalssoft/CORPS3_' + str(_classifier).split('(')[0] + '.csv', 'w')
        writer = csv.writer(outfile, delimiter=',')
        for j in range(len(best_words)):
            word, mu = best_words[j]
            row = [word,mu]
            writer.writerow(row)
            outfile.flush()
        outfile.close()