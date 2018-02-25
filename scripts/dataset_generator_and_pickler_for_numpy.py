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

#Import custom models
from CorrelationModels import CorrelationModels
from ClassificationModels import ClassificationModels

from nltk.corpus import stopwords
stopset = list(set(stopwords.words('english')))


freq_limit = 5
top_num = 10
read_limit = 10000000
chunk_size = 100
TRAIN_PERC = 1.0
VALID_PERC = 0.0
TEST_PERC = 1 - TRAIN_PERC - VALID_PERC
# NUM_SELECT = 6000
NUM_CYCLES = 1
CLASSIFIERS = [
    svm.SVC(C=1.0, kernel='linear', cache_size=10000),
    tree.DecisionTreeClassifier(),
    neighbors.KNeighborsClassifier(n_neighbors=5),
    MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
    RandomForestClassifier(n_estimators=10),
    svm.SVC(C=1.0, kernel='rbf', cache_size=2000)
]

def word_feats(words,all_word_list):
    list = [ 0 ]*len(all_word_list)
    for word in words.split():
        if word not in stopset:
            idx = all_word_list.index(word)
            list[idx] = 1
    return list

#Read the file
# outfile = open('/home/nimadaan/cmv/pythonwksp/data/naive_marginals_PD_10-fold.csv','w')
# writer = csv.writer(outfile,delimiter=',')

# reader = csv.reader(trainingfile,delimiter=',')

#Annotator Initializations
title_tokenizer_instance = title_tokenizer()
stemmer_annotator_instance = stemmer_annotator()
# pos_tag_annotator_instance = pos_tag_annotator()
# quantity_count_annotator_instance = quantity_count_annotator()
# stopword_annotator_instance = stopword_annotator()
# word_vector_annotator_instance = word_vector_annotator()

np.random.seed(0)

accuracy0 = 0
accuracies = []

num_pos_comments = 0
num_neg_comments = 0
all_word_set = set()

progress_count = 0
word_label_pairs = []
word_freq_dict = {}
posids_train = []
negids_train = []
posids_valid = []
negids_valid = []
posids_test = []
negids_test = []
# array = corps_ingest_subsampling('/home/nimadaan/cmv/pythonwksp/data/corps_full_preproc_cleaned.csv', read_limit)
array = corps_ingest_subsampling('/home/nimadaan/cmv/pythonwksp/data/PDDataSet_labelsfixed.csv', read_limit)
for data in array:
        progress_count = progress_count + 1
        if progress_count >= read_limit:
            break
        if progress_count%1000 == 0:
            print("Progress Count",progress_count)

        tag = data.label
        #Annotator Pipeline
        data = title_tokenizer_instance.process(data)
        data = stemmer_annotator_instance.process(data)

        #Doing something with annotator output
        all_word_set.update(data.annotations['stemmed_tokens'])

        #Frequency counting
        for token in data.annotations['stemmed_tokens']:
            if token in word_freq_dict:
                word_freq_dict[token] = word_freq_dict[token] + 1
            else:
                word_freq_dict[token] = 1

        rn = random.uniform(0,1)
        if tag == '1':
            if rn < TRAIN_PERC:
                posids_train.append(data.annotations["stemmed_sentence"])
            elif rn >= TRAIN_PERC and rn < TRAIN_PERC+VALID_PERC:
                posids_valid.append(data.annotations["stemmed_sentence"])
            else:
                posids_test.append(data.annotations["stemmed_sentence"])

        if tag == '0':
            if rn < TRAIN_PERC:
                negids_train.append(data.annotations["stemmed_sentence"])
            elif rn >= TRAIN_PERC and rn < TRAIN_PERC+VALID_PERC:
                negids_valid.append(data.annotations["stemmed_sentence"])
            else:
                negids_test.append(data.annotations["stemmed_sentence"])

all_word_list = []
for word in all_word_set:
    all_word_list.append(word)
with open('/home/nimadaan/cmv/pythonwksp/src_v2/wordlists/PDNUMPY_words.pkl', 'wb') as datafile:
    cPickle.dump((all_word_list, word_freq_dict), datafile)
#Train
print('Creating training features')
pos_feats = [word_feats(f,all_word_list) for f in posids_train ]
neg_feats = [word_feats(f,all_word_list) for f in negids_train ]
trainfeats = pos_feats + neg_feats
for i in range(NUM_CYCLES):
    np.random.seed(i)
    _trainfeats = []
    _trainfeats.extend(trainfeats)
    np.random.shuffle(_trainfeats)
    _trainfeats_ = _trainfeats[0:int(len(trainfeats)/NUM_CYCLES)]
    with open('/home/nimadaan/cmv/pythonwksp/src_v2/data/PDNUMPY_TrainingData_'+str(i)+'.pkl', 'wb') as datafile:
        cPickle.dump(np.asarray(_trainfeats_), datafile)

# #Valid
# print('Creating validation features')
# pos_feats = [word_feats(f,all_word_list) for f in posids_valid ]
# neg_feats = [word_feats(f,all_word_list) for f in negids_valid ]
# validfeats = pos_feats + neg_feats
# for i in range(NUM_CYCLES):
#     np.random.seed(i)
#     _feats = []
#     _feats.extend(validfeats)
#     np.random.shuffle(_feats)
#     _feats_ = _feats[0:int(len(validfeats)/NUM_CYCLES)]
#     with open('/home/nimadaan/cmv/pythonwksp/src_v2/data/CORPS3_ValidData_'+str(i)+'.pkl', 'wb') as datafile:
#         cPickle.dump(np.asarray(_feats_), datafile)
#
# #Test
# print('Creating testing features')
# pos_feats = [word_feats(f,all_word_list) for f in posids_test ]
# neg_feats = [word_feats(f,all_word_list) for f in negids_test ]
# testfeats = pos_feats + neg_feats
# for i in range(NUM_CYCLES):
#     np.random.seed(i)
#     _feats = []
#     _feats.extend(testfeats)
#     np.random.shuffle(_feats)
#     _feats_ = _feats[0:int(len(testfeats)/NUM_CYCLES)]
#     with open('/home/nimadaan/cmv/pythonwksp/src_v2/data/CORPS3_TestData_'+str(i)+'.pkl', 'wb') as datafile:
#         cPickle.dump(np.asarray(_feats_), datafile)