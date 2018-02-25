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
TRAIN_PERC = 0.8
VALID_PERC = 0.1
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

def word_feats(words):
    return dict([(word, True) for word in words.split() if word not in stopset])

def word_feats_selected(words, selected_set):
    return dict([(word, True) for word in words.split() if (word not in stopset) and (word not in selected_set)])

def word_feats_except_one_word(words, exception):
    return dict([(word, True) for word in words.split() if (word not in stopset) and (word != exception)])

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

with open('/home/nimadaan/cmv/pythonwksp/src_v2/wordsets/PD_words.pkl', 'wb') as datafile:
        cPickle.dump((all_word_set,word_freq_dict), datafile)