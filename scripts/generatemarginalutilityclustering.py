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
import sys

#Import custom models
from CorrelationModels import CorrelationModels
from ClassificationModels import ClassificationModels

from nltk.corpus import stopwords
from sklearn import metrics
stopset = list(set(stopwords.words('english')))

def remove_word_from_feats(feats,word,all_word_list):
    _feats = feats
    idx = all_word_list.index(word)
    _feats[:,idx] = 0
    return _feats

freq_limit = 5
top_num = 10
read_limit = 1000000
chunk_size = 100
TRAIN_PERC = 0.8
VALID_PERC = 0.1
TEST_PERC = 1 - TRAIN_PERC - VALID_PERC
# NUM_SELECT = 6000
NUM_CYCLES = 1
NUM_CLUSTERS = 2
NUM_ITER = 1
SAMPLE_SIZE = 5000

from sklearn.cluster import KMeans
CLUSTERERS = [
    (KMeans(init='k-means++', n_clusters=NUM_CLUSTERS), 'KMeans')
]

SCORERS = [
    (metrics.silhouette_score, 'silhoutte')
]
rem = 9
for clusterer, cname in CLUSTERERS:
    for scorer, sname in SCORERS:
        for i in range(NUM_CYCLES):
            print('Loading wordlists...')
            with open('/home/nimadaan/cmv/pythonwksp/src_v2/wordlists/PDNUMPY_words.pkl', 'rb') as datafile:
                all_word_list, word_freq_dict = cPickle.load(datafile)
            print('Loading features...')
            trainfeats = None
            with open('/home/nimadaan/cmv/pythonwksp/src_v2/data/PDNUMPY_TrainingData_'+str(i)+'.pkl', 'rb') as fid:
                trainfeats = cPickle.load(fid)
            print('Loading clusters...')
            clusterer = None
            with open('/home/nimadaan/cmv/pythonwksp/src_v2/clustermodels/PDNUMPY_K50_TrainingData_' +cname + '.pkl', 'rb') as fid:
                clusterer = cPickle.load(fid)

            score0 = np.mean(np.asarray([scorer(trainfeats, clusterer.labels_) for i in range(NUM_ITER)]))
            k = 0
            mu_dict = {}
            for word in all_word_list:
                if k%10 == rem:
                    new_feats = remove_word_from_feats(trainfeats,word,all_word_list)
                    _score = np.mean(np.asarray([scorer(new_feats, clusterer.labels_) for i in range(NUM_ITER)]))
                    print(cname,sname,rem,word,score0,_score,score0 - _score)
                    mu_dict[word] = score0 - _score
                k += 1

            best_words = sorted(mu_dict.items(), key=lambda x: x[1], reverse=True)
            outfile = open('/home/nimadaan/cmv/pythonwksp/src_v2/marginals/PDNUMPY_K50_' + cname+'_'+sname+'_'+ str(rem) + '.csv', 'w')
            writer = csv.writer(outfile, delimiter=',')
            for j in range(len(best_words)):
                word, mu = best_words[j]
                row = [word, mu]
                writer.writerow(row)
                outfile.flush()
            outfile.close()
