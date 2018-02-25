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
from sklearn import metrics
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
NUM_CLUSTERS = 50

from sklearn.cluster import KMeans
CLUSTERERS = [
    (KMeans(init='k-means++', n_clusters=NUM_CLUSTERS), 'KMeans')
]

SCORERS = [
    (metrics.silhouette_score, 'silhoutte')
]

for clusterer, name in CLUSTERERS:
    for i in range(NUM_CYCLES):
        trainfeats = None
        with open('/home/nimadaan/cmv/pythonwksp/src_v2/data/PDNUMPY_TrainingData_'+str(i)+'.pkl', 'rb') as fid:
            trainfeats = cPickle.load(fid)
        clusterer.fit(trainfeats)
        with open('/home/nimadaan/cmv/pythonwksp/src_v2/clustermodels/PDNUMPY_K50_TrainingData_' + name+ '.pkl', 'wb') as datafile:
            cPickle.dump(clusterer, datafile)
        # for scorer in SCORERS:
        #     print(scorer, scorer(trainfeats,clusterer.labels_))
