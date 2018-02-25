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
from corps_ingestor import corps_ingest_unsupervised

#Import Annotators
from title_tokenizer import title_tokenizer
from stemmer_annotator import stemmer_annotator
from pos_tag_annotator import pos_tag_annotator
from quantity_count_annotator import quantity_count_annotator
from stopword_annotator import stopword_annotator
from word_vector_annotator import word_vector_annotator


#Import custom models
from CorrelationModels import CorrelationModels
from ClassificationModels import ClassificationModels

from nltk.corpus import stopwords
stopset = list(set(stopwords.words('english')))


freq_limit = 3
top_num = 10
read_limit = 150000
chunk_size = 100
TRAIN_PERC = 0.8
VALID_PERC = 0.1
TEST_PERC = 1 - TRAIN_PERC - VALID_PERC
NUM_SELECT = 3000

def word_feats(words):
    return dict([(word, True) for word in words.split() if word not in stopset])

def word_feats_selected(words, selected_set):
    return dict([(word, True) for word in words.split() if (word not in stopset) and (word not in selected_set)])

#Read the file
#trainingfile = open('/home/nimadaan/cmv/pythonwksp/data/corps_full_preproc_cleaned.csv')


# reader = csv.reader(trainingfile,delimiter=',')

#Annotator Initializations
title_tokenizer_instance = title_tokenizer()
stemmer_annotator_instance = stemmer_annotator()
# pos_tag_annotator_instance = pos_tag_annotator()
# quantity_count_annotator_instance = quantity_count_annotator()
# stopword_annotator_instance = stopword_annotator()
# word_vector_annotator_instance = word_vector_annotator()

all_word_set = set()

progress_count = 0
word_freq_dict_source = {}
word_freq_dict_target = {}

print('Loading source data set...')
#source_array = corps_ingest_unsupervised('/home/nimadaan/cmv/pythonwksp/data/PDDataSet_labelsfixed.csv', read_limit)
source_array = corps_ingest_unsupervised('/home/nimadaan/cmv/pythonwksp/data/corps_full_preproc_cleaned.csv', read_limit)
# sarray = corps_ingest_subsampling('/home/nimadaan/cmv/pythonwksp/data/corps_full_preproc_cleaned.csv', read_limit)

num_source_tokens = 0
for data in source_array:
        progress_count = progress_count + 1
        if progress_count >= read_limit:
            break
        if progress_count%100 == 0:
            print("Progress Count",progress_count)

        data = title_tokenizer_instance.process(data)
        data = stemmer_annotator_instance.process(data)

        # all_word_set.update(data.annotations['stemmed_tokens'])

        #Frequency counting
        for token in data.annotations['stemmed_tokens']:
            num_source_tokens += 1
            if token in word_freq_dict_source:
                word_freq_dict_source[token] = word_freq_dict_source[token] + 1
            else:
                word_freq_dict_source[token] = 1

print('word_freq_dict_source',word_freq_dict_source)

#Read target data set
print('Loading target data set')
# target_array = corps_ingest_unsupervised('/home/nimadaan/cmv/pythonwksp/data/un-general-debates.csv', read_limit= 20000, col_num= 3)
target_array = corps_ingest_unsupervised('/home/nimadaan/cmv/pythonwksp/data/PDDataSet_labelsfixed.csv', read_limit= 20000, col_num= 0)
progress_count = 0
num_target_tokens = 0
for data in target_array:
    progress_count = progress_count + 1
    if progress_count >= read_limit:
        break
    if progress_count % 100 == 0:
        print("Progress Count", progress_count)

    data = title_tokenizer_instance.process(data)
    data = stemmer_annotator_instance.process(data)

    # all_word_set.update(data.annotations['stemmed_tokens'])

    # Frequency counting
    for token in data.annotations['stemmed_tokens']:
        num_target_tokens += 1
        if token in word_freq_dict_target:
            word_freq_dict_target[token] = word_freq_dict_target[token] + 1
        else:
            word_freq_dict_target[token] = 1


datasets = ['CORPS3']
classifiers = ['DecisionTreeClassifier',
               # 'KNeighborsClassifier',
                'MLPClassifier',
               'NaiveBayes',
              # 'RandomForestClassifier',
               # 'SVC'
               ]

for dataname in datasets:
    for clfname in classifiers:
        filename = dataname+"_"+clfname+".csv"

        infile = open('/home/nimadaan/cmv/pythonwksp/src_v2/marginalssoft/'+filename, 'r')
        outfile = open('/home/nimadaan/cmv/pythonwksp/src_v2/marginalssilhouttescores/'+filename, 'w')

        reader = csv.reader(x.replace('\x00', '') for x in infile)
        writer = csv.writer(outfile, delimiter=',')

        #Read Marginal Utilities
        print('Loading all marginal utilities...')
        mu_dict = {}
        for row in reader:
            word = row[0]
            mu_dict[word] = float(row[1])


        #Compute scores and write to file
        print('Write the scores to file.' +filename)
        score_dict = {}
        for word in mu_dict:
            freq_source = 1
            freq_target = 1
            try:
                freq_source = word_freq_dict_source[word]/(1.0*num_source_tokens)
                freq_target = word_freq_dict_target[word]/(1.0*num_target_tokens)
            except:
                pass

            score = (mu_dict[word] + 0.000000001)* (freq_target/(freq_source + 0.00000001))
            print('Row',[word,score,freq_source,freq_target])
            score_dict[word] = score

        best_words = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        for w,v in best_words:
            writer.writerow([w,v])
            outfile.flush()
        outfile.close()
        infile.close()
