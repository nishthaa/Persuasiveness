
import numpy as np


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


from nltk.corpus import stopwords
stopset = list(set(stopwords.words('english')))


NUM_CYCLES = 1

CLASSIFIERS = [
    tree.DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=10),
    MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
    neighbors.KNeighborsClassifier(n_neighbors=5),
    # svm.SVC(C=1.0, kernel='linear', cache_size=10000),
    # svm.SVC(C=1.0, kernel='rbf', cache_size=10000)
]


np.random.seed(0)


for i in range(NUM_CYCLES):
    trainfeats = None
    with open('/home/nimadaan/cmv/pythonwksp/src_v2/data/CORPS3_TrainingData_'+str(i)+'.pkl', 'rb') as fid:
        trainfeats = cPickle.load(fid)
    for _classifier in CLASSIFIERS:
        print('Status: Classifier',str(_classifier).split('(')[0],'CYCLE',i)
        classifier = SklearnClassifier(_classifier).train(trainfeats)
        with open('/home/nimadaan/cmv/pythonwksp/src_v2/models/CORPS3_TrainingData_' + str(i) +'_' + str(_classifier).split('(')[0]+ '.pkl', 'wb') as datafile:
            cPickle.dump(classifier, datafile)

    #Naive Bayes
    print('Status: Classifier', 'Naive Bayes', 'CYCLE', i)
    classifier = NaiveBayesClassifier.train(trainfeats)
    with open('/home/nimadaan/cmv/pythonwksp/src_v2/models/CORPS3_TrainingData_' + str(i) + '_' +
                      'NaiveBayes' + '.pkl', 'wb') as datafile:
        cPickle.dump(classifier, datafile)
