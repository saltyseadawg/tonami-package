import sklearn.pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from tonami import Classifier as c

from enum import IntFlag
class Index(IntFlag):
    SEGMENTS = 2 << 3
    PREPROCESSING = 2 << 2
    SIZE = 2 << 1
    MODEL1 = 2 << 0
    MODEL2 = 1

def get_info_from_index(index):
    def get_type(index):
        # Since they are the last 2 bits
        model_num = index % 4

        if model_num == 0:
            return 'SVM (RBF)'
        elif model_num == 1:
            return 'SVM (linear)'
        elif model_num == 2:
            return 'Decision Tree'
        else:
            return 'kNN'

    info = {}

    info['index'] = index
    info['segments'] = 5 if index & Index.SEGMENTS.value else 3
    info['preprocessing'] = 'LDA' if index & Index.PREPROCESSING.value else 'None'
    info['train_size'] = 0.8 if index & Index.SIZE.value else 0.1
    info['type'] = get_type(index)

    return info

def get_pipe_from_index(index):
    def get_type(index):
        # Since they are the last 2 bits
        model_num = index % 4

        if model_num == 0:
            return sklearn.svm.SVC(gamma='auto')
        elif model_num == 1:
            return sklearn.svm.SVC(gamma='auto', kernel='linear')
        elif model_num == 2:
            return sklearn.tree.DecisionTreeClassifier(criterion='entropy')
        else:
            return sklearn.neighbors.KNeighborsClassifier()

    pipe = []

    if index & Index.PREPROCESSING.value:
        pipe.append(('preprocessing', LDA()))

    pipe.append(('estimator', get_type(index)))

    pipe = sklearn.pipeline.Pipeline(pipe)
    return pipe

def build_all():
    print_results = False

    for index in range(16):
        print('Working on: ', index)

        info = get_info_from_index(index)
        pipe = get_pipe_from_index(index)

        c.get_data_from_pipe(pipe, info=info, print_results=print_results)
