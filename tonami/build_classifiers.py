import sklearn.pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from tonami import Classifier as c

def get_name(info):
    return "{}_{}_{}".format(info['type'], int((1-info['test_size'])*100), info['preprocessing'])

def build_svm_80_lda():
    pipe = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.StandardScaler(), 
        LDA(), 
        sklearn.svm.SVC(gamma='auto')
    )
    info = {
        'type': 'svm',
        'preprocessing': 'lda',
        'test_size': 0.2
    }
    info['name'] = get_name(info)
    c.get_data_from_pipe(pipe, info=info)

# this used to be svm_ml_times
def build_svm_10_none():
    pipe = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.StandardScaler(), 
        sklearn.svm.SVC(gamma='auto')
    )
    info = {
        'type': 'svm',
        'preprocessing': 'none',
        'test_size': 0.9
    }
    info['name'] = get_name(info)
    c.get_data_from_pipe(pipe, info=info)
