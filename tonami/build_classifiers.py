import sklearn.pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from tonami import Classifier as c

def build_svm_80_lda():
    pipe = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.StandardScaler(), 
        LDA(), 
        sklearn.svm.SVC(gamma='auto')
    )
    c.get_data_from_pipe(pipe, 'svm_80_lda', test_size=0.2)

# this used to be svm_ml_times
def build_svm_10_none():
    pipe = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.StandardScaler(), 
        sklearn.svm.SVC(gamma='auto')
    )
    c.get_data_from_pipe(pipe, 'svm_10_none', test_size=0.9)
