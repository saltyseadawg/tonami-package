import pandas as pd
from os import listdir

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC as SVM
from sklearn.tree import DecisionTreeClassifier as DT

from tonami import Classifier as c

from enum import IntFlag
class Index(IntFlag):
    SEGMENTS = 2 << 3
    PREPROCESSING = 2 << 2
    SIZE = 2 << 1
    TYPE1 = 2 << 0
    TYPE2 = 1

N_CLASSES = 4
CV_N_SPLITS = 5
NATIVE_SPEAKERS_DIR = 'data/native speakers/'
USER_AUDIO_DIR = 'data/users (balanced)/'
PICKLED_FILEPATH = 'tonami/data/pickled_models/'
MODEL_CVS_FILEPATH = 'tonami/data/model_cvs_info.json'
MODEL_PKL_FILEPATH = 'tonami/data/model_pkl_info.json'
MODEL_PREDS_FILEPATH = 'tonami/data/model_predictions.csv'

def _get_info_from_index(index):
    def _get_type(index):
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

    info = {
        "index": index,
        "segments":         5       if index & Index.SEGMENTS.value else        3,
        "preprocessing":    'LDA'   if index & Index.PREPROCESSING.value else   'None',
        "train_size":       0.8     if index & Index.SIZE.value else            0.1,
        "type": _get_type(index),
    }

    return info

def _get_pipe_from_index(index, n_segments):
    def _get_type(index):
        # Since they are the last 2 bits
        model_num = index % 4

        if model_num == 0:
            return SVM(probability=True, gamma='auto')
        elif model_num == 1:
            return SVM(probability=True, gamma='auto', kernel='linear')
        elif model_num == 2:
            return DT(criterion='entropy')
        else:
            return KNC()

    return Pipeline([
        ('preprocessing', LDA() if index & Index.PREPROCESSING.value else None),
        ('estimator', _get_type(index))
    ])

def build_all(print_results = False):  
    '''
    Performs cross-validation on all model variations except SpeakGoodChinese
    Records stats/scores of cross-validation and pickles best estimator (based on training data, not on hyperparameters)
    Only uses default hyperparameters, outlined in get_pipe_from_index
    '''
    # dummy dict to pass dataframes by ref
    json_refs = {
        "model_cvs": pd.read_json(MODEL_CVS_FILEPATH, orient="index"),
        "model_pkl": pd.read_json(MODEL_PKL_FILEPATH, orient="index"),
    }

    for index in range(32):
        print('Working on: ', index)

        info = _get_info_from_index(index)
        pipe = _get_pipe_from_index(index, info['segments'])

        scores = c.make_cvs_from_pipe(json_refs, pipe, info=info, n_splits=CV_N_SPLITS, print_results=print_results)
        c.make_pkl_from_cvs(json_refs, best_estimator_dict=scores['best_estimator_dict'], index=info['index'], print_results=print_results)

    json_refs["model_cvs"].to_json(MODEL_CVS_FILEPATH, orient="index")
    json_refs["model_pkl"].to_json(MODEL_PKL_FILEPATH, orient="index", date_format=None, date_unit='s')

def predict_all():
    model_preds = pd.read_csv(MODEL_PREDS_FILEPATH, index_col='word key')
    native_speakers_files = os.listdir(NATIVE_SPEAKERS_DIR)
    user_audio_files = os.listdir(USER_AUDIO_DIR)
    clf = c.Classifier(N_CLASSES)

    for model_num in range(16):
        model_filename = PICKLED_FILEPATH + 'pickled_' + str(model_num) + '.pkl'
        clf.load_clf(model_filename)

        for index, filename in enumerate(native_speakers_files):
            filename = NATIVE_SPEAKERS_DIR + filename
            # load audio file as utterance
            # get clf probabiliites
            # get rating as int lower_rating

            model_preds.loc[index+1, str(model_num)] = lower_rating

        for index, filename in enumerate(user_audio_files):
            filename = USER_AUDIO_DIR + filename
            # load audio file as utterance
            # get clf probabiliites
            # get rating as int lower_rating

            model_preds.loc[index+21, str(model_num)] = lower_rating

    model_preds.to_csv(MODEL_PREDS_FILEPATH, index_label='word key')