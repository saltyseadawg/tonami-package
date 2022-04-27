import os
import json
import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC as SVM
from sklearn.tree import DecisionTreeClassifier as DT

from tonami import Classifier as c
from tonami import Utterance as u
from tonami import user
from heroku.interface_utils import get_rating

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
SPEAKER_INFO_FILEPATH = 'tonami/data/speaker_max_min.txt'
INTERFACE_TEXT_FILEPATH = 'heroku/interface_text.json'

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

def _get_pipe_from_index(index):
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
        pipe = _get_pipe_from_index(index)

        scores = c.make_cvs_from_pipe(json_refs, pipe, info=info, n_splits=CV_N_SPLITS, print_results=print_results)
        c.make_pkl_from_cvs(json_refs, best_estimator_dict=scores['best_estimator_dict'], index=info['index'], print_results=print_results)

    json_refs["model_cvs"].to_json(MODEL_CVS_FILEPATH, orient="index")
    json_refs["model_pkl"].to_json(MODEL_PKL_FILEPATH, orient="index", date_format=None, date_unit='s')

def _get_all_user_profiles():
    speakers_info = pd.read_json(SPEAKER_INFO_FILEPATH)

    user_profiles={}
    for index in range(len(speakers_info.index)):
        speaker_name = speakers_info.loc[index, 'speaker_name']
        speaker_max_f0 = speakers_info.loc[index, 'max_f0']
        speaker_min_f0 = speakers_info.loc[index, 'min_f0']

        user_profiles[speaker_name] = user.User(speaker_max_f0, speaker_min_f0)

    return user_profiles

def _get_all_features(speaker_names):
    user_profiles=_get_all_user_profiles()
    native_speakers_files = os.listdir(NATIVE_SPEAKERS_DIR)
    user_audio_files = os.listdir(USER_AUDIO_DIR)

    features_3 = np.empty((70, 6))
    features_5 = np.empty((70, 15))

    for index, filename in enumerate(native_speakers_files):
        filename = NATIVE_SPEAKERS_DIR + filename
        user_utterance = u.Utterance(filename=filename)
        _, _, features_3[index] = user_utterance.pre_process(user_profiles[speaker_names[index]], 3)
        _, _, features_5[index] = user_utterance.pre_process(user_profiles[speaker_names[index]], 5)

    for index, filename in enumerate(user_audio_files):
        index += 20
        filename = USER_AUDIO_DIR + filename
        user_utterance = u.Utterance(filename=filename)
        _, _, features_3[index] = user_utterance.pre_process(user_profiles[speaker_names[index]], 3)
        _, _, features_5[index] = user_utterance.pre_process(user_profiles[speaker_names[index]], 5)

    features = {
        '3': features_3,
        '5': features_5,
    }
    return features

def predict_all():
    model_preds = pd.read_csv(MODEL_PREDS_FILEPATH, index_col='word key')
    model_cvs = pd.read_json(MODEL_CVS_FILEPATH, orient='index')
    rating_data = json.load(open(INTERFACE_TEXT_FILEPATH))['ratings']

    clf = c.Classifier(N_CLASSES)
    all_features = _get_all_features(model_preds['speaker'].values)

    for model_num, n_segments in enumerate(model_cvs['Segments'].values):
        if model_num == 32:
            break

        print("Working on: ", model_num)

        model_filename = PICKLED_FILEPATH + 'pickled_' + str(model_num) + '.pkl'
        clf.load_clf(model_filename)
        
        features_by_segments = all_features[str(n_segments)]

        for index, (target_tone, features) in enumerate(zip(model_preds['tone'], features_by_segments)):
            if np.isnan(features).any():
                lower_rating = float('NaN')
            else:
                _, classified_probs = clf.classify_tones(np.reshape(features, (1, -1)))
                lower_rating = get_rating(rating_data, target_tone, classified_probs, as_int=True)

            model_preds.loc[index+1, str(model_num)] = lower_rating

    model_preds.to_csv(MODEL_PREDS_FILEPATH, index_label='word key')