#stub for classifier class
#could be 2 or 4 for num_classes
#method is method of classification. k-means, etc.
#k-fold stuff should also go in here, PCA, etc.

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.manifold import TSNE

from tonami.pitch_process_batch import end_to_end
from tonami.CrossValidate import cross_validate_tonami

PITCH_FILEPATH = 'data/parsed/toneperfect_pitch_librosa_50-500-fminmax.json'
CONFUSION_FILEPATH = 'temp/confusion_matrix/'
PICKLED_FILEPATH = 'tonami/data/pickled_models/'

class Classifier:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def load_clf(self, filename):
        self.clf = pickle.load(open(filename, 'rb'))
    
    def classify_tones(self, features):
        prediction = self.clf.predict(features)
        probabilities = self.clf.predict_proba(features)
        return prediction, probabilities

def _insert_model_data(df, index, new_row): 
    df.loc[index] = new_row
    df = df.sort_index().reset_index(drop=True)
    return df

def _get_data_sets(speakers, n_segments):
    '''
    Reads data from json and splits data based on desired speakers and performs end_to_end
    '''
    pitch_data = pd.read_json(PITCH_FILEPATH)

    if speakers:
        pitch_data = pitch_data.loc[pitch_data['speaker'].isin(speakers)]

    label, data = end_to_end(pitch_data, n_segments)
    return data, label

def _update_model_pkl(json_refs, index, best_estimator_dict):
    '''
    Takes best estimator from cross-validation and updates model_pkl with information
    '''
    pkl_row = {
        'Date': int(datetime.now().timestamp()),
        'Accuracy': best_estimator_dict['test_score'],
        'Fit Time / sample': best_estimator_dict['fit_time']/best_estimator_dict['n_train_samples'],
        'Score Time / sample': best_estimator_dict['score_time']/best_estimator_dict['n_test_samples'],
        'Explained Variance': best_estimator_dict['explained_variance'],
        'Train Distribution': best_estimator_dict['train_dist'],
        'Test Distribution': best_estimator_dict['test_dist'],
        'Train Number': best_estimator_dict['n_train_samples'],
        'Test Number': best_estimator_dict['n_test_samples'],
    }
    json_refs['model_pkl'] = _insert_model_data(json_refs['model_pkl'], index, pkl_row)

def _update_model_cvs(json_refs, info, scores):
    '''
    Takes pipe info and stats from cross-validation and updates model_cvs with information
    '''
    cvs_row = {
        'Segments': info['segments'],
        'Preprocessing': info['preprocessing'],
        'Train Size': info['train_size'],
        'Type': info['type'],
        'mean': scores['test_score_stats']['mean'],
        'std': scores['test_score_stats']['std'],
        'min': scores['test_score_stats']['min'],
        'max': scores['test_score_stats']['max'],
        'CV Splits': info['n_splits'],
        'Segment Features': scores['best_estimator_dict']['n_segment_features'],
        'Model Features': scores['best_estimator_dict']['n_model_features'],
        'Train Score': scores['train_score'],
        'Fit Time': scores['fit_time']['mean_total'],
        'Fit Time / sample': scores['fit_time']['mean_per_sample'],
        'Score Time': scores['score_time']['mean_total'],
        'Score Time / sample': scores['score_time']['mean_per_sample'],
        'Explained Variance': scores['explained_variance']
    }
    json_refs["model_cvs"] = _insert_model_data(json_refs["model_cvs"], info['index'], cvs_row)

def _save_pipeline_pkl(pipe, index):
    '''
    Saves the pipeline data in a pickle file to be loaded later
    '''
    file_name = PICKLED_FILEPATH + "pickled_" + str(index) + ".pkl"
    pickle.dump(pipe, open(file_name, 'wb'))

def _save_confusion_matrix(y_test, y_pred, index):
    '''
    Creates a confusion matrix and saves it
    '''
    plt.figure()
    img = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4]), display_labels=["1", "2", "3", "4"])
    img.plot()
    filename = CONFUSION_FILEPATH + 'confusion_' + str(index) + '.jpg'
    plt.savefig(filename)
    plt.close()

def make_cvs_from_pipe(json_refs, pipe, info, n_splits=5, speakers=[], print_results=True):
    '''
    Takes in pipeline and name. Runs cross validation (custom) and saves info and score stats.
    '''
    info['n_splits'] = n_splits

    X, y = _get_data_sets(speakers, info['segments'])
    cv=StratifiedShuffleSplit(n_splits=n_splits, train_size=info['train_size'])
    scores = cross_validate_tonami( pipe, X, y, cv=cv, n_jobs=-1)

    _update_model_cvs(json_refs, info, scores)

    if print_results:
        score_stats = scores['test_score_stats']
        print("test_scores: %.3f ± %.3f (%.3f,%.3f)" % (score_stats['mean'], score_stats['std'], score_stats['min'], score_stats['max']))

    return scores

def make_pkl_from_cvs(json_refs, index, best_estimator_dict, print_results=True):
    '''
    Takes in pipeline and name. Gets datasets, trains and saves pipeline as pkl and stats.
    '''
    _save_pipeline_pkl(best_estimator_dict['estimator'], index)
    _save_confusion_matrix(best_estimator_dict['y_test'], best_estimator_dict['y_pred'], index)
    _update_model_pkl(json_refs, index, best_estimator_dict)

    if print_results:
        print('test_score: ',       best_estimator_dict['test_score'])
        print('train samples: ',    best_estimator_dict['n_train_samples'])
        print('train dist: ',       best_estimator_dict['train_dist'])
        print('test samples: ',     best_estimator_dict['n_test_samples'])
        print('test dist: ',        best_estimator_dict['test_dist'])

def ml_times():
    pitch_data = pd.read_json(PITCH_FILEPATH)

    # ALL THE FEMALE TONE PERFECT FILES
    pitch_data = pitch_data.loc[pitch_data['speaker'].isin(['FV1', 'FV2', 'FV3'])]
    end_to_end(pitch_data)

    # TONE 
    for i in range(1,5):
        tone = pitch_data.loc[pitch_data['tone'] == i]
        print(f'TONE: {i}')
        end_to_end(tone)
        print('\n')

def t_sne(filename="t_sne.png"):
    pitch_data = pd.read_json(PITCH_FILEPATH)
    speakers = ['FV1', 'FV2', 'FV3', 'MV1', 'MV2', 'MV3']
    # ALL THE FEMALE TONE PERFECT FILES
    # pitch_data = pitch_data.loc[pitch_data['speaker'].isin(['FV1', 'FV2', 'FV3'])]
    # TODO: suspicion that MV1 has a utterance where our first_valid_index call can't find any valid index at all
    # pitch_data = pitch_data.loc[pitch_data['speaker'].isin(['FV1', 'FV2', 'FV3', 'MV1', 'MV2','MV3'])]
    # pitch_data = pitch_data.loc[pitch_data['speaker'].isin(['MV2','MV3'])]
    feat_arrs = []
    label_arrs = []

    # normalize each speaker's pitch individually
    for i in range(len(speakers)):
        spkr_data = pitch_data.loc[pitch_data['speaker'] == speakers[i]]
        spkr_label, spkr_feats, = end_to_end(spkr_data)
        feat_arrs.append(spkr_feats)
        label_arrs.append(spkr_label)

    data = np.vstack(feat_arrs)
    label = np.concatenate(label_arrs)

    tsne = TSNE(n_components=2)
    tsne_result = tsne.fit_transform(data)
    tsne_result.shape

    plt.figure()
    fig, ax = plt.subplots()
    for g in np.unique(label):
        ix = np.where(label == g)
        ax.scatter(tsne_result[ix, 0], tsne_result[ix, 1], label = g, s = 2)
    ax.legend(bbox_to_anchor=(1, 1))
    plt.savefig(filename) #save this
    plt.close()
