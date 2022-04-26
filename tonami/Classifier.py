#stub for classifier class
#could be 2 or 4 for num_classes
#method is method of classification. k-means, etc.
#k-fold stuff should also go in here, PCA, etc.

from datetime import datetime
from collections import Counter
import pickle
import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.pipeline
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
import matplotlib.pyplot as plt

from tonami import pitch_process_batch as ppb
from tonami import CrossValidate as tcv

PITCH_FILEPATH = 'data/parsed/toneperfect_pitch_librosa_50-500-fminmax.json'
CONFUSION_FILEPATH = 'temp/confusion_matrix/'
PICKLED_FILEPATH = 'tonami/data/pickled_models/'
MODEL_CVS_FILEPATH = 'tonami/data/model_cvs_info.json'
MODEL_PKL_FILEPATH = 'tonami/data/model_pkl_info.json'

class Classifier:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def load_clf(self, filename):
        self.clf = pickle.load(open(filename, 'rb'))
    
    def classify_tones(self, features):
        prediction = self.clf.predict(features)
        probabilities = self.clf.predict_proba(features)
        return prediction, probabilities

def insert_model_data(df, index, new_row): 
    df.loc[index] = new_row
    df = df.sort_index().reset_index(drop=True)
    return df

def get_data_sets(speakers, train_size):
    '''
    Reads data from json and splits data based on desired speakers and train_size
    '''
    pitch_data = pd.read_json(PITCH_FILEPATH)

    if speakers:
        pitch_data = pitch_data.loc[pitch_data['speaker'].isin(speakers)]

    label, data = ppb.end_to_end(pitch_data)

    if train_size != 1.0:
        return sklearn.model_selection.train_test_split(data, label, train_size=train_size)
    else:
        return data, [], label, []

def get_data_set_stats(y):
    '''
    Returns y's distribution percentage with labels and length of y
    '''
    hist = Counter(y)
    num = len(y)
    dist = [(i, hist[i] / num * 100.0) for i in hist]
    dist = sorted(dist, key=lambda tup: tup[0])
    dist = [dist[0][1], dist[1][1], dist[2][1], dist[3][1]]
    dist = np.around(dist, 2)
    return dist, num

def save_pipeline_pkl(pipe, index):
    '''
    Saves the pipeline data in a pickle file to be loaded later
    '''
    file_name = PICKLED_FILEPATH + "pickled_" + str(index) + ".pkl"
    pickle.dump(pipe, open(file_name, 'wb'))

def save_pipeline_data(json_refs, index, best_est_dict):
    '''
    Saves the pipeline data as a json.
    '''

    pkl_row = {
        'Date': int(datetime.now().timestamp()),
        'Accuracy': best_est_dict['test_score'],
        'Fit Time / sample': best_est_dict['fit_time']/best_est_dict['n_train_samples'],
        'Score Time / sample': best_est_dict['score_time']/best_est_dict['n_test_samples'],
        'Explained Variance': best_est_dict['explained_variance'],
        'Train Distribution': best_est_dict['train_dist'],
        'Test Distribution': best_est_dict['test_dist'],
        'Train Number': best_est_dict['n_train_samples'],
        'Test Number': best_est_dict['n_test_samples'],
    }
    json_refs['model_pkl'] = insert_model_data(json_refs['model_pkl'], index, pkl_row)

def save_model_cvs_info(json_refs, info, scores):
    '''
    Saves the model's information and cvs as a json.
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
    json_refs["model_cvs"] = insert_model_data(json_refs["model_cvs"], info['index'], cvs_row)

def save_confusion_matrix(y_test, y_pred, index):
    '''
    Creates a confusion matrix and saves it
    '''
    plt.figure()
    img = sklearn.metrics.ConfusionMatrixDisplay(sklearn.metrics.confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4]), display_labels=["1", "2", "3", "4"])
    img.plot()
    filename = CONFUSION_FILEPATH + 'confusion_' + str(index) + '.jpg'
    plt.savefig(filename)
    plt.close()

def make_pkl_from_cvs(json_refs, scores, info, print_results=True):
    '''
    Takes in pipeline and name. Gets datasets, trains and saves pipeline as pkl and stats.
    '''
    best_est_dict = scores['best_estimator_dict']
    pipe = best_est_dict['estimator']

    save_pipeline_pkl(pipe, info['index'])
    json_refs = save_pipeline_data(json_refs, info['index'], best_est_dict)
    save_confusion_matrix(best_est_dict['y_test'], best_est_dict['y_pred'], info['index'])

    if print_results:
        print('test_score: ', best_est_dict['test_score'])
        print('train samples: ', best_est_dict['n_train_samples'])
        print('train dist: ', best_est_dict['train_dist'])
        print('test samples: ', best_est_dict['n_test_samples'])
        print('test dist: ', best_est_dict['test_dist'])

def make_cvs_from_pipe(json_refs, pipe, info, n_splits=5, speakers=[], print_results=True):
    '''
    Takes in pipeline and name. Runs cross validation (custom) and saves info and score stats.
    '''
    X_train, _, y_train, _ = get_data_sets(speakers, 1.0)
    cv = StratifiedShuffleSplit(n_splits=n_splits, train_size=info['train_size'])
    scores = tcv.cross_validate_tonami(pipe, X_train, y_train, cv=cv, n_jobs=-1)

    info['n_splits'] = n_splits
    save_model_cvs_info(json_refs, info, scores)

    if print_results:
        score_stats = scores['test_score_stats']
        print("test_scores: %.3f ± %.3f (%.3f,%.3f)" % (score_stats['mean'], score_stats['std'], score_stats['min'], score_stats['max']))

    return scores

def ml_times():
    pitch_data = pd.read_json(PITCH_FILEPATH)

    # ALL THE FEMALE TONE PERFECT FILES
    pitch_data = pitch_data.loc[pitch_data['speaker'].isin(['FV1', 'FV2', 'FV3'])]
    ppb.end_to_end(pitch_data)

    # TONE 
    for i in range(1,5):
        tone = pitch_data.loc[pitch_data['tone'] == i]
        print(f'TONE: {i}')
        ppb.end_to_end(tone)
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

    tsne = sklearn.manifold.TSNE(n_components=2)
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
