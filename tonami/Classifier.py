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
import matplotlib.pyplot as plt

from tonami import pitch_process_batch as ppb

PITCH_FILEPATH = 'data/parsed/toneperfect_pitch_librosa_50-500-fminmax.json'
CONFUSION_FILEPATH = 'temp/'
PICKLED_FILEPATH = 'tonami/data/'
MODEL_META_FILEPATH = 'tonami/data/model_metadata.json'

class Classifier:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def load_clf(self, filename):
        self.clf = pickle.load(open(filename, 'rb'))
    
    def classify_tones(self, features):
        prediction = self.clf.predict(features)
        probabilities = self.clf.predict_proba(features)
        return prediction, probabilities

def get_data_sets(speakers, train_size):
    '''
    Reads data from json and splits data based on desired speakers and train_size
    '''
    pitch_data = pd.read_json(PITCH_FILEPATH)

    if speakers:
        pitch_data = pitch_data.loc[pitch_data['speaker'].isin(speakers)]

    label, data = ppb.end_to_end(pitch_data)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, label, train_size=train_size)
    return X_train, X_test, y_train, y_test

def get_data_set_stats(y):
    '''
    Returns y's distribution percentage with labels and length of y
    '''
    num = len(y)
    hist = Counter(y)
    dist = [(i, hist[i] / num * 100.0) for i in hist]
    dist = sorted(dist, key=lambda tup: tup[0])
    dist = [dist[0][1], dist[1][1], dist[2][1], dist[3][1]]
    return dist, num

def save_pipeline_data(pipe, name):
    '''
    Saves the pipeline data in a pickle file to be loaded later
    '''
    file_name = PICKLED_FILEPATH + "pickled_" + name + ".pkl"
    pickle.dump(pipe, open(file_name, 'wb'))

def save_pipeline_meta(info, score, y_train_dist, y_test_dist, y_train_len, y_test_len):
    '''
    Saves the pipeline meta data as a json.
    '''
    data = pd.read_json(MODEL_META_FILEPATH, orient="index")
    row = {
        'Name': info['name'],
        'Date': str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")),
        'Type': info['type'],
        'Preprocessing': info['preprocessing'],
        'Train Size': info['train_size'],
        'Accuracy': score,
        'Train Number': y_train_len,
        'Train Distribution': [y_train_dist],
        'Test Number': y_test_len,
        'Test Distribtuion': [y_test_dist]
    }
    row = pd.DataFrame(row)
    data = data.drop(data[data.Name == info['name']].index)
    data = pd.concat([data, row], ignore_index=True)
    data.to_json(MODEL_META_FILEPATH, orient="index")


def save_confusion_matrix(y_test, y_pred, name):
    '''
    Creates a confusion matrix and saves it
    '''
    #TODO: labels might not be in the right order looooool could be 4 3 2 1?
    plt.figure()
    img = sklearn.metrics.ConfusionMatrixDisplay(sklearn.metrics.confusion_matrix(y_test, y_pred), display_labels=["1", "2", "3", "4"])
    img.plot()
    filename = CONFUSION_FILEPATH + 'confusion_' + name + '.jpg'
    plt.savefig(filename)

def get_data_from_pipe(pipe, info, speakers=[]):
    '''
    Takes in pipeline and name. Gets datasets, trains and saves pipeline and stats.
    '''
    X_train, X_test, y_train, y_test = get_data_sets(speakers, info['train_size'])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    score = balanced_accuracy_score(y_test, y_pred)
    y_train_dist, y_train_len = get_data_set_stats(y_train)
    y_test_dist, y_test_len = get_data_set_stats(y_test)

    save_pipeline_data(pipe, info['name'])
    save_pipeline_meta(info, score, y_train_dist, y_test_dist, y_train_len, y_test_len)
    save_confusion_matrix(y_test, y_pred, info['name'])

    print('score: ', score)
    print('train samples: ', y_train_len)
    print('train dist: ', y_train_dist)
    print('test samples: ', y_test_len)
    print('test dist: ', y_test_dist)

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
