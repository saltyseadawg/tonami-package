#stub for classifier class
#could be 2 or 4 for num_classes
#method is method of classification. k-means, etc.
#k-fold stuff should also go in here, PCA, etc.

import pickle

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.pipeline
import matplotlib.pyplot as plt

from tonami import pitch_process as pp

PITCH_FILEPATH = 'data/parsed/toneperfect_pitch_librosa_50-500-fminmax.json'


class Classifier:
    def __init__(self, num_classes, method='svm'):
        self.num_classes = num_classes
        self.method = method

    def get_probability() -> npt.NDArray[float]:
        #associated probability of being in each class
        pass
    
    def classify_tones(self, features):
        # TODO: further development is required to use different classifier models, fit the data and classify the tone
        if self.method == 'svm':
            clf = pickle.load(open('data/pickled_svm_80.pkl', 'rb'))
            prediction = clf.predict(features)
        else:
            # other classifier models
            prediction = []
        return prediction


def ml_times():
    pitch_data = pd.read_json(PITCH_FILEPATH)

    # ALL THE FEMALE TONE PERFECT FILES
    pitch_data = pitch_data.loc[pitch_data['speaker'].isin(['FV1', 'FV2', 'FV3'])]
    pp.end_to_end(pitch_data)

    # TONE 
    for i in range(1,5):
        tone = pitch_data.loc[pitch_data['tone'] == i]
        print(f'TONE: {i}')
        pp.end_to_end(tone)
        print('\n')

def svm_ml_times(filename='confusion.jpg'):
    pitch_data = pd.read_json(PITCH_FILEPATH)

    # ALL THE FEMALE TONE PERFECT FILES
    # pitch_data = pitch_data.loc[pitch_data['speaker'].isin(['FV1', 'FV2', 'FV3'])]
    # pitch_data = pitch_data.loc[pitch_data['speaker'].isin(['MV1'])]
    label, data = pp.end_to_end(pitch_data)
    
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, label, test_size=0.9)

    clf = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.svm.SVC(gamma='auto'))
    clf.fit(X_train, y_train)

    sklearn.pipeline.Pipeline(steps=[('standardscaler', sklearn.preprocessing.StandardScaler()),
                ('svc', sklearn.svm.SVC(gamma='auto'))])

    y_pred = clf.predict(X_test)
    #TODO: labels might not be in the right order looooool could be 4 3 2 1?
    plt.figure()
    img = sklearn.metrics.ConfusionMatrixDisplay(sklearn.metrics.confusion_matrix(y_test, y_pred), display_labels=["1", "2", "3", "4"])
    img.plot() #matplotlib magic hell
    # plt.show()
    plt.savefig(filename)
    # TONE 
    # for i in range(1,5):
        # tone = pitch_data.loc[pitch_data['tone'] == i]
        # print(f'TONE: {i}')
        # end_to_end(tone)
        # print('\n')

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

