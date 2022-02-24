# the line above is for jupyter notebook extension on VS code
# TODO: in final, comment out matplotlib - we want to visualize everything in visualization module
from locale import normalize
import warnings
import math

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing import Tuple, Callable
from scipy.ndimage.filters import uniform_filter1d
import scipy.signal as signal
import pandas as pd
import sklearn


# known error of package, we intend to use audioread.
warnings.filterwarnings(
    "ignore", message="PySoundFile failed. Trying audioread instead."
)

PITCH_FILEPATH = 'data/parsed/toneperfect_pitch_librosa_50-500-fminmax.json'

# VAD -> truncate -> check for nans (drop tracks with nans)
# Segment -> divide into 3
# Find mean pitch
# Find differences between means
# Final matrix (~9000,6 i.e. means, mean diffs)

def basic_feature_extraction(pitch_contours):
    features = np.empty((pitch_contours.shape[0],6))
    # calcualte features - not irregular <3
    pitch_max, pitch_min = max_min_f0(pitch_contours)
    for i in range(pitch_contours.shape[0]):
        # normalizing and sh*t
        avgd = moving_average(pitch_contours[i])
        normalized = normalize_pitch(avgd, pitch_max, pitch_min)

        features[i] = basic_feat_calc(normalized)

    return features
    
# https://www.isca-speech.org/archive_v0/archive_papers/interspeech_2010/i10_0602.pdf
def basic_feat_calc(pitch_contour):
    # pitch_contour is segmented
    # voice activity detection

    # Split into 3 bc the decision tree paper did it
    segments = np.array_split(pitch_contour, 3)
    features = np.empty(6)

    # feats 0-2: segment means
    for i in range(len(segments)):
        features[i] = segments[i].mean()
    
    # feats 3-5: diff b/w means
    features[3] = features[1] - features[0]
    features[4] = features[2] - features[1]
    features[5] = features[2] - features[0]

    return features

# https://note.nkmk.me/en/python-numpy-nan-remove/
def get_valid_mask(contours):
    # get rid of the rows with nans in the middle goddmanit
    # 1. pad the goddamn array to do black magic (np only plays nice with array that are NOT irregular)
    # 2. get the indices of pitch contours with nans in the middle
    # 3. mask that shit
    # 4. profit
    padded = np.array(pad_matrix(contours, fillval='420.69'), dtype=float)
    valid_row_mask = ~np.isnan(padded).any(axis=1)
    return valid_row_mask

def get_voice_activity(pitch_contour):
    """
    Returns voiced frames with beginning and end silences removed.

    Args:
        pitch_contour: time series of f0 returned from pitch extraction
            (ie. librosa.pyin)
    Returns:
        pitch_contour: with beginning and end silences of utterance truncated
    """
    df = pd.DataFrame(pitch_contour, dtype='float64')
    start_idx = df.first_valid_index()
    end_idx = df.last_valid_index()

    voiced = pitch_contour
    if not (start_idx is None and end_idx is None):
        voiced = pitch_contour[start_idx:end_idx + 1]

    return voiced

def get_nan_idx(arr: npt.NDArray[int]) -> Tuple[npt.NDArray[bool], Callable[[npt.NDArray[bool]], npt.NDArray[int]]]:
    """
    Helper to handle indices and logical indices of NaNs.

    Args:
        arr (ndArray): 1d array with possible NaNs
    Returns:
        nans (ndArray): logical indices of NaNs (an array of the same size as arr, 
            which marks the location of NaNs with 1, and all others with 0)
        index (lambda): a function that takes an array with 1s representing NaNs, 
            and returns the indexes of nans
    Example:
        >>> # linear interpolation of NaNs
        >>> arr = np.array([3,np.nan,4,np.nan,5)

        >>> nans, index= get_nan_idx(arr)
        >>> # nans would be [0, 1, 0, 1, 0]
        >>> # index(nans) would return [1, 3]
        >>> arr[nans]= np.interp(index(nans), index(~nans), arr[~nans])
    """
    nans = np.isnan(arr)
    index = lambda z: z.nonzero()[0] #the [0] is to extract from tuple
    return nans, index 

# https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
def interpolate_array(pitch_contour):
    """
    Interpolates over NaN values (currently, only linearly)

    Args:
        pitch_contour: 1d numpy array with possible NaNs in the middle
            must be dtype float, not object
    Returns:
        y: 1d numpy array
    """

    y= np.array(pitch_contour)

    #should run after voiced activity chops the ends off
    #so if there are nans at the ends, the entire file is nans
    #just return them for now, will be dropped by valid mask
    if not (np.isnan(y[0]) and np.isnan(y[-1])):
        nans, x = get_nan_idx(y)
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    
    return y

# https://stackoverflow.com/questions/55207719/cant-understand-the-working-of-uniform-filter1d-function-imported-from-scipy'''
# https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
def moving_average(signal, window_len: int = 5):
    """
    Uses convolution to get rolling average with a window of window_len frames.

    Args:
        signal (np.ndArray): time series to perform moving average on. most
            likely pitch contour, amplitude, etc.
        window_len (int): number of frames, 5 by default
    Returns:
        np.ndArray: times series that has been moving averaged
    """
    return uniform_filter1d(signal, size=window_len)


def normalize_pitch(pitch, max_f0, min_f0):
    """
    Normalize pitch values using typical method in literature.

    Args:
        pitch_contour: time series of f0 returned from pitch extraction
            (ie. librosa.pyin)
        max_f0: speaker's expected upper limit on pitch
        min_f0: speaker's expected lower limit on pitch
    Returns:
        normalized: speaker's pitch contour mapped to a five point scale
    """

    
    pitch_arr = np.log10(pitch)
    min_arr = np.full(pitch.shape, math.log(min_f0, 10))
    max_arr = np.full(pitch.shape, math.log(max_f0, 10))

    num = np.subtract(pitch_arr, min_arr)
    den = np.subtract(max_arr, min_arr)

    normalized = np.multiply(np.divide(num,den), 5)

    # Eqn. 7 from "A Comparison of Tone Normalization Methods..." by J. Zhang
    # normalized = (
    #     5
    #     * (math.log(pitch, 10) - math.log(min_f0, 10))
    #     / (math.log(max_f0, 10) - math.log(min_f0, 10))
    # )
    return normalized


# TODO: move this to speaker class?
def max_min_f0(pitch_contours):
    """
    Finds upper and lower limits on pitch for a speaker.

    Args:
        pitch_contours: List of pitch contours from a speaker.
    Returns:
        pitch_max: speaker's upper limit on pitch
        pitch_min: speaker's lower limit on pitch
    """
    flattened = pitch_contours
    if pitch_contours.ndim > 1:
        # turn into 1D array
        flattened = np.hstack(pitch_contours)
    pitch_max = np.nanpercentile(flattened, 95)
    pitch_min = np.nanpercentile(flattened, 5)
    return pitch_max, pitch_min


def extract_feature_vector(track, window_len):
    """Return feature vector of pitch values???"""

    feature_amplitude = extract_amplitude(track, window_len)
    feature_duration = extract_duration(track)
    # still need further development
    pass


def extract_amplitude(track, window_len):
    """
    Finds the amplitude values of an audio track.
    *currently uses max amplitude per window of calculation

    Using the word "frame" as typically used in MP3, and not in speech processing, as in
    frame = the audio at a single time stamp
    window = series of frames

    Args:
        track: audio time series (ie. y returned by librosa.load(), or pitch_contour)
        window_len: number of frames in a window used to calculate amplitude value
    Returns:
        AE: time series of amplitude values
    """
    AE = []
    # Calculate number of windows
    num_window = math.floor(len(track) / window_len)
    for frame in range(num_window):
        # Calculate bounds of each window
        # By doing this, our hop length is the same as the window length
        # Therefore, these windows are NOT overlapping.
        lower = frame * window_len
        upper = (frame + 1) * (window_len) - 1
        # Find maximum of each window and add it to our array
        AE.append(np.max(track[lower:upper]))
    return np.array(AE)


def extract_duration(track):
    duration = librosa.get_duration(track)  # TODO: make sure this y is defined
    print(duration)
    # still need to do further extraction
    pass


def filter_noise(track):
    """
    Filter noise from a signal.

    Args:
        track: audio time series to be filtered.
    Returns:
    """

    # TODO: if we are only using this on pitch_contour, then just pass in pitch_contour
    # if using it on signals in general, then don't need this part)
    # basically, should just directly pass in whatever you want to filter
    # so the function is more general purpose

    # getting pitch_contour estimation
    pitch_contour, voiced_flag, voiced_probs = librosa.pyin(
        track, fmin=50, fmax=300
    )
    plt.plot(pitch_contour)
    plt.show()

    # exploring butterworth filter
    sos = signal.butter(10, 300, "low", fs=1000, output="sos")
    filtered = signal.sosfilt(sos, track)
    plt.plot(filtered)
    plt.show()

    pitch_contour_fl, voiced_flag_fl, voiced_probs_fl = librosa.pyin(
        filtered, fmin=50, fmax=300
    )
    plt.plot(pitch_contour_fl)
    plt.show()

    # still need further development on filtering
    pass


def median_filter(track):
    """
    Filters signal by finding the median over a window.

    Args:
        track: audio time series to be filtered.
    Returns:
    """
    pass


# https://stackoverflow.com/questions/40569220/efficiently-convert-uneven-list-of-lists-to-minimal-containing-array-padded-with
def pad_matrix(v, fillval=np.nan):
    """Takes an irregular matrix and pads out each row to be equal in length.

    Args:
        v (np.array, list): a 2D matrix with rows of unequal length
        fillvall: value to insert when padding

    Returns:
        np.array: a matrix with the same number of elements in each row
    """
    lens = np.array([len(item) for item in v])
    mask = lens[:, None] > np.arange(lens.max())
    out = np.full(mask.shape, fillval)
    out[mask] = np.concatenate(v)
    return out

def preprocess(pitch_contour):
    # truncated, but irregular
    voiced = get_voice_activity(pitch_contour)
    #TODO: interp pathway
    cast_arr = np.array(voiced, dtype=float)
    nans, idx = get_nan_idx(cast_arr)
    interp = interpolate_array(cast_arr)
        # cast_arr = np.array(voiced, dtype=float)
    return interp, nans

def preprocess_all(data):
    # pitch_data = pd.read_json(PITCH_FILEPATH)
    tone = data.loc[:, 'pitch_contour'].to_numpy()
    label = data.loc[:, 'tone'].to_numpy()
    # print(tone1.to_numpy())

    truncated = []
    for i in range(tone.shape[0]):
        pitch_contour, _ = preprocess(tone[i])
        truncated.append(pitch_contour)

    truncated_np = np.array(truncated, dtype=object)
    # drop all the nan rows - still irregular
    # if interp runs, should only be dropping samples that are all nans
    valid_mask = get_valid_mask(truncated_np)
    data_valid = truncated_np[valid_mask]
    label_valid = label[valid_mask]

    return label_valid, data_valid

def end_to_end(data):
    # 1. read in the data
    # 2. stick into feature extraction of choice
    # 3. classify
    # 4. profit

    label_valid, data_valid = preprocess_all(data)

    features = basic_feature_extraction(data_valid)

    #TODO: JANKY ASS FILLER, NEEDS TO BE CHANGED LOL
    # tone1_counter = 0
    # for i in range(features.shape[0]):
        # TONE 1
        # if diffs aren't really that big == tone 1 babey
        # if features[i][3] <= 0.15 and features[i][4] <= 0.15 and features[i][5] <= 0.15:
        #     tone1_counter += 1
    
    # print(f'Total: {features.shape[0]}')
    # print(f'Correct: {tone1_counter}')
    # print(f'Wrong: {features.shape[0] - tone1_counter}')

    return label_valid, features

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

def svm_ml_times(filename='confusion.jpg'):
    import sklearn.pipeline
    pitch_data = pd.read_json(PITCH_FILEPATH)

    # ALL THE FEMALE TONE PERFECT FILES
    # pitch_data = pitch_data.loc[pitch_data['speaker'].isin(['FV1', 'FV2', 'FV3'])]
    # pitch_data = pitch_data.loc[pitch_data['speaker'].isin(['MV1'])]
    label, data = end_to_end(pitch_data)
    
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
# y is the amplitude of the waveform, sr is the sampling rate
# y, sr = librosa.load('data/pronunciation_zh_嚎.mp3')
# feature_vector = extract_feature_vector(y, 1024)
# filter_noises(y)

# %%

# TODO: move this to jupyter notebook

# y1_f1, sr1_f1 = librosa.load("tone_perfect_all_mp3/a1_FV1_MP3.mp3")
# y2_f1, sr2_f1 = librosa.load("tone_perfect_all_mp3/a2_FV1_MP3.mp3")
# y3_f1, sr3_f1 = librosa.load("tone_perfect_all_mp3/a3_FV1_MP3.mp3")
# y4_f1, sr4_f1 = librosa.load("tone_perfect_all_mp3/a4_FV1_MP3.mp3")
# plt.plot(y1_f1)
# plt.show()
# plt.plot(y2_f1)
# plt.show()
# plt.plot(y3_f1)
# plt.show()
# plt.plot(y4_f1)
# plt.show()

# y1_f2, sr1_f2 = librosa.load("tone_perfect_all_mp3/a1_FV2_MP3.mp3")
# plt.plot(y1_f1)
# plt.plot(y1_f2)
# plt.show()

# y1_f3, sr1_f3 = librosa.load("tone_perfect_all_mp3/a1_FV3_MP3.mp3")
# plt.plot(y1_f1)
# plt.plot(y1_f2)
# plt.plot(y1_f3)
# plt.show()
# # %%
# amp_1024 = extract_amplitude(y1_f1, 1024)
# amp_700 = extract_amplitude(y1_f1, 700)
# amp_300 = extract_amplitude(y1_f1, 300)
# plt.plot(y1_f1)
# plt.show()

# plt.plot(amp_1024)
# plt.show()
# plt.plot(amp_700)
# plt.show()
# plt.plot(amp_300)
# plt.show()
# # %%
# amp_1024 = extract_amplitude(y3_f1, 1024)
# amp_700 = extract_amplitude(y3_f1, 700)
# amp_300 = extract_amplitude(y3_f1, 300)
# plt.plot(y3_f1)
# plt.show()

# plt.plot(amp_1024)
# plt.show()
# plt.plot(amp_700)
# plt.show()
# plt.plot(amp_300)
# plt.show()

