# the line above is for jupyter notebook extension on VS code
# TODO: in final, comment out matplotlib - we want to visualize everything in visualization module
import warnings
import math
from typing import Tuple, Callable

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.signal as signal
import pandas as pd
from scipy.ndimage.filters import uniform_filter1d



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

def basic_feature_extraction(pitch_contours: npt.NDArray[float]) -> npt.NDArray:
    """
    Batch feature extraction for multiple pitch contours after performing
    additional processing steps in the following order:
        - smoothing by using a moving average of 5 frames
        - normalizing the pitch contour onto a 5 point scale

    Returns:
        np.array: Nx6 array of feature vectors for tone classification
    """
    features = np.empty((pitch_contours.shape[0],6))
    # calcualte features - not irregular <3
    pitch_max, pitch_min = max_min_f0(pitch_contours)
    for i in range(pitch_contours.shape[0]):
        # normalizing and sh*t
        avgd = moving_average(pitch_contours[i])
        normalized = normalize_pitch(avgd, pitch_max, pitch_min)

        features[i] = basic_feat_calc(normalized)

    return features
    
def basic_feat_calc(pitch_contour: npt.NDArray[float]) -> npt.NDArray:
    """
    Calculates the features from a single pitch contour. Refer to Liao et al.(2010)
    paper for feature details:

    https://www.isca-speech.org/archive_v0/archive_papers/interspeech_2010/i10_0602.pdf


    Args:
        pitch_contour (np.array): pitch values from a single utterance
        
    Returns:
        np.array: 1D feature vector for tone classification 
    """
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
def get_valid_mask(contours: npt.NDArray[float]) -> npt.NDArray[bool]:
    """
    Returns a boolean array of invalid pitch contours. Invalid contours only 
    have NaN values - no pitch was detected.

    Args: 
        contours (nd.array): Nx? array of pitch contours; may be irregular

    Returns:
        np.array: 1D length N boolean mask. False at invalid contour index. 
    """
    # get rid of the rows with nans in the middle goddmanit
    # 1. pad the goddamn array to do black magic (np only plays nice with array that are NOT irregular)
    # 2. get the indices of pitch contours with nans in the middle
    # 3. mask that shit
    # 4. profit
    padded = np.array(pad_matrix(contours, fillval='420.69'), dtype=float)
    valid_row_mask = ~np.isnan(padded).any(axis=1)
    return valid_row_mask

def get_voice_activity(pitch_contour: npt.NDArray[float]) -> npt.NDArray[float]:
    """
    Returns voiced frames with beginning and end silences removed.

    Args:
        pitch_contour (nd.array): time series of f0 returned from pitch extraction
            (ie. librosa.pyin)
    Returns:
        nd.array: pitch contour with beginning and end silences of utterance truncated
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
def moving_average(signal: npt.NDArray, window_len: int = 5) -> npt.NDArray:
    """
    Uses convolution to get rolling average with a window of window_len frames.

    Args:
        signal (np.array): time series to perform moving average on. most
            likely pitch contour, amplitude, etc.
        window_len (int): number of frames, 5 by default
    Returns:
        np.array: times series that has been moving averaged
    """
    return uniform_filter1d(signal, size=window_len)


def normalize_pitch(pitch: npt.NDArray[float], max_f0: float, min_f0: float) -> npt.NDArray[float]:
    """ 
    Normalize pitch values using typical method in literature.

    Args:
        pitch_contour (np.array): time series of f0 returned from pitch extraction
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
def max_min_f0(pitch_contours: npt.NDArray[float]):
    """
    Finds upper and lower limits on pitch for a speaker.

    Args:
        pitch_contours (np.array): List of pitch contours from a speaker.
    Returns:
        float: speaker's upper limit on pitch
        float : speaker's lower limit on pitch
    """
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
    if (v.ndim == 1) and (v.dtype != 'O') :
        return v
    else:
        lens = np.array([len(item) for item in v])
        mask = lens[:, None] > np.arange(lens.max())
        out = np.full(mask.shape, fillval)
        out[mask] = np.concatenate(v)
        return out

def preprocess(pitch_contour: npt.NDArray[float]) -> Tuple[npt.NDArray[float], npt.NDArray[bool]]:
    """
    Preprocesses a single raw pitch contour by:
        - truncating beginning and end silences
        - interpolating NaNs that occur in the middle of the utterance

    Args:
        pitch_contour (np.array): f0 values for a single pitch contour

    Returns:
        np.array: 1D array of a pitch contour
        np.array: logical indices of NaN values in the original raw contour

    """
    # truncated, but irregular
    voiced = get_voice_activity(pitch_contour)
    #TODO: interp pathway
    cast_arr = np.array(voiced, dtype=float)
    nans, idx = get_nan_idx(cast_arr)
    interp = interpolate_array(cast_arr)
    # cast_arr = np.array(voiced, dtype=float)
    return interp, nans

def preprocess_all(data: pd.DataFrame) -> Tuple[npt.NDArray[int], npt.NDArray]:
    """
    Batch preprocessing of raw pitch contours with tone category information.

    Args:
        data (pd.DataFrame): dataframe of raw pitch contours with tone label

    Returns:
        np.array: valid and preprocessed pitch contours
        np.array: 1D array of tone labels
    """
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

def end_to_end(data: pd.DataFrame) -> Tuple[npt.NDArray[int], npt.NDArray]:
    """
    Batch processes pitch contours with tone categories for training classifiers.

    Args:
        data (pd.DataFrame): dataframe of raw pitch contours with tone label
    
    Returns:
        np.array: 1D array of tone labels
        np.array: Nx6 array of feature vectors for tone classification
    """
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
