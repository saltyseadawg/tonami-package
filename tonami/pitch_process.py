# the line above is for jupyter notebook extension on VS code
# TODO: in final, comment out matplotlib - we want to visualize everything in visualization module
import warnings
import math

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import uniform_filter1d
import scipy.signal as signal

# known error of package, we intend to use audioread.
warnings.filterwarnings(
    "ignore", message="PySoundFile failed. Trying audioread instead."
)


def get_voice_activity(pitch_contour, voiced_flag):
    """
    Returns voiced frames with beginning and end silences removed.

    Args:
        pitch_contour: time series of f0 returned from pitch extraction 
            (ie. librosa.pyin)
        voiced_flag: time series of bools indicating voiced activity
    Returns:
        pitch_contour: with beginning and end silences of utterance truncated
    """
    # hacky method for now
    # cannot deal with creaky voice atm
    # beginning of list
    start_voiced = 0
    while not voiced_flag[start_voiced]:
        start_voiced += 1

    end_voiced = len(voiced_flag) - 1
    while not voiced_flag[end_voiced]:
        end_voiced -= 1
    voiced_flag[start_voiced:end_voiced] = True

    # in numpy you can filter an array using a boolean index list
    # https://www.w3schools.com/python/numpy/numpy_array_filter.asp#:~:text=In%20NumPy%2C%20you%20filter%20an,excluded%20from%20the%20filtered%20array.
    return pitch_contour[voiced_flag]


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


def normalize_pitch(pitch_values, max_f0, min_f0):
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

    #Eqn. 7 from "A Comparison of Tone Normalization Methods..." by J. Zhang
    normalized = []
    for p in pitch_values:
        normalized.append(
            5
            * (math.log(p, 10) - math.log(min_f0, 10))
            / (math.log(max_f0, 10) - math.log(min_f0, 10))
        )
    return normalized


# TODO: move this to speaker class?
def max_min_f0(pitch_contours):
    """
    Finds upper and lower limits on pitch for a speaker.

    Args:
        pitch_contours: List of pitch contours from a speaker.
    Returns:
        max_f0: speaker's upper limit on pitch
        min_f0: speaker's lower limit on pitch
    """
    # fudge this for now until we get the speaker data hooked up
    max_f0, min_f0 = 0, 1000
    for contour in pitch_contours:
        max_f0 = max(max_f0, max(contour))
        min_f0 = min(min_f0, min(contour))
    return max_f0, min_f0


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


# y is the amplitude of the waveform, sr is the sampling rate
# y, sr = librosa.load('data/pronunciation_zh_嚎.mp3')
# feature_vector = extract_feature_vector(y, 1024)
# filter_noises(y)

# %%

#TODO: move this to jupyter notebook

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
