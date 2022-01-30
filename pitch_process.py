#%%
# the line above is for jupyter notebook extension on VS code
import warnings
from curses import window
from encodings import normalize_encoding
from locale import normalize
import math
from cmath import log
from random import uniform

import librosa
import librosa.display
# import matplotlib
# matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import uniform_filter1d
import scipy.signal as signal

import utils

warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")

# https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
def moving_average(values, window_length: int=5):
    """Use convolution to get rolling average???"""    
    return uniform_filter1d(values, size=window_length)


def normalize_pitch(pitch_values, max_f0, min_f0):
    """Normalize pitch values using typical equation/method in literature?"""
    normalized = []
    for p in pitch_values:
        normalized.append(5 * (math.log(p, 10) - math.log(min_f0, 10)) / (math.log(max_f0, 10) - math.log(min_f0, 10)))
    return normalized

def max_min_f0(pitch_tracks):
    # fudge this for now until we get the speaker data hooked up
    max_f0, min_f0 = 0, 1000
    for track in pitch_tracks:
        max_f0 = max(max_f0, max(track))
        min_f0 = min(min_f0, min(track))
    return max_f0, min_f0

def voice_activity(f0, voiced_flag):
    """Return voiced frames. Deal with creaky voice somehow."""
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

    return f0[voiced_flag]

def extract_feature_vector(amplitude):
    """Return feature vector of pitch values???"""
    
    feature_amplitude = extract_amplitude(amplitude)
    feature_duration = extract_duration(amplitude)
    # still need further development
    pass

def extract_amplitude (amplitude):
    plt.plot(y)
    plt.show()
    # still need to do further extraction
    pass

def extract_duration (amplitude):
    duration = librosa.get_duration(y)
    print(duration)
    # still need to do further extraction
    pass

def filter_noises (amplitude):
    # getting f0 estimation
    f0, voiced_flag, voiced_probs = librosa.pyin(amplitude, fmin=50, fmax=300)
    plt.plot(f0)
    plt.show()

    # exploring butterworth filter
    sos = signal.butter(10, 300, 'low', fs=1000, output='sos')
    filtered = signal.sosfilt(sos, amplitude)
    plt.plot(filtered)
    plt.show()

    f0_f, voiced_flag_f, voiced_probs_f = librosa.pyin(filtered, fmin=50, fmax=300)
    plt.plot(f0_f)
    plt.show()

    # still need further development on filtering
    pass

# y is the amplitude of the waveform, sr is the sampling rate
y, sr = librosa.load('data/pronunciation_zh_嚎.mp3')
feature_vector = extract_feature_vector(y)
filter_noises(y)
