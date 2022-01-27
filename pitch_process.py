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
    return f0[voiced_flag]

def extract_feature_vector(pitch_values):
    """Return feature vector of pitch values???"""
    pass

