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

warnings.filterwarnings(
    "ignore", message="PySoundFile failed. Trying audioread instead."
)

# https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
def moving_average(values, window_length: int = 5):
    """Use convolution to get rolling average???"""
    return uniform_filter1d(values, size=window_length)


def normalize_pitch(pitch_values, max_f0, min_f0):
    """Normalize pitch values using typical equation/method in literature?"""
    normalized = []
    for p in pitch_values:
        normalized.append(
            5
            * (math.log(p, 10) - math.log(min_f0, 10))
            / (math.log(max_f0, 10) - math.log(min_f0, 10))
        )
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


def extract_feature_vector(amplitude, frame_length):
    """Return feature vector of pitch values???"""

    feature_amplitude = extract_amplitude(amplitude, frame_length)
    feature_duration = extract_duration(amplitude)
    # still need further development
    pass


def extract_amplitude(amplitude, frame_length):
    AE = []
    # Calculate number of frames
    num_frames = math.floor(len(amplitude) / frame_length)
    for t in range(num_frames):
        # Calculate bounds of each frame
        # By doing this, our hop length is the same as the frame length
        # Therefore, these frames are NOT overlapping.
        lower = t * frame_length
        upper = (t + 1) * (frame_length) - 1
        # Find maximum of each frame and add it to our array
        AE.append(np.max(amplitude[lower:upper]))
    return np.array(AE)


def extract_duration(amplitude):
    duration = librosa.get_duration(y)
    print(duration)
    # still need to do further extraction
    pass


def filter_noises(amplitude):
    # getting f0 estimation
    f0, voiced_flag, voiced_probs = librosa.pyin(amplitude, fmin=50, fmax=300)
    plt.plot(f0)
    plt.show()

    # exploring butterworth filter
    sos = signal.butter(10, 300, "low", fs=1000, output="sos")
    filtered = signal.sosfilt(sos, amplitude)
    plt.plot(filtered)
    plt.show()

    f0_f, voiced_flag_f, voiced_probs_f = librosa.pyin(
        filtered, fmin=50, fmax=300
    )
    plt.plot(f0_f)
    plt.show()

    # still need further development on filtering
    pass


# y is the amplitude of the waveform, sr is the sampling rate
# y, sr = librosa.load('data/pronunciation_zh_嚎.mp3')
# feature_vector = extract_feature_vector(y, 1024)
# filter_noises(y)

# %%
y1_f1, sr1_f1 = librosa.load("tone_perfect_all_mp3/a1_FV1_MP3.mp3")
y2_f1, sr2_f1 = librosa.load("tone_perfect_all_mp3/a2_FV1_MP3.mp3")
y3_f1, sr3_f1 = librosa.load("tone_perfect_all_mp3/a3_FV1_MP3.mp3")
y4_f1, sr4_f1 = librosa.load("tone_perfect_all_mp3/a4_FV1_MP3.mp3")
plt.plot(y1_f1)
plt.show()
plt.plot(y2_f1)
plt.show()
plt.plot(y3_f1)
plt.show()
plt.plot(y4_f1)
plt.show()

y1_f2, sr1_f2 = librosa.load("tone_perfect_all_mp3/a1_FV2_MP3.mp3")
plt.plot(y1_f1)
plt.plot(y1_f2)
plt.show()

y1_f3, sr1_f3 = librosa.load("tone_perfect_all_mp3/a1_FV3_MP3.mp3")
plt.plot(y1_f1)
plt.plot(y1_f2)
plt.plot(y1_f3)
plt.show()
# %%
amp_1024 = extract_amplitude(y1_f1, 1024)
amp_700 = extract_amplitude(y1_f1, 700)
amp_300 = extract_amplitude(y1_f1, 300)
plt.plot(y1_f1)
plt.show()

plt.plot(amp_1024)
plt.show()
plt.plot(amp_700)
plt.show()
plt.plot(amp_300)
plt.show()
# %%
amp_1024 = extract_amplitude(y3_f1, 1024)
amp_700 = extract_amplitude(y3_f1, 700)
amp_300 = extract_amplitude(y3_f1, 300)
plt.plot(y3_f1)
plt.show()

plt.plot(amp_1024)
plt.show()
plt.plot(amp_700)
plt.show()
plt.plot(amp_300)
plt.show()
