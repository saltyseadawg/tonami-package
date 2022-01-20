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

# pyin - F0 estimate
# filename = 'data/Zh-dàn.ogg.mp3'
# y, sr = librosa.load(f'{filename}')
# f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
# times = librosa.times_like(f0)

# D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
# fig, ax = plt.subplots()
# img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
# ax.set(title='pYIN fundamental frequency estimation')
# fig.colorbar(img, ax=ax, format="%+2.f dB")
# ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
# ax.legend(loc='upper right')
# plt.savefig('tone4.png')
# print(times)
# print(np.nanmean(f0))

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

def get_max_f0(pitch_values):
    return max(pitch_values)

def get_min_f0(pitch_values):
    return min(pitch_values)

def voice_activity(f0, voiced_flag):
    """Return voiced frames. Deal with creaky voice somehow."""
    # hacky method for now
    # cannot deal with creaky voice atm
    return f0[voiced_flag]

# def main():
# pyin - F0 estimate
filename = 'data/Zh-tā.ogg.mp3'
y, sr = librosa.load(filename)
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
# print(f0)
# print(voiced_flag)
# print(voiced_probs)
# times = librosa.times_like(f0)
# handle creaky voice somewhere here
smoothed_f0 = moving_average(voice_activity(f0, voiced_flag))
# smoothed_f0 = voice_activity(f0, voiced_flag)
max_f0 = get_max_f0(smoothed_f0)
min_f0 = get_min_f0(smoothed_f0)
normalized_values = normalize_pitch(smoothed_f0, max_f0, min_f0)
print(normalized_values)
# print(smoothed_f0)
plt.plot(normalized_values)
plt.savefig('normalize5.png')
