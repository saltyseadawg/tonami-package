import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tonami import pitch_process as pp

# helper/utility functions specifically for plotting

PITCH_FILEPATH = 'data/parsed/toneperfect_pitch_librosa_50-500-fminmax.json'

# currently, filename specifications only work for the above PITCH_FILEPATH
def plot_tones(filename='wo3_MV2_MP3.mp3'):
    """
    Plots the specified tones.

    Args:
        filename: filename of tone to be plotted
    """
    pitch_data = pd.read_json(PITCH_FILEPATH)
    pitch_data = pitch_data.loc[pitch_data['filename'].isin([filename])]

    #plot before and after interp
    pitch_contour = pitch_data.loc[:, 'pitch_contour'].to_numpy()
    plt.plot(pitch_contour[0])
    plt.savefig('tone_' + filename + ".jpg")

# could probably be combined with above at some point
def visualize_interp(filename='wo3_MV2_MP3.mp3'):
    """
    Plots the specified tones, and a version that has been interpolated

    Args:
        filename: filename of tone to be plotted
    """
    pitch_data = pd.read_json(PITCH_FILEPATH)
    pitch_data = pitch_data.loc[pitch_data['filename'].isin([filename])]

    #plot before and after interp
    pitch_contour = pitch_data.loc[:, 'pitch_contour'].to_numpy()
    interp_contour = pp.interpolate_array(np.array(pitch_contour[0], dtype=float))

    plt.plot(pitch_contour[0])
    plt.plot(interp_contour, linestyle=":")
    plt.savefig('interp_' + filename + ".jpg")

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
