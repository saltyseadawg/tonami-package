import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tonami import pitch_process as pp

# helper/utility functions specifically for plotting

PITCH_FILEPATH = 'data/parsed/toneperfect_pitch_librosa_50-500-fminmax.json'

# currently, filename specifications only work for the above PITCH_FILEPATH
def plot_pitch_contour(filename: str = 'wo3_MV2_MP3.mp3') -> None:
    """
    Plots the pitch contour of the selected file, as a line graph.

    Args:
        filename (str): filename of tone to be plotted
    """
    pitch_data = pd.read_json(PITCH_FILEPATH)
    pitch_data = pitch_data.loc[pitch_data['filename'].isin([filename])]

    plt.figure()
    plt.xlabel("Time (frames)")
    plt.ylabel("Frequency (Hz)")
    plt.title(filename + " Pitch Contour")

    #plot before and after interp
    pitch_contour = pitch_data.loc[:, 'pitch_contour'].to_numpy()
    plt.plot(pitch_contour[0])
    plt.savefig('tone_' + filename + ".jpg")

# could probably be combined with above at some point
def plot_interp(filename: str = 'wo3_MV2_MP3.mp3') -> None:
    """
    Plots the unprocessed pitch contour of the selected file,
    and then a version that has undergone linear interpolation to replace NaNs
    as an overlaid dotted line.

    If the pitch contour is all NaNs, the plot will be blank.

    Args:
        filename (str): filename of tone to be plotted
    """

    pitch_data = pd.read_json(PITCH_FILEPATH)
    pitch_data = pitch_data.loc[pitch_data['filename'].isin([filename])]

    #plot before and after interp
    pitch_contour = pitch_data.loc[:, 'pitch_contour'].to_numpy()
    interp_contour = pp.interpolate_array(np.array(pitch_contour[0], dtype=float))

    plt.figure()
    plt.xlabel("Time (frames)")
    plt.ylabel("Frequency (Hz)")
    plt.title(filename + " Interpolated")

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
