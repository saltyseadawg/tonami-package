import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# helper/utility functions specifically for plotting


def plot_tones(tones):
    """
    Plots the list of tones.

    Args:
        tones: a list/array of tones to be plotted.
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
