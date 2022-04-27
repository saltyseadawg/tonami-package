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

from tonami import pitch_process as pp

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
    padded = np.array(pp.pad_matrix(contours, fillval='420.69'), dtype=float)
    valid_row_mask = ~np.isnan(padded).any(axis=1)
    return valid_row_mask

def preprocess_all(data: pd.DataFrame, n_segments) -> Tuple[npt.NDArray[int], npt.NDArray]:
    """
    Batch preprocessing of raw pitch contours with tone category information.

    Args:
        data (pd.DataFrame): dataframe of raw pitch contours with tone label
        n_segments: number of segments

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
        pitch_contour, _ = pp.preprocess(tone[i], n_segments)
        truncated.append(pitch_contour)

    truncated_np = np.array(truncated, dtype=object)
    # drop all the nan rows - still irregular
    # if interp runs, should only be dropping samples that are all nans
    valid_mask = get_valid_mask(truncated_np)
    data_valid = truncated_np[valid_mask]
    label_valid = label[valid_mask]

    return label_valid, data_valid

def end_to_end(data: pd.DataFrame, n_segments) -> Tuple[npt.NDArray[int], npt.NDArray]:
    """
    Batch processes pitch contours with tone categories for training classifiers.

    Args:
        data (pd.DataFrame): dataframe of raw pitch contours with tone label
        n_segments: number of segments
    
    Returns:
        np.array: 1D array of tone labels
        np.array: Nx(n_segments*(n_segments+1)/2) array of feature vectors for tone classification
    """
    # 1. read in the data
    # 2. stick into feature extraction of choice
    # 3. classify
    # 4. profit

    label_valid, data_valid = preprocess_all(data, n_segments)

    features = pp.basic_feature_extraction(data_valid, n_segments)

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