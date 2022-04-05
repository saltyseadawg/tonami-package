import os

import matplotlib.pyplot as plt
import numpy.typing as npt
import numpy as np
import pandas as pd

from tonami import Utterance as u
from tonami import user
from tonami import Classifier as c
# functions that are called by the front-end to get plots, rating feedback, etc.

PITCH_FILEPATH = 'data/parsed/toneperfect_pitch_librosa_50-500-fminmax.json'
SPEAKER_INFO_FILEPATH = 'tonami/data/speaker_max_min.txt'

def load_exercise(filename: str = 'wo3_MV2_MP3.mp3'):
    """
    Takes the filename of the desired native speaker sample,
    and plots its pitch contour as a line graph.

    Args:
        filename (str): filename of tone to be plotted
    """

    sections = filename.split("_")
    speaker = sections[1]
    speakers_info = pd.read_json(SPEAKER_INFO_FILEPATH)
    speaker_max_f0 = speakers_info.loc[speakers_info['speaker_name'] == speaker, 'max_f0']
    speaker_min_f0 = speakers_info.loc[speakers_info['speaker_name'] == speaker, 'min_f0']
    speaker_info = user.User(speaker_max_f0, speaker_min_f0)

    word = u.Utterance(filename = filename, pitch_filepath=PITCH_FILEPATH)
    pitch_contour, nans, _ = word.pre_process(speaker_info)
    pitch_contour = pitch_contour[0]

    fig, ax = plt.subplots()
    # plt.xlabel("Time (frames)")
    # plt.ylabel("Frequency (Hz)")
    # plt.title(filename + " Pitch Contour")
    plt.xticks([])
    plt.yticks([])

    y_pitch = pitch_contour.copy()
    y_interp = pitch_contour.copy()
    y_pitch[nans] = np.nan
    #some gaps if we do this, since it isn't automatically drawing between those points
    # y_interp[~nans] = np.nan 

    ax.plot(y_interp, color='orange', linestyle=":", linewidth=2)
    ax.plot(y_pitch, color='orange', linewidth=3)
    # plt.show()
    #TODO: take this out when we ain't just testing
    plt.savefig('exercise_' + filename + ".jpg")

    return fig

def process_user_audio (figure, user_info: dict[user.User], filename:str, classifier:c.Classifier, tone: int=None, db_threshold=10, trim=True):
    """
    Takes the user's info, user's track and the desired tone/word

    Args:
        figure (matplotlib.figure.Figure): native speaker's figure to use as a base plot
        user_info (Class User): user information to obtain f0 min and max values
        track (np.array, 1D): audio time series to be filtered
        tone (int): integer tone value (i.e. 1, 2, 3 or 4)
    Returns:
        user_pitch_contour (np.array, 1D): processed user's audio pitch contour
        classified_tone (np.array, 1D): tone classification result from the classifier model
    """

    user_utterance = u.Utterance(filename=filename, db_threshold=db_threshold, trim=trim)
    user_pitch_contour, user_nans, features = user_utterance.pre_process(user_info)

    if not np.isnan(features).any():
        classified_tones, classified_probs = classifier.classify_tones(features)
    else:
        classified_tones = 0
        classified_probs = [[12]]

    # use the same axis as the native speaker's pitch contour plot
    ax = figure.axes[0]
    user_pitch_contour = user_pitch_contour[0]
    y_pitch = user_pitch_contour.copy()
    y_interp = user_pitch_contour.copy()
    y_pitch[user_nans] = np.nan
    
    ax = figure.axes[0]
    ax.plot(y_interp, color='blue', linestyle=":", linewidth=2)
    ax.plot(y_pitch, color='blue', linewidth=3)

    return figure, classified_tones, classified_probs
