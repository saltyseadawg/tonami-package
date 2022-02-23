import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tonami import pitch_process as pp
from tonami import Utterance as u
# functions that are called by the front-end to get plots, rating feedback, etc.

PITCH_FILEPATH = 'data/parsed/toneperfect_pitch_librosa_50-500-fminmax.json'

def load_exercise(filename: str = 'wo3_MV2_MP3.mp3') -> None:
    """
    Takes the filename of the desired native speaker sample,
    and plots its pitch contour as a line graph.

    Args:
        filename (str): filename of tone to be plotted
    """
    word = u.Utterance(filename = filename)
    pitch_contour, nans = word.pre_process()

    plt.figure()
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

    plt.plot(y_interp, color='orange', linestyle=":", linewidth=2)
    plt.plot(y_pitch, color='orange', linewidth=3)
    plt.show()
    #TODO: take this out when we ain't just testing
    plt.savefig('exercise_' + filename + ".jpg")