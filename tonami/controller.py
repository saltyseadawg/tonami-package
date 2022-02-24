import matplotlib.pyplot as plt
import numpy.typing as npt
import numpy as np
import pandas as pd

# TODO: comment this out before commit this
import librosa
import pitch_process as pp
import Utterance as u
import user
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
    pitch_contour, nans, _ = word.pre_process()

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

def send_data_to_frontend(user_info: dict[user.User], track: npt.NDArray[float]=None, tone: int=None):
    """
    Takes the user's info, user's trial, native speaker's plot and the desired tone/word

    Args:
        user_recording (str): filename of tone to be plotted
    """

    user_utterance = u.Utterance(track)
    user_pitch_contour, user_nans, features = user_utterance.pre_process(user_info)
    
    # TODO: move this to Classifier.py
    # prediction = clf.predict(features)

    # plt.plot(plot_native_speaker)
    # plt.plot(user_pitch_contour)
    # plt.show()
    # plt.savefig('../../../compare_tone_' + tone + ".jpg")

    # use user_pitch_contour to plot graphs in frontend
    return user_pitch_contour #, prediction

y1_f1, sr1_f1 = librosa.load("../data/tone_perfect/a1_FV1_MP3.mp3")
fig, ax = plt.subplots()
ax.plot(y1_f1)
test = send_data_to_frontend(user.User(500, 50), y1_f1, 1)