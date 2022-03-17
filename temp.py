from tonami import controller as cont
from tonami import user
from tonami import Utterance as u
from heroku.interface_utils import *
from tonami import Classifier as c
import json
import pandas as pd
import librosa
# from dev import parse_data
# import matplotlib.pyplot as plt
# import time
# from dev import parse_data

with open('heroku/interface_text.json') as json_file:
    text = json.load(json_file)

def get_rating_for_this_audio(text, filename, tone, speaker):

    if speaker = "LV":
        # male
        f_min = 50
        f_max = 200
    elif speaker = "HV":
        # female
        f_min = 100
        f_max = 350
    else:
        print("input either MV or FV for speaker argument")
        return    

    time_series, _ = librosa.load(filename)
    user_utterance = u.Utterance(track=time_series, sr=None, pitch_floor=f_min, pitch_ceil=f_max)
    user_info = user.User(f_max, f_min)
    user_pitch_contour, user_nans, features = user_utterance.pre_process(user_info)
    # print(features)
    clf = c.Classifier(4)
    clf.load_clf('tonami/data/pickled_svm_80.pkl') 
    classified_tones, clf_probs = clf.classify_tones(features)
    rating = get_rating(text["ratings"], tone, clf_probs)
    print(rating)
    pass

# filename = "data/tone_perfect/wan2_MV3_MP3.mp3"
filename = "data/users/fa4_LV1_ex4_R1_user-testing.mp3"
get_rating_for_this_audio(text, filename, 4, 'LV')
