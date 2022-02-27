import collections
import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import sklearn.pipeline
from pydub import AudioSegment
import librosa
import io
import matplotlib.pyplot as plt

from heroku import audio_btn
from tonami import Utterance as utt
from tonami import pitch_process as pp
from tonami import user as usr
from tonami import controller as cont
from heroku.interface_utils import *

import io
import json
from datetime import datetime

import pymongo
from tonami.audio_utils import convert_audio


if not hasattr(st, "client"):
    st.client = pymongo.MongoClient(**st.secrets["mongo"])
    st.collection = st.client.audio_files.user_test

st.set_page_config( "Tonami", "🌊", "centered", "collapsed" )

if 'key' not in st.session_state:
    st.session_state.key = 0
if 'user_audio' not in st.session_state:
    st.session_state.user_audio = None
if 'user' not in st.session_state:
    st.session_state.user = None

st.session_state.user_audio = None

# Opening JSON file
with open('heroku/interface_text.json') as json_file:
    text = json.load(json_file)

exercises = text["exercises"]
last_page = len(exercises) + 2

st.write(text['title'])

if st.session_state.key == 0:
    st.write(text['instructions'])
elif st.session_state.key == 1:
    st.write(text['calibration'])
    audio_btn.audio_btn()

    if st.session_state.user_audio is not None:
        #TODO: extract pitch max/min
        calibrate_utt = utt.Utterance(track=st.session_state.user_audio)
        st.session_state.user = usr.User(calibrate_utt.fmax, calibrate_utt.fmin)
        p = st.session_state.user.pitch_profile
        st.write(p["max_f0"], p["min_f0"])

elif st.session_state.key == last_page:
    st.write(text['end_page'])

else:
    exercise = exercises[st.session_state.key - 2]
    st.write("Test ", str(st.session_state.key - 1))
    st.write("## ", exercise["character"], " ", exercise["pinyin"], " ", str(exercise["tone"]))
    st.write(exercise["translation"])
    
    # TODO: need to double check this audio file path
    with open("data/tone_perfect/" + exercise["fileName"] + ".mp3", 'rb') as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format='audio/mp3')
    
    ns_figure = cont.load_exercise(exercise["fileName"] + ".mp3")
    st.session_state.ns_figure = ns_figure

    audio_btn.audio_btn()

    if st.session_state.user_audio is not None:
        user_bytes = convert_audio(st.session_state.user_audio, 'wav').getvalue()
        # with open(st.session_state.user_audio, "rb") as f:
            # encoded = Binary(f.read())
        # we can only insert files < 16 MB into our db
        # st.collection.insert_one(
        #     {
        #         'date': datetime.now(),
        #         'file': user_bytes
        #     }
        # )
        st.audio(user_bytes,format="audio/wav")
    
        # processing user's audio and getting the pitch contour on top of the native speaker's
        user_figure, clf_result, clf_probs = cont.process_user_audio(ns_figure, st.session_state.user, st.session_state.user_audio)
        st.session_state.user_figure = user_figure
        target_tone_prob = clf_probs[0,exercise['tone']-1]
        st.pyplot(user_figure)
        st.write("all probabilities: ", clf_probs)
        st.write("target tone's probability: ", target_tone_prob)
        st.write("Rating: ", get_rating(text["ratings"], exercise["tone"], clf_probs))

    else:
        if st.session_state.user_figure is None:
            st.pyplot(st.session_state.ns_figure)

if st.session_state.key != last_page:
    st.button('Next', on_click=on_next)
