import collections
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
import sklearn.pipeline
from pydub import AudioSegment
import librosa
import io
import matplotlib.pyplot as plt

# import sys
# sys.path.insert(1, '/app')
from heroku import audio_btn
from tonami import Utterance as utt
from tonami import pitch_process as pp
from tonami import user as usr
from tonami import controller as cont

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
if 'clf' not in st.session_state:
    st.session_state.clf = pickle.load(open('tonami/data/pickled_svm_80.pkl', 'rb'))
if 'user' not in st.session_state:
    st.session_state.user = None

st.session_state.user_audio = None

# Opening JSON file
with open('heroku/interface_text.json') as json_file:
    text = json.load(json_file)

exercises = text["exercises"]

st.write(text['title'])

if st.session_state.key == 0:
    st.write(text['instructions'])
    # st.write(st.session_state.user_audio)
elif st.session_state.key == 1:
    st.write(text['calibration'])
    # st.write(st.session_state.user_audio)
    audio_btn.audio_btn()

    if st.session_state.user_audio is not None:
        # st.write(st.session_state.user_audio)
        #TODO: extract pitch max/min
        calibrate_utt = utt.Utterance(track=st.session_state.user_audio)
        st.session_state.user = usr.User(calibrate_utt.fmax, calibrate_utt.fmin)
        p = st.session_state.user.pitch_profile
        st.write(p["max_f0"], p["min_f0"])

else:
    # st.write(st.session_state.user_audio)
    exercise = exercises[st.session_state.key - 2]
    st.write("Test ", str(st.session_state.key - 1), " - ", exercise["character"])
    
    # TODO: need to double check this audio file path
    with open("data/tone_perfect/" + exercise["fileName"] + ".mp3", 'rb') as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format='audio/mp3')
    
    ns_figure = cont.load_exercise(exercise["fileName"] + ".mp3")
    st.pyplot(ns_figure)

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
        p = st.session_state.user
        utter = utt.Utterance(st.session_state.user_audio)
        utter.pre_process(p)
        st.write(utter.normalized_pitch)
        
        # processing user's audio and getting the pitch contour on top of the native speaker's
        user_figure, clf_result = cont.send_data_to_frontend(ns_figure, p, st.session_state.user_audio)
        st.pyplot(user_figure)

        # load_clf = pickle.load(open('tonami/data/pickled_svm_80.pkl', 'rb'))

        # # Apply model to make predictions
        # prediction = load_clf.predict(df)
        # prediction_proba = load_clf.predict_proba(df)

        # st.subheader('Prediction')
        # penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
        # st.write(penguins_species[prediction])

        # st.subheader('Prediction Probability')
        # st.write(prediction_proba)

def on_next():
    st.session_state.key += 1
    st.session_state.user_audio = None

st.button('Next', on_click=on_next)
