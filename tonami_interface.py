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
from tonami import calibration as c

import json
from datetime import datetime
from bson.binary import Binary

import pymongo


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
st.session_state['record'] = False

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
        # a = np.frombuffer(st.session_state.user_audio, dtype=float)
        # sound = AudioSegment.from_file(io.BytesIO(st.session_state.user_audio))
        # sound = AudioSegment.from_file("data/Kanta.mp3")
        # samples = sound.get_array_of_samples()
        # arr = np.array(samples).astype(np.float32)
        # a_p = np.array(sound.get_array_of_samples(), dtype=np.int16)
        # a_p = a_p.astype(float)
        # st.write(type(arr))
        # st.write(a_p)
        st.session_state.user = c.get_user_from_calibration(st.session_state.user_audio)
        p = st.session_state.user.pitch_profile
        st.write(p["max_f0"], p["min_f0"])
        pass

else:
    # st.write(st.session_state.user_audio)
    exercise = exercises[st.session_state.key - 2]
    st.write("Test ", str(st.session_state.key - 1), " - ", exercise["character"])

    audio_btn.audio_file = open("data/" + exercise["fileName"] + ".mp3", 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')
    
    audio_btn()

    if st.session_state.user_audio is not None:
        # with open(st.session_state.user_audio, "rb") as f:
            # encoded = Binary(f.read())
        # we can only insert files < 16 MB into our db
        st.collection.insert_one(
            {
                'date': datetime.now(),
                'file': st.session_state.user_audio
            }
        )
        st.audio(st.session_state.user_audio,format="audio/mp3")

        # utterance = Utterance(st.session_state.user_audio, None, )

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
