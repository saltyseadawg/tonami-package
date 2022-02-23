import collections
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

from audio_btn import audio_btn

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
    audio_btn()

    if st.session_state.user_audio is not None:
        # st.write(st.session_state.user_audio)
        #TODO: extract pitch max/min
        pass

else:
    # st.write(st.session_state.user_audio)
    exercise = exercises[st.session_state.key - 2]
    st.write("Test ", str(st.session_state.key - 1), " - ", exercise["character"])

    audio_file = open("data/" + exercise["fileName"] + ".mp3", 'rb')
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

def on_next():
    st.session_state.key += 1
    st.session_state.user_audio = None

st.button('Next', on_click=on_next)
