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

USER = 'user'
PWD = 'password'

CONNECT_STRING = f'mongodb+srv://{USER}:{PWD}@cluster0.yxtrq.mongodb.net/audioDB?retryWrites=true&w=majority'

if not hasattr(st, "client"):
    st.client = pymongo.MongoClient(CONNECT_STRING)
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
    data = json.load(json_file)

exercises = data["exercises"]

st.write(data['title'])

if st.session_state.key == 0:
    st.write(data['instructions'])
    st.write(st.session_state.user_audio)
elif st.session_state.key == 1:
    st.write(data['calibration'])
    st.write(st.session_state.user_audio)
    audio_btn()

    if st.session_state.user_audio is not None:
        st.write(st.session_state.user_audio)
        audio_file = open(st.session_state.user_audio, 'rb')
        audio_bytes = audio_file.read()

else:
    st.write(st.session_state.user_audio)
    exercise = exercises[st.session_state.key - 2]
    st.write("Test ", str(st.session_state.key - 1), " - ", exercise["character"])

    audio_file = open("data/" + exercise["fileName"] + ".mp3", 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')
    
    audio_btn()

    if st.session_state.user_audio is not None:
        temp = st.session_state.user_audio
        st.write(temp)
        audio_file = open(temp, 'rb')
        audio_bytes = audio_file.read()
        st.audio(temp,format="audio/mp3")
        with open(temp, "rb") as f:
            encoded = Binary(f.read())
        # we can only insert files < 16 MB into our db
        st.collection.insert_one(
            {
                'date': datetime.now(),
                'file': encoded
            }
        )
        audio_file = open(temp, 'rb')
        audio_bytes = audio_file.read()
        st.audio(temp,format="audio/wav")

def on_next():
    st.session_state.key += 1
    st.session_state.user_audio = None

st.button('Next', on_click=on_next)
