import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

from audio_btn import audio_btn

import json

st.set_page_config( "Tonami", "🌊", "centered", "collapsed" )

if 'key' not in st.session_state:
    st.session_state.key = 0
if 'url' not in st.session_state:
    st.session_state.url = None

# Opening JSON file
with open('heroku/interface_text.json') as json_file:
    data = json.load(json_file)

exercises = data["exercises"]

st.write(data['title'])

if st.session_state.key == 0:
    st.write(data['instructions'])
elif st.session_state.key == 1:
    st.write(data['calibration'])
    st.session_state.url = audio_btn(5000)

    if st.session_state.url is not None:
        st.write()
        temp = st.session_state.url.replace("blob:","")
        st.write(temp)
        # audio_file = open(temp, 'rb')
        # audio_bytes = audio_file.read()
        # st.audio(temp,format="audio/mp3")
else:
    exercise = exercises[st.session_state.key - 2]
    st.write("Test ", str(st.session_state.key - 1), " - ", exercise["character"])

    audio_file = open("data/" + exercise["fileName"] + ".mp3", 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')

    st.session_state.url = audio_btn()
    if st.session_state.url is not None:
        temp = st.session_state.url.replace("blob:","")
        st.write(temp)
        audio_file = open(temp, 'rb')
        audio_bytes = audio_file.read()
        st.audio(temp,format="audio/mp3")
        # st.write(url, unsafe_allow_html=True)

def on_next():
    st.session_state.key += 1
    st.session_state.url = None

st.button('Next', on_click=on_next)
