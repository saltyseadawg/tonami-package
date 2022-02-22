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

# Opening JSON file
with open('heroku/interface_text.json') as json_file:
    data = json.load(json_file)

exercises = data["exercises"]

st.write(data['title'])

if st.session_state.key == 0:
    st.write(data['instructions'])
else:
    exercise = exercises[st.session_state.key - 1]
    st.write("Test ", str(st.session_state.key), " - ", exercise["character"])

    audio_file = open("data/" + exercise["fileName"] + ".mp3", 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')

    url = audio_btn()
    if url is not None:
        st.write(url.replace("blob:",""))
        # audio_file = open(url, 'rb')
        # audio_bytes = audio_file.read()
        st.audio(url.replace("blob:",""))

def increment_key():
    st.session_state.key += 1

st.button('Next', on_click=increment_key)
