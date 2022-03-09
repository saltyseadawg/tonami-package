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
import os

from heroku import audio_btn
from tonami import Utterance as utt
from tonami import pitch_process as pp
from tonami import user as usr
from tonami import controller as cont
from tonami import Classifier as c
from heroku.interface_utils import *
from azure.storage.blob import BlobServiceClient


import json

# import pymongo
from tonami.audio_utils import convert_audio

CONNECT_STR = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
EXERCISE_DIR = 'data/tone_perfect/'

st.session_state.blob_service_client = BlobServiceClient.from_connection_string(CONNECT_STR)

# if not hasattr(st, "client"):
#     st.client = pymongo.MongoClient(**st.secrets["mongo"])
#     st.collection = st.client.audio_files.user_test

st.set_page_config( "Tonami", "🌊", "centered", "collapsed" )

if 'key' not in st.session_state:
    st.session_state.key = 0
if 'user_audio' not in st.session_state:
    st.session_state.user_audio = None
if 'user' not in st.session_state:
    st.session_state.user = None
if 'clf' not in st.session_state:
    clf = c.Classifier(4)
    clf.load_clf('tonami/data/pickled_svm_80.pkl')    
    st.session_state.clf = clf
if 'text' not in st.session_state:
    with open('heroku/interface_text.json') as json_file:
        text = json.load(json_file)
    st.session_state.text = text

st.session_state.user_audio = None

text = st.session_state.text
calibration = text["calibration"]
exercises = text["exercises"]
last_page = len(exercises) + 2

st.write(text['title'])

if st.session_state.key == 0:
    st.write(text['instructions'])
elif st.session_state.key == 1: 
    voice_label = st.radio(calibration['options_instructions'], calibration['options_labels'], 1)
    voice_info = calibration['options_info'][calibration['options_labels'].index(voice_label)]
    st.session_state.user = usr.User(voice_info['max_f0'], voice_info['min_f0'])

elif st.session_state.key == last_page:
    st.write(text['end_page'])

else:
    exercise = exercises[st.session_state.key - 2]
    st.write("Exercise ", str(st.session_state.key - 1))
    st.write("## ", exercise["character"], " ", exercise["pinyin"], " ", str(exercise["tone"]))
    st.write(exercise["translation"])
    
    # TODO: need to double check this audio file path
    exercise_path = os.path.join(EXERCISE_DIR, f'{exercise["fileName"]}.mp3')
    with open(exercise_path, 'rb') as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format='audio/mp3')
    
    ns_figure = cont.load_exercise(f'{exercise["fileName"]}.mp3')
    st.session_state.ns_figure = ns_figure

    filename_sections = exercise["fileName"].split("_")
    audio_btn.audio_btn(str(st.session_state.key - 1) + "_" + filename_sections[0])

    if st.session_state.user_audio is not None:
        # user_bytes = convert_audio(st.session_state.user_audio, 'wav').getvalue()
        # with open(st.session_state.user_audio, 'rb') as f:
        #     user_audio_bytes = f.read()
        upload_file(st.session_state.blob_service_client, st.session_state.user_audio)
        st.audio(st.session_state.user_audio, format="audio/mp3")

        # with open(st.session_state.user_audio, "rb") as f:
            # encoded = Binary(f.read())
        # we can only insert files < 16 MB into our db
        # st.collection.insert_one(
        #     {
        #         'date': datetime.now(),
        #         'file': user_bytes
        #     }
        # )
    
        # processing user's audio and getting the pitch contour on top of the native speaker's
        user_figure, clf_result, clf_probs = cont.process_user_audio(ns_figure, st.session_state.user, st.session_state.user_audio, st.session_state.clf)
        st.session_state.user_figure = user_figure
        target_tone_prob = clf_probs[0,exercise['tone']-1]
        st.pyplot(user_figure)
        # st.write("all probabilities: ", clf_probs)
        # st.write("target tone's probability: ", target_tone_prob)
        st.write("###", "Rating: ", get_rating(text["ratings"], exercise['tone'], clf_probs))

    else:
        if st.session_state.user_figure is None:
            st.pyplot(st.session_state.ns_figure)

if st.session_state.key != last_page:
    st.button('Next', on_click=on_next)
