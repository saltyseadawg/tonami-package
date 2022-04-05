import collections
import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import sklearn.pipeline
import librosa
import io
import matplotlib.pyplot as plt
import os
import json
import time

from azure.storage.blob import BlobServiceClient
from heroku import audio_btn
from heroku.interface_utils import *
from pydub import AudioSegment
from scipy import stats
from tonami import Utterance as utt
from tonami import pitch_process as pp
from tonami import user as usr
from tonami import controller as cont
from tonami import Classifier as c
from tonami.audio_utils import convert_audio

st.set_page_config( "Tonami", "🌊", "centered", "collapsed" )

if 'user_audio' not in st.session_state:
    st.session_state.user_audio = None
if 'user' not in st.session_state:
    st.session_state.user = usr.User(50, 350)
if 'db_threshold' not in st.session_state:
    st.session_state.db_threshold = int(10)
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

######################################################################################################################################################
# setup
USER_TESTING_FILEPATH = 'data/user_audio/testing/'
# USER_TESTING_FILEPATH = 'data/native speakers/'
SAMPLERATE = 22050
user_audio_dict = {}
raw_data_dict = {}
times_dict = {}
placeholder_graph = st.empty()

######################################################################################################################################################
st.write("get all files")
filenames = os.listdir(USER_TESTING_FILEPATH)
# filenames = filenames[0:25]
num_files = len(filenames)

user_audio_dict['Filename'] = filenames

######################################################################################################################################################
st.write("get target tones")
targets = [0] * num_files

for i, filename in enumerate(filenames):
    if i > 2:
        targets[i] = int(filename.split("_")[1][-1])
    else:
        targets[i] = 2

user_audio_dict['Target'] = targets

######################################################################################################################################################
st.write("load files")

time_series_arr = [None] * num_files
load_times = [0] * num_files

for i, filename in enumerate(filenames):
    start = time.time()
    time_series_arr[i], _ = librosa.load(USER_TESTING_FILEPATH + filename)  
    load_times[i] = time.time() - start

raw_data_dict['Time Series'] = time_series_arr
times_dict['load'] = load_times

######################################################################################################################################################
st.write("get durations")
durations = [0] * num_files

for i, time_series in enumerate(time_series_arr):
    durations[i] = len(time_series)/SAMPLERATE

user_audio_dict['Durations'] = durations

######################################################################################################################################################
st.write("trim files")

f_time_series_arr = [None] * num_files
trim_times = [0] * num_files

for i, time_series in enumerate(time_series_arr):
    start = time.time()
    f_time_series_arr[i], _ = librosa.effects.trim(y=time_series, top_db=st.session_state.db_threshold) 
    trim_times[i] = time.time() - start

raw_data_dict['Filtered Time Series'] = f_time_series_arr
times_dict['trim'] = trim_times

######################################################################################################################################################
st.write("get filtered durations")
f_durations = [0] * num_files

for i, f_time_series in enumerate(f_time_series_arr):
    f_durations[i] = len(f_time_series)/SAMPLERATE

user_audio_dict['Filtered Durations'] = f_durations

######################################################################################################################################################
st.write("get pitch contours")
pitch_contour_arr = [None] * num_files
contour_times = [0] * num_files

for i, f_time_series in enumerate(f_time_series_arr):
    start = time.time()
    pitch_contour_arr[i] , _, _ = librosa.pyin(f_time_series, fmin=50, fmax=400)
    contour_times[i] = time.time() - start

raw_data_dict['Pitch Contours'] = pitch_contour_arr
times_dict['contour'] = contour_times

######################################################################################################################################################
st.write("Tonami's feature extraction")
normalized_pitch_arr = [None] * num_files
nans_arr = [None] * num_files
features_arr = [None] * num_files
extraction_times = [0] * num_files

for i, pitch_contour in enumerate(pitch_contour_arr):
    start = time.time()
    interp, nans_arr[i] = pp.preprocess(pitch_contour)
    interp_np = np.array([interp], dtype=float)
    profile = st.session_state.user.get_pitch_profile()
    avgd = pp.moving_average(interp_np)
    normalized_pitch = pp.normalize_pitch(avgd, profile['max_f0'], profile['min_f0'])
    features_arr[i] = np.array([pp.basic_feat_calc(normalized_pitch[0])])
    normalized_pitch_arr[i] = normalized_pitch
    extraction_times[i] = time.time() - start
    
raw_data_dict['Normalized Pitch'] = normalized_pitch_arr
raw_data_dict['Nans'] = nans_arr
raw_data_dict['Features'] = features_arr
times_dict['extraction'] = extraction_times

######################################################################################################################################################
st.write("Classifier time (svm_80_lda)")
class_times = [0] * num_files
classified_tones = [0] * num_files
classified_probs = [None] * num_files
classification_errors = 0

for i, features in enumerate(features_arr):
    start = time.time()
    try:
        classified_tones[i], classified_probs[i] = st.session_state.clf.classify_tones(features)
    except:
        classification_errors += 1
    class_times[i] = time.time() - start

user_audio_dict['Classified Tones'] = classified_tones
user_audio_dict['Classified Probs'] = classified_probs
times_dict['class'] = class_times

######################################################################################################################################################
st.write("get rating")
rating_times = [0] * num_files
ratings_arr = [""] * num_files

for i, clf_probs in enumerate(classified_probs):
    start = time.time()
    if classified_tones[i] != 0:
        ratings_arr[i] = get_rating(text["ratings"], targets[i], clf_probs)
    else:
        ratings_arr[i] = "Unavailable"
    rating_times[i] = time.time() - start

user_audio_dict['Rating'] = ratings_arr
times_dict['rate'] = rating_times

######################################################################################################################################################
st.write("display feedback")
display_times = [0] * num_files

for i, pitch_contour in enumerate(pitch_contour_arr):
    fig, ax = plt.subplots()
    plt.xticks([])
    plt.yticks([])
    start = time.time()
    user_pitch_contour = pitch_contour
    y_pitch = user_pitch_contour.copy()
    y_interp = user_pitch_contour.copy()
    y_pitch[nans_arr[i][0]] = np.nan
    ax.plot(y_interp, color='blue', linestyle=":", linewidth=2)
    ax.plot(y_pitch, color='blue', linewidth=3)
    placeholder_graph.pyplot(fig, clear_figure = True)
    st.write("###", "Rating: ", ratings_arr[i])
    display_times[i] = time.time() - start
    placeholder_graph.empty()
    plt.close()

times_dict['display'] = display_times

######################################################################################################################################################
st.write("Processing time (svm_80_lda)")
process_times = [0] * num_files

for i, filename in enumerate(filenames):
    fig, ax = plt.subplots()
    plt.xticks([])
    plt.yticks([])
    ns_figure = fig
    start = time.time()
    user_figure, clf_result, clf_probs = cont.process_user_audio(ns_figure, st.session_state.user, USER_TESTING_FILEPATH + filename, st.session_state.clf, db_threshold=st.session_state.db_threshold)
    st.session_state.user_figure = user_figure
    placeholder_graph.pyplot(user_figure, clear_figure = True)

    if clf_result != 0:
        st.write("###", "Rating: ", get_rating(text["ratings"], targets[i], clf_probs))
    else:
        st.write("### Error with rating")

    process_times[i] = time.time() - start
    placeholder_graph.empty()
    plt.close()

times_dict['total'] = process_times

######################################################################################################################################################
st.write("Processing time (svm_80_lda) (untrimmed)")
process_times = [0] * num_files

for i, filename in enumerate(filenames):
    fig, ax = plt.subplots()
    plt.xticks([])
    plt.yticks([])
    ns_figure = fig
    start = time.time()
    user_figure, clf_result, clf_probs = cont.process_user_audio(ns_figure, st.session_state.user, USER_TESTING_FILEPATH + filename, st.session_state.clf, db_threshold=st.session_state.db_threshold, trim=False)
    st.session_state.user_figure = user_figure
    placeholder_graph.pyplot(user_figure, clear_figure = True)

    if clf_result != 0:
        st.write("###", "Rating: ", get_rating(text["ratings"], targets[i], clf_probs))
    else:
        st.write("### Error with rating")

    process_times[i] = time.time() - start
    placeholder_graph.empty()
    plt.close()

times_dict['total (untrimmed)'] = process_times

######################################################################################################################################################
st.write("consolidate data")
user_audio_df = pd.DataFrame(user_audio_dict)
raw_data_df = pd.DataFrame(raw_data_dict)
times_df = pd.DataFrame(times_dict)

######################################################################################################################################################
st.write("remove outliers")
outliers = user_audio_df[user_audio_df['Filtered Durations'] > 0.8].index
st.write("Num of outliers: ", len(outliers))
user_audio_df_fix = user_audio_df.drop(outliers)
raw_data_df_fix = raw_data_df.drop(outliers)
times_df_fix = times_df.drop(outliers)

######################################################################################################################################################
# File report
st.write("### Audio File Statistics")
st.write("Quantity :", user_audio_df.shape[0])
st.write("Duration :", "%.2f" % round(user_audio_df['Durations'].mean(), 2), u"\u00B1", "%.2f" % round(user_audio_df['Durations'].std(), 2) )
st.write("Filtered :", "%.2f" % round(user_audio_df['Filtered Durations'].mean(), 2), u"\u00B1", "%.2f" % round(user_audio_df['Filtered Durations'].std(), 2) )

st.write("### Audio File Statistics (w/o outliers)")
st.write("Quantity :", user_audio_df_fix.shape[0])
st.write("Duration :", "%.2f" % round(user_audio_df_fix['Durations'].mean(), 2), u"\u00B1", "%.2f" % round(user_audio_df_fix['Durations'].std(), 2) )
st.write("Filtered :", "%.2f" % round(user_audio_df_fix['Filtered Durations'].mean(), 2), u"\u00B1", "%.2f" % round(user_audio_df_fix['Filtered Durations'].std(), 2) )

######################################################################################################################################################
# Time breakdown reports
st.write("### Task Time Statistics (microseconds)")
for name, values in times_df.items():
    st.write("%10s :" % name, 
          "%6i" % (values.mean()*1000000), 
          u"\u00B1", 
          "%6i" % (values.std()*1000000)
         )

st.write("### Task Time Statistics (w/o outliers) (microseconds)")
for name, values in times_df_fix.items():
    st.write("%10s :" % name, 
          "%6i" % (values.mean()*1000000),
          u"\u00B1", 
          "%6i" % (values.std()*1000000)
         )

######################################################################################################################################################
st.write("### Classification errors: ", classification_errors)
t, p = stats.ttest_rel(times_df_fix['total'], times_df_fix['total (untrimmed)'])
st.write("### t-statistic: ", t)
st.write("### p-value: ", p)