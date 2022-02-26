import streamlit as st
import numpy as np

def on_next():
    st.session_state.key += 1
    st.session_state.user_audio = None
    st.session_state.user_figure = None
    st.session_state.ns_figure = None

def get_rating(rating_meta, target_tone, clf_probs):
    target_prob = clf_probs[0][target_tone-1]

    # if the probability of any other tone is >= 90%, return try again
    for prob in clf_probs[0][np.arange(len(clf_probs[0]))!=(target_tone-1)]:
        if prob >= 0.9:
            return rating_meta[-1]["label"]

    # if your target probability is more than 50%, you are good
    if target_prob > 0.5:
        return rating_meta[0]["label"]
    else:
        return rating_meta[1]["label"]

    # for rating in rating_meta:
    #     if prob >= rating["upperlimit"]:
    #         return rating["label"]
    # return rating[-1]["label"] 