import streamlit as st

def on_next():
    st.session_state.key += 1
    st.session_state.user_audio = None
    st.session_state.user_figure = None
    st.session_state.ns_figure = None

def get_rating(rating_meta, prob):
    for rating in rating_meta:
        if prob >= rating["upperlimit"]:
            return rating["label"]
    return rating[-1]["label"]