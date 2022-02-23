from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    WebRtcStreamerContext,
    RTCConfiguration,
)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import queue
from pathlib import Path
import time
import pydub

# from streamlit_lottie import st_lottie
import json

TMP_DIR = Path('temp')
if not TMP_DIR.exists():
    TMP_DIR.mkdir(exist_ok=True, parents=True)



MEDIA_STREAM_CONSTRAINTS = {
    "video": False,
    "audio": {
        # these setting doesn't work
        # "sampleRate": 48000,
        # "sampleSize": 16,
        # "channelCount": 1,
        "echoCancellation": False,  # don't turn on else it would reduce wav quality
        "noiseSuppression": True,
        "autoGainControl": True,
    },
}

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# https://github.com/whitphx/streamlit-webrtc/issues/357

def save_frames_from_audio_receiver(wavpath):
    webrtc_ctx = webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
        rtc_configuration=RTC_CONFIGURATION,
    )

    if "audio_buffer" not in st.session_state:
        st.session_state["audio_buffer"] = pydub.AudioSegment.empty()

    status_indicator = st.empty()
    # if not webrtc_ctx.state.playing:
    #     st.write('AHHHH')
    #     return False
    lottie = False
    while True:
        # save audio AFTER user has clicked start
        if webrtc_ctx.audio_receiver and webrtc_ctx.state.playing:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                status_indicator.info("No frame arrived.")
                continue

            # if not lottie:  # voice gif
            #     st_lottie(lottie_json, height=80)
            #     lottie = True

            for i, audio_frame in enumerate(audio_frames):
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                # st.markdown(f'{len(audio_frame.layout.channels)}, {audio_frame.format.bytes}, {audio_frame.sample_rate}')
                # 2, 2, 48000
                st.session_state["audio_buffer"] += sound
        else:
            lottie = True
            break

    audio_buffer = st.session_state["audio_buffer"]

    if not webrtc_ctx.state.playing and len(audio_buffer) > 0:
        audio_buffer.export(wavpath, format="mp3")
        st.session_state["audio_buffer"] = pydub.AudioSegment.empty()
        return True
    return False

def audio_btn():
    st.markdown('# Recorder')
    cur_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    tmp_wavpath = TMP_DIR / f'{cur_time}.mp3'
    audio_file = str(tmp_wavpath)

    if audio_file and save_frames_from_audio_receiver(audio_file):  # second way
        st.session_state['user_audio'] = audio_file
