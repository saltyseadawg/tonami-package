import queue
import time
from pathlib import Path

import pydub
import streamlit as st
import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
)

# from streamlit_lottie import st_lottie


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


# https://github.com/whitphx/streamlit-webrtc/issues/357

def save_frames_from_audio_receiver(filename):
    webrtc_ctx = webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
        # COMMENT THIS OUT WHEN TESTING LOCALLY
        rtc_configuration={"iceServers": [{"urls": ["stun:stun3.l.google.com:19302"]}]} # needed when hosting remotely
    )

    if "audio_buffer" not in st.session_state:
        st.session_state["audio_buffer"] = pydub.AudioSegment.empty()

    status_indicator = st.empty()
    lottie = False
    is_connected = False

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
                if not is_connected:
                    is_connected = True
                    st.write('Recording...')
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
        st.session_state.user_audio = filename
        audio_buffer.export(filename, format="mp3")
        st.session_state["audio_buffer"] = pydub.AudioSegment.empty()

def audio_btn(exercise_info=None):
    st.markdown('# Recorder')
    st.write("After pressing \"Start\" button, please wait until \"Recording...\" message appears below the button.")
    cur_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    if exercise_info is not None:
        tmp_wavpath = TMP_DIR / f'ex{exercise_info}_{cur_time}.mp3'
    else:
        tmp_wavpath = TMP_DIR / f'{cur_time}.mp3'
    audio_file = str(tmp_wavpath)
    if audio_file:
        save_frames_from_audio_receiver(audio_file)  # second way
    
