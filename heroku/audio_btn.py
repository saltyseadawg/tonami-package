import streamlit as st

from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

from bokeh.io import curdoc
from bokeh.plotting import figure, output_file, show

def audio_btn(duration = 3000):
    stt_button = Button(label="Record", button_type="primary", id="stt_button")
    stt_button.js_on_event("button_click", CustomJS(args=dict(btn=stt_button, duration=duration), code="""
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia ({ audio: true })

                // Success callback
                .then(function(stream) {
                    btn.label = "Recording in progress..."
                    btn.button_type = "danger"
                    btn.change.emit()

                    const mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.start();

                    const audioChunks = [];
                    mediaRecorder.addEventListener("dataavailable", event => {
                        audioChunks.push(event.data);
                    });

                    mediaRecorder.addEventListener("stop", () => {
                        btn.label = "Record"
                        btn.button_type = "primary"
                        btn.change.emit()

                        const audioBlob = new Blob(audioChunks, {'type': 'audio/mp3'});
                        const audioUrl = URL.createObjectURL(audioBlob);
                        const audio = new Audio(audioUrl);
                        document.dispatchEvent(new CustomEvent("ON_RECORD", {detail: {url: audioUrl}}));
                    });

                    setTimeout(() => {
                        mediaRecorder.stop();
                    }, duration);
                })

                // Error callback
                .catch(function(err) {
                    console.log('The following getUserMedia error occurred: ' + err);
                }
            );
        } else {
            console.log('getUserMedia not supported on your browser!');
        }

        
    """))

    result = streamlit_bokeh_events(
        stt_button,
        events="ON_RECORD",
        key="listen",
        refresh_on_update=False,
        override_height=42,
        debounce_time=0)

    if result is not None:
        if "ON_RECORD" in result:
            url = result.get("ON_RECORD")["url"]
            return url
