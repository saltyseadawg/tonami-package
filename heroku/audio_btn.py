import streamlit as st

from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

from bokeh.io import curdoc
from bokeh.plotting import figure, output_file, show

def audio_btn():  
    recording = False
    stt_button = Button(label="Record", button_type="primary", id="stt_button")
    stt_button.js_on_event("button_click", CustomJS(args=dict(btn=stt_button), code="""
        if (btn.label == "Record") {
            btn.label = "Stop"
            btn.button_type = "danger"
            document.dispatchEvent(new CustomEvent("ON_RECORD", {detail: {url: null}}));
        } else {
            btn.label = "Record"
            btn.button_type = "primary"
            document.dispatchEvent(new CustomEvent("ON_RECORD", {detail: {url: "url"}}));
        }
        btn.change.emit()
    """))

    result = streamlit_bokeh_events(
        stt_button,
        events="ON_RECORD",
        key="listen",
        refresh_on_update=False,
        override_height=75,
        debounce_time=0)

    if result is not None:
        st.write("here")
        if "ON_RECORD" in result:
            url = result.get("ON_RECORD")["url"]
            if url is None:
                st.write("none")
            else:
                st.write(url)

    st.write(result)
    st.write(recording)
