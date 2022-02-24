#loads desired audio from either files or parsed data
#these guys could return a list of utterances or tracks? i'm not sure yet.
#depends on rest of code?
from ast import Bytes
from pydub import AudioSegment
import io

def convert_audio(audio: AudioSegment, format: str):
    """Converts an audio track into the specified audio format. 
    """
    buf = io.BytesIO()
    # librosa can't load mp3 buffer? -> use wav
    audio.export(buf, format=format)
    return buf
