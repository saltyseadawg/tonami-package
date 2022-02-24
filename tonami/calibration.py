import librosa
import numpy as np
from pydub import AudioSegment

from tonami import pitch_process as pp
from tonami import user as usr

def get_user_from_calibration(user_audio):
    filename='data/Haoba.mp3'
    # y, sr = librosa.load(filename)
    user_audio = AudioSegment.from_file(filename)
    samples = user_audio.get_array_of_samples()
    arr = np.array(samples).astype(np.float32)
    pitch_contour, _, _ = librosa.pyin(arr, fmin=100, fmax=300)
    pitch_contour, _ = pp.preprocess(pitch_contour)
    max_f0, min_f0 = pp.max_min_f0(pitch_contour)
    user = usr.User(max_f0, min_f0)
    return user