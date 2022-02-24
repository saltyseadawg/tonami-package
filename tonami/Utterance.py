import librosa
import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Tuple, Union
from pydub import AudioSegment

from tonami import pitch_process as pp
from tonami import audio_utils
# stub for utterance class
# idea right now is that it creates an utterance that has gone through all
# the pre-processing and can be passed to our classifier or visualization model
# possibly stores its own classification/probability as well?

class Utterance:
    def __init__(self, track : Union[npt.NDArray[float], AudioSegment] = None, sr=None, pitch_floor=50, pitch_ceil=400, filename : str =None, pitch_contour : npt.NDArray[float] = None, pitch_filepath : str = 'data/parsed/toneperfect_pitch_librosa_50-500-fminmax.json'):
        #constructor overloading - if track is a time series array
        if track != None and isinstance(track, np.ndarray):
            self.track = track
            self.sr = sr

            #we might want to just pass the pitch contour and stuff in directly as constructor? unsure
            #depends on how rest of code is written i guess
            self.pitch_contour, self.voiced_flag, self.voiced_prob = librosa.pyin(track, pitch_floor, pitch_ceil)

        # if track is an 
        elif isinstance(track, AudioSegment):
            # librosa can't load mp3 buffer? -> use wav
            wav = audio_utils.convert_audio(track, 'wav')
            y, sr = librosa.load(wav)

            # need to figure out how to adjust the floor and ceiling
            self.pitch_contour, self.voiced_flag, self.voiced_prob = librosa.pyin(y, fmin=pitch_floor, fmax=pitch_ceil)

        elif filename != None:
            self.PITCH_FILEPATH = pitch_filepath
            self.filename = filename

            self.pitch_data = pd.read_json(self.PITCH_FILEPATH)
            self.pitch_data = self.pitch_data.loc[self.pitch_data['filename'].isin([self.filename])]
            self.pitch_contour = np.array(self.pitch_data.loc[:, 'pitch_contour'].to_numpy()[0], dtype=float)
            self.label = self.pitch_data.loc[:, 'tone'].to_numpy()

            #TODO: code for getting the fmin and fmax
            #TODO: handle filename not found
        
        #TODO: construct with just pitch_contour?
        else:
            print("attempted to create invalid Utterance")

    def pre_process(self, spkr_max, spkr_min) -> Tuple[npt.NDArray[float], npt.NDArray[int]]:
        """Prepares the audio track for classification and visualization.
        """
        interp = pp.preprocess(self.pitch_contour)

        # need to normalize with speaker max and min -> different from librosa estimating pitch floor and ceil
        self.pitch_contour = pp.moving_average(pp.normalize_pitch(interp, spkr_max, spkr_min))    

