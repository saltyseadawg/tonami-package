from pickle import NONE
import librosa
import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Tuple, Union
from pydub import AudioSegment

from tonami import pitch_process as pp
from tonami import audio_utils
from tonami.Classifier import PITCH_FILEPATH
# stub for utterance class
# idea right now is that it creates an utterance that has gone through all
# the pre-processing and can be passed to our classifier or visualization model
# possibly stores its own classification/probability as well?

class Utterance:
    def __init__(self, track : Union[npt.NDArray[float], AudioSegment] = None, sr=None, pitch_floor=50, pitch_ceil=400, filename : str =None, pitch_contour : npt.NDArray[float] = None, pitch_filepath : str = None, db_threshold=10, trim=True):
        #constructor overloading - if track is a time series array
        if track is not None and isinstance(track, np.ndarray):
            self.track = track
            self.sr = sr

            #we might want to just pass the pitch contour and stuff in directly as constructor? unsure
            #depends on how rest of code is written i guess
            self.pitch_contour, self.voiced_flag, self.voiced_prob = librosa.pyin(track, pitch_floor, pitch_ceil)
            self.fmax, self.fmin = pp.max_min_f0(self.pitch_contour)

        # if track is an 
        elif isinstance(track, AudioSegment):
            # librosa can't load mp3 buffer? -> use wav
            wav = audio_utils.convert_audio(track, 'wav')
            y, sr = librosa.load(wav)

            # need to figure out how to adjust the floor and ceiling
            self.pitch_contour, self.voiced_flag, self.voiced_prob = librosa.pyin(y, fmin=pitch_floor, fmax=pitch_ceil)
            self.fmax, self.fmin = pp.max_min_f0(self.pitch_contour)

        elif filename is not None:

            self.filename = filename

            if pitch_filepath is None: #TODO: this is deranged
                time_series, _ = librosa.load(filename)
                
                if trim:
                    time_series, _ = librosa.effects.trim(y=time_series, top_db=db_threshold)

                self.pitch_contour, _, _ = librosa.pyin(time_series, fmin=pitch_floor, fmax=pitch_ceil) #guessing
                self.fmax, self.fmin = pp.max_min_f0(self.pitch_contour) #accurate for normalizing
            else:
                self.PITCH_FILEPATH = pitch_filepath
                
                self.pitch_data = pd.read_json(self.PITCH_FILEPATH)
                self.pitch_data = self.pitch_data.loc[self.pitch_data['filename'].isin([self.filename])]
                self.pitch_contour = np.array(self.pitch_data.loc[:, 'pitch_contour'].to_numpy()[0], dtype=float)
                self.label = self.pitch_data.loc[:, 'tone'].to_numpy()

                self.fmax, self.fmin = pp.max_min_f0(self.pitch_contour)
            #TODO: handle filename not found
        
        #TODO: construct with just pitch_contour?        
        else:
            print("attempted to create invalid Utterance")

    def pre_process(self, user) -> Tuple[npt.NDArray[float], npt.NDArray[bool], npt.NDArray[float]]:
        """Prepares the audio track for classification and visualization.
        """
        interp, nans = pp.preprocess(self.pitch_contour)
        interp_np = np.array([interp], dtype=float)
        profile = user.get_pitch_profile()
        
        avgd = pp.moving_average(interp_np)
        normalized_pitch = pp.normalize_pitch(avgd, profile['max_f0'], profile['min_f0'])
        features = np.array([pp.basic_feat_calc(normalized_pitch[0])])
        self.normalized_pitch = normalized_pitch

        return normalized_pitch, nans, features #nans - mask
        

