import librosa
import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Tuple

from tonami import pitch_process as pp
# stub for utterance class
# idea right now is that it creates an utterance that has gone through all
# the pre-processing and can be passed to our classifier or visualization model
# possibly stores its own classification/probability as well?

class Utterance:
    def __init__(self, track : npt.NDArray[float] = None, sr=None, fmin=50, fmax=400, filename : str =None, pitch_contour : npt.NDArray[float] = None, pitch_filepath : str = 'data/parsed/toneperfect_pitch_librosa_50-500-fminmax.json'):
        
        self.fmin = fmin
        self.fmax = fmax
        
        #constructor overlaoding
        if track != None:
            self.track = track
            self.sr = sr

            #we might want to just pass the pitch contour and stuff in directly as constructor? unsure
            #depends on how rest of code is written i guess
            self.pitch_contour, self.voiced_flag, self.voiced_prob = librosa.pyin(track, fmin, fmax)

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

    def pre_process(self) -> Tuple[npt.NDArray[float], npt.NDArray[int]]:
        interp = pp.preprocess(self.pitch_contour)

        self.pitch_contour = pp.moving_average(pp.normalize_pitch(interp, self.fmax, self.fmin))

        return self.pitch_contour, nans #mask
        

