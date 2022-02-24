import numpy as np
import pandas as pd
from pytest import approx

from tonami import pitch_process as pp
from tonami import Utterance as u
from tonami import user

PITCH_FILEPATH = 'data/parsed/toneperfect_pitch_librosa_50-500-fminmax.json'

def test_utterance_constructor():
    #TODO: add test for the other methods of constructing

    # from filename:
    filename = 'wo3_MV2_MP3.mp3'
    word = u.Utterance(filename=filename)

    pitch_data = pd.read_json(PITCH_FILEPATH)
    pitch_data = pitch_data.loc[pitch_data['filename'].isin([filename])]
    result = np.array(pitch_data.loc[:, 'pitch_contour'].to_numpy()[0], dtype=float)

    #TODO: add check for fmin and fmax
    nan_test, _ = pp.get_nan_idx(word.pitch_contour)
    nan_result, _ = pp.get_nan_idx(result)
    #np.nan == np.nan will return false!
    assert(nan_test == nan_test).all()
    assert (word.pitch_contour[~nan_test] == approx(result[~nan_result], abs=1e-3)) #approx doesn't need .all() for arrays

def test_pre_process():
    filename = 'wo3_MV2_MP3.mp3'
    word = u.Utterance(filename=filename)

    pitch_data = pd.read_json(PITCH_FILEPATH)
    pitch_data = pitch_data.loc[pitch_data['filename'].isin([filename])]
    pitch_contour = np.array(pitch_data.loc[:, 'pitch_contour'].to_numpy()[0], dtype=float)

    voiced = pp.get_voice_activity(pitch_contour)
    cast_arr = np.array(voiced, dtype=float)
    interp = pp.interpolate_array(cast_arr)

    #TODO: change the fmax and fmin to get the speakers profile
    result = pp.moving_average(interp)
    result = pp.normalize_pitch(result, word.fmax, word.fmin)
    speaker_info = user.User(word.fmax, word.fmin)
    pre_processed = word.pre_process(speaker_info)

    assert (pre_processed[0] == result).all()
    # assert pre_processed[0] == result