from lib2to3.refactor import get_all_fix_names
import numpy as np
import math
from pytest import approx

from tonami import pitch_process as pp

def test_moving_average():
    arr = [1,2,3,4,5,6]
    result = [1,2,3,4,5,5]
    assert (pp.moving_average(arr, 3) == result).all()

def test_normalize_pitch():
    arr = np.array([100, 30, 10, 500, 50, 250])
    min_f0 = 50
    max_f0 = 500
    result = np.empty(arr.shape)
    for i in range(arr.shape[0]):
        result[i] = (5 * (math.log(arr[i], 10) - math.log(min_f0, 10)) / (math.log(max_f0, 10) - math.log(min_f0, 10)))

    normalized = pp.normalize_pitch(arr, max_f0, min_f0)
    assert (normalized == approx(result, abs=1e-10))

def test_pad_matrix():
    arr = np.array([[], [1], [1,2,], [1,2,3,4]], dtype=object)
    result = np.array([[0,0,0,0],[1,0,0,0], [1,2,0,0], [1,2,3,4]])
    assert (pp.pad_matrix(arr, 0) == result).all()

def test_get_valid_mask():
    rows = [
        [None, None, None],
        [1, 2, None],
        [None, 1, 2],
        [1, None, 2],
        [1, 2, 3]
    ]
    arr = np.array(rows, dtype=float)
    results = [False, False, False, False, True]
    assert (pp.get_valid_mask(arr) == results).all()

def test_get_voice_activity():
    no_voice = np.array(
        [None, None, None],
    )

    truncate_voice = np.array(
        [None, 3, 4, None, 5, 6, None, None]
    )
    truncate_voice_result = np.array(
        [3,4,None,5,6]
    )

    assert (pp.get_voice_activity(no_voice) == no_voice).all()
    assert (pp.get_voice_activity(truncate_voice) == truncate_voice_result).all()

def test_interp():
    arr = np.array([3,np.nan,4,np.nan,np.nan,7,10])
    result = np.array([3,3.5,4,5,6,7,10])
    interp = pp.interpolate_array(arr)
    assert (interp.shape == result.shape) 
    # below throws hard to understand error if the arrays are not the same size
    assert (interp == result).all()
