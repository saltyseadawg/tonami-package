from lib2to3.refactor import get_all_fix_names
import numpy as np

from tonami import pitch_process as pp

def test_moving_average():
    arr = [1,2,3,4,5,6]
    result = [1,2,3,4,5,5]
    assert (pp.moving_average(arr, 3) == result).all()

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

