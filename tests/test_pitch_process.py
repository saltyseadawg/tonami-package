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
    pass


