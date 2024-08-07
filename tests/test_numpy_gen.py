import pytest
import numpy as np


@pytest.mark.config_kairos("np_var", dtype=np.int32)
def test_rand_numpy_int(np_var):
    assert np_var.dtype == np.int32
    assert np_var.shape == (1, 1)
    assert np_var.size == 1


@pytest.mark.config_kairos("np_array", dtype=np.int8, size=(2, 2))
def test_rand_numpy_int8_array(np_array):
    assert np_array.dtype == np.int8
    assert np_array.shape == (2, 2)
    assert np_array.size == 4


@pytest.mark.config_kairos(
    "np_array", dtype=np.int64, size=(3, 3), min=0, max=5, repeat=100
)
def test_rand_numpy_int64_min_max(np_array):
    assert np_array.dtype == np.int64
    assert np_array.shape == (3, 3)
    assert np_array.size == 9
    assert ((np_array >= 0) & (np_array <= 5)).all()


@pytest.mark.config_kairos("np_float", dtype=np.float64, size=(3, 3))
def test_rand_np_float128(np_float):
    assert np_float.dtype == np.float64
    assert np_float.shape == (3, 3)
    assert np_float.size == 9


@pytest.mark.config_kairos(
    "np_float", dtype=np.float32, size=(2, 2), min=0, max=10, repeat=100
)
def test_rand_np_float_min_max(np_float):
    assert np_float.dtype == np.float32
    assert np_float.shape == (2, 2)
    assert np_float.size == 4
    assert ((np_float >= 0) & (np_float < 10)).all()
