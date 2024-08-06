import pytest
import numpy as np
import polars as pl


@pytest.mark.config_kairos("var", dtype=int, min=10, max=20)
def test_single_rand_int(var):
    assert isinstance(var, int)
    assert var >= 10
    assert var <= 20


@pytest.mark.config_kairos("var1", dtype=int, min=0, max=2)
@pytest.mark.config_kairos("var2", dtype=int, min=-10, max=-5)
def test_rand_int_multiple_values(var1, var2):
    assert 0 <= var1 <= 2
    assert -10 <= var2 <= -5
    assert isinstance(var1, int)
    assert isinstance(var2, int)


@pytest.mark.config_kairos("kairos_num2", dtype=int, repeat=3)
@pytest.mark.config_kairos("kairos_num1", dtype=int, repeat=5)
def test_multiple_parametrized_rand_type_int(kairos_num1, kairos_num2):
    assert isinstance(kairos_num1, int)
    assert isinstance(kairos_num2, int)


@pytest.mark.config_kairos("var1", dtype=int, min=100, max=102)
@pytest.mark.config_kairos("var2", dtype=float, min=-1, max=1.5)
def test_rand_int_float_mixed(var1, var2):
    assert isinstance(var1, int)
    assert isinstance(var2, float)


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


@pytest.mark.config_kairos("pl_float", dtype=pl.Float64, size=(2, 2))
def test_pl_float(pl_float):
    assert pl_float.dtypes == [pl.Float64] * 2
    assert pl_float.shape == (2, 2)


@pytest.mark.config_kairos(
    "pl_float", dtype=pl.Float32, size=(3, 3), min=-1, max=4, repeat=100
)
def test_pl_float_min_max(pl_float):
    assert pl_float.dtypes == [pl.Float32] * 3
    assert pl_float.shape == (3, 3)
    assert ((pl_float["column_0"] >= -1) & (pl_float["column_0"] < 4)).all()
    assert ((pl_float["column_1"] >= -1) & (pl_float["column_1"] < 4)).all()
    assert ((pl_float["column_2"] >= -1) & (pl_float["column_2"] < 4)).all()


@pytest.mark.config_kairos(
    "pl_float", dtype=pl.Float32, size=(3, 3), min=0, max=3, repeat=100
)
def test_pl_int_min_max(pl_float):
    assert pl_float.dtypes == [pl.Float32] * 3
    assert pl_float.shape == (3, 3)
    assert ((pl_float["column_0"] >= 0) & (pl_float["column_0"] < 3)).all()
    assert ((pl_float["column_1"] >= 0) & (pl_float["column_1"] < 3)).all()
    assert ((pl_float["column_2"] >= 0) & (pl_float["column_2"] < 3)).all()


@pytest.mark.config_kairos(
    {
        "var1": {
            "dtype": float,
            "min": 0,
            "max": 10,
        },
        "var2": {
            "dtype": int,
            "min": -10,
            "max": 5,
        },
    },
    repeat=100,
)
def test_config_as_dict(var1, var2):
    assert isinstance(var1, float)
    assert (var1 >= 0) and (var1 <= 10)

    assert isinstance(var2, int)
    assert (var2 >= -10) and (var2 <= 5)


@pytest.mark.parametrize("param1, param2", [(1, True), (2, False)])
@pytest.mark.config_kairos("kairos1")
@pytest.mark.config_kairos("kairos2")
def test_parametrize_with_config_kairos(param1, param2, kairos1, kairos2):
    assert isinstance(kairos1, int)
    assert isinstance(kairos2, int)
    assert param1 in [1, 2]
