import pytest
import polars as pl


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
