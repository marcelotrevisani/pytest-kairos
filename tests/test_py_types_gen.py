import pytest


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
