import pytest


@pytest.mark.config_random(["va1", "va2"])
def test_rand_int(var1, var2):
    assert isinstance(var1, int)
    assert isinstance(var2, int)
