import random
from contextlib import suppress
from functools import singledispatch

import pytest

try:
    import numpy as np
except ModuleNotFoundError:
    from pytest_kairos._fake_module import np

try:
    import polars as pl
except ModuleNotFoundError:
    from pytest_kairos._fake_module import pl
import sys

KAIROS_MIN_SEED_VAL = 1
KAIROS_MAX_SEED_VAL = sys.maxsize

KAIROS_INT_MIN_VAL = -(sys.maxsize - 1)
KAIROS_INT_MAX_VAL = sys.maxsize


def pytest_addoption(parser):
    parser.addoption(
        "--seed",
        action="store",
        default=None,
        type=int,
        help="Seed for random number generator",
    )
    parser.addoption(
        "--repeat",
        action="store",
        default=1,
        type=int,
        help="Number of times to repeat each test",
    )


@pytest.fixture(scope="session", autouse=True)
def kairos_seed(request):
    seed = request.config.getoption(
        "--seed", random.randint(KAIROS_MIN_SEED_VAL, KAIROS_MAX_SEED_VAL)
    )
    random.seed(seed)

    with suppress(ModuleNotFoundError):
        import numpy as np

        np.random.seed(seed)
        np.random.default_rng(seed=seed)

    return seed


@singledispatch
def dispatch_configuration(args, metafunc, mark):
    raise NotImplementedError("Parameters received not recognized. {args}")


@dispatch_configuration.register(list)
def dispatch_list_args(args, metafunc, mark):
    for kairos_var in args:
        min_val = mark.kwargs.get("min", None)
        max_val = mark.kwargs.get("max", None)
        repeat = mark.kwargs.get("repeat", 1)
        dtype = mark.kwargs.get("dtype", int)
        size = mark.kwargs.get("size", None)
        metafunc.parametrize(
            kairos_var,
            [
                generate_random_number(dtype(), min=min_val, max=max_val, size=size)
                for _ in range(repeat)
            ],
        )


@dispatch_configuration.register(str)
def dispatch_str_args(args, metafunc, mark):
    dispatch_list_args([args], metafunc, mark)


@dispatch_configuration.register(dict)
def dispatch_dict_args(args, metafunc, mark):
    kairos_vars = list(args.keys())
    values = []
    for _ in range(mark.kwargs.get("repeat", 1)):
        parametrized_val = []
        for key, options in args.items():
            parametrized_val.append(
                generate_random_number(
                    options["dtype"](),
                    min=options.get("min", None),
                    max=options.get("max", None),
                    size=options.get("size", None),
                )
            )
        values.append(parametrized_val)
    metafunc.parametrize(", ".join(kairos_vars), values)


def pytest_generate_tests(metafunc):
    for mark in metafunc.definition.iter_markers("config_kairos"):
        args = mark.args[0]
        dispatch_configuration(args, metafunc, mark)


@singledispatch
def generate_random_number(dtype, **kwargs):
    raise NotImplementedError(f"Unsupported: {dtype}")


@generate_random_number.register(int)
def _generate_py_rand_int(dtype, min=None, max=None, size=None) -> int:
    min = KAIROS_INT_MIN_VAL if min is None else min
    max = KAIROS_INT_MAX_VAL if max is None else max
    return random.randint(min, max)


@generate_random_number.register(float)
def _generate_py_rand_float(dtype, min=None, max=None, size=None) -> float:
    min = sys.float_info.min if min is None else min
    max = sys.float_info.max if max is None else max
    return random.uniform(min, max)


@generate_random_number.register(np.integer)
def _generate_np_int_array(dtype, min=None, max=None, size=None):
    size = (1, 1) if size is None else size
    info = np.iinfo(dtype)
    min = info.min if min is None else min
    max = info.max if max is None else max
    rng = np.random.default_rng()
    return rng.integers(low=min, high=max, size=size, dtype=type(dtype))


@generate_random_number.register(np.floating)
def _generate_np_float_array(dtype, min=None, max=None, size=None):
    size = (1, 1) if size is None else size
    rng = np.random.default_rng()
    random_matrix = rng.random(size=size, dtype=type(dtype))
    return random_matrix * generate_random_number(
        np.int64(), min=min, max=max, size=size
    ).astype(type(dtype))


@generate_random_number.register(pl.datatypes.classes.FloatType)
def _generate_pl_float_dataframe(dtype, min=None, max=None, size=None):
    np_array = _generate_np_float_array(dtype=np.float64(), min=min, max=max, size=size)
    pl_array = pl.DataFrame(np_array)
    if not isinstance(pl_array.dtypes[0], type(dtype)):
        pl_array = pl_array.with_columns(
            [pl.col(col).cast(type(dtype)) for col in pl_array.columns]
        )
    return pl_array


@generate_random_number.register(pl.datatypes.classes.IntegerType)
def _generate_pl_int_dataframe(dtype, min=None, max=None, size=None):
    np_array = _generate_np_int_array(np.int128, min=min, max=max, size=size)
    pl_array = pl.DataFrame(np_array)
    if not isinstance(pl_array.dtypes[0], type(dtype)):
        pl_array = pl_array.with_columns(
            [pl.col(col).cast(type(dtype)) for col in pl_array.columns]
        )
    return pl_array
