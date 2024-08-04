import random
from contextlib import suppress

import pytest
import sys

KAIROS_MIN_SEED_VAL = 1
KAIROS_MAX_SEED_VAL = sys.maxsize


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

    return seed


def pytest_generate_tests(metafunc):
    if hasattr(metafunc.function, "config_random"):
        all_fixture_names = metafunc.function.config_random
        for fixture_name in all_fixture_names:
            pass
