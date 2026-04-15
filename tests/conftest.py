import pytest
import numpy as np
import sparsediffpy as sp


@pytest.fixture
def scope():
    return sp.Scope()


@pytest.fixture
def rng():
    return np.random.default_rng(42)
