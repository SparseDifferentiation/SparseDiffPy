import numpy as np
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_point


def test_sub_vectors_forward(scope, rng):
    x = scope.Variable(3, 1)
    y = scope.Variable(3, 1)
    f = x - y
    fn = sp.compile(f)
    x0 = random_point(scope, rng)
    x_val = x.value
    y_val = y.value
    np.testing.assert_allclose(fn.forward(), x_val - y_val)


def test_sub_vectors_jacobian(scope, rng):
    x = scope.Variable(3, 1)
    y = scope.Variable(3, 1)
    f = x - y
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_sub_scalar_constant(scope, rng):
    x = scope.Variable(3, 1)
    f = x - 1.0
    fn = sp.compile(f)
    x0 = random_point(scope, rng)
    np.testing.assert_allclose(fn.forward(), x0 - 1.0)


def test_rsub_constant(scope, rng):
    x = scope.Variable(3, 1)
    f = 1.0 - x
    fn = sp.compile(f)
    x0 = random_point(scope, rng)
    np.testing.assert_allclose(fn.forward(), 1.0 - x0)
