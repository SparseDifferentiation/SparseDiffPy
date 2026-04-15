import numpy as np
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_point


def test_vector_mult_constant_jacobian(scope, rng):
    x = scope.Variable(3, 1)
    c = np.array([1.0, 2.0, 3.0])
    f = c * x
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_vector_mult_constant_forward(scope, rng):
    x = scope.Variable(3, 1)
    c = np.array([1.0, 2.0, 3.0])
    f = c * x
    fn = sp.compile(f)
    x0 = random_point(scope, rng)
    np.testing.assert_allclose(fn.forward(), c * x0)


def test_vector_mult_parameter_jacobian(scope, rng):
    x = scope.Variable(3, 1)
    a = scope.Parameter(3, 1)
    a.value = np.array([1.0, 2.0, 3.0])
    f = a * x
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_vector_mult_parameter_update(scope, rng):
    x = scope.Variable(3, 1)
    a = scope.Parameter(3, 1)
    a.value = np.array([1.0, 1.0, 1.0])
    f = a * x
    fn = sp.compile(f)
    x0 = random_point(scope, rng)
    np.testing.assert_allclose(fn.forward(), x0)
    a.value = np.array([2.0, 3.0, 4.0])
    np.testing.assert_allclose(fn.forward(), np.array([2.0, 3.0, 4.0]) * x0)
