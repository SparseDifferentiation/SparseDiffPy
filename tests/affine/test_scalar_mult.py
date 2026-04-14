import numpy as np
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_point


def test_scalar_mult_constant_jacobian(scope, rng):
    x = scope.Variable(4, 1)
    f = 3.0 * x
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_scalar_mult_constant_forward(scope, rng):
    x = scope.Variable(3, 1)
    f = 2.5 * x
    fn = sp.compile(f)
    x0 = random_point(scope, rng)
    np.testing.assert_allclose(fn.forward(), 2.5 * x0)


def test_scalar_mult_parameter_jacobian(scope, rng):
    x = scope.Variable(4, 1)
    a = scope.Parameter(1, 1, value=np.array([[3.0]]))
    f = a * x
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_scalar_mult_parameter_update(scope, rng):
    x = scope.Variable(3, 1)
    a = scope.Parameter(1, 1, value=np.array([[2.0]]))
    f = a * x
    fn = sp.compile(f)
    x0 = random_point(scope, rng)
    np.testing.assert_allclose(fn.forward(), 2.0 * x0)
    a.value = np.array([[5.0]])
    np.testing.assert_allclose(fn.forward(), 5.0 * x0)


def test_scalar_mult_matrix_jacobian(scope, rng):
    X = scope.Variable(3, 2)
    f = 2.0 * X
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_right_scalar_mult_jacobian(scope, rng):
    x = scope.Variable(4, 1)
    f = x * 3.0
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))
