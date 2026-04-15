import numpy as np
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_point


def test_multiply_vectors_jacobian(scope, rng):
    x = scope.Variable(4, 1)
    y = scope.Variable(4, 1)
    f = x * y
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_multiply_vectors_hessian(scope, rng):
    x = scope.Variable(4, 1)
    y = scope.Variable(4, 1)
    f = x * y
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    x0 = random_point(scope, rng)
    checker.check_hessian(x0, rng.standard_normal(4))


def test_multiply_vectors_forward(scope, rng):
    x = scope.Variable(3, 1)
    y = scope.Variable(3, 1)
    f = x * y
    fn = sp.compile(f)
    x0 = random_point(scope, rng)
    x_val = x.value
    y_val = y.value
    np.testing.assert_allclose(fn.forward(), x_val * y_val)


def test_multiply_matrices_jacobian(scope, rng):
    X = scope.Variable(3, 2)
    Y = scope.Variable(3, 2)
    f = X * Y
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))
