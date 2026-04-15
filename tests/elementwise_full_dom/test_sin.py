import numpy as np
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_point


def test_sin_vector_forward(scope, rng):
    x = scope.Variable(4, 1)
    f = sp.sin(x)
    fn = sp.compile(f)
    x0 = random_point(scope, rng)
    np.testing.assert_allclose(fn.forward(), np.sin(x0))


def test_sin_vector_jacobian(scope, rng):
    x = scope.Variable(4, 1)
    f = sp.sin(x)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_sin_vector_hessian(scope, rng):
    x = scope.Variable(4, 1)
    f = sp.sin(x)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    x0 = random_point(scope, rng)
    checker.check_hessian(x0, rng.standard_normal(4))


def test_sin_matrix_jacobian(scope, rng):
    X = scope.Variable(3, 2)
    f = sp.sin(X)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))
