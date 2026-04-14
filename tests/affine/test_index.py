import numpy as np
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_point


def test_index_scalar(scope, rng):
    x = scope.Variable(4, 1)
    f = x[0]
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_index_scalar_forward(scope, rng):
    x = scope.Variable(4, 1)
    f = x[2]
    fn = sp.compile(f)
    x0 = random_point(scope, rng)
    np.testing.assert_allclose(fn.forward(), [x0[2]])


def test_index_slice(scope, rng):
    x = scope.Variable(4, 1)
    f = x[1:3]
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_index_slice_forward(scope, rng):
    x = scope.Variable(4, 1)
    f = x[1:3]
    fn = sp.compile(f)
    x0 = random_point(scope, rng)
    np.testing.assert_allclose(fn.forward(), x0[1:3])


def test_index_fancy(scope, rng):
    x = scope.Variable(4, 1)
    f = x[[0, 3]]
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_index_matrix_element(scope, rng):
    X = scope.Variable(3, 2)
    f = X[1, 0]
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_index_matrix_row_slice(scope, rng):
    X = scope.Variable(3, 2)
    f = X[0:2, :]
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_index_matrix_column(scope, rng):
    X = scope.Variable(3, 2)
    f = X[:, 1]
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))
