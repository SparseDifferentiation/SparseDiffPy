import numpy as np
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_point


def test_reshape_jacobian(scope, rng):
    x = scope.Variable(6, 1)
    f = sp.reshape(x, 2, 3)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_reshape_matrix_jacobian(scope, rng):
    X = scope.Variable(3, 2)
    f = sp.reshape(X, 6, 1)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))
