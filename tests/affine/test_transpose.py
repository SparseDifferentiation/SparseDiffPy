import numpy as np
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_point


def test_transpose_matrix_jacobian(scope, rng):
    X = scope.Variable(3, 2)
    f = X.T
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_transpose_vector_jacobian(scope, rng):
    x = scope.Variable(4, 1)
    f = x.T
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))
