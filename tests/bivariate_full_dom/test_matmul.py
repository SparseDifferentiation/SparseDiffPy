import numpy as np
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_point


def test_matmul_jacobian(scope, rng):
    X = scope.Variable(2, 3)
    Y = scope.Variable(3, 2)
    f = X @ Y
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_matmul_hessian(scope, rng):
    X = scope.Variable(2, 3)
    Y = scope.Variable(3, 2)
    f = X @ Y
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    x0 = random_point(scope, rng)
    checker.check_hessian(x0, rng.standard_normal(4))


def test_matmul_vec_jacobian(scope, rng):
    """Row vector @ column vector = scalar."""
    x = scope.Variable(1, 3)
    y = scope.Variable(3, 1)
    f = x @ y
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))
