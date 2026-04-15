import numpy as np
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_positive_point


def test_prod_all_jacobian(scope, rng):
    x = scope.Variable(3, 1)
    f = sp.prod(x)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_positive_point(scope, rng))


def test_prod_all_hessian(scope, rng):
    x = scope.Variable(3, 1)
    f = sp.prod(x)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    x0 = random_positive_point(scope, rng)
    checker.check_hessian(x0, np.array([1.0]))


def test_prod_all_forward(scope, rng):
    x = scope.Variable(3, 1)
    f = sp.prod(x)
    fn = sp.compile(f)
    x0 = random_positive_point(scope, rng)
    np.testing.assert_allclose(fn.forward(), [np.prod(x0)])


def test_prod_axis_zero_jacobian(scope, rng):
    X = scope.Variable(3, 2)
    f = sp.prod(X, axis=0)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_positive_point(scope, rng))


def test_prod_axis_one_jacobian(scope, rng):
    X = scope.Variable(3, 2)
    f = sp.prod(X, axis=1)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_positive_point(scope, rng))
