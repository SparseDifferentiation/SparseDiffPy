import numpy as np
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_positive_point


def test_log_vector_forward(scope, rng):
    x = scope.Variable(4, 1)
    f = sp.log(x)
    fn = sp.compile(f)
    x0 = random_positive_point(scope, rng)
    np.testing.assert_allclose(fn.forward(), np.log(x0))


def test_log_vector_jacobian(scope, rng):
    x = scope.Variable(4, 1)
    f = sp.log(x)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_positive_point(scope, rng))


def test_log_vector_hessian(scope, rng):
    x = scope.Variable(4, 1)
    f = sp.log(x)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    x0 = random_positive_point(scope, rng)
    checker.check_hessian(x0, rng.standard_normal(4))


def test_log_matrix_jacobian(scope, rng):
    X = scope.Variable(3, 2)
    f = sp.log(X)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_positive_point(scope, rng))
