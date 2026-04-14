import numpy as np
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_point


def test_logistic_vector_jacobian(scope, rng):
    x = scope.Variable(4, 1)
    f = sp.logistic(x)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_logistic_vector_hessian(scope, rng):
    x = scope.Variable(4, 1)
    f = sp.logistic(x)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    x0 = random_point(scope, rng)
    checker.check_hessian(x0, rng.standard_normal(4))


def test_logistic_forward(scope, rng):
    """The C logistic function is the softplus: log(1 + exp(x))."""
    x = scope.Variable(3, 1)
    f = sp.logistic(x)
    fn = sp.compile(f)
    x0 = random_point(scope, rng)
    expected = np.log(1.0 + np.exp(x0))
    np.testing.assert_allclose(fn.forward(), expected)
