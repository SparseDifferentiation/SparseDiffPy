import numpy as np
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_point


def test_hstack_vectors_jacobian(scope, rng):
    x = scope.Variable(3, 1)
    y = scope.Variable(3, 1)
    f = sp.hstack([x, y])
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_hstack_vectors_forward(scope, rng):
    x = scope.Variable(3, 1)
    y = scope.Variable(3, 1)
    f = sp.hstack([x, y])
    fn = sp.compile(f)
    x0 = random_point(scope, rng)
    x_val = x.value
    y_val = y.value
    np.testing.assert_allclose(fn.forward(), np.concatenate([x_val, y_val]))


def test_hstack_three_jacobian(scope, rng):
    a = scope.Variable(2, 1)
    b = scope.Variable(2, 1)
    c = scope.Variable(2, 1)
    f = sp.hstack([a, b, c])
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))
