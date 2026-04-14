import numpy as np
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_positive_point


def test_rel_entr_jacobian(scope, rng):
    x = scope.Variable(3, 1)
    y = scope.Variable(3, 1)
    f = sp.rel_entr(x, y)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_positive_point(scope, rng))


def test_rel_entr_hessian(scope, rng):
    x = scope.Variable(3, 1)
    y = scope.Variable(3, 1)
    f = sp.rel_entr(x, y)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    x0 = random_positive_point(scope, rng)
    checker.check_hessian(x0, rng.standard_normal(3))


def test_rel_entr_forward(scope, rng):
    x = scope.Variable(3, 1)
    y = scope.Variable(3, 1)
    f = sp.rel_entr(x, y)
    fn = sp.compile(f)
    x0 = random_positive_point(scope, rng)
    x_val = x.value
    y_val = y.value
    expected = x_val * np.log(x_val / y_val)
    np.testing.assert_allclose(fn.forward(), expected)
