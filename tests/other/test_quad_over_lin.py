import numpy as np
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_positive_point


def test_quad_over_lin_jacobian(scope, rng):
    x = scope.Variable(3, 1)
    z = scope.Variable(1, 1)
    f = sp.quad_over_lin(x, z)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_positive_point(scope, rng))


def test_quad_over_lin_hessian(scope, rng):
    x = scope.Variable(3, 1)
    z = scope.Variable(1, 1)
    f = sp.quad_over_lin(x, z)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    x0 = random_positive_point(scope, rng)
    checker.check_hessian(x0, np.array([1.0]))


def test_quad_over_lin_forward(scope, rng):
    x = scope.Variable(3, 1)
    z = scope.Variable(1, 1)
    f = sp.quad_over_lin(x, z)
    fn = sp.compile(f)
    x0 = random_positive_point(scope, rng)
    x_val = x.value
    z_val = z.value[0]
    expected = np.sum(x_val ** 2) / z_val
    np.testing.assert_allclose(fn.forward(), [expected], rtol=1e-10)
