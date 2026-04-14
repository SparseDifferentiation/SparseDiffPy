import numpy as np
import scipy.sparse
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_point


def test_quad_form_identity_jacobian(scope, rng):
    x = scope.Variable(3, 1)
    Q = scipy.sparse.eye(3, format="csr")
    f = sp.quad_form(x, Q)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_quad_form_identity_hessian(scope, rng):
    x = scope.Variable(3, 1)
    Q = scipy.sparse.eye(3, format="csr")
    f = sp.quad_form(x, Q)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    x0 = random_point(scope, rng)
    checker.check_hessian(x0, np.array([1.0]))


def test_quad_form_dense_jacobian(scope, rng):
    x = scope.Variable(3, 1)
    Q = rng.standard_normal((3, 3))
    Q = Q.T @ Q  # make positive definite
    f = sp.quad_form(x, Q)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_quad_form_forward(scope, rng):
    x = scope.Variable(3, 1)
    Q = np.eye(3) * 2.0
    f = sp.quad_form(x, Q)
    fn = sp.compile(f)
    x0 = random_point(scope, rng)
    expected = x0 @ (2.0 * x0)
    np.testing.assert_allclose(fn.forward(), [expected], rtol=1e-10)
