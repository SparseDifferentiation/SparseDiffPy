import numpy as np
import scipy.sparse
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_point


def test_left_matmul_dense_jacobian(scope, rng):
    x = scope.Variable(3, 1)
    A = rng.standard_normal((4, 3))
    f = A @ x
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_left_matmul_dense_forward(scope, rng):
    x = scope.Variable(3, 1)
    A = rng.standard_normal((4, 3))
    f = A @ x
    fn = sp.compile(f)
    x0 = random_point(scope, rng)
    np.testing.assert_allclose(fn.forward(), A @ x0, rtol=1e-10)


def test_left_matmul_sparse_jacobian(scope, rng):
    x = scope.Variable(3, 1)
    A = scipy.sparse.eye(3, format="csr") * 2.0
    f = A @ x
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_left_matmul_sparse_forward(scope, rng):
    x = scope.Variable(3, 1)
    A = scipy.sparse.eye(3, format="csr") * 3.0
    f = A @ x
    fn = sp.compile(f)
    x0 = random_point(scope, rng)
    np.testing.assert_allclose(fn.forward(), 3.0 * x0)


def test_left_matmul_parameter_jacobian(scope, rng):
    x = scope.Variable(3, 1)
    A = scope.Parameter(4, 3)
    A.value = rng.standard_normal((4, 3))
    f = A @ x
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_left_matmul_parameter_update(scope, rng):
    x = scope.Variable(3, 1)
    A = scope.Parameter(3, 3)
    A.value = np.eye(3)
    f = A @ x
    fn = sp.compile(f)
    x0 = random_point(scope, rng)
    np.testing.assert_allclose(fn.forward(), x0)
    A.value = 2 * np.eye(3)
    np.testing.assert_allclose(fn.forward(), 2 * x0)
