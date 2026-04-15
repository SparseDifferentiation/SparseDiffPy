import numpy as np
import scipy.sparse
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_point


def test_right_matmul_dense_jacobian(scope, rng):
    x = scope.Variable(1, 3)
    A = rng.standard_normal((3, 4))
    f = x @ A
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_right_matmul_dense_forward(scope, rng):
    x = scope.Variable(1, 3)
    A = rng.standard_normal((3, 4))
    f = x @ A
    fn = sp.compile(f)
    x0 = random_point(scope, rng)
    x_mat = x0.reshape(1, 3)
    expected = (x_mat @ A).ravel(order="F")
    np.testing.assert_allclose(fn.forward(), expected, rtol=1e-10)


def test_right_matmul_sparse_jacobian(scope, rng):
    x = scope.Variable(1, 3)
    A = scipy.sparse.eye(3, format="csr") * 2.0
    f = x @ A
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))
