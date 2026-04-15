import numpy as np
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_point


def test_diag_vec_jacobian(scope, rng):
    x = scope.Variable(3, 1)
    f = sp.diag_vec(x)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_diag_vec_forward(scope, rng):
    x = scope.Variable(3, 1)
    f = sp.diag_vec(x)
    fn = sp.compile(f)
    x0 = random_point(scope, rng)
    result = fn.forward()
    # diag_vec produces a 3x3 matrix, flattened column-major
    expected = np.diag(x0).ravel(order="F")
    np.testing.assert_allclose(result, expected)
