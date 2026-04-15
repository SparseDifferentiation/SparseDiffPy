"""Tests with row vectors (1, n) to exercise different code paths."""

import numpy as np
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_point


class TestRowVectorBasics:
    def test_row_variable_forward(self, scope, rng):
        r = scope.Variable(1, 4)
        fn = sp.compile(r)
        x0 = random_point(scope, rng)
        np.testing.assert_allclose(fn.forward(), x0)

    def test_row_sin_jacobian(self, scope, rng):
        r = scope.Variable(1, 4)
        f = sp.sin(r)
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng))

    def test_row_sin_hessian(self, scope, rng):
        r = scope.Variable(1, 4)
        f = sp.sin(r)
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        x0 = random_point(scope, rng)
        checker.check_hessian(x0, rng.standard_normal(4))


class TestRowVectorIndexing:
    def test_row_scalar_index(self, scope, rng):
        r = scope.Variable(1, 4)
        f = r[2]
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng))

    def test_row_scalar_index_forward(self, scope, rng):
        r = scope.Variable(1, 4)
        f = r[2]
        fn = sp.compile(f)
        x0 = random_point(scope, rng)
        np.testing.assert_allclose(fn.forward(), [x0[2]])

    def test_row_slice_index(self, scope, rng):
        r = scope.Variable(1, 4)
        f = r[1:3]
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng))


class TestRowVectorBroadcast:
    def test_row_plus_scalar(self, scope, rng):
        r = scope.Variable(1, 3)
        f = r + 1.0
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng))

    def test_row_plus_matrix(self, scope, rng):
        r = scope.Variable(1, 3)
        M = scope.Variable(4, 3)
        f = r + M
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng))

    def test_scalar_times_row(self, scope, rng):
        r = scope.Variable(1, 4)
        f = 2.5 * r
        fn = sp.compile(f)
        x0 = random_point(scope, rng)
        np.testing.assert_allclose(fn.forward(), 2.5 * x0)


class TestRowVectorMatmul:
    def test_row_times_matrix(self, scope, rng):
        """(1,3) @ (3,2) = (1,2)"""
        r = scope.Variable(1, 3)
        A = rng.standard_normal((3, 2))
        f = r @ A
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng))

    def test_row_times_column(self, scope, rng):
        """(1,3) @ (3,1) = (1,1) — dot product."""
        r = scope.Variable(1, 3)
        c = scope.Variable(3, 1)
        f = r @ c
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        x0 = random_point(scope, rng)
        checker.check_jacobian(x0)
        checker.check_hessian(x0, np.array([1.0]))

    def test_matrix_times_row_transpose(self, scope, rng):
        """A @ r.T where r is (1,3) -> r.T is (3,1)."""
        r = scope.Variable(1, 3)
        A = rng.standard_normal((4, 3))
        f = A @ r.T
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng))


class TestRowVectorTranspose:
    def test_row_transpose_is_column(self, scope, rng):
        r = scope.Variable(1, 4)
        f = r.T
        assert f.shape == (4, 1)
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng))

    def test_column_transpose_is_row(self, scope, rng):
        c = scope.Variable(4, 1)
        f = c.T
        assert f.shape == (1, 4)
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng))


class TestRowVectorReductions:
    def test_sum_row(self, scope, rng):
        r = scope.Variable(1, 4)
        f = sp.sum(r)
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng))

    def test_sum_row_forward(self, scope, rng):
        r = scope.Variable(1, 4)
        f = sp.sum(r)
        fn = sp.compile(f)
        x0 = random_point(scope, rng)
        np.testing.assert_allclose(fn.forward(), [np.sum(x0)])
