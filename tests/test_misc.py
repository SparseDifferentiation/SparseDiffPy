"""Miscellaneous tests: hessians for affine atoms, re-evaluation, negative indexing,
parameter jacobian after update, x.T@x, sub with broadcast, scope roundtrip,
compile twice, degenerate cases."""

import numpy as np
import pytest
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_point, random_positive_point


# ---------------------------------------------------------------------------
# Hessian for affine atoms (should be zero, but verify through compositions)
# ---------------------------------------------------------------------------

class TestAffineHessians:
    def test_neg_hessian_is_zero(self, scope, rng):
        x = scope.Variable(3, 1)
        f = -x
        fn = sp.compile(f)
        x0 = random_point(scope, rng)
        fn.forward()
        fn.jacobian()
        H = fn.hessian(rng.standard_normal(3))
        np.testing.assert_allclose(H.toarray(), np.zeros((3, 3)), atol=1e-14)

    def test_add_hessian_is_zero(self, scope, rng):
        x = scope.Variable(3, 1)
        y = scope.Variable(3, 1)
        f = x + y
        fn = sp.compile(f)
        x0 = random_point(scope, rng)
        fn.forward()
        fn.jacobian()
        H = fn.hessian(rng.standard_normal(3))
        np.testing.assert_allclose(H.toarray(), np.zeros((6, 6)), atol=1e-14)

    def test_hstack_hessian_is_zero(self, scope, rng):
        x = scope.Variable(3, 1)
        y = scope.Variable(3, 1)
        f = sp.hstack([x, y])
        fn = sp.compile(f)
        x0 = random_point(scope, rng)
        fn.forward()
        fn.jacobian()
        H = fn.hessian(rng.standard_normal(6))
        np.testing.assert_allclose(H.toarray(), np.zeros((6, 6)), atol=1e-14)

    def test_index_hessian_is_zero(self, scope, rng):
        x = scope.Variable(4, 1)
        f = x[1:3]
        fn = sp.compile(f)
        x0 = random_point(scope, rng)
        fn.forward()
        fn.jacobian()
        H = fn.hessian(rng.standard_normal(2))
        np.testing.assert_allclose(H.toarray(), np.zeros((4, 4)), atol=1e-14)

    def test_sum_hessian_is_zero(self, scope, rng):
        x = scope.Variable(3, 2)
        f = sp.sum(x)
        fn = sp.compile(f)
        x0 = random_point(scope, rng)
        fn.forward()
        fn.jacobian()
        H = fn.hessian(np.array([1.0]))
        np.testing.assert_allclose(H.toarray(), np.zeros((6, 6)), atol=1e-14)

    def test_sin_of_sum_hessian_nonzero(self, scope, rng):
        """sin(sum(x)) — affine feeding into nonlinear produces nonzero Hessian."""
        x = scope.Variable(3, 1)
        f = sp.sin(sp.sum(x))
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        x0 = random_point(scope, rng)
        checker.check_hessian(x0, np.array([1.0]))


# ---------------------------------------------------------------------------
# Re-evaluation after value change
# ---------------------------------------------------------------------------

class TestReEvaluation:
    def test_forward_updates_with_new_values(self, scope, rng):
        x = scope.Variable(3, 1)
        f = sp.sin(x)
        fn = sp.compile(f)

        x.value = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(fn.forward(), np.sin([0, 0, 0]))

        x.value = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(fn.forward(), np.sin([1, 2, 3]))

    def test_jacobian_updates_with_new_values(self, scope, rng):
        x = scope.Variable(3, 1)
        f = sp.sin(x)
        fn = sp.compile(f)

        x.value = np.array([0.0, 0.0, 0.0])
        fn.forward()
        J1 = fn.jacobian().toarray()
        np.testing.assert_allclose(np.diag(J1), np.cos([0, 0, 0]))

        x.value = np.array([1.0, 2.0, 3.0])
        fn.forward()
        J2 = fn.jacobian().toarray()
        np.testing.assert_allclose(np.diag(J2), np.cos([1, 2, 3]))

    def test_hessian_updates_with_new_values(self, scope, rng):
        x = scope.Variable(3, 1)
        f = sp.sin(x)
        fn = sp.compile(f)
        w = np.ones(3)

        x.value = np.array([0.0, 0.0, 0.0])
        fn.forward()
        fn.jacobian()
        H1 = fn.hessian(w).toarray()
        np.testing.assert_allclose(np.diag(H1), -np.sin([0, 0, 0]), atol=1e-14)

        x.value = np.array([1.0, 2.0, 3.0])
        fn.forward()
        fn.jacobian()
        H2 = fn.hessian(w).toarray()
        np.testing.assert_allclose(np.diag(H2), -np.sin([1, 2, 3]))


# ---------------------------------------------------------------------------
# Negative indexing
# ---------------------------------------------------------------------------

class TestNegativeIndexing:
    def test_negative_scalar_index(self, scope, rng):
        x = scope.Variable(4, 1)
        f = x[-1]
        fn = sp.compile(f)
        x0 = random_point(scope, rng)
        np.testing.assert_allclose(fn.forward(), [x0[-1]])

    def test_negative_scalar_index_jacobian(self, scope, rng):
        x = scope.Variable(4, 1)
        f = x[-1]
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng))

    def test_negative_slice(self, scope, rng):
        x = scope.Variable(4, 1)
        f = x[-2:]
        fn = sp.compile(f)
        x0 = random_point(scope, rng)
        np.testing.assert_allclose(fn.forward(), x0[-2:])

    def test_negative_fancy(self, scope, rng):
        x = scope.Variable(4, 1)
        f = x[[-1, -3]]
        fn = sp.compile(f)
        x0 = random_point(scope, rng)
        np.testing.assert_allclose(fn.forward(), x0[[-1, -3]])


# ---------------------------------------------------------------------------
# Parameter Jacobian after update
# ---------------------------------------------------------------------------

class TestParameterJacobianAfterUpdate:
    def test_left_matmul_jacobian_after_update(self, scope, rng):
        x = scope.Variable(3, 1)
        A = scope.Parameter(3, 3)
        A.value = np.eye(3)
        f = A @ x
        fn = sp.compile(f)

        x0 = random_point(scope, rng)
        fn.forward()
        J1 = fn.jacobian().toarray()
        np.testing.assert_allclose(J1, np.eye(3), atol=1e-14)

        A.value = 2 * np.eye(3)
        fn.forward()
        J2 = fn.jacobian().toarray()
        np.testing.assert_allclose(J2, 2 * np.eye(3), atol=1e-14)

    def test_scalar_mult_jacobian_after_update(self, scope, rng):
        x = scope.Variable(3, 1)
        a = scope.Parameter(1, 1)
        a.value = np.array([[3.0]])
        f = a * x
        fn = sp.compile(f)

        x0 = random_point(scope, rng)
        fn.forward()
        J1 = fn.jacobian().toarray()
        np.testing.assert_allclose(J1, 3.0 * np.eye(3), atol=1e-14)

        a.value = np.array([[7.0]])
        fn.forward()
        J2 = fn.jacobian().toarray()
        np.testing.assert_allclose(J2, 7.0 * np.eye(3), atol=1e-14)


# ---------------------------------------------------------------------------
# x.T @ x pattern
# ---------------------------------------------------------------------------

class TestTransposeMatmul:
    def test_xT_x_forward(self, scope, rng):
        x = scope.Variable(3, 1)
        f = x.T @ x
        fn = sp.compile(f)
        x0 = random_point(scope, rng)
        np.testing.assert_allclose(fn.forward(), [x0 @ x0], rtol=1e-10)

    def test_xT_x_jacobian(self, scope, rng):
        x = scope.Variable(3, 1)
        f = x.T @ x
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng))

    def test_xT_x_hessian(self, scope, rng):
        x = scope.Variable(3, 1)
        f = x.T @ x
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        x0 = random_point(scope, rng)
        checker.check_hessian(x0, np.array([1.0]))

    def test_xT_A_x(self, scope, rng):
        """x.T @ A @ x where A is a constant matrix."""
        x = scope.Variable(3, 1)
        A = rng.standard_normal((3, 3))
        A = A + A.T  # symmetric
        f = x.T @ (A @ x)
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        x0 = random_point(scope, rng)
        checker.check_jacobian(x0)
        checker.check_hessian(x0, np.array([1.0]))


# ---------------------------------------------------------------------------
# Subtraction with broadcasting
# ---------------------------------------------------------------------------

class TestSubBroadcast:
    def test_sub_scalar_broadcast(self, scope, rng):
        X = scope.Variable(3, 2)
        a = scope.Variable(1, 1)
        f = X - a
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng))

    def test_sub_column_broadcast(self, scope, rng):
        X = scope.Variable(3, 2)
        c = scope.Variable(3, 1)
        f = X - c
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng))

    def test_rsub_broadcast(self, scope, rng):
        x = scope.Variable(3, 1)
        f = np.array([10.0, 20.0, 30.0]) - x
        fn = sp.compile(f)
        x0 = random_point(scope, rng)
        np.testing.assert_allclose(
            fn.forward(), np.array([10.0, 20.0, 30.0]) - x0
        )


# ---------------------------------------------------------------------------
# Scope set_values / get_values roundtrip
# ---------------------------------------------------------------------------

class TestScopeRoundtrip:
    def test_set_get_roundtrip(self, scope, rng):
        x = scope.Variable(3, 1)
        y = scope.Variable(2, 1)
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        scope.set_values(vals)
        np.testing.assert_allclose(scope.get_values(), vals)
        np.testing.assert_allclose(x.value, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(y.value, [4.0, 5.0])

    def test_variable_value_writes_to_scope(self, scope, rng):
        x = scope.Variable(3, 1)
        y = scope.Variable(2, 1)
        x.value = np.array([10.0, 20.0, 30.0])
        y.value = np.array([40.0, 50.0])
        np.testing.assert_allclose(
            scope.get_values(), [10.0, 20.0, 30.0, 40.0, 50.0]
        )


# ---------------------------------------------------------------------------
# Compile same expression twice
# ---------------------------------------------------------------------------

class TestCompileTwice:
    def test_two_compiles_independent(self, scope, rng):
        x = scope.Variable(3, 1)
        f = sp.sin(x)
        fn1 = sp.compile(f)
        fn2 = sp.compile(f)

        x.value = np.array([1.0, 2.0, 3.0])
        f1 = fn1.forward()
        f2 = fn2.forward()
        np.testing.assert_allclose(f1, f2)
        np.testing.assert_allclose(
            fn1.jacobian().toarray(), fn2.jacobian().toarray()
        )


# ---------------------------------------------------------------------------
# Degenerate / edge cases
# ---------------------------------------------------------------------------

class TestDegenerateCases:
    def test_hstack_single(self, scope, rng):
        x = scope.Variable(3, 1)
        f = sp.hstack([x])
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng))

    def test_vstack_single(self, scope, rng):
        x = scope.Variable(3, 1)
        f = sp.vstack([x])
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng))

    def test_sum_scalar(self, scope, rng):
        x = scope.Variable(1, 1)
        f = sp.sum(x)
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng))

    def test_scalar_variable(self, scope, rng):
        x = scope.Variable(1, 1)
        f = sp.sin(x)
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        x0 = random_point(scope, rng)
        checker.check_jacobian(x0)
        checker.check_hessian(x0, np.array([1.0]))

    def test_identity_expression(self, scope, rng):
        """Compiling just a variable."""
        x = scope.Variable(3, 1)
        fn = sp.compile(x)
        x0 = random_point(scope, rng)
        np.testing.assert_allclose(fn.forward(), x0)
        J = fn.jacobian().toarray()  # forward() was just called above
        np.testing.assert_allclose(J, np.eye(3))

    def test_constant_expression_raises(self, scope, rng):
        """Compiling a constant (no variables) should raise."""
        from sparsediffpy._core._constants import Constant
        c = Constant(np.array([1.0, 2.0, 3.0]), (3, 1))
        with pytest.raises(ValueError, match="at least one Variable"):
            sp.compile(c)

    def test_nested_transpose(self, scope, rng):
        """x.T.T should be x."""
        x = scope.Variable(3, 1)
        f = x.T.T
        assert f.shape == (3, 1)
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng))

    def test_power_one(self, scope, rng):
        """x**1 should be identity."""
        x = scope.Variable(3, 1)
        f = x ** 1
        fn = sp.compile(f)
        x0 = random_point(scope, rng)
        np.testing.assert_allclose(fn.forward(), x0)

    def test_power_zero(self, scope, rng):
        """x**0 should be ones."""
        x = scope.Variable(3, 1)
        f = x ** 0
        fn = sp.compile(f)
        x0 = random_positive_point(scope, rng)
        np.testing.assert_allclose(fn.forward(), np.ones(3))
