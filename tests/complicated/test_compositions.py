"""Complicated composition tests.

Each test builds a deep or wide expression involving many atoms, then
verifies forward (against manual NumPy), Jacobian, and Hessian.
"""

import numpy as np
import pytest
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_point, random_positive_point


# -----------------------------------------------------------------------
# 1. Affine chain: A @ x + b
# -----------------------------------------------------------------------

class TestAffineChain:
    def _build(self, scope, rng):
        x = scope.Variable(3, 1)
        A = rng.standard_normal((3, 3))
        b = rng.standard_normal(3)
        f = A @ x + b
        fn = sp.compile(f)
        return fn, x, A, b

    def test_forward(self, scope, rng):
        fn, x, A, b = self._build(scope, rng)
        x0 = random_point(scope, rng)
        np.testing.assert_allclose(fn.forward(), A @ x0 + b, rtol=1e-10)

    def test_jacobian(self, scope, rng):
        fn, x, A, b = self._build(scope, rng)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng))

    def test_hessian(self, scope, rng):
        fn, x, A, b = self._build(scope, rng)
        checker = NumericalDerivativeChecker(fn, scope)
        x0 = random_point(scope, rng)
        checker.check_hessian(x0, rng.standard_normal(3))


# -----------------------------------------------------------------------
# 2. Nonlinear composition: exp(A @ x) + sin(x)
# -----------------------------------------------------------------------

class TestNonlinearComposition:
    def _build(self, scope, rng):
        x = scope.Variable(3, 1)
        A = rng.standard_normal((3, 3))
        f = sp.exp(A @ x) + sp.sin(x)
        fn = sp.compile(f)
        return fn, x, A

    def test_forward(self, scope, rng):
        fn, x, A = self._build(scope, rng)
        x0 = random_point(scope, rng, low=-0.5, high=0.5)
        np.testing.assert_allclose(
            fn.forward(), np.exp(A @ x0) + np.sin(x0), rtol=1e-10
        )

    def test_jacobian(self, scope, rng):
        fn, x, A = self._build(scope, rng)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng, low=-0.5, high=0.5))

    def test_hessian(self, scope, rng):
        fn, x, A = self._build(scope, rng)
        checker = NumericalDerivativeChecker(fn, scope)
        x0 = random_point(scope, rng, low=-0.5, high=0.5)
        checker.check_hessian(x0, rng.standard_normal(3))


# -----------------------------------------------------------------------
# 3. Matrix expression: sin(X) * Y  and  X.T @ Y  (tested separately)
# -----------------------------------------------------------------------

class TestMatrixExpression:
    def _build_elementwise(self, scope, rng):
        X = scope.Variable(3, 2)
        Y = scope.Variable(3, 2)
        f = sp.sin(X) * Y
        fn = sp.compile(f)
        return fn, X, Y

    def test_elementwise_forward(self, scope, rng):
        fn, X, Y = self._build_elementwise(scope, rng)
        x0 = random_point(scope, rng)
        X_val = X.value.reshape(3, 2, order="F")
        Y_val = Y.value.reshape(3, 2, order="F")
        expected = (np.sin(X_val) * Y_val).ravel(order="F")
        np.testing.assert_allclose(fn.forward(), expected, rtol=1e-10)

    def test_elementwise_jacobian(self, scope, rng):
        fn, X, Y = self._build_elementwise(scope, rng)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng))

    def test_elementwise_hessian(self, scope, rng):
        fn, X, Y = self._build_elementwise(scope, rng)
        checker = NumericalDerivativeChecker(fn, scope)
        x0 = random_point(scope, rng)
        checker.check_hessian(x0, rng.standard_normal(6))

    def _build_matmul(self, scope, rng):
        X = scope.Variable(3, 2)
        Y = scope.Variable(3, 2)
        f = X.T @ Y  # (2,3) @ (3,2) = (2,2)
        fn = sp.compile(f)
        return fn, X, Y

    def test_matmul_jacobian(self, scope, rng):
        fn, X, Y = self._build_matmul(scope, rng)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng))

    def test_matmul_hessian(self, scope, rng):
        fn, X, Y = self._build_matmul(scope, rng)
        checker = NumericalDerivativeChecker(fn, scope)
        x0 = random_point(scope, rng)
        checker.check_hessian(x0, rng.standard_normal(4))


# -----------------------------------------------------------------------
# 4. Broadcast heavy: a * X + r + c
#    a: scalar (1,1), X: matrix (3,2), r: row (1,2), c: column (3,1)
# -----------------------------------------------------------------------

class TestBroadcastHeavy:
    def _build(self, scope, rng):
        a = scope.Variable(1, 1)
        X = scope.Variable(3, 2)
        r = scope.Variable(1, 2)
        c = scope.Variable(3, 1)
        f = a * X + r + c
        fn = sp.compile(f)
        return fn, a, X, r, c

    def test_forward(self, scope, rng):
        fn, a, X, r, c = self._build(scope, rng)
        x0 = random_point(scope, rng)
        a_val = a.value[0]
        X_val = X.value.reshape(3, 2, order="F")
        r_val = r.value.reshape(1, 2, order="F")
        c_val = c.value.reshape(3, 1, order="F")
        expected = (a_val * X_val + r_val + c_val).ravel(order="F")
        np.testing.assert_allclose(fn.forward(), expected, rtol=1e-10)

    def test_jacobian(self, scope, rng):
        fn, *_ = self._build(scope, rng)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng))

    def test_hessian(self, scope, rng):
        fn, *_ = self._build(scope, rng)
        checker = NumericalDerivativeChecker(fn, scope)
        x0 = random_point(scope, rng)
        checker.check_hessian(x0, rng.standard_normal(6))


# -----------------------------------------------------------------------
# 5. Index into composition: sum(exp(x[0:3]) + log(x[3:6]))
# -----------------------------------------------------------------------

class TestIndexIntoComposition:
    """Index + elementwise composition tests.

    Full-domain ops (exp, sin, etc.) work on indexed expressions.
    Restricted-domain ops (log, tan, atanh, entr) raise an error when
    applied directly to an index node (C engine limitation).
    """
    def test_full_domain_on_index(self, scope, rng):
        """exp on indexed variable works correctly."""
        x = scope.Variable(6, 1)
        f = sp.exp(x[3:6])
        fn = sp.compile(f)
        x0 = random_positive_point(scope, rng)
        np.testing.assert_allclose(fn.forward(), np.exp(x0[3:6]))
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(x0)

    def test_restricted_domain_on_index_raises(self, scope, rng):
        """log on indexed variable raises ValueError."""
        x = scope.Variable(6, 1)
        with pytest.raises(ValueError, match="log cannot be applied directly"):
            sp.log(x[3:6])

    def test_workaround_separate_variables(self, scope, rng):
        """Use separate variables as workaround for restricted-domain + index."""
        a = scope.Variable(3, 1)
        b = scope.Variable(3, 1)
        f = sp.sum(sp.exp(a) + sp.log(b))
        fn = sp.compile(f)
        x0 = random_positive_point(scope, rng)
        a_val = a.value
        b_val = b.value
        expected = np.sum(np.exp(a_val) + np.log(b_val))
        np.testing.assert_allclose(fn.forward(), [expected], rtol=1e-10)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(x0)
        checker.check_hessian(x0, np.array([1.0]))


# -----------------------------------------------------------------------
# 6. Hstack mixed: hstack([sin(x), A @ x, y])
# -----------------------------------------------------------------------

class TestHstackMixed:
    def _build(self, scope, rng):
        x = scope.Variable(3, 1)
        y = scope.Variable(3, 1)
        A = rng.standard_normal((3, 3))
        f = sp.hstack([sp.sin(x), A @ x, y])
        fn = sp.compile(f)
        return fn, x, y, A

    def test_forward(self, scope, rng):
        fn, x, y, A = self._build(scope, rng)
        x0 = random_point(scope, rng)
        x_val = x.value
        y_val = y.value
        expected = np.concatenate([np.sin(x_val), A @ x_val, y_val])
        np.testing.assert_allclose(fn.forward(), expected, rtol=1e-10)

    def test_jacobian(self, scope, rng):
        fn, *_ = self._build(scope, rng)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng))

    def test_hessian(self, scope, rng):
        fn, *_ = self._build(scope, rng)
        checker = NumericalDerivativeChecker(fn, scope)
        x0 = random_point(scope, rng)
        checker.check_hessian(x0, rng.standard_normal(9))


# -----------------------------------------------------------------------
# 7. Multi-compile shared scope
# -----------------------------------------------------------------------

class TestMultiCompileSharedScope:
    def test_two_expressions_same_scope(self, scope, rng):
        x = scope.Variable(3, 1)
        f = sp.sin(x)
        g = sp.exp(x)
        fn_f = sp.compile(f)
        fn_g = sp.compile(g)

        x0 = random_point(scope, rng)
        np.testing.assert_allclose(fn_f.forward(), np.sin(x0), rtol=1e-10)
        np.testing.assert_allclose(fn_g.forward(), np.exp(x0), rtol=1e-10)

        # Jacobians are independent
        J_f = fn_f.jacobian().toarray()
        J_g = fn_g.jacobian().toarray()
        np.testing.assert_allclose(J_f, np.diag(np.cos(x0)), rtol=1e-10)
        np.testing.assert_allclose(J_g, np.diag(np.exp(x0)), rtol=1e-10)

    def test_shared_scope_different_variables(self, scope, rng):
        x = scope.Variable(3, 1)
        y = scope.Variable(2, 1)
        f = sp.sin(x)
        g = sp.exp(y)
        fn_f = sp.compile(f)
        fn_g = sp.compile(g)

        x0 = random_point(scope, rng)
        x_val = x.value
        y_val = y.value
        np.testing.assert_allclose(fn_f.forward(), np.sin(x_val), rtol=1e-10)
        np.testing.assert_allclose(fn_g.forward(), np.exp(y_val), rtol=1e-10)

        # f's Jacobian is 3x5 (3 outputs, 5 total vars), g's is 2x5
        J_f = fn_f.jacobian().toarray()
        J_g = fn_g.jacobian().toarray()
        assert J_f.shape == (3, 5)
        assert J_g.shape == (2, 5)
        # f depends on x (cols 0-2), not y (cols 3-4)
        np.testing.assert_allclose(J_f[:, 3:], 0.0)
        # g depends on y (cols 3-4), not x (cols 0-2)
        np.testing.assert_allclose(J_g[:, 0:3], 0.0)


# -----------------------------------------------------------------------
# 8. Deep chain: exp(sin(tanh(A @ x + b)))
# -----------------------------------------------------------------------

class TestDeepChain:
    def _build(self, scope, rng):
        x = scope.Variable(3, 1)
        A = rng.standard_normal((3, 3)) * 0.5  # scale down to avoid exp overflow
        b = rng.standard_normal(3) * 0.1
        f = sp.exp(sp.sin(sp.tanh(A @ x + b)))
        fn = sp.compile(f)
        return fn, x, A, b

    def test_forward(self, scope, rng):
        fn, x, A, b = self._build(scope, rng)
        x0 = random_point(scope, rng, low=-0.5, high=0.5)
        expected = np.exp(np.sin(np.tanh(A @ x0 + b)))
        np.testing.assert_allclose(fn.forward(), expected, rtol=1e-10)

    def test_jacobian(self, scope, rng):
        fn, *_ = self._build(scope, rng)
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(random_point(scope, rng, low=-0.5, high=0.5))

    def test_hessian(self, scope, rng):
        fn, *_ = self._build(scope, rng)
        checker = NumericalDerivativeChecker(fn, scope)
        x0 = random_point(scope, rng, low=-0.5, high=0.5)
        checker.check_hessian(x0, rng.standard_normal(3))


# -----------------------------------------------------------------------
# 9. Matrix hessian: sin(A @ X) with matrix variable and 2D weights
# -----------------------------------------------------------------------

class TestMatrixHessian:
    def test_sin_AX_hessian(self, scope, rng):
        X = scope.Variable(3, 3)
        A = rng.standard_normal((3, 3))
        f = sp.sin(A @ X)
        fn = sp.compile(f)
        checker = NumericalDerivativeChecker(fn, scope)
        x0 = random_point(scope, rng, low=-0.5, high=0.5)
        weights = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        checker.check_hessian(x0, weights.ravel(order='F'))

    def test_sin_AX_hessian_2d_weights(self, scope, rng):
        """Passing weights as a 2D array — hessian() should flatten column-major."""
        X = scope.Variable(3, 3)
        A = rng.standard_normal((3, 3))
        f = sp.sin(A @ X)
        fn = sp.compile(f)
        x0 = random_point(scope, rng, low=-0.5, high=0.5)
        weights = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)

        # 2D weights should be flattened column-major internally
        fn.forward()
        fn.jacobian()
        H_2d = fn.hessian(weights)

        # Compare against explicitly flattened F-order weights
        fn.forward()
        fn.jacobian()
        H_flat = fn.hessian(weights.ravel(order='F'))

        np.testing.assert_allclose(H_2d.toarray(), H_flat.toarray())
