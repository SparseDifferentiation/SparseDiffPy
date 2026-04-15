"""Test utilities: numerical derivative checker and helpers."""

import numpy as np
import sparsediffpy as sp


class NumericalDerivativeChecker:
    """Check Jacobian and Hessian of a compiled expression against
    central finite differences.

    Usage:
        checker = NumericalDerivativeChecker(fn, scope)
        checker.check_jacobian(x0)
        checker.check_hessian(x0, weights)
    """

    def __init__(self, compiled_expr, scope, h=1e-5, rtol=1e-5, atol=1e-8):
        self._fn = compiled_expr
        self._scope = scope
        self._h = h
        self._rtol = rtol
        self._atol = atol

    def check_jacobian(self, x0):
        """Compare analytical Jacobian against central finite differences.

        J_approx[:, j] = (f(x + h*e_j) - f(x - h*e_j)) / (2h)
        """
        x0 = np.asarray(x0, dtype=np.float64).ravel()
        n = x0.size
        self._scope.set_values(x0)

        # Analytical Jacobian
        self._fn.forward()
        J_analytical = self._fn.jacobian().toarray()
        m = J_analytical.shape[0]

        # Numerical Jacobian via central differences
        J_numerical = np.zeros((m, n))
        for j in range(n):
            x_plus = x0.copy()
            x_minus = x0.copy()
            x_plus[j] += self._h
            x_minus[j] -= self._h

            self._scope.set_values(x_plus)
            f_plus = self._fn.forward().copy()

            self._scope.set_values(x_minus)
            f_minus = self._fn.forward().copy()

            J_numerical[:, j] = (f_plus - f_minus) / (2 * self._h)

        # Restore original point
        self._scope.set_values(x0)

        np.testing.assert_allclose(
            J_analytical, J_numerical,
            rtol=self._rtol, atol=self._atol,
            err_msg="Jacobian mismatch between analytical and numerical",
        )

    def check_hessian(self, x0, weights):
        """Compare analytical Hessian against numerical Hessian.

        For phi(x) = w^T f(x), the Hessian is computed by perturbing x_j
        and recomputing the gradient grad_phi = J^T w:

        H_approx[:, j] = (J(x+h*e_j)^T w - J(x-h*e_j)^T w) / (2h)
        """
        x0 = np.asarray(x0, dtype=np.float64).ravel()
        weights = np.asarray(weights, dtype=np.float64).ravel()
        n = x0.size

        # Analytical Hessian
        self._scope.set_values(x0)
        self._fn.forward()
        self._fn.jacobian()
        H_analytical = self._fn.hessian(weights).toarray()

        # Numerical Hessian via central differences on the gradient
        H_numerical = np.zeros((n, n))
        for j in range(n):
            x_plus = x0.copy()
            x_minus = x0.copy()
            x_plus[j] += self._h
            x_minus[j] -= self._h

            self._scope.set_values(x_plus)
            self._fn.forward()
            J_plus = self._fn.jacobian().toarray()
            grad_plus = J_plus.T @ weights

            self._scope.set_values(x_minus)
            self._fn.forward()
            J_minus = self._fn.jacobian().toarray()
            grad_minus = J_minus.T @ weights

            H_numerical[:, j] = (grad_plus - grad_minus) / (2 * self._h)

        # Restore original point
        self._scope.set_values(x0)

        np.testing.assert_allclose(
            H_analytical, H_numerical,
            rtol=self._rtol, atol=self._atol,
            err_msg="Hessian mismatch between analytical and numerical",
        )


def random_point(scope, rng, low=-1.0, high=1.0):
    """Set all variables to random values and return the flat vector."""
    n = scope.total_var_size
    x0 = rng.uniform(low, high, size=n)
    scope.set_values(x0)
    return x0


def random_positive_point(scope, rng, low=0.1, high=2.0):
    """Set all variables to positive random values (for restricted domains)."""
    n = scope.total_var_size
    x0 = rng.uniform(low, high, size=n)
    scope.set_values(x0)
    return x0
