import numpy as np
import pytest

import sparsediffpy as sp


@pytest.fixture
def scope():
    return sp.Scope()


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def check_derivatives(fn, scope, x0, w, h=1e-6, rtol=1e-5, atol=1e-6):
    """Check fn.jacobian() and fn.hessian(w) against central differences."""
    x0 = np.asarray(x0, dtype=np.float64)

    def at(x, f):
        scope.set_values(x)
        return f()

    def jac(x):
        return at(x, fn.jacobian).toarray()

    J = jac(x0)
    H = at(x0, lambda: fn.hessian(w)).toarray()
    J_fd = np.zeros_like(J)
    H_fd = np.zeros_like(H)
    for j in range(x0.size):
        e = np.zeros_like(x0)
        e[j] = h
        J_fd[:, j] = (at(x0 + e, fn.forward) - at(x0 - e, fn.forward)) / (2 * h)
        H_fd[:, j] = (jac(x0 + e) - jac(x0 - e)).T @ w / (2 * h)
    scope.set_values(x0)
    np.testing.assert_allclose(J, J_fd, rtol=rtol, atol=atol, err_msg="Jacobian")
    np.testing.assert_allclose(H, H_fd, rtol=rtol, atol=atol, err_msg="Hessian")
