"""Bivariate named functions: sp.quad_form, sp.quad_over_lin, sp.rel_entr."""

import scipy.sparse

from sparsediffpy._core._nodes_bivariate import QuadOverLin, RelEntr
from sparsediffpy._core._nodes_other import QuadForm


def quad_form(x, Q):
    """Quadratic form xT Q x with x (n, 1) and Q (n, n)"""
    if not isinstance(Q, scipy.sparse.csr_matrix):
        Q = scipy.sparse.csr_matrix(Q)

    n = x.shape[0]
    if x.shape[1] != 1 or Q.shape != (n, n):
        raise ValueError(f"quad_form: need x (n, 1) and Q (n, n) "
                           ", got x {x.shape} and Q {Q.shape}")

    return QuadForm(x, Q)


def quad_over_lin(x, z):
    """sum(x^2) / z where z is a scalar variable.

    Both arguments must be variable-dependent expressions.
    z must be a plain Variable (not a composition).
    """
    return QuadOverLin(x, z)


def rel_entr(x, y):
    """x * log(x / y) elementwise.

    Both arguments must be variable-dependent expressions.
    The C engine does not support constant arguments.
    """
    return RelEntr(x, y)
