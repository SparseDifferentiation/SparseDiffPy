"""Bivariate named functions: sp.quad_form, sp.quad_over_lin, sp.rel_entr."""

import numpy as np
import scipy.sparse

from sparsediffpy._core._expression import _wrap_constant
from sparsediffpy._core._nodes_bivariate import QuadOverLin, RelEntr
from sparsediffpy._core._nodes_other import QuadForm


def quad_form(x, Q):
    """Quadratic form x' Q x.

    x must be a column vector (n, 1).
    Q must be a scipy.sparse matrix or np.ndarray of shape (n, n).
    """
    x = _wrap_constant(x)
    if x.shape[1] != 1:
        raise ValueError(f"quad_form: x must be a column vector, got shape {x.shape}")

    if not scipy.sparse.issparse(Q):
        Q = scipy.sparse.csr_matrix(Q)
    else:
        Q = Q.tocsr()

    n = x.shape[0]
    if Q.shape != (n, n):
        raise ValueError(
            f"quad_form: Q shape {Q.shape} doesn't match x shape {x.shape}"
        )

    return QuadForm(
        x,
        Q_csr_data=np.asarray(Q.data, dtype=np.float64),
        Q_csr_indices=np.asarray(Q.indices, dtype=np.int32),
        Q_csr_indptr=np.asarray(Q.indptr, dtype=np.int32),
        Q_shape=Q.shape,
    )


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
