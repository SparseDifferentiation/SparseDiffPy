"""Module-level named functions: sp.sin, sp.exp, sp.hstack, etc."""

import numpy as np
import scipy.sparse

from sparsediffpy._core._constants import _wrap_constant
from sparsediffpy._core._nodes_affine import (
    DiagVec, HStack, Reshape, Sum, Trace, Transpose,
)
from sparsediffpy._core._nodes_bivariate import QuadOverLin, RelEntr
from sparsediffpy._core._nodes_elementwise import (
    Asinh, Atanh, Cos, Entr, Exp, Log, Logistic, NormalCdf, Power,
    Sin, Sinh, Tan, Tanh, Xexp,
)
from sparsediffpy._core._nodes_other import Prod, ProdAxisOne, ProdAxisZero, QuadForm
from sparsediffpy._core._shapes import validate_shape


def _ensure_expr(x):
    if hasattr(x, "_is_sparsediff_expr"):
        return x
    return _wrap_constant(x)


# ---------------------------------------------------------------------------
# Unary elementwise functions
# ---------------------------------------------------------------------------

def sin(x):
    return Sin(_ensure_expr(x))

def cos(x):
    return Cos(_ensure_expr(x))

def exp(x):
    return Exp(_ensure_expr(x))

def log(x):
    return Log(_ensure_expr(x))

def tan(x):
    return Tan(_ensure_expr(x))

def sinh(x):
    return Sinh(_ensure_expr(x))

def tanh(x):
    return Tanh(_ensure_expr(x))

def asinh(x):
    return Asinh(_ensure_expr(x))

def atanh(x):
    return Atanh(_ensure_expr(x))

def logistic(x):
    return Logistic(_ensure_expr(x))

def normal_cdf(x):
    return NormalCdf(_ensure_expr(x))

def entr(x):
    return Entr(_ensure_expr(x))

def xexp(x):
    return Xexp(_ensure_expr(x))

def diag_vec(x):
    return DiagVec(_ensure_expr(x))


# ---------------------------------------------------------------------------
# Unary with extra arguments
# ---------------------------------------------------------------------------

def power(x, p):
    return Power(_ensure_expr(x), float(p))


def sum(x, axis=None):
    """Sum reduction.

    axis=None: sum all elements -> (1,1)
    axis=0: sum along rows (collapse d1) -> (1, d2)
    axis=1: sum along columns (collapse d2) -> (d1, 1)
    """
    c_axis = -1 if axis is None else axis
    return Sum(_ensure_expr(x), c_axis)


def prod(x, axis=None):
    """Product reduction.

    axis=None: product of all elements -> (1,1)
    axis=0: product along rows -> (1, d2)
    axis=1: product along columns -> (d1, 1)
    """
    x = _ensure_expr(x)
    if axis is None:
        return Prod(x)
    elif axis == 0:
        return ProdAxisZero(x)
    elif axis == 1:
        return ProdAxisOne(x)
    else:
        raise ValueError(f"Invalid axis {axis}, must be None, 0, or 1")


def reshape(x, d1, d2):
    validate_shape(d1, d2)
    return Reshape(_ensure_expr(x), (d1, d2))


def trace(x):
    return Trace(_ensure_expr(x))


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------

def hstack(expressions):
    """Horizontally stack expressions. All must have the same d1 (rows).

    Result shape: (d1, sum of all d2).
    """
    exprs = [_ensure_expr(e) for e in expressions]
    if not exprs:
        raise ValueError("hstack requires at least one expression")

    d1 = exprs[0].shape[0]
    for e in exprs[1:]:
        if e.shape[0] != d1:
            raise ValueError(
                f"hstack: all expressions must have the same number of rows, "
                f"got {d1} and {e.shape[0]}"
            )

    total_d2 = builtins_sum(e.shape[1] for e in exprs)
    return HStack(exprs, (d1, total_d2))


def vstack(expressions):
    """Vertically stack expressions. All must have the same d2 (columns).

    Implemented as transpose(hstack(transpose(each))).
    """
    exprs = [_ensure_expr(e) for e in expressions]
    if not exprs:
        raise ValueError("vstack requires at least one expression")

    d2 = exprs[0].shape[1]
    for e in exprs[1:]:
        if e.shape[1] != d2:
            raise ValueError(
                f"vstack: all expressions must have the same number of columns, "
                f"got {d2} and {e.shape[1]}"
            )

    transposed = [Transpose(e) for e in exprs]
    total_d1 = builtins_sum(e.shape[0] for e in exprs)
    h = HStack(transposed, (d2, total_d1))
    return Transpose(h)


# Keep a reference to Python's built-in sum (shadowed by our sum function)
import builtins as _builtins
builtins_sum = _builtins.sum


# ---------------------------------------------------------------------------
# Special functions
# ---------------------------------------------------------------------------

def quad_form(x, Q):
    """Quadratic form x' Q x.

    x must be a column vector (n, 1).
    Q must be a scipy.sparse matrix or np.ndarray of shape (n, n).
    """
    x = _ensure_expr(x)
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
    """sum(x^2) / z where z is a scalar expression."""
    x = _ensure_expr(x)
    z = _ensure_expr(z)
    return QuadOverLin(x, z)


def rel_entr(x, y):
    """x * log(x / y) elementwise."""
    x = _ensure_expr(x)
    y = _ensure_expr(y)
    return RelEntr(x, y)
