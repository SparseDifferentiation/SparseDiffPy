"""Affine named functions: sp.sum, sp.prod, sp.reshape, sp.hstack, etc."""

import builtins as _builtins

from sparsediffpy._core._expression import _wrap_constant
from sparsediffpy._core._nodes_affine import (
    DiagVec, HStack, Reshape, Sum, Trace, Transpose,
)
from sparsediffpy._core._nodes_other import Prod, ProdAxisOne, ProdAxisZero
from sparsediffpy._core._shapes import validate_shape

_builtin_sum = _builtins.sum


def diag_vec(x):
    return DiagVec(x)

def trace(x):
    return Trace(x)

def reshape(x, d1, d2):
    validate_shape(d1, d2)
    return Reshape(x, (d1, d2))


def sum(x, axis=None):
    """Sum reduction.

    axis=None: sum all elements -> (1,1)
    axis=0: sum along rows (collapse d1) -> (1, d2)
    axis=1: sum along columns (collapse d2) -> (1, d1)
    """
    c_axis = -1 if axis is None else axis
    return Sum(x, c_axis)


def prod(x, axis=None):
    """Product reduction.

    axis=None: product of all elements -> (1,1)
    axis=0: product along rows -> (1, d2)
    axis=1: product along columns -> (1, d1)
    """
    if axis is None:
        return Prod(x)
    elif axis == 0:
        return ProdAxisZero(x)
    elif axis == 1:
        return ProdAxisOne(x)
    else:
        raise ValueError(f"Invalid axis {axis}, must be None, 0, or 1")


def hstack(expressions):
    """Horizontally stack expressions. All must have the same d1 (rows).

    Result shape: (d1, sum of all d2).
    """
    exprs = [_wrap_constant(e) for e in expressions]
    if not exprs:
        raise ValueError("hstack requires at least one expression")

    d1 = exprs[0].shape[0]
    for e in exprs[1:]:
        if e.shape[0] != d1:
            raise ValueError(
                f"hstack: all expressions must have the same number of rows, "
                f"got {d1} and {e.shape[0]}"
            )

    total_d2 = _builtin_sum(e.shape[1] for e in exprs)
    return HStack(exprs, (d1, total_d2))


def vstack(expressions):
    """Vertically stack expressions. All must have the same d2 (columns).

    Implemented as transpose(hstack(transpose(each))).
    """
    exprs = [_wrap_constant(e) for e in expressions]
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
    total_d1 = _builtin_sum(e.shape[0] for e in exprs)
    h = HStack(transposed, (d2, total_d1))
    return Transpose(h)
