"""Affine named functions: sp.sum, sp.prod, sp.reshape, sp.hstack, etc."""

import builtins as _builtins

import numpy as np

from sparsediffpy._core._expression import _wrap_constant
from sparsediffpy._core._nodes_affine import (
    Broadcast, DiagVec, HStack, Index, Reshape, Sum, Trace, Transpose,
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


def broadcast(x, shape):
    """Broadcast a scalar or smaller-shaped expression to `shape`.

    If `x.shape == shape`, returns `x` unchanged.
    """
    x = _wrap_constant(x)
    shape = tuple(shape)
    validate_shape(shape[0], shape[1])
    if x.shape == shape:
        return x
    return Broadcast(x, shape)


def index_flat(x, flat_indices, result_shape):
    """Gather elements by pre-computed Fortran-flat indices into `x`.

    `flat_indices` is an array of column-major indices into `x` (treated as a
    flat buffer of size d1*d2). `result_shape` is the 2-D shape of the output.
    """
    flat_indices = np.asarray(flat_indices, dtype=np.int32)
    result_shape = tuple(result_shape)
    validate_shape(result_shape[0], result_shape[1])
    if flat_indices.size != result_shape[0] * result_shape[1]:
        raise ValueError(
            f"flat_indices length {flat_indices.size} does not match "
            f"result_shape {result_shape} (size {result_shape[0] * result_shape[1]})"
        )
    return Index(x, flat_indices, result_shape)


def sum(x, axis=None):
    """Sum reduction.

    axis=None: sum all elements -> (1,1)
    axis=0: sum along rows (collapse d1) -> (1, d2)
    axis=1: sum along columns (collapse d2) -> (1, d1)
    """
    if axis is None:
        return Sum(x, -1)
    elif axis == 0:
        return Sum(x, 0)
    elif axis == 1:
        return Sum(x, 1)
    else:
        raise ValueError(f"Invalid axis {axis}, must be None, 0, or 1")


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
        raise ValueError("hstack: empty argument")

    d1 = exprs[0].shape[0]
    for e in exprs[1:]:
        if e.shape[0] != d1:
            raise ValueError(f"hstack: row mismatch, {d1} vs {e.shape[0]}")

    total_d2 = _builtin_sum(e.shape[1] for e in exprs)
    return HStack(exprs, (d1, total_d2))


def vstack(expressions):
    """Vertically stack expressions. All must have the same d2 (columns).

    Implemented as transpose(hstack(transpose(each))).
    """
    exprs = [_wrap_constant(e) for e in expressions]
    if not exprs:
        raise ValueError("vstack: empty argument")

    d2 = exprs[0].shape[1]
    for e in exprs[1:]:
        if e.shape[1] != d2:
            raise ValueError(f"vstack: column mismatch, {d2} vs {e.shape[1]}")

    transposed = [Transpose(e) for e in exprs]
    total_d1 = _builtin_sum(e.shape[0] for e in exprs)
    h = HStack(transposed, (d2, total_d1))
    return Transpose(h)
