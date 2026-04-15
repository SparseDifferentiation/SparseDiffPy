"""Operator dispatch: routes +, -, *, @, [] to the correct expression nodes.

Separated from _expression.py to avoid circular imports — this module
imports from both _expression and _nodes_*, but neither imports this module.
_expression.py calls these functions via late binding (module-level reference).
"""

import numpy as np

from sparsediffpy._core._constants import Constant, SparseConstant
from sparsediffpy._core._expression import _wrap_constant
from sparsediffpy._core._nodes_affine import (
    Add, Broadcast, Index, LeftMatMul, Neg, ParamScalarMult,
    ParamVectorMult, RightMatMul, Transpose,
)
from sparsediffpy._core._nodes_bivariate import MatMul, Multiply
from sparsediffpy._core._nodes_elementwise import Power
from sparsediffpy._core._shapes import broadcast_shape, check_matmul_shapes, is_scalar


def _is_param_like(node):
    # Lazy import to avoid circular: _scope -> _expression -> _dispatch -> _scope
    from sparsediffpy._core._scope import Parameter
    return isinstance(node, (Constant, SparseConstant, Parameter))


def make_add(left, right):
    result_shape, left_bc, right_bc = broadcast_shape(left.shape, right.shape)
    if left_bc:
        left = Broadcast(left, result_shape)
    if right_bc:
        right = Broadcast(right, result_shape)
    return Add(left, right)


def make_sub(left, right):
    return make_add(left, Neg(right))


def make_neg(node):
    return Neg(node)


def make_mul(left, right):
    if _is_param_like(left) and is_scalar(left.shape):
        return ParamScalarMult(left, right)
    if _is_param_like(right) and is_scalar(right.shape):
        return ParamScalarMult(right, left)

    if _is_param_like(left) and left.shape == right.shape:
        return ParamVectorMult(left, right)
    if _is_param_like(right) and right.shape == left.shape:
        return ParamVectorMult(right, left)

    result_shape, left_bc, right_bc = broadcast_shape(left.shape, right.shape)
    if left_bc:
        left = Broadcast(left, result_shape)
    if right_bc:
        right = Broadcast(right, result_shape)

    if _is_param_like(left):
        return ParamVectorMult(left, right)
    if _is_param_like(right):
        return ParamVectorMult(right, left)

    return Multiply(left, right)


def make_matmul(left, right):
    result_shape = check_matmul_shapes(left.shape, right.shape)

    if _is_param_like(left):
        return LeftMatMul(left, right, result_shape)
    if _is_param_like(right):
        return RightMatMul(right, left, result_shape)
    return MatMul(left, right, result_shape)


def make_pow(node, exponent):
    if not isinstance(exponent, (int, float)):
        raise TypeError("Exponent must be a constant number")
    return Power(node, float(exponent))


def make_transpose(node):
    return Transpose(node)


def make_index(node, key):
    d1, d2 = node.shape

    if isinstance(key, tuple):
        if len(key) != 2:
            raise IndexError("Only 1D or 2D indexing supported")
        row_key, col_key = key
        row_indices = _resolve_axis_index(row_key, d1)
        col_indices = _resolve_axis_index(col_key, d2)
        flat_indices = []
        for c in col_indices:
            for r in row_indices:
                flat_indices.append(r + c * d1)
        out_d1 = len(row_indices)
        out_d2 = len(col_indices)
    else:
        if d2 == 1:
            flat_indices = _resolve_axis_index(key, d1)
            out_d1 = len(flat_indices)
            out_d2 = 1
        elif d1 == 1:
            flat_indices = _resolve_axis_index(key, d2)
            out_d1 = 1
            out_d2 = len(flat_indices)
        else:
            flat_indices = _resolve_axis_index(key, d1 * d2)
            out_d1 = len(flat_indices)
            out_d2 = 1

    flat_arr = np.array(flat_indices, dtype=np.int32)
    return Index(node, flat_arr, (out_d1, out_d2))


def _resolve_axis_index(key, length):
    if isinstance(key, (int, np.integer)):
        idx = int(key)
        if idx < 0:
            idx += length
        if idx < 0 or idx >= length:
            raise IndexError(f"Index {key} out of range for axis of length {length}")
        return [idx]
    if isinstance(key, slice):
        return list(range(*key.indices(length)))
    if isinstance(key, (list, np.ndarray)):
        out = []
        for i in key:
            idx = int(i)
            if idx < 0:
                idx += length
            if idx < 0 or idx >= length:
                raise IndexError(f"Index {i} out of range for axis of length {length}")
            out.append(idx)
        return out
    raise IndexError(f"Unsupported index type: {type(key).__name__}")
