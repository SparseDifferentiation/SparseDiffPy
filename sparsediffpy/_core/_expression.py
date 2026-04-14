"""Expression base class, operator overloading, dispatch helpers, and _wrap_constant.

Node types are defined in _nodes_affine.py, _nodes_elementwise.py,
_nodes_bivariate.py, and _nodes_other.py.
"""

import numpy as np
import scipy.sparse

from sparsediffpy._core._constants import Constant, SparseConstant
from sparsediffpy._core._shapes import (
    broadcast_shape,
    check_matmul_shapes,
    is_scalar,
)


# ---------------------------------------------------------------------------
# _wrap_constant: converts raw values into expression nodes
# ---------------------------------------------------------------------------

def _wrap_constant(value):
    """Wrap a raw Python/NumPy/SciPy value into an expression node.

    Called by operators and node constructors so users can write
    ``x + 1.0`` or ``A @ x`` with raw scalars/arrays.

    - Expression subclass -> return as-is
    - int / float -> Constant with shape (1, 1)
    - np.ndarray 1D (n,) -> Constant with shape (n, 1) (column vector)
    - np.ndarray 2D (m, n) -> Constant with shape (m, n)
    - scipy.sparse -> SparseConstant
    """
    if hasattr(value, "_is_sparsediff_expr"):
        return value

    if isinstance(value, (int, float)):
        return Constant(np.array([float(value)]), (1, 1))

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return Constant(np.array([value.item()]), (1, 1))
        if value.ndim == 1:
            return Constant(value, (value.shape[0], 1))
        if value.ndim == 2:
            return Constant(value, (value.shape[0], value.shape[1]))
        raise ValueError(f"Cannot wrap {value.ndim}D array as constant")

    if scipy.sparse.issparse(value):
        return SparseConstant(value)

    raise TypeError(f"Cannot convert {type(value).__name__} to expression")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Expression:
    """Base class for all expression tree nodes."""

    _is_sparsediff_expr = True
    shape = None  # (d1, d2), set by subclasses

    # Tell NumPy to defer to our operators instead of trying its own
    __array_ufunc__ = None
    __array_priority__ = 20

    def __add__(self, other):
        other = _wrap_constant(other)
        return _make_add(self, other)

    def __radd__(self, other):
        other = _wrap_constant(other)
        return _make_add(other, self)

    def __sub__(self, other):
        other = _wrap_constant(other)
        from sparsediffpy._core._nodes_affine import Neg
        return _make_add(self, Neg(other))

    def __rsub__(self, other):
        other = _wrap_constant(other)
        from sparsediffpy._core._nodes_affine import Neg
        return _make_add(other, Neg(self))

    def __neg__(self):
        from sparsediffpy._core._nodes_affine import Neg
        return Neg(self)

    def __mul__(self, other):
        other = _wrap_constant(other)
        return _make_mul(self, other)

    def __rmul__(self, other):
        other = _wrap_constant(other)
        return _make_mul(other, self)

    def __matmul__(self, other):
        other = _wrap_constant(other)
        return _make_matmul(self, other)

    def __rmatmul__(self, other):
        other = _wrap_constant(other)
        return _make_matmul(other, self)

    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)):
            raise TypeError("Exponent must be a constant number")
        from sparsediffpy._core._nodes_elementwise import Power
        return Power(self, float(exponent))

    def __getitem__(self, key):
        return _make_index(self, key)

    @property
    def T(self):
        from sparsediffpy._core._nodes_affine import Transpose
        return Transpose(self)

    @property
    def size(self):
        return self.shape[0] * self.shape[1]


# ---------------------------------------------------------------------------
# Make Constant and SparseConstant behave as expressions
# ---------------------------------------------------------------------------

for _cls in (Constant, SparseConstant):
    _cls._is_sparsediff_expr = True
    _cls.__array_ufunc__ = None
    _cls.__array_priority__ = 20
    _cls.__add__ = Expression.__add__
    _cls.__radd__ = Expression.__radd__
    _cls.__sub__ = Expression.__sub__
    _cls.__rsub__ = Expression.__rsub__
    _cls.__neg__ = Expression.__neg__
    _cls.__mul__ = Expression.__mul__
    _cls.__rmul__ = Expression.__rmul__
    _cls.__matmul__ = Expression.__matmul__
    _cls.__rmatmul__ = Expression.__rmatmul__
    _cls.__pow__ = Expression.__pow__
    _cls.__getitem__ = Expression.__getitem__
    _cls.T = Expression.T
    _cls.size = Expression.size


# ---------------------------------------------------------------------------
# Operator dispatch helpers
# ---------------------------------------------------------------------------

def _maybe_broadcast(node, target_shape):
    from sparsediffpy._core._nodes_affine import Broadcast
    if node.shape == target_shape:
        return node
    return Broadcast(node, target_shape)


def _is_param_like(node):
    from sparsediffpy._core._scope import Parameter
    return isinstance(node, (Constant, SparseConstant, Parameter))


def _make_add(left, right):
    from sparsediffpy._core._nodes_affine import Add
    result_shape, left_bc, right_bc = broadcast_shape(left.shape, right.shape)
    if left_bc:
        left = _maybe_broadcast(left, result_shape)
    if right_bc:
        right = _maybe_broadcast(right, result_shape)
    return Add(left, right)


def _make_mul(left, right):
    from sparsediffpy._core._nodes_affine import ParamScalarMult, ParamVectorMult
    from sparsediffpy._core._nodes_bivariate import Multiply
    from sparsediffpy._core._scope import Parameter

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
        left = _maybe_broadcast(left, result_shape)
    if right_bc:
        right = _maybe_broadcast(right, result_shape)

    if _is_param_like(left):
        return ParamVectorMult(left, right)
    if _is_param_like(right):
        return ParamVectorMult(right, left)

    return Multiply(left, right)


def _make_matmul(left, right):
    from sparsediffpy._core._nodes_affine import LeftMatMul, RightMatMul
    from sparsediffpy._core._nodes_bivariate import MatMul
    from sparsediffpy._core._scope import Parameter

    result_shape = check_matmul_shapes(left.shape, right.shape)
    left_is_param = isinstance(left, (Constant, SparseConstant, Parameter))
    right_is_param = isinstance(right, (Constant, SparseConstant, Parameter))

    if left_is_param and not right_is_param:
        return LeftMatMul(left, right, result_shape)
    if right_is_param and not left_is_param:
        return RightMatMul(right, left, result_shape)
    return MatMul(left, right, result_shape)


def _make_index(node, key):
    from sparsediffpy._core._nodes_affine import Index

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
            indices = _resolve_axis_index(key, d1)
            flat_indices = indices
            out_d1 = len(indices)
            out_d2 = 1
        elif d1 == 1:
            indices = _resolve_axis_index(key, d2)
            flat_indices = [i * d1 for i in indices]
            out_d1 = 1
            out_d2 = len(indices)
        else:
            total = d1 * d2
            indices = _resolve_axis_index(key, total)
            flat_indices = indices
            out_d1 = len(indices)
            out_d2 = 1

    result_shape = (out_d1, out_d2)
    flat_arr = np.array(flat_indices, dtype=np.int32)
    return Index(node, flat_arr, result_shape)


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
