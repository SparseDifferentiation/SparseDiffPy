"""Expression base class and _wrap_constant.

Operator dispatch lives in _dispatch.py (avoids circular imports).
Node types are defined in _nodes_affine.py, _nodes_elementwise.py,
_nodes_bivariate.py, and _nodes_other.py.
"""

import numpy as np
import scipy.sparse

from sparsediffpy._core._constants import Constant, SparseConstant


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
        return _dispatch.make_add(self, _wrap_constant(other))

    def __radd__(self, other):
        return _dispatch.make_add(_wrap_constant(other), self)

    def __sub__(self, other):
        return _dispatch.make_sub(self, _wrap_constant(other))

    def __rsub__(self, other):
        return _dispatch.make_sub(_wrap_constant(other), self)

    def __neg__(self):
        return _dispatch.make_neg(self)

    def __mul__(self, other):
        return _dispatch.make_mul(self, _wrap_constant(other))

    def __rmul__(self, other):
        return _dispatch.make_mul(_wrap_constant(other), self)

    def __matmul__(self, other):
        return _dispatch.make_matmul(self, _wrap_constant(other))

    def __rmatmul__(self, other):
        return _dispatch.make_matmul(_wrap_constant(other), self)

    def __pow__(self, exponent):
        return _dispatch.make_pow(self, exponent)

    def __getitem__(self, key):
        return _dispatch.make_index(self, key)

    @property
    def T(self):
        return _dispatch.make_transpose(self)

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
# Import _dispatch at the bottom to avoid circular imports.
# By this point Expression is fully defined, so _dispatch.py (which imports
# from _nodes_*.py which inherit from Expression) can resolve everything.
# ---------------------------------------------------------------------------

from sparsediffpy._core import _dispatch  # noqa: E402
