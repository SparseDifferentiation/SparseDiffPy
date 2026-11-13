"""Expression graph: Scope, leaves (Variable/Parameter/Constant), and operators.

Every non-leaf is a generic ``Node(make, args, shape)``: ``make`` receives the
C capsules of ``args`` and returns the node's own capsule. Shapes are always
2-D ``(d1, d2)`` and values are flattened column-major, like the C engine.
"""

import numbers

import numpy as np
import scipy.sparse as sps

from sparsediffpy import _sparsediffengine as _C


class DimensionError(ValueError):
    """Raised when a value has the wrong number of elements."""


def _check_shape(d1, d2):
    """Return (d1, d2) as a tuple of positive Python ints, or raise."""
    if not all(isinstance(d, (int, np.integer)) for d in (d1, d2)):
        raise TypeError(f"Shape dimensions must be integers, got ({d1!r}, {d2!r})")
    if d1 <= 0 or d2 <= 0:
        raise ValueError(f"Shape dimensions must be positive, got ({d1}, {d2})")
    shape = (int(d1), int(d2))
    _check_size(shape)
    return shape


_MAX_SIZE = 2**31 - 1  # the engine stores sizes as C int


def _check_size(shape):
    """Raise if a shape has more entries than the engine can index."""
    if shape[0] * shape[1] > _MAX_SIZE:
        raise ValueError(f"Shape {shape} has {shape[0] * shape[1]} entries; "
                         f"the engine supports at most {_MAX_SIZE}")


class Expression:
    """Base class for all expressions. Supports + - * @ ** [] and .T."""

    __array_ufunc__ = None  # make NumPy defer to our reflected operators
    shape = None
    args = ()

    @property
    def size(self):
        """Number of entries, d1 * d2."""
        return self.shape[0] * self.shape[1]

    @property
    def T(self):
        """Transpose."""
        return Node(_C.make_transpose, [self], self.shape[::-1])

    def __add__(self, other):
        return _add(self, cast_to_const(other))

    def __radd__(self, other):
        return _add(cast_to_const(other), self)

    def __sub__(self, other):
        return _add(self, -cast_to_const(other))

    def __rsub__(self, other):
        return _add(cast_to_const(other), -self)

    def __neg__(self):
        return Node(_C.make_neg, [self], self.shape)

    def __mul__(self, other):
        return _mul(self, cast_to_const(other))

    def __rmul__(self, other):
        return _mul(cast_to_const(other), self)

    def __truediv__(self, other):
        """self / c for a constant c (a number or array), as self * (1 / c)."""
        if isinstance(other, Expression) or sps.issparse(other):
            raise TypeError("Can only divide by a constant number or dense array; "
                            "for an expression d use x * d ** -1")
        c = np.asarray(other, dtype=np.float64)
        if np.any(c == 0):
            raise ZeroDivisionError("division of an expression by zero")
        return self * (1.0 / c)

    def __matmul__(self, other):
        return _matmul(self, cast_to_const(other))

    def __rmatmul__(self, other):
        return _matmul(cast_to_const(other), self)

    def __pow__(self, p):
        if not isinstance(p, numbers.Real):
            raise TypeError("Exponent must be a constant number")
        return Node(lambda c: _C.make_power(c, float(p)), [self], self.shape)

    def __getitem__(self, key):
        return _index(self, key)

    def __iter__(self):
        # Without this, Python would iterate via __getitem__ until IndexError.
        raise TypeError("Expressions are not iterable; index with x[i, j] or "
                        "pass a list of expressions")


class Node(Expression):
    """An operation: ``make(*child_capsules)`` builds its C capsule."""

    def __init__(self, make, args, shape):
        self.make = make
        self.args = tuple(args)
        self.shape = tuple(shape)
        _check_size(self.shape)


class Constant(Expression):
    """Fixed dense or scipy.sparse data."""

    def __init__(self, value):
        if value is None:
            raise TypeError("Constant value cannot be None")
        if sps.issparse(value):
            self.value = sps.csr_matrix(value, dtype=np.float64)
        else:
            value = np.asarray(value, dtype=np.float64)
            if value.size == 1:
                value = value.reshape(1, 1)
            elif value.ndim != 2:
                raise ValueError(
                    f"Constants must be scalars or 2-D arrays, (n, 1) or (1, n); "
                    f"got shape {value.shape}. Use a[:, None] or a[None, :].")
            self.value = value
        self.shape = _check_shape(*self.value.shape)

    def dense(self):
        """The value as a dense array."""
        return self.value.toarray() if sps.issparse(self.value) else self.value


class Variable(Expression):
    """A block of the scope's flat variable vector. Create via Scope.Variable."""

    def __init__(self, scope, offset, shape):
        self._scope, self._offset, self.shape = scope, offset, shape

    @property
    def value(self):
        """The variable's current entries, shaped like the variable."""
        flat = self._scope._values[self._offset:self._offset + self.size]
        return _read_only_copy(flat.reshape(self.shape, order="F"))

    @value.setter
    def value(self, val):
        self._scope._values[self._offset:self._offset + self.size] = \
            _flatten_shaped(val, self.shape)
        self._scope._values_version += 1


class Parameter(Expression):
    """Data that can change without recompiling. Create via Scope.Parameter."""

    def __init__(self, scope, offset, shape, value=None):
        self._scope, self._offset, self.shape = scope, offset, shape
        self._flat = None
        self._symmetric = False  # set by quad_form; the engine assumes Q = Q'
        if value is not None:
            self.value = value

    @property
    def value(self):
        """The parameter's current value, or None if it was never set."""
        if self._flat is None:
            return None
        return _read_only_copy(self._flat.reshape(self.shape, order="F"))

    @value.setter
    def value(self, val):
        flat = _flatten_shaped(val, self.shape).copy()  # don't alias the caller's array
        if self._symmetric:
            _check_symmetric(flat, self.shape)
        self._flat = flat
        self._scope._param_version += 1


def _check_symmetric(flat, shape):
    """Raise unless the column-major ``flat`` is a symmetric matrix of ``shape``."""
    M = flat.reshape(shape, order="F")
    if not np.allclose(M, M.T):
        raise ValueError("Parameter used in quad_form must be symmetric")


def _flatten_shaped(val, shape):
    """Flatten a value of shape ``shape`` column-major.

    Also accepts a 1-D array for a vector shape (n, 1) or (1, n), and a scalar
    for (1, 1). Any other shape raises, even with the right element count.
    """
    if val is None:
        raise TypeError("value cannot be None")
    arr = np.asarray(val, dtype=np.float64)
    if not (arr.shape == shape
            or (arr.ndim == 0 and shape == (1, 1))
            or (arr.ndim == 1 and min(shape) == 1
                and arr.size == shape[0] * shape[1])):
        raise DimensionError(f"expected shape {shape}, got {arr.shape}")
    return arr.ravel(order="F")


def _read_only_copy(arr):
    """A copy of arr that raises on assignment (edits would be silently lost)."""
    arr = arr.copy()
    arr.flags.writeable = False
    return arr


def _flatten_vector(val, n, name):
    """Return ``val`` as a float64 1-D array of exactly n entries."""
    arr = np.asarray(val, dtype=np.float64)
    if arr.ndim != 1 or arr.size != n:
        raise DimensionError(f"{name} must be a 1-D array of {n} elements, "
                             f"got shape {arr.shape}")
    return arr


class Scope:
    """Owns the flat variable vector and the parameter vector."""

    def __init__(self):
        self._values = np.zeros(0)
        self._values_version = 0  # bumped on every write; problems skip stale work
        self._n_params = 0
        self._param_version = 0

    def Variable(self, d1, d2):
        """Create a (d1, d2) variable, appending its entries to the flat vector."""
        shape = _check_shape(d1, d2)
        var = Variable(self, self._values.size, shape)
        self._values = np.concatenate([self._values, np.zeros(var.size)])
        return var

    def Parameter(self, d1, d2, value=None):
        """Create a (d1, d2) parameter, optionally with an initial value."""
        shape = _check_shape(d1, d2)
        param = Parameter(self, self._n_params, shape, value)
        self._n_params += param.size
        return param

    @property
    def total_var_size(self):
        """Length of the flat variable vector."""
        return self._values.size

    def set_values(self, array):
        """Set all variables at once from a flat array (declaration order)."""
        values = _flatten_vector(array, self._values.size, "values")
        if not np.array_equal(values, self._values):  # same point: keep caches
            self._values[:] = values
            self._values_version += 1

    def get_values(self):
        """A copy of the flat variable vector (declaration order)."""
        return self._values.copy()


def cast_to_const(value):
    """Return ``value`` as an Expression, wrapping non-Expressions in a Constant."""
    return value if isinstance(value, Expression) else Constant(value)


def _is_data(e):
    """Whether e is data (Constant or Parameter) rather than a function of x."""
    return isinstance(e, (Constant, Parameter))


def _broadcast_shape(a, b):
    """Shape of a op b when scalars, rows and columns broadcast, or raise."""
    out = []
    for da, db in zip(a, b):
        if da != db and 1 not in (da, db):
            raise ValueError(f"Cannot broadcast shapes {a} and {b}")
        out.append(max(da, db))
    return tuple(out)


def broadcast(x, shape):
    """Broadcast a scalar, row or column expression to ``shape``."""
    x, shape = cast_to_const(x), _check_shape(*shape)
    if x.shape == shape:
        return x
    if _broadcast_shape(x.shape, shape) != shape:
        raise ValueError(f"Cannot broadcast {x.shape} to {shape}")
    if isinstance(x, Constant):
        return Constant(np.broadcast_to(x.dense(), shape))
    return Node(lambda c: _C.make_broadcast(c, *shape), [x], shape)


def _add(a, b):
    """a + b, broadcasting both operands to the common shape first."""
    shape = _broadcast_shape(a.shape, b.shape)
    return Node(_C.make_add, [broadcast(a, shape), broadcast(b, shape)], shape)


def _mul(a, b):
    """Elementwise a * b.

    A data operand (constant or parameter) uses the engine's scalar or vector
    scaling atoms; two functions of x use the general product.
    """
    if _is_data(b) and not _is_data(a):
        a, b = b, a  # put the constant/parameter operand (if any) first
    if _is_data(a) and a.shape == (1, 1):
        return Node(_C.make_param_scalar_mult, [a, b], b.shape)
    shape = _broadcast_shape(a.shape, b.shape)
    a, b = broadcast(a, shape), broadcast(b, shape)
    make = _C.make_param_vector_mult if _is_data(a) else _C.make_multiply
    return Node(make, [a, b], shape)


def _matmul(a, b):
    """a @ b: data times a function of x (either side), or two functions of x."""
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Matmul shape mismatch: {a.shape} @ {b.shape}")
    shape = (a.shape[0], b.shape[1])
    if _is_data(a):
        return Node(_matmul_with_data(_C.make_left_matmul, a), _data_args(a, b),
                    shape)
    if _is_data(b):
        return Node(_matmul_with_data(_C.make_right_matmul, b), _data_args(b, a),
                    shape)
    return Node(_C.make_matmul, [a, b], shape)


def _data_args(data, x):
    """Children of a data matmul: a Parameter is a child, a Constant is baked in."""
    return [data, x] if isinstance(data, Parameter) else [x]


def _matmul_with_data(make, A):
    """Return make(child) / make(param, child) for a constant/parameter A."""
    m, n = A.shape
    if isinstance(A, Parameter):
        return lambda p, c: make(p, c, "dense", None, m, n)
    if sps.issparse(A.value):
        V = A.value
        return lambda c: make(None, c, "sparse", V.data, V.indices.astype(np.int32),
                              V.indptr.astype(np.int32), m, n)
    data = np.ascontiguousarray(A.value).ravel()  # row-major
    return lambda c: make(None, c, "dense", data, m, n)


def index_flat(x, flat_indices, shape):
    """Gather entries of ``x`` by column-major flat indices into ``shape``."""
    x = cast_to_const(x)
    idx = np.asarray(flat_indices).ravel()
    if idx.size and not np.issubdtype(idx.dtype, np.integer):
        raise TypeError(f"flat indices must be integers, got dtype {idx.dtype}")
    shape = _check_shape(*shape)
    if idx.size != shape[0] * shape[1]:
        raise ValueError(f"{idx.size} indices do not fill shape {shape}")
    if idx.size and (idx.min() < 0 or idx.max() >= x.size):
        raise IndexError(f"flat index out of range for size {x.size}")
    idx = idx.astype(np.int32)  # safe now that all are in range; also copies
    return Node(lambda c: _C.make_index(c, shape[0], shape[1], idx), [x], shape)


def _axis_indices(key, n):
    """Positions along an axis of length n chosen by an int, slice or int list."""
    if isinstance(key, (bool, np.bool_)):  # bool subclasses int; x[True] is not x[1]
        raise IndexError("Boolean indices are not supported; use integer indices")
    if isinstance(key, (int, np.integer)):
        if not -n <= key < n:
            raise IndexError(f"Index {key} out of range for axis of length {n}")
        return np.array([key % n])
    if isinstance(key, slice):
        return np.arange(n)[key]
    if isinstance(key, (list, np.ndarray)):
        key = np.asarray(key)
        if key.ndim != 1 or not np.issubdtype(key.dtype, np.integer):
            raise IndexError("Index arrays must be 1-D with integer entries "
                             f"(got shape {key.shape}, dtype {key.dtype})")
        if key.size and (key.min() < -n or key.max() >= n):
            raise IndexError(f"Index out of range for axis of length {n}")
        return key % n
    raise IndexError(f"Unsupported index type: {type(key).__name__}")


def _index(x, key):
    """x[i, j] on anything; x[i] only on vectors, where it indexes along the axis."""
    d1, d2 = x.shape
    if isinstance(key, tuple):
        if len(key) != 2:
            raise IndexError(
                f"Expected 2 indices for shape {x.shape}, got {len(key)}")
        rows, cols = _axis_indices(key[0], d1), _axis_indices(key[1], d2)
        flat = (rows[:, None] + cols[None, :] * d1).ravel(order="F")
        return index_flat(x, flat, (rows.size, cols.size))
    if d1 > 1 and d2 > 1:
        raise IndexError(f"A single index on shape {x.shape} is ambiguous; "
                         "use X[i, j], X[i, :] or X[:, j]")
    flat = _axis_indices(key, x.size)
    return index_flat(x, flat, (1, flat.size) if d1 == 1 else (flat.size, 1))
