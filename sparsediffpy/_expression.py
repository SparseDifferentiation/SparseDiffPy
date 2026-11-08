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
    if not all(isinstance(d, (int, np.integer)) for d in (d1, d2)):
        raise TypeError(f"Shape dimensions must be integers, got ({d1!r}, {d2!r})")
    if d1 <= 0 or d2 <= 0:
        raise ValueError(f"Shape dimensions must be positive, got ({d1}, {d2})")
    return (int(d1), int(d2))


class Expression:
    """Base class for all expressions. Supports + - * @ ** [] and .T."""

    __array_ufunc__ = None  # make NumPy defer to our reflected operators
    shape = None
    args = ()

    @property
    def size(self):
        return self.shape[0] * self.shape[1]

    @property
    def T(self):
        return Node(_C.make_transpose, [self], self.shape[::-1], "transpose")

    def __add__(self, other):
        return _add(self, cast_to_const(other))

    def __radd__(self, other):
        return _add(cast_to_const(other), self)

    def __sub__(self, other):
        return _add(self, -cast_to_const(other))

    def __rsub__(self, other):
        return _add(cast_to_const(other), -self)

    def __neg__(self):
        return Node(_C.make_neg, [self], self.shape, "neg")

    def __mul__(self, other):
        return _mul(self, cast_to_const(other))

    def __rmul__(self, other):
        return _mul(cast_to_const(other), self)

    def __matmul__(self, other):
        return _matmul(self, cast_to_const(other))

    def __rmatmul__(self, other):
        return _matmul(cast_to_const(other), self)

    def __pow__(self, p):
        if not isinstance(p, numbers.Real):
            raise TypeError("Exponent must be a constant number")
        return Node(lambda c: _C.make_power(c, float(p)), [self], self.shape,
                    "power", {"p": float(p)})

    def __getitem__(self, key):
        return _index(self, key)

    def __iter__(self):
        # Without this, Python would iterate via __getitem__ until IndexError.
        raise TypeError("Expressions are not iterable; index with x[i, j] or "
                        "pass a list of expressions")


class Node(Expression):
    """An operation: ``make(*child_capsules)`` builds its C capsule.

    ``op`` names the operation and ``attrs`` holds the parameters that rewrites
    such as :func:`compact` read (``idx`` for index, ``p`` for power).
    Compilation only uses ``make``.
    """

    def __init__(self, make, args, shape, op=None, attrs=None):
        self.make = make
        self.args = tuple(args)
        self.shape = tuple(shape)
        self.op = op
        self.attrs = attrs


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
        return self.value.toarray() if sps.issparse(self.value) else self.value


class Variable(Expression):
    """A block of the scope's flat variable vector. Create via Scope.Variable."""

    def __init__(self, scope, offset, shape):
        self._scope, self._offset, self.shape = scope, offset, shape

    @property
    def value(self):
        flat = self._scope._values[self._offset:self._offset + self.size]
        return flat.reshape(self.shape, order="F").copy()

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
        if self._flat is None:
            return None
        return self._flat.reshape(self.shape, order="F").copy()

    @value.setter
    def value(self, val):
        flat = _flatten_shaped(val, self.shape)
        if self._symmetric:
            _check_symmetric(flat, self.shape)
        self._flat = flat
        self._scope._param_version += 1


def _check_symmetric(flat, shape):
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
            or (arr.ndim == 1 and min(shape) == 1 and arr.size == shape[0] * shape[1])):
        raise DimensionError(f"expected shape {shape}, got {arr.shape}")
    return arr.ravel(order="F")


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
        self._values_version = 0  # bumped on every write, so problems can skip stale work
        self._n_params = 0
        self._param_version = 0

    def Variable(self, d1, d2):
        """Create a (d1, d2) variable; its entries are appended to the flat vector."""
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
        return self._values.size

    def set_values(self, array):
        """Set all variables at once from a flat array (declaration order)."""
        self._values[:] = _flatten_vector(array, self._values.size, "values")
        self._values_version += 1

    def get_values(self):
        return self._values.copy()


def cast_to_const(value):
    """Return ``value`` as an Expression, wrapping non-Expressions in a Constant."""
    return value if isinstance(value, Expression) else Constant(value)


def _is_data(e):
    return isinstance(e, (Constant, Parameter))


# ---------------------------------------------------------------------------
# Operator dispatch
# ---------------------------------------------------------------------------

def _broadcast_shape(a, b):
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
    return Node(lambda c: _C.make_broadcast(c, *shape), [x], shape,
                "broadcast")


def _add(a, b):
    shape = _broadcast_shape(a.shape, b.shape)
    return Node(_C.make_add, [broadcast(a, shape), broadcast(b, shape)], shape,
                "add")


def _mul(a, b):
    if _is_data(b) and not _is_data(a):
        a, b = b, a  # put the constant/parameter operand (if any) first
    if _is_data(a) and a.shape == (1, 1):
        return Node(_C.make_param_scalar_mult, [a, b], b.shape, "scalar_mult")
    shape = _broadcast_shape(a.shape, b.shape)
    a, b = broadcast(a, shape), broadcast(b, shape)
    if _is_data(a):
        return Node(_C.make_param_vector_mult, [a, b], shape, "vector_mult")
    return Node(_C.make_multiply, [a, b], shape, "multiply")


def _matmul(a, b):
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Matmul shape mismatch: {a.shape} @ {b.shape}")
    shape = (a.shape[0], b.shape[1])
    if _is_data(a):
        return Node(_matmul_with_data(_C.make_left_matmul, a), _data_args(a, b),
                    shape, "left_matmul")
    if _is_data(b):
        return Node(_matmul_with_data(_C.make_right_matmul, b), _data_args(b, a),
                    shape, "right_matmul")
    return Node(_C.make_matmul, [a, b], shape, "matmul")


def _data_args(data, x):
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
    idx = np.asarray(flat_indices, dtype=np.int32).ravel()
    shape = _check_shape(*shape)
    if idx.size != shape[0] * shape[1]:
        raise ValueError(f"{idx.size} indices do not fill shape {shape}")
    if idx.size and (idx.min() < 0 or idx.max() >= x.size):
        raise IndexError(f"flat index out of range for size {x.size}")
    return Node(lambda c: _C.make_index(c, shape[0], shape[1], idx), [x], shape,
                "index", {"idx": idx})


def _axis_indices(key, n):
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
            raise IndexError(f"Expected 2 indices for shape {x.shape}, got {len(key)}")
        rows, cols = _axis_indices(key[0], d1), _axis_indices(key[1], d2)
        flat = (rows[:, None] + cols[None, :] * d1).ravel(order="F")
        return index_flat(x, flat, (rows.size, cols.size))
    if d1 > 1 and d2 > 1:
        raise IndexError(f"A single index on shape {x.shape} is ambiguous; "
                         "use X[i, j], X[i, :] or X[:, j]")
    flat = _axis_indices(key, x.size)
    return index_flat(x, flat, (1, flat.size) if d1 == 1 else (flat.size, 1))
