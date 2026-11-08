"""Named functions: elementwise atoms, reductions, stacking, and special forms."""

import numpy as np
import scipy.sparse as sps

from sparsediffpy import _sparsediffengine as _C
from sparsediffpy._expression import (
    Constant, Node, Parameter, Variable, _check_shape, _check_symmetric, _wrap,
)


def _require_variables(name, *xs):
    """Restricted-domain atoms in the engine only accept plain Variables."""
    if not all(isinstance(x, Variable) for x in xs):
        raise ValueError(f"{name} cannot be applied directly to a composite "
                         "expression; its arguments must be plain Variables")
    if len(xs) == 2 and xs[0] is xs[1]:
        raise ValueError(f"{name}: arguments must be distinct Variables")


def _elementwise(make, name, restricted=False):
    def atom(x):
        x = _wrap(x)
        if restricted:
            _require_variables(name, x)
        return Node(make, [x], x.shape, name)
    atom.__name__ = atom.__qualname__ = name
    atom.__doc__ = f"Elementwise {name}."
    return atom


sin = _elementwise(_C.make_sin, "sin")
cos = _elementwise(_C.make_cos, "cos")
tan = _elementwise(_C.make_tan, "tan", restricted=True)
exp = _elementwise(_C.make_exp, "exp")
log = _elementwise(_C.make_log, "log", restricted=True)
sinh = _elementwise(_C.make_sinh, "sinh")
tanh = _elementwise(_C.make_tanh, "tanh")
asinh = _elementwise(_C.make_asinh, "asinh")
atanh = _elementwise(_C.make_atanh, "atanh", restricted=True)
atan = _elementwise(_C.make_atan, "atan")
logistic = _elementwise(_C.make_logistic, "logistic")
normal_cdf = _elementwise(_C.make_normal_cdf, "normal_cdf")
entr = _elementwise(_C.make_entr, "entr", restricted=True)
xexp = _elementwise(_C.make_xexp, "xexp")


def power(x, p):
    """Elementwise x ** p for a constant exponent p."""
    return _wrap(x) ** p


def atan2(y, x):
    """Elementwise atan2(y, x) for distinct Variables of the same shape."""
    _require_variables("atan2", y, x)
    if y.shape != x.shape:
        raise ValueError(f"atan2: shape mismatch {y.shape} vs {x.shape}")
    return Node(_C.make_atan2, [y, x], y.shape, "atan2")


def _reduction_shape(x, axis):
    """Output shape of sum/prod. The engine always returns a row vector."""
    d1, d2 = x.shape
    shapes = {None: (1, 1), 0: (1, d2), 1: (1, d1)}
    if axis not in shapes:
        raise ValueError(f"Invalid axis {axis}, must be None, 0, or 1")
    return shapes[axis]


def sum(x, axis=None):
    """Sum of all entries (axis=None) -> (1, 1); axis=0 -> (1, d2); axis=1 -> (1, d1)."""
    x = _wrap(x)
    shape = _reduction_shape(x, axis)
    c_axis = -1 if axis is None else axis
    return Node(lambda c: _C.make_sum(c, c_axis), [x], shape, "sum")


def prod(x, axis=None):
    """Product of entries; shapes as for :func:`sum`. ``x`` must be a Variable."""
    if not isinstance(x, Variable):
        raise ValueError("prod requires its argument to be a plain Variable "
                         "(the engine has no chain rule for prod yet)")
    shape = _reduction_shape(x, axis)
    make = {None: _C.make_prod, 0: _C.make_prod_axis_zero,
            1: _C.make_prod_axis_one}[axis]
    return Node(make, [x], shape, "prod")


def reshape(x, d1, d2):
    """Column-major reshape."""
    x, shape = _wrap(x), _check_shape(d1, d2)
    if x.size != shape[0] * shape[1]:
        raise ValueError(f"Cannot reshape {x.shape} to {shape}")
    return Node(lambda c: _C.make_reshape(c, *shape), [x], shape, "reshape")


def trace(x):
    x = _wrap(x)
    if x.shape[0] != x.shape[1]:
        raise ValueError(f"trace requires a square matrix, got {x.shape}")
    return Node(_C.make_trace, [x], (1, 1), "trace")


def diag_vec(x):
    """(n, 1) column vector -> (n, n) diagonal matrix."""
    x = _wrap(x)
    if x.shape[1] != 1:
        raise ValueError(f"diag_vec requires a column vector, got {x.shape}")
    return Node(_C.make_diag_vec, [x], (x.shape[0], x.shape[0]), "diag_vec")


def hstack(exprs):
    """Concatenate horizontally; all arguments need the same number of rows."""
    exprs = [_wrap(e) for e in exprs]
    if not exprs or len({e.shape[0] for e in exprs}) != 1:
        raise ValueError(f"hstack: need >=1 args with equal rows, got "
                         f"{[e.shape for e in exprs]}")
    shape = (exprs[0].shape[0], np.sum([e.shape[1] for e in exprs]).item())
    return Node(lambda *caps: _C.make_hstack(list(caps)), exprs, shape, "hstack")


def vstack(exprs):
    """Concatenate vertically; all arguments need the same number of columns."""
    exprs = [_wrap(e) for e in exprs]
    if not exprs or len({e.shape[1] for e in exprs}) != 1:
        raise ValueError(f"vstack: need >=1 args with equal columns, got "
                         f"{[e.shape for e in exprs]}")
    shape = (np.sum([e.shape[0] for e in exprs]).item(), exprs[0].shape[1])
    return Node(lambda *caps: _C.make_vstack(list(caps)), exprs, shape, "vstack")


def quad_form(x, Q):
    """x' Q x for a column vector x (n, 1) and a constant or Parameter Q (n, n).

    A NumPy Q uses the engine's dense (BLAS) path, a scipy.sparse Q the sparse
    path. Constant Q is symmetrized; a Parameter Q must be symmetric, which is
    checked here and on every later ``Q.value`` update.
    """
    x, Q = _wrap(x), _wrap(Q)
    n = x.shape[0]
    if x.shape[1] != 1 or Q.shape != (n, n):
        raise ValueError(f"quad_form: need x (n, 1) and Q (n, n), "
                         f"got {x.shape} and {Q.shape}")
    if isinstance(Q, Parameter):
        Q._symmetric = True  # checked now and on every later Q.value update
        if Q._flat is not None:
            _check_symmetric(Q._flat, Q.shape)
        return Node(lambda p, c: _C.make_quad_form(p, c, "dense", None, n),
                    [Q, x], (1, 1), "quad_form")
    if not isinstance(Q, Constant):
        raise TypeError("quad_form: Q must be a constant or a Parameter")
    V = (Q.value + Q.value.T) / 2  # the engine assumes symmetric Q
    if not sps.issparse(V):
        data = np.ascontiguousarray(V).ravel()
        return Node(lambda c: _C.make_quad_form(None, c, "dense", data, n),
                    [x], (1, 1), "quad_form")
    V = sps.csr_matrix(V)
    return Node(lambda c: _C.make_quad_form(
        None, c, "sparse", V.data, V.indices.astype(np.int32),
        V.indptr.astype(np.int32), n, n), [x], (1, 1), "quad_form")


def quad_over_lin(x, z):
    """sum(x**2) / z for distinct Variables x and scalar z."""
    _require_variables("quad_over_lin", x, z)
    if z.shape != (1, 1):
        raise ValueError(f"quad_over_lin: z must be scalar, got {z.shape}")
    return Node(_C.make_quad_over_lin, [x, z], (1, 1), "quad_over_lin")


def rel_entr(x, y):
    """Elementwise x * log(x / y) for distinct Variables x, y.

    Shapes must match, or one of them must be scalar.
    """
    _require_variables("rel_entr", x, y)
    if x.shape == y.shape or y.shape == (1, 1):
        shape = x.shape
    elif x.shape == (1, 1):
        shape = y.shape
    else:
        raise ValueError(f"rel_entr: incompatible shapes {x.shape} and {y.shape}")
    return Node(_C.make_rel_entr, [x, y], shape, "rel_entr")
