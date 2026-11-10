"""Each atom is wired to the right engine call.

A case is ``(name, shapes, f, positive)``. ``f(m, *xs)`` is evaluated once
with ``m = sp`` on Variables (the expression under test) and once with
``m = NP`` on their values (the expected result). A shape that is an array
declares a Parameter with that value. The value must match NumPy, and the
Jacobian and Hessian must match finite differences.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sps
from conftest import check_derivatives
from scipy.special import ndtr

import sparsediffpy as sp

A = np.array([[1.0, -2.0, 0.5], [0.0, 3.0, 1.0]])
Q = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 4.0]])
N = Q + np.triu(np.ones((3, 3)), 1)  # not symmetric


def _reduce(f):
    """NumPy sum/prod with sp's shapes: (1, 1), or a row for axis 0 or 1."""
    return lambda X, axis=None: np.reshape(f(X, axis=axis), (1, -1))


def _dense(M):
    return M.toarray() if sps.issparse(M) else M


NP = SimpleNamespace(
    sin=np.sin, cos=np.cos, exp=np.exp, sinh=np.sinh, tanh=np.tanh, log=np.log,
    tan=np.tan, asinh=np.arcsinh, atan=np.arctan, atanh=np.arctanh,
    logistic=lambda x: np.logaddexp(0, x), normal_cdf=ndtr,
    xexp=lambda x: x * np.exp(x), entr=lambda x: -x * np.log(x),
    power=lambda x, p: x ** p, atan2=np.arctan2,
    sum=_reduce(np.sum), prod=_reduce(np.prod), hstack=np.hstack, vstack=np.vstack,
    reshape=lambda x, d1, d2: x.reshape(d1, d2, order="F"),
    trace=lambda X: np.trace(X).reshape(1, 1), diag_vec=lambda x: np.diag(x.ravel()),
    quad_form=lambda x, P: x.T @ _dense(P) @ x,
    quad_over_lin=lambda x, z: (x ** 2).sum() / z,
    rel_entr=lambda x, y: x * np.log(x / y),
)

ELEMENTWISE = ["sin", "cos", "exp", "sinh", "tanh", "asinh", "atan", "logistic",
               "normal_cdf", "xexp"]
POSITIVE_ELEMENTWISE = ["log", "tan", "atanh", "entr"]

CASES = [
    *[(a, [(3, 2)], lambda m, x, a=a: getattr(m, a)(x), False) for a in ELEMENTWISE],
    *[(a, [(3, 2)], lambda m, x, a=a: getattr(m, a)(x), True)
      for a in POSITIVE_ELEMENTWISE],
    ("power", [(3, 2)], lambda m, x: x ** 3 + m.power(x, 0.5), True),
    ("atan2", [(3, 1), (3, 1)], lambda m, y, x: m.atan2(y, x), True),
    # affine
    ("add_broadcast", [(1, 1), (1, 2), (3, 1), (3, 2)],
     lambda m, a, r, c, X: a * X + r + c - 1.0, False),
    ("neg_sub", [(3, 1), (3, 1)], lambda m, x, y: -x - y + np.c_[0:3.0], False),
    ("constant_mult", [(3, 2)], lambda m, X: 2.0 * X * np.arange(6.0).reshape(3, 2),
     False),
    ("constant_div", [(3, 2), (1, 1)],
     lambda m, X, a: X / 2.0 + a / np.arange(1.0, 7.0).reshape(3, 2), False),
    ("matmul_dense", [(3, 1), (2, 3)], lambda m, x, Y: A @ x + (Y @ A.T)[:, 0:1],
     False),
    ("matmul_sparse", [(3, 1), (2, 3)],
     lambda m, x, Y: sps.csr_matrix(A) @ x + (Y @ sps.csr_matrix(A.T))[:, 0:1],
     False),
    ("transpose_reshape", [(3, 2)], lambda m, X: m.reshape(X.T, 6, 1), False),
    ("sums", [(3, 2)],
     lambda m, X: m.hstack([m.sum(X), m.sum(X, axis=0), m.sum(X, axis=1)]), False),
    ("trace_diag_vec", [(3, 3), (3, 1)], lambda m, X, x: m.trace(X) + m.diag_vec(x),
     False),
    ("stacking", [(2, 1), (2, 2)],
     lambda m, x, y: m.vstack([m.hstack([x, y]), m.hstack([y, x])]), False),
    ("indexing", [(3, 3), (1, 4)],
     lambda m, X, r: m.vstack([X[0:2, [2, 0]], X[2:3, 1:], r[:, 1:3]]
                              if m is NP else [X[0:2, [2, 0]], X[-1, 1:], r[1:3]]),
     False),
    # bivariate and other
    ("multiply", [(3, 1), (3, 2)], lambda m, x, Y: m.sin(x) * Y, False),
    ("matmul_variables", [(2, 3), (3, 2)], lambda m, X, Y: X @ Y, False),
    ("x_transpose_x", [(3, 1)], lambda m, x: x.T @ x, False),
    ("quad_form", [(3, 1), 2 * Q],
     lambda m, x, P: m.vstack([m.quad_form(x, Q), m.quad_form(x, sps.csr_matrix(Q)),
                               m.quad_form(x, P)]), False),
    ("quad_form_nonsymmetric", [(3, 1)],
     lambda m, x: m.quad_form(x, N) + m.quad_form(x, sps.csr_matrix(N)), False),
    ("quad_over_lin", [(3, 1), (1, 1)], lambda m, x, z: m.quad_over_lin(x, z), True),
    ("rel_entr", [(3, 1), (3, 1)], lambda m, x, y: m.rel_entr(x, y), True),
    ("rel_entr_scalar_vector", [(1, 1), (3, 1)], lambda m, x, y: m.rel_entr(x, y),
     True),
    ("rel_entr_vector_scalar", [(3, 1), (1, 1)], lambda m, x, y: m.rel_entr(x, y),
     True),
    ("prods", [(3, 2)],
     lambda m, X: m.hstack([m.prod(X), m.prod(X, axis=0), m.prod(X, axis=1)]), True),
    # compositions
    ("parameter_matmul_and_scale", [(3, 1), A, np.array([[3.0]])],
     lambda m, x, B, c: c * m.exp(B @ x) + B @ m.sin(x), False),
    ("shared_subexpression", [(3, 1)],
     lambda m, x: (lambda u: u * u + m.exp(u))(m.sin(A @ x)), False),
]


@pytest.mark.parametrize("name, shapes, f, positive", CASES,
                         ids=[c[0] for c in CASES])
def test_atom(name, shapes, f, positive, rng):
    scope = sp.Scope()
    leaves = [scope.Parameter(*s.shape, value=s) if isinstance(s, np.ndarray)
              else scope.Variable(*s) for s in shapes]
    expr = f(sp, *leaves)
    fn = sp.compile(expr)
    low, high = (0.2, 0.9) if positive else (-1.0, 1.0)
    x0 = rng.uniform(low, high, scope.total_var_size)
    scope.set_values(x0)
    expected = f(NP, *[leaf.value for leaf in leaves])
    np.testing.assert_allclose(fn.forward(), np.broadcast_to(expected, expr.shape),
                               rtol=1e-10, atol=1e-12)
    check_derivatives(fn, scope, x0, rng.standard_normal(expr.size))
