"""Every atom: forward value vs NumPy, Jacobian and Hessian vs finite differences.

Each case builds (expr, reference) on a fresh scope; ``reference()`` computes
the expected value from the variables' current values with NumPy.
"""

import numpy as np
import pytest
import scipy.sparse as sps
from conftest import check_derivatives
from scipy.special import ndtr

import sparsediffpy as sp

A = np.array([[1.0, -2.0, 0.5], [0.0, 3.0, 1.0]])
Q = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 4.0]])
CASES = {}  # name -> (build, positive_domain)


def case(positive=False):
    def register(build):
        CASES[build.__name__] = (build, positive)
        return build
    return register


def _elementwise(name, f, ref, positive=False):
    def build(s):
        X = s.Variable(3, 2)
        return f(X), lambda: ref(X.value)
    build.__name__ = name
    case(positive)(build)


for _name, _f, _ref in [
    ("sin", sp.sin, np.sin), ("cos", sp.cos, np.cos), ("exp", sp.exp, np.exp),
    ("sinh", sp.sinh, np.sinh), ("tanh", sp.tanh, np.tanh),
    ("asinh", sp.asinh, np.arcsinh), ("atan", sp.atan, np.arctan),
    ("logistic", sp.logistic, lambda x: np.log1p(np.exp(x))),
    ("normal_cdf", sp.normal_cdf, ndtr), ("xexp", sp.xexp, lambda x: x * np.exp(x)),
    ("power3", lambda x: x ** 3, lambda x: x ** 3),
]:
    _elementwise(_name, _f, _ref)

for _name, _f, _ref in [
    ("log", sp.log, np.log), ("tan", sp.tan, np.tan),
    ("atanh", sp.atanh, np.arctanh), ("entr", sp.entr, lambda x: -x * np.log(x)),
    ("power_half", lambda x: sp.power(x, 0.5), np.sqrt),
]:
    _elementwise(_name, _f, _ref, positive=True)


@case(positive=True)
def atan2(s):
    y, x = s.Variable(3, 1), s.Variable(3, 1)
    return sp.atan2(y, x), lambda: np.arctan2(y.value, x.value)


# --- affine -----------------------------------------------------------------

@case()
def add_broadcast(s):
    a, r, c, X = s.Variable(1, 1), s.Variable(1, 2), s.Variable(3, 1), s.Variable(3, 2)
    return a * X + r + c - 1.0, lambda: a.value * X.value + r.value + c.value - 1


@case()
def neg_sub(s):
    x, y = s.Variable(3, 1), s.Variable(3, 1)
    return -x - y + np.arange(3.0), lambda: -x.value - y.value + np.c_[0:3.0]


@case()
def constant_mult(s):
    X = s.Variable(3, 2)
    B = np.arange(6.0).reshape(3, 2)
    return 2.0 * X * B, lambda: 2 * X.value * B


@case()
def matmul_dense_left_right(s):
    x, Y = s.Variable(3, 1), s.Variable(2, 3)
    return A @ x + (Y @ A.T)[:, 0:1], lambda: A @ x.value + (Y.value @ A.T)[:, 0:1]


@case()
def matmul_sparse_left_right(s):
    x, Y = s.Variable(3, 1), s.Variable(2, 3)
    S = sps.csr_matrix(A)
    return S @ x + (Y @ S.T)[:, 0:1], lambda: A @ x.value + (Y.value @ A.T)[:, 0:1]


@case()
def transpose_reshape(s):
    X = s.Variable(3, 2)
    return sp.reshape(X.T, 6, 1), lambda: X.value.T.reshape(6, 1, order="F")


@case()
def sums(s):
    X = s.Variable(3, 2)
    return sp.hstack([sp.sum(X), sp.sum(X, axis=0), sp.sum(X, axis=1)]), \
        lambda: np.r_[X.value.sum(), X.value.sum(0), X.value.sum(1)][None, :]


@case()
def trace_diag_vec(s):
    X, x = s.Variable(3, 3), s.Variable(3, 1)
    return sp.trace(X) + sp.diag_vec(x), \
        lambda: np.trace(X.value) + np.diag(x.value.ravel())


@case()
def stacking(s):
    x, y = s.Variable(2, 1), s.Variable(2, 2)
    return sp.vstack([sp.hstack([x, y]), sp.hstack([y, x])]), \
        lambda: np.block([[x.value, y.value], [y.value, x.value]])


@case()
def indexing(s):
    X, r = s.Variable(3, 3), s.Variable(1, 4)
    return sp.vstack([X[0:2, [2, 0]], X[-1, 1:], r[1:3]]), \
        lambda: np.r_[X.value[0:2][:, [2, 0]], X.value[-1:, 1:], r.value[:, 1:3]]


# --- bivariate / other --------------------------------------------------------

@case()
def multiply(s):
    x, Y = s.Variable(3, 1), s.Variable(3, 2)
    return sp.sin(x) * Y, lambda: np.sin(x.value) * Y.value


@case()
def matmul_variables(s):
    X, Y = s.Variable(2, 3), s.Variable(3, 2)
    return X @ Y, lambda: X.value @ Y.value


@case()
def x_transpose_x(s):
    x = s.Variable(3, 1)
    return x.T @ x, lambda: x.value.T @ x.value


@case()
def quad_forms(s):
    x = s.Variable(3, 1)
    P = s.Parameter(3, 3, value=2 * Q)
    xv = lambda: x.value.ravel()  # noqa: E731
    return sp.vstack([sp.quad_form(x, Q), sp.quad_form(x, sps.csr_matrix(Q)),
                      sp.quad_form(x, P)]), \
        lambda: np.c_[[xv() @ Q @ xv(), xv() @ Q @ xv(), 2 * xv() @ Q @ xv()]]


@case(positive=True)
def quad_over_lin(s):
    x, z = s.Variable(3, 1), s.Variable(1, 1)
    return sp.quad_over_lin(x, z), lambda: (x.value ** 2).sum() / z.value


@case(positive=True)
def rel_entr_vectors(s):
    x, y = s.Variable(3, 1), s.Variable(3, 1)
    return sp.rel_entr(x, y), lambda: x.value * np.log(x.value / y.value)


@case(positive=True)
def rel_entr_scalar_vector(s):
    a, y = s.Variable(1, 1), s.Variable(3, 1)
    return sp.rel_entr(a, y), lambda: a.value * np.log(a.value / y.value)


@case(positive=True)
def rel_entr_vector_scalar(s):
    x, b = s.Variable(3, 1), s.Variable(1, 1)
    return sp.rel_entr(x, b), lambda: x.value * np.log(x.value / b.value)


@case(positive=True)
def prods(s):
    X = s.Variable(3, 2)
    return sp.hstack([sp.prod(X), sp.prod(X, axis=0), sp.prod(X, axis=1)]), \
        lambda: np.r_[X.value.prod(), X.value.prod(0), X.value.prod(1)][None, :]


# --- compositions -------------------------------------------------------------

@case()
def parameter_matmul_and_scale(s):
    x = s.Variable(3, 1)
    B, c = s.Parameter(2, 3, value=A), s.Parameter(1, 1, value=3.0)
    return c * sp.exp(B @ x) + B @ sp.sin(x), \
        lambda: 3 * np.exp(A @ x.value) + A @ np.sin(x.value)


@case()
def deep_chain(s):
    x = s.Variable(4, 1)
    f = x
    for _ in range(5):
        f = sp.tanh(0.9 * f + 0.1)
    return sp.sum(f * f) + sp.cos(x[0]), None


@case()
def shared_subexpression(s):
    x = s.Variable(3, 1)
    u = sp.sin(A @ x)
    return u * u + sp.exp(u), lambda: np.sin(A @ x.value) ** 2 + np.exp(np.sin(A @ x.value))


@pytest.mark.parametrize("name", list(CASES))
def test_atom(name, rng):
    build, positive = CASES[name]
    scope = sp.Scope()
    f, ref = build(scope)
    fn = sp.compile(f)
    low, high = (0.2, 0.9) if positive else (-1.0, 1.0)
    x0 = rng.uniform(low, high, scope.total_var_size)
    scope.set_values(x0)
    out = fn.forward()
    assert out.shape == (f.size,)
    if ref is not None:
        expected = np.broadcast_to(ref(), f.shape).ravel(order="F")
        np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-12)
    check_derivatives(fn, scope, x0, rng.standard_normal(f.size))
