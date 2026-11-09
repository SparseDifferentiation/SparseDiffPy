"""Scope/values, shape rules, validation errors, parameters, and Problem."""

import sys

import numpy as np
import pytest

import sparsediffpy as sp
from sparsediffpy import _compile


# --- scope and values ---------------------------------------------------------

def test_values_are_column_major(scope):
    X, y = scope.Variable(2, 2), scope.Variable(1, 1)
    X.value = np.array([[1.0, 2.0], [3.0, 4.0]])
    y.value = 5.0
    np.testing.assert_array_equal(scope.get_values(), [1, 3, 2, 4, 5])
    scope.set_values(np.arange(5.0))
    np.testing.assert_array_equal(X.value, [[0, 2], [1, 3]])


def test_wrong_sizes(scope):
    x = scope.Variable(3, 1)
    p = scope.Parameter(2, 1)
    with pytest.raises(sp.DimensionError):
        x.value = [1.0, 2.0]
    with pytest.raises(sp.DimensionError):
        p.value = [1.0, 2.0, 3.0]
    with pytest.raises(sp.DimensionError):
        scope.set_values(np.zeros(4))
    assert p.value is None


def test_value_shapes(scope):
    """Values need the exact shape, not just the right number of elements."""
    X, r, c, y = (scope.Variable(2, 3), scope.Variable(1, 3),
                  scope.Variable(3, 1), scope.Variable(1, 1))
    P = scope.Parameter(2, 3)
    for leaf in (X, P):
        for bad in (np.zeros(6), np.zeros((3, 2)), np.zeros((6, 1)), 1.0):
            with pytest.raises(sp.DimensionError, match="expected shape"):
                leaf.value = bad
        leaf.value = np.arange(6.0).reshape(2, 3)
        np.testing.assert_array_equal(leaf.value, np.arange(6.0).reshape(2, 3))
    with pytest.raises(sp.DimensionError):
        r.value = np.zeros((3, 1))
    with pytest.raises(sp.DimensionError):
        c.value = np.zeros((1, 3))
    r.value, c.value, y.value = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 7.0
    np.testing.assert_array_equal(r.value, [[1, 2, 3]])
    np.testing.assert_array_equal(c.value, [[4], [5], [6]])
    y.value = [8.0]
    assert y.value.shape == (1, 1)

    n = scope.total_var_size
    with pytest.raises(sp.DimensionError, match="values"):
        scope.set_values(np.zeros((n, 1)))
    scope.set_values(np.zeros(n))

    fn = sp.compile(sp.exp(X))
    with pytest.raises(sp.DimensionError):
        fn.hessian(np.ones(6))
    assert fn.hessian(np.ones((2, 3))).shape == (n, n)
    fn = sp.compile(sp.exp(c))
    assert fn.hessian(np.ones(3)).shape == fn.hessian(np.ones((3, 1))).shape
    with pytest.raises(sp.DimensionError):
        fn.hessian(np.ones((1, 3)))


@pytest.mark.parametrize("shape", [(0, 1), (-1, 2), (2.0, 1)])
def test_invalid_shapes(scope, shape):
    with pytest.raises((ValueError, TypeError)):
        scope.Variable(*shape)


def test_reevaluates_at_new_values(scope):
    x = scope.Variable(2, 1)
    fn = sp.compile(sp.exp(x))
    for v in ([0.0, 1.0], [2.0, -1.0]):
        x.value = v
        np.testing.assert_allclose(fn.forward(), np.exp(v)[:, None])
        np.testing.assert_allclose(fn.jacobian().diagonal(), np.exp(v))
        np.testing.assert_allclose(fn.hessian([1, 1]).diagonal(), np.exp(v))


def test_two_compiles_share_scope(scope):
    x, y = scope.Variable(2, 1), scope.Variable(2, 1)
    f, g = sp.compile(sp.sin(x)), sp.compile(x * y)
    scope.set_values([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(f.forward(), np.sin([[1], [2]]))
    np.testing.assert_allclose(g.forward(), [[3], [8]])
    assert f.jacobian().shape == g.jacobian().shape == (2, 4)


def test_forward_has_expression_shape(scope):
    x, M = scope.Variable(1, 3), scope.Variable(2, 2)
    scope.set_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert sp.compile(2.0 * x).forward().shape == (1, 3)
    out = sp.compile(sp.exp(M)).forward()
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, np.exp([[4.0, 6.0], [5.0, 7.0]]))  # column-major


def test_constants_must_be_2d(scope):
    x = scope.Variable(3, 1)
    for bad in (np.ones(3), [1.0, 2.0, 3.0], np.zeros((2, 2, 2))):
        with pytest.raises(ValueError, match="2-D"):
            x + bad
        with pytest.raises(ValueError, match="2-D"):
            bad * x
    assert (x + np.array([2.0])).shape == (x + 2.0).shape == (3, 1)
    assert (x + np.ones((3, 1))).shape == (3, 1)


def test_variable_declared_after_compile(scope):
    x = scope.Variable(2, 1)
    fn = sp.compile(sp.exp(x))
    scope.Variable(3, 1)
    x.value = [0.0, 1.0]
    np.testing.assert_allclose(fn.forward(), np.exp([[0.0], [1.0]]))
    assert fn.jacobian().shape == (2, 2)


def test_affine_hessian_is_zero(scope):
    x = scope.Variable(3, 1)
    fn = sp.compile(sp.sum(-x[0:2]) + 2.0)
    x.value = [1.0, 2.0, 3.0]
    assert not fn.hessian([1.0]).toarray().any()


# --- shapes and validation ----------------------------------------------------

def test_shapes(scope):
    x, r, X = scope.Variable(3, 1), scope.Variable(1, 2), scope.Variable(3, 2)
    assert (x + r).shape == (3, 2)
    assert (X @ r.T).shape == (3, 1)
    assert x.T.shape == (1, 3)
    assert x[1].shape == (1, 1) and x[-2:].shape == (2, 1)
    assert r[1].shape == (1, 1) and X[:, 1].shape == (3, 1)
    assert sp.sum(X, axis=1).shape == (1, 3)


@pytest.mark.parametrize("make", [
    lambda s: s.Variable(3, 1) + s.Variable(2, 1),
    lambda s: s.Variable(3, 2) @ s.Variable(3, 2),
    lambda s: sp.trace(s.Variable(2, 3)),
    lambda s: sp.diag_vec(s.Variable(1, 3)),
    lambda s: sp.reshape(s.Variable(3, 2), 4, 2),
    lambda s: sp.hstack([]),
    lambda s: sp.vstack([s.Variable(2, 1), s.Variable(2, 2)]),
    lambda s: sp.quad_form(s.Variable(3, 1), np.eye(2)),
    lambda s: sp.sum(s.Variable(2, 2), axis=2),
    lambda s: sp.rel_entr(s.Variable(3, 1), s.Variable(2, 1)),
])
def test_shape_errors(scope, make):
    with pytest.raises(ValueError):
        make(scope)


@pytest.mark.parametrize("make", [
    lambda x: sp.log(x[1:3]),
    lambda x: sp.entr(2.0 * x),
    lambda x: sp.prod(sp.exp(x)),
    lambda x: sp.quad_over_lin(x, x[0]),
    lambda x: sp.rel_entr(x, x),
])
def test_variable_only_atoms(scope, make):
    with pytest.raises(ValueError):
        make(scope.Variable(4, 1))


@pytest.mark.parametrize("key", [4, -5, [0, 4], (0, 3), (3, 0)])
def test_index_out_of_range(scope, key):
    X = scope.Variable(2, 2) if isinstance(key, tuple) else scope.Variable(4, 1)
    with pytest.raises(IndexError):
        X[key]


def test_index_must_be_explicit(scope):
    x, X = scope.Variable(4, 1), scope.Variable(2, 3)
    for key in (1, slice(0, 2), [0, 1]):
        with pytest.raises(IndexError, match="ambiguous"):
            X[key]
    assert X[1, :].shape == (1, 3) and X[:, 1].shape == (2, 1)
    with pytest.raises(IndexError, match="integer"):
        x[[True, False, True, False]]
    for key in (True, False, np.True_, (True, 0), (0, False)):
        with pytest.raises(IndexError, match="Boolean"):
            x[key]
    with pytest.raises(IndexError, match="1-D"):
        x[np.array([[0, 1]])]
    with pytest.raises(IndexError, match="2 indices"):
        X[0, 0, 0]


def test_compile_errors(scope):
    with pytest.raises(ValueError, match="at least one Variable"):
        sp.compile(sp.Scope().Parameter(1, 1, value=1.0) * 2.0)
    with pytest.raises(ValueError, match="same Scope"):
        sp.compile(scope.Variable(1, 1) + sp.Scope().Variable(1, 1))
    with pytest.raises(TypeError):
        scope.Variable(1, 1) ** scope.Variable(1, 1)


def test_expressions_are_not_iterable(scope):
    x, X = scope.Variable(3, 1), scope.Variable(2, 2)
    with pytest.raises(TypeError, match="not iterable"):
        list(x)
    with pytest.raises(TypeError, match="not iterable"):
        sp.Problem(sp.sum(X), X)  # constraints must be a list
    assert sp.Problem(sp.sum(X), [X]).total_constraint_size == 4


def test_none_is_rejected(scope):
    x, p = scope.Variable(2, 1), scope.Parameter(2, 1)
    with pytest.raises(TypeError, match="None"):
        x + None
    with pytest.raises(TypeError, match="None"):
        x.value = None
    with pytest.raises(TypeError, match="None"):
        p.value = None


def test_index_flat_wraps_constants(scope):
    x = scope.Variable(2, 1)
    fn = sp.compile(sp.index_flat(np.array([[5.0], [6.0]]), [1], (1, 1)) + x[0])
    x.value = [1.0, 2.0]
    np.testing.assert_array_equal(fn.forward(), [[7.0]])


def test_numpy_scalar_exponent(scope):
    x = scope.Variable(2, 1)
    f1, f2 = sp.compile(x ** np.int64(2)), sp.compile(x ** np.float32(2))
    x.value = [1.5, -2.0]
    np.testing.assert_array_equal(f1.forward(), x.value ** 2)
    np.testing.assert_array_equal(f2.forward(), x.value ** 2)


def test_quad_form_parameter_must_be_symmetric(scope):
    x = scope.Variable(2, 1)
    N, S = np.array([[1.0, 5.0], [0.0, 1.0]]), np.array([[1.0, 2.0], [2.0, 3.0]])
    with pytest.raises(ValueError, match="symmetric"):
        sp.quad_form(x, scope.Parameter(2, 2, value=N))
    P = scope.Parameter(2, 2, value=S)
    fn = sp.compile(sp.quad_form(x, P))
    with pytest.raises(ValueError, match="symmetric"):
        P.value = N
    x.value = [1.0, 1.0]
    P.value = 2 * S
    np.testing.assert_allclose(fn.jacobian().toarray(), [2 * (2 * S) @ [1.0, 1.0]])


# --- parameters ----------------------------------------------------------------

def test_parameter_update_without_recompile(scope):
    x = scope.Variable(3, 1)
    A = scope.Parameter(2, 3, value=np.ones((2, 3)))
    c = scope.Parameter(1, 1, value=1.0)
    fn = sp.compile(c * sp.exp(A @ x))
    x.value = [0.1, 0.2, 0.3]
    for Av, cv in [(np.ones((2, 3)), 1.0), (np.arange(6.0).reshape(2, 3), -2.0)]:
        A.value, c.value = Av, cv
        np.testing.assert_allclose(fn.forward(), cv * np.exp(Av @ x.value))
        J = cv * np.exp(Av @ x.value) * Av
        np.testing.assert_allclose(fn.jacobian().toarray(), J)


def test_unset_parameter_raises(scope):
    x, p = scope.Variable(2, 1), scope.Parameter(2, 1)
    fn = sp.compile(p * x)
    with pytest.raises(ValueError, match="no value"):
        fn.forward()
    p.value = [1.0, 2.0]
    np.testing.assert_allclose(fn.forward(), [[0], [0]])


# --- Caching ------------------------------------------------------------------

def count_calls(monkeypatch, name):
    """Count calls to the engine function `name` as seen from _compile."""
    calls, real = [], getattr(_compile._C, name)

    def counted(*args):
        calls.append(1)
        return real(*args)
    monkeypatch.setattr(_compile._C, name, counted)
    return calls


def test_forward_runs_once_per_point(scope, monkeypatch):
    n_fwd = count_calls(monkeypatch, "problem_constraint_forward")
    x, p = scope.Variable(2, 1), scope.Parameter(1, 1, value=1.0)
    fn = sp.compile(p * sp.exp(x))
    x.value = [0.0, 1.0]
    for _ in range(3):
        np.testing.assert_allclose(fn.forward(), np.exp([[0.0], [1.0]]))
    assert len(n_fwd) == 1
    x.value = [1.0, 1.0]
    fn.forward()
    scope.set_values([2.0, 2.0])
    fn.forward()
    assert len(n_fwd) == 3
    p.value = 3.0
    np.testing.assert_allclose(fn.forward(), 3.0 * np.exp([[2.0], [2.0]]))
    assert len(n_fwd) == 4


def test_derivatives_reuse_forward_and_jacobian(scope, monkeypatch):
    n_fwd = count_calls(monkeypatch, "problem_constraint_forward")
    n_jac = count_calls(monkeypatch, "problem_jacobian")
    n_hess = count_calls(monkeypatch, "problem_hessian")
    x = scope.Variable(2, 1)
    fn = sp.compile(sp.exp(x))
    x.value = [0.5, -0.5]
    fn.forward()
    J = fn.jacobian()
    assert fn.jacobian() is J
    fn.hessian([1.0, 1.0])
    fn.hessian([2.0, 0.0])
    assert (len(n_fwd), len(n_jac), len(n_hess)) == (1, 1, 2)
    x.value = [0.0, 0.0]
    fn.hessian([1.0, 1.0])
    assert (len(n_fwd), len(n_jac), len(n_hess)) == (2, 2, 3)


def test_problem_skips_repeated_u(scope, monkeypatch):
    n_fwd = count_calls(monkeypatch, "problem_constraint_forward")
    x = scope.Variable(2, 1)
    prob = sp.Problem(sp.sum(x), [sp.exp(x)])
    prob.constraint_forward(np.array([1.0, 2.0]))
    prob.constraint_forward(np.array([1.0, 2.0]))  # equal, distinct array
    assert len(n_fwd) == 1
    prob.constraint_forward(np.array([1.0, 3.0]))
    assert len(n_fwd) == 2
    prob.constraint_forward()  # scope values are a different point
    assert len(n_fwd) == 3
    np.testing.assert_allclose(prob.constraint_forward(), [1.0, 1.0])
    assert len(n_fwd) == 3


def test_derivatives_require_forward(scope):
    x = scope.Variable(2, 1)
    prob = sp.Problem(sp.sum(x), [sp.exp(x)])
    with pytest.raises(RuntimeError, match="objective_forward"):
        prob.gradient()
    with pytest.raises(RuntimeError, match="constraint_forward"):
        prob.jacobian()
    prob.objective_forward([1.0, 2.0])
    prob.constraint_forward([1.0, 3.0])
    with pytest.raises(RuntimeError, match="same point"):
        prob.hessian(1.0, [1.0, 1.0])
    prob.objective_forward([1.0, 2.0])  # a point is "the last u": redo both forwards
    prob.constraint_forward([1.0, 2.0])
    assert prob.hessian(1.0, [1.0, 1.0]).shape == (2, 2)


# --- Problem ----------------------------------------------------------------------

def test_problem(scope):
    x = scope.Variable(3, 1)
    shared = sp.sin(x)
    prob = sp.Problem(sp.sum(sp.exp(shared)), [x[0:2] * x[1:3], shared])
    u = np.array([0.3, -0.5, 0.8])
    s, c = np.sin(u), np.cos(u)

    assert prob.objective_forward(u) == pytest.approx(np.exp(s).sum())
    np.testing.assert_allclose(prob.gradient(), np.exp(s) * c)
    np.testing.assert_allclose(prob.constraint_forward(u),
                               np.r_[u[:2] * u[1:], s])

    J = prob.jacobian().toarray()
    np.testing.assert_allclose(J[:2], [[u[1], u[0], 0], [0, u[2], u[1]]])
    np.testing.assert_allclose(J[2:], np.diag(c))

    lam = np.array([1.0, 2.0, 0.5, -1.0, 3.0])
    H = prob.hessian(2.0, lam).toarray()
    H_obj = np.diag(np.exp(s) * (c ** 2 - s))
    H_con = np.array([[0, 1, 0], [1, 0, 2], [0, 2, 0]]) + np.diag(-s * lam[2:])
    np.testing.assert_allclose(H, 2.0 * H_obj + H_con)

    with pytest.raises(ValueError, match="lagrange"):
        prob.hessian(1.0, np.ones(4))
    for bad in (np.zeros(2), np.zeros(4)):
        with pytest.raises(sp.DimensionError, match="1-D array of 3 elements"):
            prob.objective_forward(bad)
        with pytest.raises(sp.DimensionError, match="1-D array of 3 elements"):
            prob.constraint_forward(bad)
    with pytest.raises(sp.DimensionError):
        prob.objective_forward(u[:, None])
    with pytest.raises(sp.DimensionError, match="lagrange"):
        prob.hessian(1.0, lam[:, None])
    with pytest.raises(ValueError, match="scalar"):
        sp.Problem(x)


def test_problem_parameters(scope):
    x, a = scope.Variable(2, 1), scope.Parameter(1, 1, value=2.0)
    prob = sp.Problem(sp.sum(a * x ** 2))
    u = np.array([1.0, 3.0])
    assert prob.objective_forward(u) == pytest.approx(20.0)
    a.value = 0.5
    assert prob.objective_forward(u) == pytest.approx(5.0)
    np.testing.assert_allclose(prob.gradient(), u)


def test_objective_and_constraints_at_different_points(scope):
    x, a = scope.Variable(3, 1), scope.Parameter(1, 1, value=2.0)
    e = sp.exp(a * x)  # shared by the objective and the constraint
    prob = sp.Problem(sp.sum(e), [e])
    prob.objective_forward(np.zeros(3))
    prob.constraint_forward(np.ones(3))
    np.testing.assert_allclose(prob.gradient(), [2.0, 2.0, 2.0])
    np.testing.assert_allclose(prob.jacobian().diagonal(), 2 * np.exp([2.0] * 3))


def test_hessian_without_prior_jacobian(scope):
    """hessian() must refresh the Jacobian values the engine's Hessian reuses."""
    x, y = scope.Variable(2, 1), scope.Variable(2, 1)
    fn = sp.compile(x * y + sp.sin(x * y))
    prob = sp.Problem(sp.sum(x * y), [x * y])
    for u in ([0.1, 0.2, 0.3, 0.4], [1.0, -2.0, 0.5, 3.0]):
        scope.set_values(u)
        xy, xv, yv = np.multiply(u[:2], u[2:]), np.array(u[:2]), np.array(u[2:])
        w = np.array([1.0, 2.0])
        # d2/dx_i dy_i of w_i * (x_i y_i + sin(x_i y_i))
        d2 = lambda a, b: w * (-np.sin(xy) * a * b)  # noqa: E731
        off = np.diag(w * (1 + np.cos(xy)) + d2(xv, yv))
        H = np.block([[np.diag(d2(yv, yv)), off], [off, np.diag(d2(xv, xv))]])
        np.testing.assert_allclose(fn.hessian(w).toarray(), H, atol=1e-12)

        prob.objective_forward(u)
        prob.constraint_forward(u)
        H_prob = np.block([[np.zeros((2, 2)), np.diag(2.0 + w)],
                           [np.diag(2.0 + w), np.zeros((2, 2))]])
        np.testing.assert_allclose(prob.hessian(2.0, w).toarray(), H_prob,
                                   atol=1e-12)


# --- deep expressions -------------------------------------------------------------

def test_compile_deep_sum(scope):
    """A loop-built sum is a chain of adds deeper than the recursion limit."""
    x, p = scope.Variable(3, 1), scope.Parameter(1, 1, value=2.0)
    n_adds = 5000
    assert n_adds > sys.getrecursionlimit()  # each add is one level deeper
    f = x[0]
    for i in range(1, n_adds + 1):
        f = f + p * sp.sin(x[i % 3])
    fn = sp.compile(f)
    u = np.array([0.3, -0.7, 1.1])
    x.value = u
    counts = np.bincount(np.arange(1, n_adds + 1) % 3, minlength=3)
    for pv in (2.0, 3.0):
        p.value = pv
        np.testing.assert_allclose(fn.forward(), [[u[0] + pv * counts @ np.sin(u)]])
        np.testing.assert_allclose(fn.jacobian().toarray(),
                                   [np.eye(3)[0] + pv * counts * np.cos(u)])
        np.testing.assert_allclose(fn.hessian(1.0).toarray(),
                                   np.diag(-pv * counts * np.sin(u)), atol=1e-9)


def test_compile_deep_nonlinear_chain(scope):
    """A deep chain of nonlinear atoms that all reuse one shared subexpression.

    The shared node is shallow on purpose: the engine re-evaluates a shared
    subexpression once per use, so reusing a deep one would take exponential time.
    """
    x = scope.Variable(1, 1)
    n_levels, c = 3000, 1e-3
    shared = sp.sin(x)
    g = x
    for _ in range(n_levels):
        g = sp.sin(g) + c * shared

    x0 = 0.7
    v, d1, d2 = x0, 1.0, 0.0  # g, dg/dx, d2g/dx2 by the chain rule
    for _ in range(n_levels):
        d2 = np.cos(v) * d2 - np.sin(v) * d1 ** 2 - c * np.sin(x0)
        d1 = np.cos(v) * d1 + c * np.cos(x0)
        v = np.sin(v) + c * np.sin(x0)

    x.value = x0
    fn = sp.compile(g)
    np.testing.assert_allclose(fn.forward(), [[v]], rtol=1e-12)
    np.testing.assert_allclose(fn.jacobian().toarray(), [[d1]], rtol=1e-10)
    np.testing.assert_allclose(fn.hessian(1.0).toarray(), [[d2]], rtol=1e-8)

    prob = sp.Problem(sp.sum(g), [g])
    u = np.array([x0])
    assert prob.objective_forward(u) == pytest.approx(v, rel=1e-12)
    np.testing.assert_allclose(prob.gradient(), [d1], rtol=1e-10)
    np.testing.assert_allclose(prob.constraint_forward(u), [v], rtol=1e-12)
    np.testing.assert_allclose(prob.jacobian().toarray(), [[d1]], rtol=1e-10)
    np.testing.assert_allclose(prob.hessian(1.0, [2.0]).toarray(), [[3 * d2]],
                               rtol=1e-8)
