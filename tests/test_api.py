"""Scope/values, shape rules, validation errors, parameters, and Problem."""

import numpy as np
import pytest

import sparsediffpy as sp


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


def test_compile_errors(scope):
    with pytest.raises(ValueError, match="at least one Variable"):
        sp.compile(sp.Scope().Parameter(1, 1, value=1.0) * 2.0)
    with pytest.raises(ValueError, match="same Scope"):
        sp.compile(scope.Variable(1, 1) + sp.Scope().Variable(1, 1))
    with pytest.raises(TypeError):
        scope.Variable(1, 1) ** scope.Variable(1, 1)


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
        with pytest.raises(ValueError, match="expected 3 elements"):
            prob.objective_forward(bad)
        with pytest.raises(ValueError, match="expected 3 elements"):
            prob.constraint_forward(bad)
    assert prob.objective_forward(u[:, None]) == pytest.approx(np.exp(s).sum())
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


def test_hessian_without_prior_jacobian(scope):
    """The engine's Hessian reuses cached Jacobian values; hessian() must refresh them."""
    x, y = scope.Variable(2, 1), scope.Variable(2, 1)
    fn = sp.compile(x * y + sp.sin(x * y))
    prob = sp.Problem(sp.sum(x * y), [x * y])
    for u in ([0.1, 0.2, 0.3, 0.4], [1.0, -2.0, 0.5, 3.0]):
        scope.set_values(u)
        xy, xv, yv = np.multiply(u[:2], u[2:]), np.array(u[:2]), np.array(u[2:])
        w = np.array([1.0, 2.0])
        # d2/dx_i dy_i of w_i * (x_i y_i + sin(x_i y_i))
        d2 = lambda a, b: w * (-np.sin(xy) * a * b)  # noqa: E731
        H = np.block([[np.diag(d2(yv, yv)), np.diag(w * (1 + np.cos(xy)) + d2(xv, yv))],
                      [np.diag(w * (1 + np.cos(xy)) + d2(xv, yv)), np.diag(d2(xv, xv))]])
        np.testing.assert_allclose(fn.hessian(w).toarray(), H, atol=1e-12)

        prob.objective_forward(u)
        prob.constraint_forward(u)
        H_prob = np.block([[np.zeros((2, 2)), np.diag(2.0 + w)],
                           [np.diag(2.0 + w), np.zeros((2, 2))]])
        np.testing.assert_allclose(prob.hessian(2.0, w).toarray(), H_prob, atol=1e-12)
