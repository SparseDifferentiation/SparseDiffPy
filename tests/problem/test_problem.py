"""Tests for sp.Problem: objective + list of constraints, derivatives via COO."""

import numpy as np
import pytest
import scipy.sparse as sparse

import sparsediffpy as sp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assemble_jacobian(problem, u, m, n):
    """Evaluate the problem and assemble its COO Jacobian into a dense (m, n)."""
    problem.objective_forward(u)
    problem.constraint_forward(u)
    rows, cols = problem.get_jacobian_sparsity_coo()
    vals = problem.eval_jacobian_vals()
    if m == 0:
        return np.zeros((0, n))
    return sparse.coo_matrix((vals, (rows, cols)), shape=(m, n)).toarray()


def _assemble_hessian(problem, u, obj_factor, lagrange, n):
    """Evaluate the problem and assemble its lower-triangular COO Hessian into
    a dense symmetric (n, n)."""
    problem.objective_forward(u)
    problem.constraint_forward(u)
    problem.gradient()            # required before hessian (populates obj adjoints)
    problem.eval_jacobian_vals()  # required before hessian (populates constraint adjoints)
    rows, cols = problem.get_problem_hessian_sparsity_coo()
    vals = problem.eval_hessian_vals_coo_lower_tri(obj_factor, lagrange)
    H_lower = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).toarray()
    return H_lower + H_lower.T - np.diag(np.diag(H_lower))


def _numerical_gradient(problem, u, h=1e-6):
    n = u.size
    g = np.zeros(n)
    for j in range(n):
        u_p = u.copy(); u_p[j] += h
        u_m = u.copy(); u_m[j] -= h
        f_p = problem.objective_forward(u_p)
        f_m = problem.objective_forward(u_m)
        g[j] = (f_p - f_m) / (2 * h)
    return g


def _numerical_jacobian(problem, u, m, h=1e-6):
    n = u.size
    J = np.zeros((m, n))
    for j in range(n):
        u_p = u.copy(); u_p[j] += h
        u_m = u.copy(); u_m[j] -= h
        c_p = problem.constraint_forward(u_p)
        c_m = problem.constraint_forward(u_m)
        J[:, j] = (c_p - c_m) / (2 * h)
    return J


def _numerical_lagrangian_hessian(problem, u, obj_factor, lagrange, m, h=1e-5):
    """Central differences on the Lagrangian gradient grad_L = obj_factor*grad_f + J^T lambda."""
    n = u.size

    def lag_grad(u_):
        problem.objective_forward(u_)
        problem.constraint_forward(u_)
        gf = problem.gradient()
        if m > 0:
            rows, cols = problem.get_jacobian_sparsity_coo()
            vals = problem.eval_jacobian_vals()
            J = sparse.coo_matrix((vals, (rows, cols)), shape=(m, n)).toarray()
            return obj_factor * gf + J.T @ lagrange
        return obj_factor * gf

    H = np.zeros((n, n))
    for j in range(n):
        u_p = u.copy(); u_p[j] += h
        u_m = u.copy(); u_m[j] -= h
        H[:, j] = (lag_grad(u_p) - lag_grad(u_m)) / (2 * h)
    return (H + H.T) / 2


# ---------------------------------------------------------------------------
# Objective only, no constraints
# ---------------------------------------------------------------------------

def test_problem_objective_only_objective_forward(scope, rng):
    x = scope.Variable(3, 1)
    obj = sp.sum(sp.power(x, 2))
    problem = sp.Problem(obj, [])
    u = rng.standard_normal(3)
    np.testing.assert_allclose(problem.objective_forward(u), float(np.sum(u ** 2)))


def test_problem_objective_only_gradient(scope, rng):
    x = scope.Variable(4, 1)
    obj = sp.sum(sp.power(x, 2)) + sp.sum(sp.sin(x))
    problem = sp.Problem(obj, [])
    problem.init_jacobian_coo()
    problem.init_hessian_coo_lower_tri()
    u = rng.uniform(-0.5, 0.5, size=4)
    problem.objective_forward(u)
    np.testing.assert_allclose(problem.gradient(),
                               _numerical_gradient(problem, u), rtol=1e-5, atol=1e-6)


def test_problem_objective_only_hessian(scope, rng):
    x = scope.Variable(3, 1)
    obj = sp.sum(sp.power(x, 2)) + sp.sum(sp.sin(x))
    problem = sp.Problem(obj, [])
    problem.init_jacobian_coo()
    problem.init_hessian_coo_lower_tri()

    u = rng.uniform(-0.5, 0.5, size=3)
    H = _assemble_hessian(problem, u, obj_factor=1.0, lagrange=np.zeros(0), n=3)
    H_num = _numerical_lagrangian_hessian(problem, u, 1.0, np.zeros(0), m=0)
    np.testing.assert_allclose(H, H_num, rtol=1e-4, atol=1e-6)


def test_problem_zero_constraint_size(scope):
    x = scope.Variable(2, 1)
    obj = sp.sum(sp.power(x, 2))
    problem = sp.Problem(obj, [])
    assert problem._total_constraint_size == 0
    u = np.array([1.0, 2.0])
    c = problem.constraint_forward(u)
    assert c.shape == (0,)


# ---------------------------------------------------------------------------
# Objective + single vector constraint
# ---------------------------------------------------------------------------

def test_problem_constraint_forward_and_jacobian(scope, rng):
    x = scope.Variable(3, 1)
    A = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
    b = np.array([[0.1], [-0.2]])
    obj = sp.sum(sp.exp(x))
    c = A @ x + b + sp.sin(x[:2])
    problem = sp.Problem(obj, [c])
    problem.init_jacobian_coo()

    u = rng.uniform(-0.3, 0.3, size=3)

    # constraint_forward value
    c_val = problem.constraint_forward(u)
    expected = A @ u + b.ravel() + np.sin(u[:2])
    np.testing.assert_allclose(c_val, expected, rtol=1e-10)

    # Jacobian (assembled from COO) vs numerical
    J_analytic = _assemble_jacobian(problem, u, m=2, n=3)
    J_numeric = _numerical_jacobian(problem, u, m=2)
    np.testing.assert_allclose(J_analytic, J_numeric, rtol=1e-5, atol=1e-6)


def test_problem_lagrangian_hessian(scope, rng):
    x = scope.Variable(3, 1)
    obj = sp.sum(sp.power(x, 2)) + sp.sum(sp.sin(x))
    c1 = sp.exp(x) + x       # (3, 1)
    c2 = sp.sum(sp.power(x, 3))  # (1, 1)
    problem = sp.Problem(obj, [c1, c2])
    problem.init_jacobian_coo()
    problem.init_hessian_coo_lower_tri()

    u = rng.uniform(-0.3, 0.3, size=3)
    lagrange = rng.standard_normal(4)  # 3 + 1 = 4 constraint rows
    obj_factor = 0.7

    H = _assemble_hessian(problem, u, obj_factor, lagrange, n=3)
    H_num = _numerical_lagrangian_hessian(problem, u, obj_factor, lagrange, m=4)
    np.testing.assert_allclose(H, H_num, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# Shared subexpression across obj + constraint
# ---------------------------------------------------------------------------

def test_problem_shared_subexpression(scope, rng):
    """`t = sp.sin(x)` reused in both obj and constraint. The Python node is
    shared but per-root atom caches ensure each root gets its own capsule, which
    the C engine requires for correct reverse-mode accumulation."""
    x = scope.Variable(3, 1)
    t = sp.sin(x)               # shared Python node
    obj = sp.sum(sp.power(t, 2))
    c = t + x                   # also uses t
    problem = sp.Problem(obj, [c])
    problem.init_jacobian_coo()
    problem.init_hessian_coo_lower_tri()

    u = rng.uniform(-0.5, 0.5, size=3)

    np.testing.assert_allclose(problem.objective_forward(u),
                               float(np.sum(np.sin(u) ** 2)), rtol=1e-10)
    np.testing.assert_allclose(problem.constraint_forward(u),
                               np.sin(u) + u, rtol=1e-10)

    J_analytic = _assemble_jacobian(problem, u, m=3, n=3)
    np.testing.assert_allclose(J_analytic, _numerical_jacobian(problem, u, m=3),
                               rtol=1e-5, atol=1e-6)

    lagrange = rng.standard_normal(3)
    H = _assemble_hessian(problem, u, 1.0, lagrange, n=3)
    H_num = _numerical_lagrangian_hessian(problem, u, 1.0, lagrange, m=3)
    np.testing.assert_allclose(H, H_num, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# Parameter update flow
# ---------------------------------------------------------------------------

def test_problem_parameter_update(scope, rng):
    x = scope.Variable(3, 1)
    p = scope.Parameter(3, 1)
    p.value = np.array([1.0, 2.0, 3.0])
    obj = sp.sum(sp.power(x - p, 2))
    problem = sp.Problem(obj, [])

    u = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(problem.objective_forward(u), 1.0 + 4.0 + 9.0)

    new_theta = np.array([4.0, 5.0, 6.0])
    problem.update_params(new_theta)
    np.testing.assert_allclose(problem.objective_forward(u), 16.0 + 25.0 + 36.0)


def test_problem_parameter_initial_value_required(scope):
    x = scope.Variable(3, 1)
    p = scope.Parameter(3, 1)  # value never set
    obj = sp.sum(sp.power(x - p, 2))
    with pytest.raises(ValueError, match="has no value set"):
        sp.Problem(obj, [])


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_problem_non_scalar_objective(scope):
    x = scope.Variable(3, 1)
    with pytest.raises(ValueError, match="scalar"):
        sp.Problem(x, [])


def test_problem_cross_scope_variables():
    s1 = sp.Scope()
    s2 = sp.Scope()
    x = s1.Variable(2, 1)
    y = s2.Variable(2, 1)
    obj = sp.sum(sp.power(x, 2)) + sp.sum(sp.power(y, 2))
    with pytest.raises(ValueError, match="same Scope"):
        sp.Problem(obj, [])


def test_problem_wrong_length_lagrange(scope):
    x = scope.Variable(2, 1)
    obj = sp.sum(sp.power(x, 2))
    c = sp.exp(x)  # 2 constraints
    problem = sp.Problem(obj, [c])
    problem.init_jacobian_coo()
    problem.init_hessian_coo_lower_tri()
    problem.objective_forward(np.zeros(2))
    problem.constraint_forward(np.zeros(2))
    problem.eval_jacobian_vals()

    with pytest.raises(ValueError, match="lagrange length"):
        problem.eval_hessian_vals_coo_lower_tri(1.0, np.zeros(5))


def test_problem_no_variables():
    """A constant-only objective has no Variables — must raise."""
    from sparsediffpy._core._constants import Constant
    obj = Constant(np.array([[5.0]]), shape=(1, 1))
    with pytest.raises(ValueError, match="at least one Variable"):
        sp.Problem(obj, [])
