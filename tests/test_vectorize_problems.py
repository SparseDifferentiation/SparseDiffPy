"""compact() on the benchmark NLPs (Luksan-Vlcek, Brusselator, logistic
regression, Van der Pol RK4 shooting).

Each problem is written twice with identical math, as in the paper's
experiments: ``vec`` (slices, gathers, matmul) and ``scalar`` (a Python loop
producing one scalar term per row, JuMP/Pyomo style). For the scalar form we
build the objective and constraints three ways and require the NLP oracles
f, grad f, g, J_g and the Lagrangian Hessian to agree:

  * vec      - the hand-vectorized formulation (independent reference)
  * vstack   - scalar terms stacked as they are
  * compact  - scalar terms merged by sp.compact, rows mapped back via perm

We also pin down how many groups compact finds, so a regression that stops
merging (and silently falls back to one node per term) is caught.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import sparsediffpy as sp
from sparsediffpy.expression import Constant

TOL = dict(rtol=1e-10, atol=1e-10)


# ------------------------------------------------------------------ problems


def luksan_data(n):
    return {"n": n}


def luksan_vec(x, d, ops):
    n = d["n"]
    x0, x1 = x[0:n - 1], x[1:n]
    f = ops.sum(100.0 * (x0 ** 2 - x1) ** 2 + (x0 - 1.0) ** 2)
    a, b, c = x[0:n - 2], x[1:n - 1], x[2:n]
    g = (3.0 * b ** 3 + 2.0 * c - 5.0 + ops.sin(b - c) * ops.sin(b + c)
         + 4.0 * b - a * ops.exp(a - b) - 3.0)
    return f, [g]


def luksan_scalar(x, d, ops):
    n = d["n"]
    xs = [x[i] for i in range(n)]
    f = [100.0 * (xs[i] ** 2 - xs[i + 1]) ** 2 + (xs[i] - 1.0) ** 2
         for i in range(n - 1)]
    g = [3.0 * xs[k + 1] ** 3 + 2.0 * xs[k + 2] - 5.0
         + ops.sin(xs[k + 1] - xs[k + 2]) * ops.sin(xs[k + 1] + xs[k + 2])
         + 4.0 * xs[k + 1] - xs[k] * ops.exp(xs[k] - xs[k + 1]) - 3.0
         for k in range(n - 2)]
    return f, g


def brusselator_data(N):
    idx = np.arange(N * N).reshape(N, N)
    xs = np.linspace(0, 1, N)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    # a forcing that is nonzero on some cells only: a const that varies by lane
    forcing = (((X - 0.3) ** 2 + (Y - 0.6) ** 2) <= 0.2 ** 2).astype(float) * 5.0
    return {"N": N, "n": 2 * N * N, "alpha": 10.0 * (N - 1) ** 2 / 1e4,
            "forcing": forcing.ravel(),
            "ip": np.roll(idx, -1, axis=0).ravel(),
            "im": np.roll(idx, 1, axis=0).ravel(),
            "jp": np.roll(idx, -1, axis=1).ravel(),
            "jm": np.roll(idx, 1, axis=1).ravel()}


def brusselator_vec(x, d, ops):
    M = d["N"] ** 2
    u, v = x[0:M], x[M:2 * M]

    def lap(w):
        return (w[d["ip"]] + w[d["im"]] + w[d["jp"]] + w[d["jm"]] - 4.0 * w)

    u2v = u ** 2 * v
    fu = (1.0 + u2v - 4.4 * u + d["alpha"] * lap(u)
          + d["forcing"].reshape(-1, 1))
    fv = 3.4 * u - u2v + d["alpha"] * lap(v)
    return None, [sp.vstack([fu, fv])]


def _brusselator_rows(x, d):
    M = d["N"] ** 2
    xs = [x[i] for i in range(2 * M)]
    u, v = xs[:M], xs[M:]
    a, F = d["alpha"], d["forcing"]
    ip, im, jp, jm = d["ip"], d["im"], d["jp"], d["jm"]
    fu = [1.0 + u[k] ** 2 * v[k] - 4.4 * u[k]
          + a * (u[ip[k]] + u[im[k]] + u[jp[k]] + u[jm[k]] - 4.0 * u[k]) + F[k]
          for k in range(M)]
    fv = [3.4 * u[k] - u[k] ** 2 * v[k]
          + a * (v[ip[k]] + v[im[k]] + v[jp[k]] + v[jm[k]] - 4.0 * v[k])
          for k in range(M)]
    return fu, fv


def brusselator_scalar(x, d, ops):
    fu, fv = _brusselator_rows(x, d)
    return None, fu + fv


def brusselator_interleaved(x, d, ops):
    # same rows, written per grid cell: (fu_k, fv_k) for each k
    fu, fv = _brusselator_rows(x, d)
    return None, [t for pair in zip(fu, fv) for t in pair]


def logreg_data(n, seed=0):
    rng = np.random.default_rng(seed)
    m = 2 * n
    A = rng.standard_normal((m, n)) / np.sqrt(n)
    y = (rng.random(m) < 0.5).astype(float)
    return {"n": n, "m": m, "A": A, "c": A.T @ y, "mu": 1e-2}


def logreg_vec(x, d, ops):
    n = d["n"]
    f = (ops.sum(ops.softplus(Constant(d["A"]) @ x))
         - Constant(d["c"].reshape(1, -1)) @ x
         + 0.5 * d["mu"] * ops.sum(x ** 2))
    return f, [x[0:n - 1] ** 2 + x[1:n] ** 2]


def logreg_scalar(x, d, ops):
    # the row dot products are matmuls: compact keeps those terms as they are
    n, m, A = d["n"], d["m"], d["A"]
    xs = [x[i] for i in range(n)]
    f = [ops.softplus(Constant(A[i].reshape(1, -1)) @ x) for i in range(m)]
    f += [-d["c"][j] * xs[j] + 0.5 * d["mu"] * xs[j] ** 2 for j in range(n)]
    return f, [xs[i] ** 2 + xs[i + 1] ** 2 for i in range(n - 1)]


def vdp_data(N):
    return {"N": N, "n": 3 * N + 2, "h": 10.0 / N}


def _vdp_rhs(a, b, u):
    return (1.0 - b ** 2) * a - b + u, a


def _vdp_step(a, b, u, h):
    k1a, k1b = _vdp_rhs(a, b, u)
    k2a, k2b = _vdp_rhs(a + 0.5 * h * k1a, b + 0.5 * h * k1b, u)
    k3a, k3b = _vdp_rhs(a + 0.5 * h * k2a, b + 0.5 * h * k2b, u)
    k4a, k4b = _vdp_rhs(a + h * k3a, b + h * k3b, u)
    return (a + (h / 6.0) * (k1a + 2.0 * k2a + 2.0 * k3a + k4a),
            b + (h / 6.0) * (k1b + 2.0 * k2b + 2.0 * k3b + k4b))


def vdp_vec(x, d, ops):
    N, h = d["N"], d["h"]
    x1, x2, u = x[0:N + 1], x[N + 1:2 * N + 2], x[2 * N + 2:3 * N + 2]
    f = ops.sum(x1 ** 2) + ops.sum(x2 ** 2) + ops.sum(u ** 2)
    na, nb = _vdp_step(x1[0:N], x2[0:N], u, h)
    return f, [sp.vstack([x1[1:N + 1] - na, x2[1:N + 1] - nb])]


def vdp_scalar(x, d, ops):
    # deep terms with heavy sharing: x1[k], x2[k], u[k] and the RK stages
    # are reused many times inside each defect
    N, h = d["N"], d["h"]
    xs = [x[i] for i in range(3 * N + 2)]
    x1, x2, u = xs[0:N + 1], xs[N + 1:2 * N + 2], xs[2 * N + 2:]
    g1, g2 = [], []
    for k in range(N):
        na, nb = _vdp_step(x1[k], x2[k], u[k], h)
        g1.append(x1[k + 1] - na)
        g2.append(x2[k + 1] - nb)
    return [t ** 2 for t in xs], g1 + g2


OPS = SimpleNamespace(sum=sp.sum, sin=sp.sin, exp=sp.exp, softplus=sp.logistic)

# name: (data, vec, scalar, size, objective groups, constraint groups)
PROBLEMS = {
    "luksan": (luksan_data, luksan_vec, luksan_scalar, 12, 1, 1),
    "brusselator": (brusselator_data, brusselator_vec, brusselator_scalar,
                    4, None, 2),
    "logreg": (logreg_data, logreg_vec, logreg_scalar, 6, 2 * 6 + 1, 1),
    "vdp": (vdp_data, vdp_vec, vdp_scalar, 6, 1, 2),
}


# ------------------------------------------------------------------ harness


def _n_groups(g):
    return len(g.args) if g.opcode == "vstack" else 1


def _oracles(f, g, u, sigma, lam):
    prob = sp.Problem(0.0 if f is None else f, [g])
    fv = prob.objective_forward(u)
    c = np.asarray(prob.constraint_forward(u)).ravel().copy()
    grad = np.asarray(prob.gradient()).ravel().copy()
    J = prob.jacobian().toarray()
    H = prob.hessian(sigma, lam).toarray()
    return float(fv), grad, c, J, H


def _build(form, d, compact):
    """Objective and stacked constraints; perm maps compact rows to terms."""
    x = sp.Scope().Variable(d["n"], 1)
    f, g = form(x, d, OPS)
    if isinstance(f, list):  # scalar objective terms: sum of a stack
        f = sp.sum(sp.compact(f)[0] if compact else sp.vstack(f))
    if not compact:
        return f, sp.vstack(g) if len(g) > 1 else g[0], None
    gc, perm = sp.compact(g)
    return f, gc, perm


def _point(d, m, seed=1):
    rng = np.random.default_rng(seed)
    return 0.5 + 0.3 * rng.standard_normal(d["n"]), 1.3, rng.standard_normal(m)


def _assert_same(got, expected, perm=None):
    rows = slice(None) if perm is None else perm
    exp_f, exp_grad, exp_c, exp_J, exp_H = expected
    for name, a, b in zip(
            ["objective", "gradient", "constraints", "jacobian", "hessian"],
            got, (exp_f, exp_grad, exp_c[rows], exp_J[rows], exp_H)):
        np.testing.assert_allclose(a, b, err_msg=name, **TOL)


# ------------------------------------------------------------------ tests


@pytest.mark.parametrize("name", PROBLEMS)
def test_compact_matches_vstack_and_vec(name):
    data, vec, scalar, size, f_groups, g_groups = PROBLEMS[name]
    d = data(size)

    f_vec, g_vec, _ = _build(vec, d, compact=False)
    f_vs, g_vs, _ = _build(scalar, d, compact=False)
    f_c, g_c, perm = _build(scalar, d, compact=True)
    m = g_vec.shape[0]
    assert g_vs.shape == g_c.shape == (m, 1)
    if perm is not None:
        assert sorted(perm) == list(range(m))

    u, sigma, lam = _point(d, m)
    ref = _oracles(f_vec, g_vec, u, sigma, lam)
    # the scalar form is the same math as the vectorized one ...
    _assert_same(_oracles(f_vs, g_vs, u, sigma, lam), ref)
    # ... and compact keeps it, with multipliers given in compact's row order
    lam_c = lam if perm is None else lam[perm]
    _assert_same(_oracles(f_c, g_c, u, sigma, lam_c), ref, perm)

    assert _n_groups(g_c) == g_groups
    if f_groups is not None:
        assert _n_groups(f_c.args[0]) == f_groups


@pytest.mark.parametrize("name", PROBLEMS)
def test_compact_at_several_points(name):
    # forward/derivative caches must not leak between evaluations
    data, vec, scalar, size, _, _ = PROBLEMS[name]
    d = data(size)
    f_vec, g_vec, _ = _build(vec, d, compact=False)
    f_c, g_c, perm = _build(scalar, d, compact=True)
    p_vec = sp.Problem(0.0 if f_vec is None else f_vec, [g_vec])
    p_c = sp.Problem(0.0 if f_c is None else f_c, [g_c])
    rows = slice(None) if perm is None else perm
    for seed in range(4):
        u, sigma, lam = _point(d, g_vec.shape[0], seed)
        np.testing.assert_allclose(p_c.objective_forward(u),
                                   p_vec.objective_forward(u), **TOL)
        np.testing.assert_allclose(p_c.gradient(), p_vec.gradient(), **TOL)
        np.testing.assert_allclose(
            np.asarray(p_c.constraint_forward(u)).ravel(),
            np.asarray(p_vec.constraint_forward(u)).ravel()[rows], **TOL)
        np.testing.assert_allclose(p_c.jacobian().toarray(),
                                   p_vec.jacobian().toarray()[rows], **TOL)
        np.testing.assert_allclose(p_c.hessian(sigma, lam[rows]).toarray(),
                                   p_vec.hessian(sigma, lam).toarray(), **TOL)


def test_brusselator_interleaved_rows_get_permuted():
    d = brusselator_data(3)
    M = 9
    x = sp.Scope().Variable(d["n"], 1)
    _, terms = brusselator_interleaved(x, d, OPS)
    g_c, perm = sp.compact(terms)
    assert _n_groups(g_c) == 2
    # all fu rows (even positions) first, then all fv rows
    assert list(perm) == list(range(0, 2 * M, 2)) + list(range(1, 2 * M, 2))

    _, g_vec, _ = _build(brusselator_vec, d, compact=False)
    u, sigma, lam = _point(d, 2 * M)
    # interleaved row r is vec row (r // 2) + M * (r % 2)
    to_vec = np.array([r // 2 + M * (r % 2) for r in range(2 * M)])
    ref = _oracles(None, g_vec, u, sigma, lam)
    got = _oracles(None, g_c, u, sigma, lam[to_vec[perm]])
    _assert_same(got, ref, to_vec[perm])


def test_brusselator_varying_forcing_is_a_vector_constant():
    # forcing is 0 on most cells and 5 on some: the merged const must keep
    # per-lane values rather than collapse to the first lane's
    d = brusselator_data(4)
    assert 0 < np.count_nonzero(d["forcing"]) < d["forcing"].size
    d0 = dict(d, forcing=np.zeros_like(d["forcing"]))
    u, _, _ = _point(d, 2 * 16)
    _, g, _ = _build(brusselator_scalar, d, compact=True)
    _, g0, _ = _build(brusselator_scalar, d0, compact=True)
    c = sp.Problem(0.0, [g]).constraint_forward(u)
    c0 = sp.Problem(0.0, [g0]).constraint_forward(u)
    np.testing.assert_allclose(np.asarray(c - c0).ravel()[:16], d["forcing"],
                               **TOL)


@pytest.mark.parametrize("N", [1, 2, 20])
def test_vdp_horizon_lengths(N):
    # a one-stage horizon (one lane per group) up to a longer one
    d = vdp_data(N)
    f_vec, g_vec, _ = _build(vdp_vec, d, compact=False)
    f_c, g_c, perm = _build(vdp_scalar, d, compact=True)
    assert perm is None
    u, sigma, lam = _point(d, 2 * N)
    _assert_same(_oracles(f_c, g_c, u, sigma, lam),
                 _oracles(f_vec, g_vec, u, sigma, lam))


@pytest.mark.parametrize("n", [3, 4, 50])
def test_luksan_sizes(n):
    # n = 3 leaves a single constraint term: nothing to merge it with
    d = luksan_data(n)
    f_vec, g_vec, _ = _build(luksan_vec, d, compact=False)
    f_c, g_c, perm = _build(luksan_scalar, d, compact=True)
    assert perm is None
    u, sigma, lam = _point(d, n - 2)
    _assert_same(_oracles(f_c, g_c, u, sigma, lam),
                 _oracles(f_vec, g_vec, u, sigma, lam))


def test_compact_shrinks_the_graph():
    # the point of compact: far fewer nodes than one subtree per term
    def count(e, seen=None):
        seen = set() if seen is None else seen
        if id(e) in seen:
            return 0
        seen.add(id(e))
        return 1 + sum(count(a, seen) for a in getattr(e, "args", ()))

    d = luksan_data(200)
    _, g_vs, _ = _build(luksan_scalar, d, compact=False)
    _, g_c, _ = _build(luksan_scalar, d, compact=True)
    assert count(g_c) * 50 < count(g_vs)
