"""compact() on benchmark NLPs must match the hand-vectorized formulation."""

import numpy as np
import pytest

import sparsediffpy as sp
from conftest import count_nodes
from sparsediffpy.expression import Constant

TOL = dict(rtol=1e-10, atol=1e-10)


def luksan(n):
    def vec(x):
        x0, x1 = x[0:n - 1], x[1:n]
        f = sp.sum(100.0 * (x0 ** 2 - x1) ** 2 + (x0 - 1.0) ** 2)
        a, b, c = x[0:n - 2], x[1:n - 1], x[2:n]
        g = (3.0 * b ** 3 + 2.0 * c - 5.0 + sp.sin(b - c) * sp.sin(b + c)
             + 4.0 * b - a * sp.exp(a - b) - 3.0)
        return f, g

    def scalar(x):
        xs = [x[i] for i in range(n)]
        f = [100.0 * (xs[i] ** 2 - xs[i + 1]) ** 2 + (xs[i] - 1.0) ** 2
             for i in range(n - 1)]
        g = [3.0 * xs[k + 1] ** 3 + 2.0 * xs[k + 2] - 5.0
             + sp.sin(xs[k + 1] - xs[k + 2]) * sp.sin(xs[k + 1] + xs[k + 2])
             + 4.0 * xs[k + 1] - xs[k] * sp.exp(xs[k] - xs[k + 1]) - 3.0
             for k in range(n - 2)]
        return f, g

    return n, vec, scalar


def _forcing(N):
    X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N), indexing="ij")
    return 5.0 * (((X - 0.3) ** 2 + (Y - 0.6) ** 2) <= 0.04).ravel()


def brusselator(N, forcing=True, interleaved=False):
    M, alpha = N * N, 10.0 * (N - 1) ** 2 / 1e4
    F = _forcing(N) if forcing else np.zeros(M)
    idx = np.arange(M).reshape(N, N)
    nbrs = [np.roll(idx, s, axis=a).ravel() for s in (-1, 1) for a in (0, 1)]

    def vec(x):
        u, v = x[0:M], x[M:2 * M]

        def lap(w):
            return sum(w[i] for i in nbrs) - 4.0 * w

        u2v = u ** 2 * v
        fu = 1.0 + u2v - 4.4 * u + alpha * lap(u) + F.reshape(-1, 1)
        fv = 3.4 * u - u2v + alpha * lap(v)
        return None, sp.vstack([fu, fv])

    def scalar(x):
        u, v = [x[i] for i in range(M)], [x[M + i] for i in range(M)]

        def lap(w, k):
            return sum(w[i[k]] for i in nbrs) - 4.0 * w[k]

        fu = [1.0 + u[k] ** 2 * v[k] - 4.4 * u[k] + alpha * lap(u, k) + F[k]
              for k in range(M)]
        fv = [3.4 * u[k] - u[k] ** 2 * v[k] + alpha * lap(v, k)
              for k in range(M)]
        if interleaved:
            return None, [t for pair in zip(fu, fv) for t in pair]
        return None, fu + fv

    return 2 * M, vec, scalar


def logreg(n):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((2 * n, n)) / np.sqrt(n)
    c = A.T @ (rng.random(2 * n) < 0.5)
    mu = 1e-2

    def vec(x):
        f = (sp.sum(sp.logistic(Constant(A) @ x)) - Constant(c[None, :]) @ x
             + 0.5 * mu * sp.sum(x ** 2))
        return f, x[0:n - 1] ** 2 + x[1:n] ** 2

    def scalar(x):
        xs = [x[i] for i in range(n)]
        f = [sp.logistic(Constant(a[None, :]) @ x) for a in A]
        f += [-c[j] * xs[j] + 0.5 * mu * xs[j] ** 2 for j in range(n)]
        return f, [xs[i] ** 2 + xs[i + 1] ** 2 for i in range(n - 1)]

    return n, vec, scalar


def vdp(N):
    h = 10.0 / N

    def rk4(a, b, u):
        def rhs(a, b):
            return (1.0 - b ** 2) * a - b + u, a

        k1 = rhs(a, b)
        k2 = rhs(a + 0.5 * h * k1[0], b + 0.5 * h * k1[1])
        k3 = rhs(a + 0.5 * h * k2[0], b + 0.5 * h * k2[1])
        k4 = rhs(a + h * k3[0], b + h * k3[1])
        return [s + (h / 6.0) * (p + 2.0 * q + 2.0 * r + t)
                for s, p, q, r, t in zip((a, b), k1, k2, k3, k4)]

    def vec(x):
        x1, x2, u = x[0:N + 1], x[N + 1:2 * N + 2], x[2 * N + 2:3 * N + 2]
        na, nb = rk4(x1[0:N], x2[0:N], u)
        return (sp.sum(x1 ** 2) + sp.sum(x2 ** 2) + sp.sum(u ** 2),
                sp.vstack([x1[1:N + 1] - na, x2[1:N + 1] - nb]))

    def scalar(x):
        xs = [x[i] for i in range(3 * N + 2)]
        x1, x2, u = xs[0:N + 1], xs[N + 1:2 * N + 2], xs[2 * N + 2:]
        steps = [rk4(x1[k], x2[k], u[k]) for k in range(N)]
        g = ([x1[k + 1] - s[0] for k, s in enumerate(steps)]
             + [x2[k + 1] - s[1] for k, s in enumerate(steps)])
        return [t ** 2 for t in xs], g

    return 3 * N + 2, vec, scalar


def _build(problem):
    n, vec, scalar = problem
    f, g = vec(sp.Scope().Variable(n, 1))
    ref = sp.Problem(0.0 if f is None else f, [g])
    f_terms, g_terms = scalar(sp.Scope().Variable(n, 1))
    f = None if f_terms is None else sp.compact(f_terms)[0]
    g, perm = sp.compact(g_terms)
    p = sp.Problem(0.0 if f is None else sp.sum(f), [g])
    return n, ref, p, f, g, perm


def _assert_same(n, ref, p, perm):
    rows = slice(None) if perm is None else perm
    for seed in range(3):
        rng = np.random.default_rng(seed)
        u = 0.5 + 0.3 * rng.standard_normal(n)
        lam = rng.standard_normal(ref.total_constraint_size)
        for a, b in [
                (p.objective_forward(u), ref.objective_forward(u)),
                (p.gradient(), ref.gradient()),
                (p.constraint_forward(u), ref.constraint_forward(u)[rows]),
                (p.jacobian().toarray(), ref.jacobian().toarray()[rows]),
                (p.hessian(1.3, lam[rows]).toarray(),
                 ref.hessian(1.3, lam).toarray())]:
            np.testing.assert_allclose(a, b, **TOL)


@pytest.mark.parametrize("make, size", [
    (luksan, 3), (luksan, 12), (luksan, 50), (brusselator, 4), (logreg, 6),
    (vdp, 1), (vdp, 6), (vdp, 20),
], ids=lambda v: getattr(v, "__name__", str(v)))
def test_compact_matches_vectorized(make, size):
    n, ref, p, f, g, perm = _build(make(size))
    assert perm is None
    _assert_same(n, ref, p, perm)


@pytest.mark.parametrize("make, size", [
    (luksan, 10), (brusselator, 3), (logreg, 6), (vdp, 3),
], ids=lambda v: getattr(v, "__name__", str(v)))
def test_graph_size_does_not_grow(make, size):
    *_, f1, g1, _ = _build(make(size))
    *_, f2, g2, _ = _build(make(2 * size))
    assert count_nodes(g1) == count_nodes(g2)
    if f1 is not None and make is not logreg:
        assert count_nodes(f1) == count_nodes(f2)


def test_interleaved_rows_are_grouped():
    n, ref, p, f, g, perm = _build(brusselator(3, interleaved=True))
    assert list(perm) == list(range(0, 18, 2)) + list(range(1, 18, 2))
    _assert_same(n, ref, p, None)


def test_varying_constant_stays_per_row():
    F = _forcing(4)
    assert 0 < np.count_nonzero(F) < F.size
    u = np.random.default_rng(0).uniform(0.5, 1.5, 32)
    c, c0 = [_build(brusselator(4, forcing=b))[2].constraint_forward(u)
             for b in (True, False)]
    np.testing.assert_allclose((c - c0)[:16], F, **TOL)


def test_compact_shrinks_the_graph():
    n, _, scalar = luksan(200)
    terms = scalar(sp.Scope().Variable(n, 1))[1]
    assert 50 * count_nodes(sp.compact(terms)[0]) < count_nodes(sp.vstack(terms))
