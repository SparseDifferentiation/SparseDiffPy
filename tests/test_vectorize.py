"""compact(terms) must merge what it can and match vstack(terms) exactly."""

import numpy as np
import pytest
import scipy.sparse as sps

import sparsediffpy as sp
from conftest import count_nodes
from sparsediffpy.expression import Constant


def _evaluate(g, u, lam):
    prob = sp.Problem(sp.sum(g), [g])
    f, c = prob.objective_forward(u), prob.constraint_forward(u)
    return (f, prob.gradient().copy(), c.copy(), prob.jacobian().toarray(),
            prob.hessian(0.7, lam).toarray())


def assert_compact(build, n, rng):
    """build(scope, n) -> scalar terms over n + extra variables.

    The compacted graph must not grow with n (the terms were merged), and
    value, gradient, Jacobian and Hessian must match vstack(terms), with rows
    mapped by perm.
    """
    assert count_nodes(sp.compact(build(sp.Scope(), n))[0]) == \
        count_nodes(sp.compact(build(sp.Scope(), 2 * n))[0])

    scope = sp.Scope()
    terms = build(scope, n)
    u = rng.uniform(0.5, 1.5, scope.total_var_size)
    lam = rng.standard_normal(len(terms))
    f, grad, c, J, H = _evaluate(sp.vstack(terms), u, lam)

    g, perm = sp.compact(build(sp.Scope(), n))
    rows = slice(None) if perm is None else perm
    got = _evaluate(g, u, lam[rows])
    expected = (f, grad, c[rows], J[rows], H)
    for name, a, b in zip(["objective", "gradient", "constraints", "jacobian",
                           "hessian"], expected, got):
        np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-12, err_msg=name)
    return perm


def test_loop_merges_in_place(rng):
    def build(scope, n):
        x = scope.Variable(n + 2, 1)
        return [3 * x[k + 1] ** 3 + 2 * x[k + 2] - x[k] * sp.exp(x[k] - x[k + 1])
                for k in range(n)]
    assert assert_compact(build, 6, rng) is None


def test_varying_constants(rng):
    def build(scope, n):
        x = scope.Variable(n, 1)
        return [(k + 1.0) * (x[k] - k) ** 2 for k in range(n)]
    assert_compact(build, 5, rng)


def test_interleaved_groups_and_unsupported_terms(rng):
    A = rng.standard_normal((2, 6))

    def build(scope, n):
        x, z = scope.Variable(n + 1, 1), scope.Variable(6, 1)
        terms = []
        for k in range(n):
            terms += [sp.sin(x[k]) * x[k + 1], x[k] ** 2 - 2.0 * k]
        terms += [sp.exp(Constant(A[:1]) @ z),
                  sp.exp(Constant(sps.csr_matrix(A[1:])) @ z)]
        return terms
    perm = assert_compact(build, 5, rng)
    assert list(perm) == [0, 2, 4, 6, 8, 1, 3, 5, 7, 9, 10, 11]


def test_shared_subexpression_and_parameter(rng):
    def build(scope, n):
        x = scope.Variable(n + 1, 1)
        p = scope.Parameter(1, 1, value=2.5)
        terms = []
        for k in range(n):
            d = x[k] - x[k + 1]
            terms.append(p * d * d + sp.tanh(d))
        return terms
    assert_compact(build, 4, rng)


def test_terms_without_args(rng):
    def build(scope, n):
        y = scope.Variable(1, 1)
        return [y] * n + [sp.exp(y) for _ in range(n)]
    assert_compact(build, 2, rng)


def test_different_structure_does_not_merge(rng):
    scope = sp.Scope()
    x, z = scope.Variable(3, 1), scope.Variable(3, 1)
    terms = [x[0] ** 2, x[1] ** 2, z[0] ** 2, x[2] ** 3, x[0] * x[1], x[1] * x[1]]
    g, perm = sp.compact(terms)
    assert perm is None
    a, b, c, d = rng.uniform(0.5, 1.5, 4)
    u = np.array([a, b, c, d, 0.0, 0.0])
    np.testing.assert_allclose(sp.Problem(sp.sum(g), [g]).constraint_forward(u),
                               [a * a, b * b, d * d, c ** 3, a * b, b * b])


def test_parameter_update_after_compact():
    scope = sp.Scope()
    x = scope.Variable(3, 1)
    p = scope.Parameter(1, 1, value=2.0)
    g, _ = sp.compact([p * x[k] for k in range(3)])
    prob = sp.Problem(sp.sum(g), [g])
    u = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(prob.constraint_forward(u), 2.0 * u)
    p.value = 5.0
    np.testing.assert_allclose(prob.constraint_forward(u), 5.0 * u)


def test_rejects_non_scalar_and_empty():
    x = sp.Scope().Variable(3, 1)
    with pytest.raises(ValueError):
        sp.compact([x[0], x])
    with pytest.raises(ValueError):
        sp.compact([])
