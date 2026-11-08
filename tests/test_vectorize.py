"""compact(terms) must merge what it can and match vstack(terms) exactly."""

import numpy as np
import pytest
import scipy.sparse as sps

import sparsediffpy as sp
from sparsediffpy.expression import Constant


def _evaluate(g, u, lam):
    # sum(g) is unchanged by a row permutation, so the objective needs no mapping
    prob = sp.Problem(sp.sum(g), [g])
    f, c = prob.objective_forward(u), prob.constraint_forward(u)
    return (f, prob.gradient().copy(), c.copy(), prob.jacobian().toarray(),
            prob.hessian(0.7, lam).toarray())


def _n_groups(g):
    # compact returns one part per group, stacked only if there are several
    return len(g.args) if g.opcode == "vstack" else 1


def assert_compact(build, n_vars, rng, groups):
    """build(scope) -> scalar terms. Check the group count, then compare value,
    gradient, Jacobian and Hessian with vstack(terms), mapping rows by perm."""
    terms = build(sp.Scope())
    u = rng.uniform(0.5, 1.5, n_vars)
    lam = rng.standard_normal(len(terms))
    f, grad, c, J, H = _evaluate(sp.vstack(terms), u, lam)

    g, perm = sp.compact(build(sp.Scope()))
    assert _n_groups(g) == groups
    rows = slice(None) if perm is None else perm
    got = _evaluate(g, u, lam[rows])
    expected = (f, grad, c[rows], J[rows], H)
    for name, a, b in zip(["objective", "gradient", "constraints", "jacobian",
                           "hessian"], expected, got):
        np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-12, err_msg=name)
    return perm


def test_loop_merges_into_one_vector(rng):
    def build(scope):
        x = scope.Variable(8, 1)
        return [3 * x[k + 1] ** 3 + 2 * x[k + 2] - x[k] * sp.exp(x[k] - x[k + 1])
                for k in range(6)]
    assert assert_compact(build, 8, rng, groups=1) is None  # rows in place


def test_varying_constants(rng):
    def build(scope):
        x = scope.Variable(5, 1)
        return [(k + 1.0) * (x[k] - k) ** 2 for k in range(5)]
    assert_compact(build, 5, rng, groups=1)


def test_interleaved_groups_and_unsupported_terms(rng):
    A = rng.standard_normal((2, 6))

    def build(scope):
        x = scope.Variable(6, 1)
        terms = []
        for k in range(5):
            terms += [sp.sin(x[k]) * x[k + 1], x[k] ** 2 - 2.0 * k]
        # row @ x is not merged: kept as is, but still placed via perm
        terms += [sp.exp(Constant(A[:1]) @ x),
                  sp.exp(Constant(sps.csr_matrix(A[1:])) @ x)]
        return terms
    perm = assert_compact(build, 6, rng, groups=4)
    assert list(perm) == [0, 2, 4, 6, 8, 1, 3, 5, 7, 9, 10, 11]


def test_shared_subexpression_and_parameter(rng):
    def build(scope):
        x = scope.Variable(5, 1)
        p = scope.Parameter(1, 1, value=2.5)
        terms = []
        for k in range(4):
            d = x[k] - x[k + 1]  # shared inside the term
            terms.append(p * d * d + sp.tanh(d))
        return terms
    assert_compact(build, 5, rng, groups=1)


def test_terms_without_args(rng):
    def build(scope):
        y = scope.Variable(1, 1)
        return [y, y, sp.exp(y), sp.exp(y)]
    assert_compact(build, 1, rng, groups=2)


def test_different_structure_does_not_merge(rng):
    # different variable, exponent or sharing pattern: separate groups
    def build(scope):
        x, z = scope.Variable(3, 1), scope.Variable(3, 1)
        return [x[0] ** 2, x[1] ** 2, z[0] ** 2, x[2] ** 3,
                x[0] * x[1], x[1] * x[1]]
    assert_compact(build, 6, rng, groups=4)


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
