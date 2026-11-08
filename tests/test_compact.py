"""compact(terms) must match vstack(terms) in value and derivatives."""

import numpy as np
import scipy.sparse as sps

import sparsediffpy as sp
from sparsediffpy._expression import Constant


def _evaluate(g, u, lam):
    # sum(g) is unchanged by a row permutation, so the objective needs no mapping
    prob = sp.Problem(sp.sum(g), [g])
    f, c = prob.objective_forward(u), prob.constraint_forward(u)
    return (f, prob.gradient().copy(), c.copy(), prob.jacobian().toarray(),
            prob.hessian(0.7, lam).toarray())


def assert_matches_vstack(build, n_vars, rng):
    """build(scope) -> list of scalar terms; compare compact with vstack."""
    terms = build(sp.Scope())
    u = rng.uniform(0.5, 1.5, n_vars)
    lam = rng.standard_normal(len(terms))
    f, grad, c, J, H = _evaluate(sp.vstack(terms), u, lam)

    g, perm = sp.compact(build(sp.Scope()))
    rows = slice(None) if perm is None else perm
    got = _evaluate(g, u, lam[rows])
    expected = (f, grad, c[rows], J[rows], H)
    for name, a, b in zip(["objective", "gradient", "constraints", "jacobian",
                           "hessian"], expected, got):
        np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-12, err_msg=name)
    return g, perm


def test_loop_merges_into_one_vector(rng):
    def build(scope):
        x = scope.Variable(8, 1)
        return [3 * x[k + 1] ** 3 + 2 * x[k + 2] - x[k] * sp.exp(x[k] - x[k + 1])
                for k in range(6)]
    g, perm = assert_matches_vstack(build, 8, rng)
    assert perm is None and g.op != "vstack"  # a single template, in place


def test_varying_constants(rng):
    def build(scope):
        x = scope.Variable(5, 1)
        return [(k + 1.0) * (x[k] - k) ** 2 for k in range(5)]
    assert_matches_vstack(build, 5, rng)


def test_interleaved_templates_and_unsupported_terms(rng):
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
    _, perm = assert_matches_vstack(build, 6, rng)
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
    assert_matches_vstack(build, 5, rng)


def test_templates_without_params(rng):
    def build(scope):
        y = scope.Variable(1, 1)
        return [y, y, sp.exp(y), sp.exp(y)]
    assert_matches_vstack(build, 1, rng)
