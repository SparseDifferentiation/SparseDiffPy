"""Misuse of the raw problem bindings raises instead of crashing the process."""

import numpy as np
import pytest

from sparsediffpy import _sparsediffengine as C

N = 3


def make_problem(param=None):
    """minimize sum(exp(x)) subject to exp(x), optionally with x scaled by param."""
    x = C.make_variable(N, 1, 0, N)
    if param is not None:
        x = C.make_param_scalar_mult(param, x)
    con = C.make_exp(C.make_variable(N, 1, 0, N))
    return C.make_problem(C.make_sum(C.make_exp(x), -1), [con], False)


def forwarded(prob):
    u = np.ones(N)
    C.problem_objective_forward(prob, u)
    C.problem_constraint_forward(prob, u)
    return prob


@pytest.mark.parametrize("call", [
    lambda p: C.problem_gradient(p),
    lambda p: C.problem_jacobian(p),
    lambda p: C.problem_eval_jacobian_vals(p),
    lambda p: C.problem_hessian(p, 1.0, np.ones(N)),
], ids=["gradient", "jacobian", "eval_jacobian_vals", "hessian"])
def test_derivatives_before_init(call):
    with pytest.raises(RuntimeError, match="problem_init_derivatives"):
        call(forwarded(make_problem()))


def test_hessian_coo_values_before_coo_init():
    prob = forwarded(make_problem())
    C.problem_init_derivatives(prob)
    with pytest.raises(RuntimeError, match="coo"):
        C.problem_eval_hessian_vals_coo(prob, 1.0, np.ones(N))


@pytest.mark.parametrize("call", [
    lambda p: C.problem_objective_forward(p, np.ones(N - 1)),
    lambda p: C.problem_constraint_forward(p, np.ones(N + 1)),
    lambda p: C.problem_hessian(p, 1.0, np.ones(1)),
], ids=["objective_u", "constraint_u", "lagrange"])
def test_wrong_lengths(call):
    prob = forwarded(make_problem())
    C.problem_init_derivatives(prob)
    C.problem_gradient(prob)
    C.problem_jacobian(prob)
    with pytest.raises(ValueError, match="must have"):
        call(prob)


def test_update_params_without_params():
    with pytest.raises(ValueError, match="no parameters"):
        C.problem_update_params(make_problem(), np.ones(1))


def test_short_theta():
    param = C.make_parameter(1, 1, 2, N, np.ones(1))  # reads theta[2]
    prob = make_problem(param)
    C.problem_register_params(prob, [param])
    with pytest.raises(ValueError, match="at least 3"):
        C.problem_update_params(prob, np.ones(2))
    C.problem_update_params(prob, np.full(3, 2.0))
    assert C.problem_objective_forward(prob, np.ones(N)) == pytest.approx(
        N * np.exp(2.0))


def test_register_constant_as_parameter():
    const = C.make_parameter(1, 1, -1, N, np.ones(1))
    with pytest.raises(ValueError, match="constant"):
        C.problem_register_params(make_problem(), [const])


def test_correct_sequence_still_works():
    prob = make_problem()
    C.problem_init_derivatives(prob)
    forwarded(prob)
    np.testing.assert_allclose(C.problem_gradient(prob), np.e * np.ones(N))
    data, _, _, shape = C.problem_jacobian(prob)
    assert shape == (N, N)
    np.testing.assert_allclose(data, np.e * np.ones(N))
    data, _, _, _ = C.problem_hessian(prob, 1.0, np.ones(N))
    np.testing.assert_allclose(data, 2 * np.e * np.ones(N))
