"""Turn expressions into an engine problem and evaluate it.

``compile(f)`` is a thin wrapper around ``Problem``: f becomes the only
constraint of a problem with a zero objective, so the engine's constraint
Jacobian is J_f and its Lagrangian Hessian (objective weight 0) is the Hessian
of w' f.
"""

import numpy as np
import scipy.sparse as sps

from sparsediffpy import _sparsediffengine as _C
from sparsediffpy._expression import Constant, Parameter, Variable, _flatten, _wrap


def _find_scope(exprs):
    """Return the single Scope shared by all Variables/Parameters in exprs."""
    scopes, has_var, seen, stack = set(), False, set(), list(exprs)
    while stack:
        e = stack.pop()
        if id(e) in seen:
            continue
        seen.add(id(e))
        if isinstance(e, (Variable, Parameter)):
            scopes.add(e._scope)
            has_var |= isinstance(e, Variable)
        stack.extend(e.args)
    if not has_var:
        raise ValueError("Expression must contain at least one Variable")
    if len(scopes) > 1:
        raise ValueError("All variables must belong to the same Scope")
    return scopes.pop()


class _Builder:
    """Builds C capsules bottom-up, sharing repeated subexpressions."""

    def __init__(self, scope):
        self.n_vars = scope.total_var_size
        self.cache = {}
        self.params = []  # (Parameter, capsule), each once

    def __call__(self, e):
        if id(e) in self.cache:
            return self.cache[id(e)]
        d1, d2 = e.shape
        if isinstance(e, Variable):
            cap = _C.make_variable(d1, d2, e._offset, self.n_vars)
        elif isinstance(e, Parameter):
            values = e._flat if e._flat is not None else np.zeros(e.size)
            cap = _C.make_parameter(d1, d2, e._offset, self.n_vars, values)
            self.params.append((e, cap))
        elif isinstance(e, Constant):
            values = e.dense().ravel(order="F")
            cap = _C.make_parameter(d1, d2, -1, self.n_vars, values)
        else:
            cap = e.make(*[self(a) for a in e.args])
        if _C.get_expr_dimensions(cap) != e.shape:
            raise ValueError(f"Engine shape {_C.get_expr_dimensions(cap)} != "
                             f"expected {e.shape}")
        self.cache[id(e)] = cap
        return cap


def _csr(triple):
    data, indices, indptr, shape = triple
    return sps.csr_matrix((data, indices, indptr), shape=shape)


class Problem:
    """A scalar objective plus a list of constraint expressions.

    Typical solver loop (u is the flat variable vector)::

        prob.objective_forward(u); prob.constraint_forward(u)
        prob.gradient(); prob.jacobian(); prob.hessian(obj_factor, lagrange)
    """

    def __init__(self, objective, constraints=(), verbose=False):
        objective = _wrap(objective)
        constraints = [_wrap(c) for c in constraints]
        if objective.shape != (1, 1):
            raise ValueError(f"Objective must be scalar, got {objective.shape}")
        self._scope = _find_scope([objective, *constraints])
        build = _Builder(self._scope)
        obj_cap = build(objective)
        con_caps = [build(c) for c in constraints]
        self._capsule = _C.make_problem(obj_cap, con_caps, verbose)
        self._params = [p for p, _ in build.params]
        if build.params:
            _C.problem_register_params(self._capsule, [c for _, c in build.params])
        _C.problem_init_derivatives(self._capsule)
        self._param_version = None
        self.n_vars = self._scope.total_var_size
        self.total_constraint_size = np.sum([c.size for c in constraints]).item()

    def _sync_params(self):
        """Push Parameter values to the engine if any changed since last time."""
        if not self._params or self._param_version == self._scope._param_version:
            return
        theta = np.zeros(self._scope._n_params)
        for p in self._params:
            if p._flat is None:
                raise ValueError(f"Parameter with shape {p.shape} has no value; "
                                 "set parameter.value before evaluating")
            theta[p._offset:p._offset + p.size] = p._flat
        _C.problem_update_params(self._capsule, theta)
        self._param_version = self._scope._param_version

    def _u(self, u):
        """Flat variable vector of length n_vars (the C side does not check it)."""
        if u is None:
            return self._scope._values[:self.n_vars]
        return _flatten(u, self.n_vars)

    def objective_forward(self, u=None):
        """Objective value at u (default: the scope's current values)."""
        self._sync_params()
        return _C.problem_objective_forward(self._capsule, self._u(u))

    def constraint_forward(self, u=None):
        """Stacked constraint values at u (default: the scope's current values)."""
        self._sync_params()
        return _C.problem_constraint_forward(self._capsule, self._u(u))

    def gradient(self):
        """Objective gradient at the point of the last objective_forward."""
        return _C.problem_gradient(self._capsule)

    def jacobian(self):
        """Constraint Jacobian (csr_matrix) at the last constraint_forward point."""
        return _csr(_C.problem_jacobian(self._capsule))

    def hessian(self, obj_factor, lagrange):
        """Lagrangian Hessian obj_factor * H_f + sum_i lagrange_i * H_gi (csr_matrix).

        Call objective_forward and constraint_forward at the point first.
        """
        lagrange = np.asarray(lagrange, dtype=np.float64).ravel(order="F")
        if lagrange.size != self.total_constraint_size:
            raise ValueError(f"lagrange has {lagrange.size} entries, expected "
                             f"{self.total_constraint_size}")
        # The engine's Hessian reuses Jacobian values cached by eval_jacobian,
        # so those must be current at this point.
        _C.problem_gradient(self._capsule)
        _C.problem_jacobian(self._capsule)
        return _csr(_C.problem_hessian(self._capsule, float(obj_factor), lagrange))


class CompiledExpression:
    """A compiled expression f, evaluated at the scope's current values."""

    def __init__(self, expr):
        self.shape = expr.shape
        self._problem = Problem(Constant(0.0), [expr])

    def forward(self):
        """f(x) as an array of shape f.shape."""
        return self._problem.constraint_forward().reshape(self.shape, order="F")

    def jacobian(self):
        """Sparse Jacobian df/dx, shape (f.size, n_vars)."""
        self.forward()
        return self._problem.jacobian()

    def hessian(self, weights):
        """Sparse Hessian of w' f(x), shape (n_vars, n_vars)."""
        self.forward()
        return self._problem.hessian(0.0, weights)


def compile(expr):
    """Analyze the sparsity of ``expr`` once and return a CompiledExpression."""
    return CompiledExpression(_wrap(expr))
