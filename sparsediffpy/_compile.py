"""Turn expressions into an engine problem and evaluate it.

``compile(f)`` is a thin wrapper around ``Problem``: f becomes the only
constraint of a problem with a zero objective, so the engine's constraint
Jacobian is J_f and its Lagrangian Hessian (objective weight 0) is the Hessian
of w' f.
"""

import numpy as np
import scipy.sparse as sps

from sparsediffpy import _sparsediffengine as _C
from sparsediffpy._expression import (
    Constant, Parameter, Variable, _flatten_shaped, _flatten_vector, cast_to_const,
)


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
    """Builds C capsules bottom-up, sharing repeated subexpressions.

    The walk uses an explicit stack rather than recursion, so deep expressions
    (e.g. a long sum built in a Python loop) do not hit Python's recursion
    limit. Nodes are built in the same order as a recursive left-to-right
    post-order traversal.
    """

    def __init__(self, scope):
        self.n_vars = scope.total_var_size
        self.cache = {}
        self.params = []  # (Parameter, capsule), each once

    def __call__(self, root):
        """Return root's capsule, building any children not built yet."""
        stack = [root]
        while stack:
            e = stack[-1]
            if id(e) in self.cache:
                stack.pop()
                continue
            pending = [a for a in e.args if id(a) not in self.cache]
            if pending:
                stack.extend(reversed(pending))  # build the leftmost child first
                continue
            stack.pop()
            self.cache[id(e)] = self._build_one(e)
        return self.cache[id(root)]

    def _build_one(self, e):
        """Build e's capsule; all of e's children must already be cached."""
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
            cap = e.make(*[self.cache[id(a)] for a in e.args])
        if _C.get_expr_dimensions(cap) != e.shape:
            raise ValueError(f"Engine shape {_C.get_expr_dimensions(cap)} != "
                             f"expected {e.shape}")
        return cap


def _csr(triple):
    """scipy CSR matrix from an engine (data, indices, indptr, shape) tuple."""
    data, indices, indptr, shape = triple
    return sps.csr_matrix((data, indices, indptr), shape=shape)


def _read_only(arr):
    """Mark arr read-only and return it, so cached results cannot be mutated."""
    arr.flags.writeable = False
    return arr


class Problem:
    """A scalar objective plus a list of constraint expressions.

    Everything is evaluated at the scope's current values: set ``x.value`` (or
    ``scope.set_values(u)`` with the flat vector) to move the point, then call
    any method in any order::

        scope.set_values(u)
        prob.objective(); prob.gradient()
        prob.constraints(); prob.jacobian()
        prob.hessian(obj_factor, lagrange)

    Each engine call runs at most once per point (scope values and
    parameters), and the results are read-only arrays.
    """

    def __init__(self, objective, constraints=(), verbose=False):
        objective = cast_to_const(objective)
        constraints = [cast_to_const(c) for c in constraints]
        if objective.shape != (1, 1):
            raise ValueError(f"Objective must be scalar, got {objective.shape}")
        self._scope = _find_scope([objective, *constraints])
        # Separate engine nodes for the objective and the constraints, so that
        # neither evaluation can overwrite values the other one reads.
        obj_build, con_build = _Builder(self._scope), _Builder(self._scope)
        obj_cap = obj_build(objective)
        con_caps = [con_build(c) for c in constraints]
        self._capsule = _C.make_problem(obj_cap, con_caps, verbose)
        params = obj_build.params + con_build.params
        self._params = [p for p, _ in params]
        if params:
            _C.problem_register_params(self._capsule, [c for _, c in params])
        _C.problem_init_derivatives(self._capsule)
        self._param_version = None
        self.n_vars = self._scope.total_var_size
        self.total_constraint_size = sum(c.size for c in constraints)
        # The engine holds one point; _key identifies it (scope values and
        # parameters). Results computed there are cached until it moves.
        self._key = None
        self._obj = self._con = self._grad = self._jac = None

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

    def _refresh(self):
        """Forget cached results if the scope's values or parameters changed."""
        key = (self._scope._values_version, self._scope._param_version)
        if key != self._key:
            self._sync_params()
            self._key = key
            self._obj = self._con = self._grad = self._jac = None

    def _u(self):
        """The flat variable vector this problem is evaluated at."""
        return self._scope._values[:self.n_vars]

    def objective(self):
        """Objective value (float)."""
        self._refresh()
        if self._obj is None:
            self._obj = _C.problem_objective_forward(self._capsule, self._u())
        return self._obj

    def constraints(self):
        """Stacked constraint values, shape (total_constraint_size,)."""
        self._refresh()
        if self._con is None:
            self._con = _read_only(
                _C.problem_constraint_forward(self._capsule, self._u()))
        return self._con

    def gradient(self):
        """Objective gradient, shape (n_vars,)."""
        self.objective()
        if self._grad is None:
            self._grad = _read_only(_C.problem_gradient(self._capsule))
        return self._grad

    def jacobian(self):
        """Constraint Jacobian, csr_matrix of shape (total_constraint_size, n_vars)."""
        self.constraints()
        if self._jac is None:
            self._jac = _csr(_C.problem_jacobian(self._capsule))
            _read_only(self._jac.data)
            _read_only(self._jac.indices)
            _read_only(self._jac.indptr)
        return self._jac

    def hessian(self, obj_factor, lagrange):
        """Lagrangian Hessian obj_factor H_f + sum_i lagrange_i H_gi (csr_matrix)."""
        lagrange = _flatten_vector(lagrange, self.total_constraint_size, "lagrange")
        # The engine's Hessian reads values left by the forwards, the gradient
        # and the Jacobian, so all four must be current (no-ops if they are).
        self.gradient()
        self.jacobian()
        return _csr(_C.problem_hessian(self._capsule, float(obj_factor), lagrange))


class CompiledExpression:
    """A compiled expression f, evaluated at the scope's current values."""

    def __init__(self, expr):
        self.shape = expr.shape
        self._problem = Problem(Constant(0.0), [expr])

    def forward(self):
        """f(x) as a read-only array of shape f.shape."""
        return self._problem.constraints().reshape(self.shape, order="F")

    def jacobian(self):
        """Sparse Jacobian df/dx, shape (f.size, n_vars)."""
        return self._problem.jacobian()

    def hessian(self, weights):
        """Sparse Hessian of sum(weights * f(x)), shape (n_vars, n_vars).

        weights has f's shape (a 1-D array is also accepted when f is a vector).
        """
        return self._problem.hessian(0.0, _flatten_shaped(weights, self.shape))


def compile(expr):
    """Analyze the sparsity of ``expr`` once and return a CompiledExpression."""
    return CompiledExpression(cast_to_const(expr))
