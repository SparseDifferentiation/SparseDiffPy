"""Problem class: wraps a C problem capsule (objective + list of constraints).

Takes SparseDiffPy expressions (not CVXPY, not pre-built capsules). CVXPY-facing
adapters live in downstream libraries (e.g. DNLP).
"""

import numpy as np

from sparsediffpy import _sparsediffengine as _C
from sparsediffpy._core._compile import _build_capsule, _collect_leaves


class Problem:
    """A compiled NLP-style problem: one scalar objective plus a list of constraints.

    Method names mirror DNLP's `C_problem` so a CVXPY adapter can return a
    Problem and existing solver callsites keep working.
    """

    def __init__(self, objective, constraints=None, verbose=False):
        constraints = list(constraints) if constraints else []

        if objective.shape != (1, 1):
            raise ValueError(
                f"Objective must be scalar (shape (1, 1)), got {objective.shape}"
            )

        variables, parameters = [], []
        visited = set()
        _collect_leaves(objective, variables, parameters, visited)
        for c in constraints:
            _collect_leaves(c, variables, parameters, visited)

        if not variables:
            raise ValueError("Problem must contain at least one Variable")

        scope = variables[0]._scope
        for v in variables[1:]:
            if v._scope is not scope:
                raise ValueError("All variables must belong to the same Scope")

        n_vars = scope._next_var_offset

        # One shared cache across objective + all constraints: CSE in both
        # directions (within an expression, and across the obj/constraint
        # boundary) is safe, and each Parameter capsule is appended to
        # param_caps exactly once.
        cache = {}
        param_caps, param_objs = [], []
        obj_cap = _build_capsule(objective, n_vars, cache, param_caps, param_objs)
        constraint_caps = [
            _build_capsule(c, n_vars, cache, param_caps, param_objs)
            for c in constraints
        ]

        self._capsule = _C.make_problem(obj_cap, constraint_caps, verbose)
        if param_caps:
            _C.problem_register_params(self._capsule, param_caps)

        self._scope = scope
        self._param_capsules = param_caps
        self._param_objects = param_objs
        self._n_vars = n_vars
        self._total_constraint_size = sum(c.size for c in constraints)
        self._jacobian_coo_initialized = False
        self._hessian_coo_initialized = False

        if param_caps:
            self._sync_params()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sync_params(self):
        """Push current Parameter values to the C problem.

        Called once at construction. After construction, callers invoke
        update_params(theta) explicitly (matching DNLP's solver-loop contract).
        """
        for p in self._param_objects:
            if p._value_flat is None:
                raise ValueError(
                    f"Parameter with shape {p.shape} has no value set. "
                    f"Assign a value via parameter.value = ... before constructing Problem."
                )
        theta = np.concatenate([p._value_flat for p in self._param_objects])
        _C.problem_update_params(self._capsule, theta)
        self._scope._params_dirty = False

    # ------------------------------------------------------------------
    # Parameter updates
    # ------------------------------------------------------------------

    def update_params(self, theta):
        """Update parameter values in the C DAG from a flat theta vector.

        Sparsity structures (Jacobian/Hessian) remain valid after this call.
        """
        theta = np.asarray(theta, dtype=np.float64)
        _C.problem_update_params(self._capsule, theta)
        self._scope._params_dirty = False

    # ------------------------------------------------------------------
    # Sparsity initialization (COO)
    # ------------------------------------------------------------------

    def init_jacobian_coo(self):
        """Fill sparsity for the constraint Jacobian in COO format.

        Must be called once before get_jacobian_sparsity_coo() or eval_jacobian_vals().
        """
        _C.problem_init_jacobian_coo(self._capsule)
        self._jacobian_coo_initialized = True

    def init_hessian_coo_lower_tri(self):
        """Fill sparsity for the Lagrangian Hessian (lower triangle, COO).

        Must be called once before get_problem_hessian_sparsity_coo() or
        eval_hessian_vals_coo_lower_tri().
        """
        _C.problem_init_hessian_coo_lower_triangular(self._capsule)
        self._hessian_coo_initialized = True

    # ------------------------------------------------------------------
    # Forward evaluation
    # ------------------------------------------------------------------

    def objective_forward(self, u):
        """Evaluate the objective at variable values `u`. Returns a float."""
        u = np.asarray(u, dtype=np.float64)
        return _C.problem_objective_forward(self._capsule, u)

    def constraint_forward(self, u):
        """Evaluate constraints at variable values `u`. Returns an np.ndarray."""
        u = np.asarray(u, dtype=np.float64)
        return _C.problem_constraint_forward(self._capsule, u)

    def gradient(self):
        """Compute gradient of the objective. Call objective_forward first."""
        return _C.problem_gradient(self._capsule)

    # ------------------------------------------------------------------
    # Jacobian (COO path)
    # ------------------------------------------------------------------

    def get_jacobian_sparsity_coo(self):
        """Return the sparsity pattern (rows, cols) of the constraint Jacobian.

        Call init_jacobian_coo() first.
        """
        rows, cols, _shape = _C.get_jacobian_sparsity_coo(self._capsule)
        return rows, cols

    def eval_jacobian_vals(self):
        """Evaluate the constraint Jacobian and return its nonzero values.

        Values correspond to the sparsity pattern from get_jacobian_sparsity_coo().
        Call constraint_forward() first to set the evaluation point.
        """
        return _C.problem_eval_jacobian_vals(self._capsule)

    # ------------------------------------------------------------------
    # Lagrangian Hessian (COO lower-triangular path)
    # ------------------------------------------------------------------

    def get_problem_hessian_sparsity_coo(self):
        """Return the sparsity pattern (rows, cols) of the lower-triangular
        Lagrangian Hessian.

        Call init_hessian_coo_lower_tri() first.
        """
        rows, cols, _shape = _C.get_problem_hessian_sparsity_coo(self._capsule)
        return rows, cols

    def eval_hessian_vals_coo_lower_tri(self, obj_factor, lagrange):
        """Evaluate the lower-triangular Lagrangian Hessian values.

        Computes obj_factor * H_f + sum_i lagrange[i] * H_gi, where f is the
        objective and g_i are the constraints. Values correspond to the sparsity
        pattern from get_problem_hessian_sparsity_coo().

        Call objective_forward() and constraint_forward() first to set the
        evaluation point.
        """
        lagrange = np.asarray(lagrange, dtype=np.float64)
        if lagrange.size != self._total_constraint_size:
            raise ValueError(
                f"lagrange length {lagrange.size} != total_constraint_size "
                f"{self._total_constraint_size}"
            )
        return _C.problem_eval_hessian_vals_coo(
            self._capsule, float(obj_factor), lagrange
        )
