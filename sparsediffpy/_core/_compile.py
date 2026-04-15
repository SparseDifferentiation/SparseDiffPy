"""compile() and CompiledExpression.

The recursive tree walker that converts Python expression nodes to C capsules.
Node-type-to-C-call mapping lives in _registry.py.
"""

import numpy as np
import scipy.sparse

from sparsediffpy import _sparsediffengine as _C
from sparsediffpy._core._constants import Constant, SparseConstant
from sparsediffpy._core._nodes_affine import (
    LeftMatMul,
    ParamScalarMult,
    ParamVectorMult,
    RightMatMul,
)
from sparsediffpy._core._registry import (
    ATOM_CONVERTERS,
    _to_dense_row_major,
    make_dense_left_matmul,
    make_dense_right_matmul,
    make_sparse_left_matmul,
    make_sparse_right_matmul,
)
from sparsediffpy._core._scope import Parameter, Variable


def compile(expr):
    """Compile an expression tree into a CompiledExpression.

    Walks the Python expression tree, discovers all Variables and Parameters,
    builds C capsules bottom-up, creates a C problem, and initializes
    sparsity patterns for Jacobian and Hessian computation.
    """
    # 1. Collect all Variable and Parameter leaves
    variables = []
    parameters = []
    _collect_leaves(expr, variables, parameters, set())

    # 2. Determine the scope
    scope = None
    for v in variables:
        if scope is None:
            scope = v._scope
        elif v._scope is not scope:
            raise ValueError("All variables must belong to the same Scope")

    if scope is None:
        from sparsediffpy._core._scope import Scope
        scope = Scope()

    n_vars = scope._next_var_offset
    if n_vars == 0:
        n_vars = 1  # C layer needs at least 1 variable

    # 3. Build C capsules bottom-up
    capsule_cache = {}
    param_capsules_ordered = []
    param_objects_ordered = []
    root_capsule = _build_capsule(
        expr, n_vars, capsule_cache, param_capsules_ordered, param_objects_ordered
    )

    # 4. Create dummy zero objective (scalar)
    dummy_obj = _C.make_parameter(1, 1, -1, n_vars, np.array([0.0]))

    # 5. Create C problem with expr as the single constraint
    problem = _C.make_problem(dummy_obj, [root_capsule], False)

    # 6. Register parameters if any
    if param_capsules_ordered:
        _C.problem_register_params(problem, param_capsules_ordered)

    # 7. Init sparsity patterns
    _C.problem_init_jacobian(problem)
    _C.problem_init_hessian(problem)

    return CompiledExpression(
        problem_capsule=problem,
        scope=scope,
        param_capsules=param_capsules_ordered,
        param_objects=param_objects_ordered,
        expr_shape=expr.shape,
        n_vars=n_vars,
    )


# ---------------------------------------------------------------------------
# Tree walking
# ---------------------------------------------------------------------------

def _collect_leaves(node, variables, parameters, visited):
    """Walk the expression tree to collect Variable and Parameter leaves."""
    node_id = id(node)
    if node_id in visited:
        return
    visited.add(node_id)

    if isinstance(node, Variable):
        variables.append(node)
        return
    if isinstance(node, Parameter):
        parameters.append(node)
        return
    if isinstance(node, (Constant, SparseConstant)):
        return

    # Walk children
    if hasattr(node, "child"):
        _collect_leaves(node.child, variables, parameters, visited)
    if hasattr(node, "left"):
        _collect_leaves(node.left, variables, parameters, visited)
    if hasattr(node, "right"):
        _collect_leaves(node.right, variables, parameters, visited)
    if hasattr(node, "x") and hasattr(node, "z"):
        _collect_leaves(node.x, variables, parameters, visited)
        _collect_leaves(node.z, variables, parameters, visited)
    elif hasattr(node, "x") and hasattr(node, "y"):
        _collect_leaves(node.x, variables, parameters, visited)
        _collect_leaves(node.y, variables, parameters, visited)
    if hasattr(node, "param_expr"):
        _collect_leaves(node.param_expr, variables, parameters, visited)
    if hasattr(node, "matrix_expr"):
        _collect_leaves(node.matrix_expr, variables, parameters, visited)
    if hasattr(node, "children"):
        for c in node.children:
            _collect_leaves(c, variables, parameters, visited)


# ---------------------------------------------------------------------------
# Capsule building
# ---------------------------------------------------------------------------

def _build_capsule(node, n_vars, cache, param_caps, param_objs):
    """Recursively build C capsules for the expression tree."""
    node_id = id(node)
    if node_id in cache:
        return cache[node_id]

    cap = _convert_node(node, n_vars, cache, param_caps, param_objs)

    # Post-conversion dimension check
    d1_c, d2_c = _C.get_expr_dimensions(cap)
    d1_py, d2_py = node.shape
    if d1_c != d1_py or d2_c != d2_py:
        raise ValueError(
            f"Dimension mismatch for {type(node).__name__}: "
            f"C dimensions ({d1_c}, {d2_c}) vs Python dimensions ({d1_py}, {d2_py})"
        )

    cache[node_id] = cap
    return cap


def _convert_node(node, n_vars, cache, param_caps, param_objs):
    """Convert a single Python expression node to a C capsule."""

    # --- Leaves ---
    if isinstance(node, Variable):
        return _C.make_variable(
            node.shape[0], node.shape[1], node._var_id, n_vars
        )

    if isinstance(node, Parameter):
        # Use current values if set, otherwise zeros as placeholder.
        # Real values are synced via problem_update_params before evaluation.
        size = node.shape[0] * node.shape[1]
        values = node._value_flat if node._value_flat is not None else np.zeros(size)
        cap = _C.make_parameter(
            node.shape[0], node.shape[1], node._param_id, n_vars, values,
        )
        param_caps.append(cap)
        param_objs.append(node)
        return cap

    if isinstance(node, Constant):
        return _C.make_parameter(
            node.shape[0], node.shape[1], -1, n_vars, node._value_flat
        )

    if isinstance(node, SparseConstant):
        return _C.make_parameter(
            node.shape[0], node.shape[1], -1, n_vars, node._to_dense_flat()
        )

    # --- Matmul and multiply with parameter dispatch ---
    # These need special handling because they access matrix_expr / param_expr
    # directly rather than going through a uniform children list.
    if isinstance(node, LeftMatMul):
        return _convert_left_matmul(node, n_vars, cache, param_caps, param_objs)

    if isinstance(node, RightMatMul):
        return _convert_right_matmul(node, n_vars, cache, param_caps, param_objs)

    if isinstance(node, ParamScalarMult):
        param_cap = _build_capsule(node.param_expr, n_vars, cache, param_caps, param_objs)
        child_cap = _build_capsule(node.child, n_vars, cache, param_caps, param_objs)
        return _C.make_param_scalar_mult(param_cap, child_cap)

    if isinstance(node, ParamVectorMult):
        param_cap = _build_capsule(node.param_expr, n_vars, cache, param_caps, param_objs)
        child_cap = _build_capsule(node.child, n_vars, cache, param_caps, param_objs)
        return _C.make_param_vector_mult(param_cap, child_cap)

    # --- Registry lookup ---
    node_type = type(node)
    if node_type in ATOM_CONVERTERS:
        child_caps = _build_children(node, n_vars, cache, param_caps, param_objs)
        return ATOM_CONVERTERS[node_type](node, child_caps)

    raise TypeError(f"Unknown expression node type: {node_type.__name__}")


def _build_children(node, n_vars, cache, param_caps, param_objs):
    """Build C capsules for all children of a node, returned as a list."""
    caps = []
    # Unary: .child
    if hasattr(node, "child"):
        caps.append(_build_capsule(node.child, n_vars, cache, param_caps, param_objs))
    # Binary: .left, .right
    if hasattr(node, "left"):
        caps.append(_build_capsule(node.left, n_vars, cache, param_caps, param_objs))
    if hasattr(node, "right"):
        caps.append(_build_capsule(node.right, n_vars, cache, param_caps, param_objs))
    # QuadOverLin/RelEntr: .x, .y or .x, .z
    if hasattr(node, "x") and not caps:
        caps.append(_build_capsule(node.x, n_vars, cache, param_caps, param_objs))
        if hasattr(node, "z"):
            caps.append(_build_capsule(node.z, n_vars, cache, param_caps, param_objs))
        elif hasattr(node, "y"):
            caps.append(_build_capsule(node.y, n_vars, cache, param_caps, param_objs))
    # HStack: .children
    if hasattr(node, "children"):
        for c in node.children:
            caps.append(_build_capsule(c, n_vars, cache, param_caps, param_objs))
    return caps


# ---------------------------------------------------------------------------
# Left/right matmul converters
# ---------------------------------------------------------------------------

def _convert_left_matmul(node, n_vars, cache, param_caps, param_objs):
    """Convert A @ f(x)."""
    child_cap = _build_capsule(node.child, n_vars, cache, param_caps, param_objs)
    matrix = node.matrix_expr
    m, n = matrix.shape

    if isinstance(matrix, SparseConstant):
        return make_sparse_left_matmul(None, child_cap, matrix)

    if isinstance(matrix, Parameter):
        param_cap = _build_capsule(matrix, n_vars, cache, param_caps, param_objs)
        return make_dense_left_matmul(
            param_cap, child_cap, _to_dense_row_major(matrix), m, n
        )

    if isinstance(matrix, Constant):
        return make_dense_left_matmul(
            None, child_cap, _to_dense_row_major(matrix), m, n
        )

    raise TypeError(f"LeftMatMul matrix must be Constant, SparseConstant, or Parameter")


def _convert_right_matmul(node, n_vars, cache, param_caps, param_objs):
    """Convert f(x) @ A."""
    child_cap = _build_capsule(node.child, n_vars, cache, param_caps, param_objs)
    matrix = node.matrix_expr
    m, n = matrix.shape

    if isinstance(matrix, SparseConstant):
        return make_sparse_right_matmul(None, child_cap, matrix)

    if isinstance(matrix, Parameter):
        param_cap = _build_capsule(matrix, n_vars, cache, param_caps, param_objs)
        return make_dense_right_matmul(
            param_cap, child_cap, _to_dense_row_major(matrix), m, n
        )

    if isinstance(matrix, Constant):
        return make_dense_right_matmul(
            None, child_cap, _to_dense_row_major(matrix), m, n
        )

    raise TypeError(f"RightMatMul matrix must be Constant, SparseConstant, or Parameter")


# ---------------------------------------------------------------------------
# CompiledExpression
# ---------------------------------------------------------------------------

class CompiledExpression:
    """A compiled expression ready for evaluation.

    Reads variable values from the scope's flat buffer.
    Reads parameter values from the Parameter objects.
    """

    def __init__(self, problem_capsule, scope, param_capsules, param_objects,
                 expr_shape, n_vars):
        self._problem = problem_capsule
        self._scope = scope
        self._param_capsules = param_capsules
        self._param_objects = param_objects
        self._expr_shape = expr_shape
        self._n_vars = n_vars

    def _sync_params(self):
        """Push current parameter values to the C problem."""
        if not self._param_objects:
            return
        for p in self._param_objects:
            if p._value_flat is None:
                raise ValueError(
                    f"Parameter with shape {p.shape} has no value set. "
                    f"Assign a value via parameter.value = ... before evaluating."
                )
        theta_parts = [p._value_flat for p in self._param_objects]
        theta = np.concatenate(theta_parts)
        _C.problem_update_params(self._problem, theta)

    def _set_point(self):
        """Push variable values and evaluate forward pass."""
        self._sync_params()
        _C.problem_objective_forward(self._problem, self._scope._flat_values)
        _C.problem_constraint_forward(self._problem, self._scope._flat_values)

    def forward(self):
        """Evaluate the expression at the current variable values."""
        self._set_point()
        result = _C.problem_constraint_forward(self._problem, self._scope._flat_values)
        return result

    def jacobian(self):
        """Compute the sparse Jacobian at the current variable values.

        Returns scipy.sparse.csr_matrix of shape (expr_size, n_vars).
        """
        self._set_point()
        data, indices, indptr, (m, n) = _C.problem_jacobian(self._problem)
        return scipy.sparse.csr_matrix((data, indices, indptr), shape=(m, n))

    def hessian(self, weights):
        """Compute the sparse Hessian of the weighted expression.

        The Hessian is of the scalar function w^T f(x), where w is the
        weights vector and f is the compiled expression.

        Args:
            weights: array of length expr_size

        Returns scipy.sparse.csr_matrix of shape (n_vars, n_vars).
        """
        weights = np.asarray(weights, dtype=np.float64).ravel()
        self._set_point()
        _C.problem_jacobian(self._problem)
        data, indices, indptr, (m, n) = _C.problem_hessian(
            self._problem, 0.0, weights
        )
        return scipy.sparse.csr_matrix((data, indices, indptr), shape=(m, n))
