"""Node converter registry: maps expression node types to C diff engine constructors.

Each converter receives (node, children_caps) where node is the Python expression
node and children_caps are already-converted C capsules. matmul and multiply are
handled separately in _compile.py (they need param_dict for parameter support).

Modelled after DNLP's registry.py.
"""

import numpy as np

from sparsediffpy import _sparsediffengine as _C
from sparsediffpy._core._constants import Constant, SparseConstant
from sparsediffpy._core._nodes_affine import (
    Add, Broadcast, DiagVec, HStack, Index, LeftMatMul, Neg,
    ParamScalarMult, ParamVectorMult, Reshape, RightMatMul, Sum, Trace,
    Transpose,
)
from sparsediffpy._core._nodes_bivariate import (
    MatMul, Multiply, QuadOverLin, RelEntr,
)
from sparsediffpy._core._nodes_elementwise import (
    Asinh, Atanh, Cos, Entr, Exp, Log, Logistic, NormalCdf, Power,
    Sin, Sinh, Tan, Tanh, Xexp,
)
from sparsediffpy._core._nodes_other import (
    Prod, ProdAxisOne, ProdAxisZero, QuadForm,
)
from sparsediffpy._core._scope import Parameter


# ---------------------------------------------------------------------------
# Matmul helpers (matching DNLP's helpers.py)
# ---------------------------------------------------------------------------

def make_sparse_left_matmul(param_node, child_cap, matrix):
    """A @ f(x) with sparse constant A."""
    return _C.make_left_matmul(
        param_node, child_cap, "sparse",
        matrix._csr_data, matrix._csr_indices, matrix._csr_indptr,
        matrix.shape[0], matrix.shape[1],
    )


def make_dense_left_matmul(param_node, child_cap, A_flat, m, n):
    """A @ f(x) with dense constant A."""
    return _C.make_left_matmul(param_node, child_cap, "dense", A_flat, m, n)


def make_sparse_right_matmul(param_node, child_cap, matrix):
    """f(x) @ A with sparse constant A."""
    return _C.make_right_matmul(
        param_node, child_cap, "sparse",
        matrix._csr_data, matrix._csr_indices, matrix._csr_indptr,
        matrix.shape[0], matrix.shape[1],
    )


def make_dense_right_matmul(param_node, child_cap, A_flat, m, n):
    """f(x) @ A with dense constant A."""
    return _C.make_right_matmul(param_node, child_cap, "dense", A_flat, m, n)


def _to_dense_row_major(matrix):
    """Convert a Constant or Parameter to row-major flat data for dense matmul."""
    m, n = matrix.shape
    return matrix._value_flat.reshape((m, n), order="F").flatten(order="C")


# ---------------------------------------------------------------------------
# Individual converters for nodes needing special handling
# ---------------------------------------------------------------------------

def convert_hstack(node, child_caps):
    return _C.make_hstack(child_caps)


def convert_index(node, child_caps):
    return _C.make_index(
        child_caps[0], node.shape[0], node.shape[1], node.flat_indices
    )


def convert_sum(node, child_caps):
    return _C.make_sum(child_caps[0], node.axis)


def convert_power(node, child_caps):
    return _C.make_power(child_caps[0], node.exponent)


def convert_reshape(node, child_caps):
    return _C.make_reshape(child_caps[0], node.shape[0], node.shape[1])


def convert_broadcast(node, child_caps):
    return _C.make_broadcast(child_caps[0], node.shape[0], node.shape[1])


def convert_quad_over_lin(node, child_caps):
    return _C.make_quad_over_lin(child_caps[0], child_caps[1])


def convert_rel_entr(node, child_caps):
    return _C.make_rel_entr(child_caps[0], child_caps[1])


def convert_quad_form(node, child_caps):
    return _C.make_quad_form(
        child_caps[0],
        node.Q_csr_data, node.Q_csr_indices, node.Q_csr_indptr,
        node.Q_shape[0], node.Q_shape[1],
    )



# ---------------------------------------------------------------------------
# Registry dict
# ---------------------------------------------------------------------------

ATOM_CONVERTERS = {
    # Elementwise unary (full domain)
    Neg: lambda _node, caps: _C.make_neg(caps[0]),
    Exp: lambda _node, caps: _C.make_exp(caps[0]),
    Sin: lambda _node, caps: _C.make_sin(caps[0]),
    Cos: lambda _node, caps: _C.make_cos(caps[0]),
    Sinh: lambda _node, caps: _C.make_sinh(caps[0]),
    Tanh: lambda _node, caps: _C.make_tanh(caps[0]),
    Asinh: lambda _node, caps: _C.make_asinh(caps[0]),
    Logistic: lambda _node, caps: _C.make_logistic(caps[0]),
    NormalCdf: lambda _node, caps: _C.make_normal_cdf(caps[0]),
    Xexp: lambda _node, caps: _C.make_xexp(caps[0]),

    # Elementwise unary (restricted domain)
    Log: lambda _node, caps: _C.make_log(caps[0]),
    Tan: lambda _node, caps: _C.make_tan(caps[0]),
    Atanh: lambda _node, caps: _C.make_atanh(caps[0]),
    Entr: lambda _node, caps: _C.make_entr(caps[0]),

    # Elementwise unary with extra args
    Power: convert_power,

    # Affine unary
    Transpose: lambda _node, caps: _C.make_transpose(caps[0]),
    DiagVec: lambda _node, caps: _C.make_diag_vec(caps[0]),
    Trace: lambda _node, caps: _C.make_trace(caps[0]),

    # Reductions
    Sum: convert_sum,
    Prod: lambda _node, caps: _C.make_prod(caps[0]),
    ProdAxisZero: lambda _node, caps: _C.make_prod_axis_zero(caps[0]),
    ProdAxisOne: lambda _node, caps: _C.make_prod_axis_one(caps[0]),

    # Shape operations
    Reshape: convert_reshape,
    Broadcast: convert_broadcast,

    # Binary (both variable-dependent)
    Add: lambda _node, caps: _C.make_add(caps[0], caps[1]),
    Multiply: lambda _node, caps: _C.make_multiply(caps[0], caps[1]),
    MatMul: lambda _node, caps: _C.make_matmul(caps[0], caps[1]),

    # Bivariate
    QuadOverLin: convert_quad_over_lin,
    RelEntr: convert_rel_entr,
    QuadForm: convert_quad_form,

    # Structural
    HStack: convert_hstack,
    Index: convert_index,
}
