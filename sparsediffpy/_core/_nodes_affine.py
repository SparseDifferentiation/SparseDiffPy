"""Affine expression nodes: linear/affine operations on expressions."""

import numpy as np

from sparsediffpy._core._expression import Expression, _wrap_constant


class _UnaryOp(Expression):
    def __init__(self, child):
        child = _wrap_constant(child)
        self.child = child
        self.shape = child.shape


class Neg(_UnaryOp):
    pass


class Transpose(Expression):
    def __init__(self, child):
        self.child = child
        self.shape = (child.shape[1], child.shape[0])


class DiagVec(Expression):
    """Create a diagonal matrix from a column vector (n,1) -> (n,n)."""
    def __init__(self, child):
        if child.shape[1] != 1:
            raise ValueError(f"diag_vec requires a column vector, got shape {child.shape}")
        self.child = child
        self.shape = (child.shape[0], child.shape[0])


class Trace(Expression):
    def __init__(self, child):
        if child.shape[0] != child.shape[1]:
            raise ValueError(f"trace requires a square matrix, got shape {child.shape}")
        self.child = child
        self.shape = (1, 1)


class Reshape(Expression):
    def __init__(self, child, new_shape):
        old_size = child.shape[0] * child.shape[1]
        new_size = new_shape[0] * new_shape[1]
        if old_size != new_size:
            raise ValueError(
                f"Cannot reshape {child.shape} (size {old_size}) to "
                f"{new_shape} (size {new_size})"
            )
        self.child = child
        self.shape = new_shape


class Broadcast(Expression):
    """Broadcast scalar/row/column to a target shape."""
    def __init__(self, child, target_shape):
        self.child = child
        self.shape = target_shape


class Sum(Expression):
    """Sum reduction. C layer always returns row vectors:
    axis=-1: (1,1), axis=0: (1,d2), axis=1: (1,d1)."""
    def __init__(self, child, axis):
        self.child = child
        self.axis = axis
        d1, d2 = child.shape
        if axis == -1:
            self.shape = (1, 1)
        elif axis == 0:
            self.shape = (1, d2)
        elif axis == 1:
            self.shape = (1, d1)
        else:
            raise ValueError(f"Invalid axis {axis}, must be -1, 0, or 1")


class Add(Expression):
    def __init__(self, left, right):
        assert left.shape == right.shape, f"Add shape mismatch: {left.shape} vs {right.shape}"
        self.left = left
        self.right = right
        self.shape = left.shape


class HStack(Expression):
    """Horizontal concatenation."""
    def __init__(self, children, result_shape):
        self.children = children
        self.shape = result_shape


class Index(Expression):
    """Indexing with flat column-major indices."""
    def __init__(self, child, flat_indices, result_shape):
        self.child = child
        self.flat_indices = flat_indices
        self.shape = result_shape


class ParamScalarMult(Expression):
    """a * f(x) where a is a scalar constant/parameter."""
    def __init__(self, param_expr, child):
        self.param_expr = param_expr
        self.child = child
        self.shape = child.shape


class ParamVectorMult(Expression):
    """a . f(x) elementwise where a is a constant/parameter of matching shape."""
    def __init__(self, param_expr, child):
        self.param_expr = param_expr
        self.child = child
        self.shape = child.shape


class LeftMatMul(Expression):
    """A @ f(x) where A is a constant/sparse constant/parameter matrix."""
    def __init__(self, matrix_expr, child, result_shape):
        self.matrix_expr = matrix_expr
        self.child = child
        self.shape = result_shape


class RightMatMul(Expression):
    """f(x) @ A where A is a constant/sparse constant/parameter matrix."""
    def __init__(self, matrix_expr, child, result_shape):
        self.matrix_expr = matrix_expr
        self.child = child
        self.shape = result_shape
