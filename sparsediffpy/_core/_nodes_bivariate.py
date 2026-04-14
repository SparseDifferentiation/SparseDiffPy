"""Bivariate expression nodes: operations on two variable-dependent expressions."""

from sparsediffpy._core._expression import Expression
from sparsediffpy._core._shapes import is_scalar


class Multiply(Expression):
    """Elementwise multiply (both operands are variable-dependent)."""
    def __init__(self, left, right):
        assert left.shape == right.shape, f"Multiply shape mismatch: {left.shape} vs {right.shape}"
        self.left = left
        self.right = right
        self.shape = left.shape


class MatMul(Expression):
    """Matrix multiply where both operands are variable-dependent."""
    def __init__(self, left, right, result_shape):
        self.left = left
        self.right = right
        self.shape = result_shape


class QuadOverLin(Expression):
    """sum(x^2) / z where z is a scalar."""
    def __init__(self, x, z):
        if not is_scalar(z.shape):
            raise ValueError(f"quad_over_lin: z must be scalar, got shape {z.shape}")
        self.x = x
        self.z = z
        self.shape = (1, 1)


class RelEntr(Expression):
    """x * log(x / y) elementwise."""
    def __init__(self, x, y):
        if x.shape != y.shape:
            raise ValueError(f"rel_entr: shape mismatch {x.shape} vs {y.shape}")
        self.x = x
        self.y = y
        self.shape = x.shape
