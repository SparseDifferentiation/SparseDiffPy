"""Bivariate expression nodes: operations on two variable-dependent expressions."""

from sparsediffpy._core._expression import Expression
from sparsediffpy._core._shapes import is_scalar


def _expr_contains_variable(expr, var):
    """Check if a specific Variable object appears anywhere in an expression tree."""
    if expr is var:
        return True
    for attr in ("child", "left", "right", "x", "y", "z", "param_expr", "matrix_expr"):
        child = getattr(expr, attr, None)
        if child is not None and _expr_contains_variable(child, var):
            return True
    for child in getattr(expr, "children", []):
        if _expr_contains_variable(child, var):
            return True
    return False


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
    """sum(x^2) / z where both x and z are plain Variables.

    z must be scalar and must not appear in x.
    """
    def __init__(self, x, z):
        from sparsediffpy._core._scope import Variable
        if not isinstance(x, Variable):
            raise ValueError(
                "quad_over_lin: x (first argument) must be a plain Variable."
            )
        if not isinstance(z, Variable):
            raise ValueError(
                "quad_over_lin: z (second argument) must be a plain scalar Variable."
            )
        if not is_scalar(z.shape):
            raise ValueError(f"quad_over_lin: z must be scalar, got {z.shape}")
        if _expr_contains_variable(x, z):
            raise ValueError(
                "quad_over_lin: z must not appear in x."
            )
        self.x = x
        self.z = z
        self.shape = (1, 1)


class RelEntr(Expression):
    """x * log(x / y) elementwise.

    Supports three variants (auto-dispatched by the C layer):
    - Both same shape: elementwise
    - Scalar x, vector y: x * log(x / y_i) for each i
    - Vector x, scalar y: x_i * log(x_i / y) for each i
    """
    def __init__(self, x, y):
        if x.shape == y.shape:
            self.shape = x.shape
        elif is_scalar(x.shape):
            self.shape = y.shape
        elif is_scalar(y.shape):
            self.shape = x.shape
        else:
            raise ValueError(
                f"rel_entr: shapes must match or one must be scalar, "
                f"got {x.shape} and {y.shape}"
            )
        self.x = x
        self.y = y
