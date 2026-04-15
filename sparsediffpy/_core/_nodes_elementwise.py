"""Elementwise expression nodes: unary operations applied element-by-element."""

from sparsediffpy._core._expression import Expression
from sparsediffpy._core._nodes_affine import _UnaryOp, Index


# --- Full domain ---

class Exp(_UnaryOp):
    pass


class Sin(_UnaryOp):
    pass


class Cos(_UnaryOp):
    pass


class Sinh(_UnaryOp):
    pass


class Tanh(_UnaryOp):
    pass


class Asinh(_UnaryOp):
    pass


class Logistic(_UnaryOp):
    pass


class NormalCdf(_UnaryOp):
    pass


class Xexp(_UnaryOp):
    pass


class Power(Expression):
    def __init__(self, child, exponent):
        self.child = child
        self.exponent = exponent
        self.shape = child.shape


# --- Restricted domain ---
# The C engine's restricted-domain Jacobian code does not correctly handle
# children with non-trivial Jacobian structure (e.g., index nodes with
# nonzero offset). These ops require the child to be a plain variable or
# a full-domain composition.

def _check_no_index_child(child, op_name):
    """Raise if the immediate child is an Index node.

    The C engine's restricted-domain atoms assume the child's Jacobian has
    columns starting at offset 0. Applying them directly to an Index node
    with nonzero offset produces wrong Jacobian column positions.
    """
    if isinstance(child, Index):
        raise ValueError(
            f"{op_name} cannot be applied directly to an indexed expression. "
            f"This is a known limitation of the C engine's restricted-domain "
            f"Jacobian computation. As a workaround, use a separate variable "
            f"for the indexed slice."
        )


class Log(_UnaryOp):
    def __init__(self, child):
        _check_no_index_child(child, "log")
        super().__init__(child)


class Tan(_UnaryOp):
    def __init__(self, child):
        _check_no_index_child(child, "tan")
        super().__init__(child)


class Atanh(_UnaryOp):
    def __init__(self, child):
        _check_no_index_child(child, "atanh")
        super().__init__(child)


class Entr(_UnaryOp):
    def __init__(self, child):
        _check_no_index_child(child, "entr")
        super().__init__(child)
