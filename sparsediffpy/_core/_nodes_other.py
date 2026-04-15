"""Other expression nodes: quad_form, prod variants."""

from sparsediffpy._core._expression import Expression


class QuadForm(Expression):
    """x' Q x where Q is a constant CSR sparse matrix."""
    def __init__(self, child, Q):
        self.child = child
        self.Q = Q
        self.shape = (1, 1)


def _check_prod_child_is_variable(child):
    """Require prod's argument to be a plain Variable.

    Temporary limitation: the C engine does not implement the chain rule
    for prod, so it only works correctly when the argument is a variable
    (Jacobian is identity-like). Compositions like prod(f(x)) will give
    wrong derivatives.
    """
    from sparsediffpy._core._scope import Variable
    if not isinstance(child, Variable):
        raise ValueError(
            "prod requires its argument to be a plain Variable. "
            "The C engine does not currently implement the chain rule for prod, "
            "so compositions like prod(f(x)) are not supported."
        )


class Prod(Expression):
    """Product of all elements -> (1, 1)."""
    def __init__(self, child):
        _check_prod_child_is_variable(child)
        self.child = child
        self.shape = (1, 1)


class ProdAxisZero(Expression):
    """Product along axis 0 -> (1, d2)."""
    def __init__(self, child):
        _check_prod_child_is_variable(child)
        self.child = child
        self.shape = (1, child.shape[1])


class ProdAxisOne(Expression):
    """Product along axis 1 -> (1, d1). C layer returns row vector."""
    def __init__(self, child):
        _check_prod_child_is_variable(child)
        self.child = child
        self.shape = (1, child.shape[0])
