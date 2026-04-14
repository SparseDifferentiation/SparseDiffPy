"""Other expression nodes: quad_form, prod variants."""

from sparsediffpy._core._expression import Expression


class QuadForm(Expression):
    """x' Q x where Q is a constant sparse matrix."""
    def __init__(self, child, Q_csr_data, Q_csr_indices, Q_csr_indptr, Q_shape):
        self.child = child
        self.Q_csr_data = Q_csr_data
        self.Q_csr_indices = Q_csr_indices
        self.Q_csr_indptr = Q_csr_indptr
        self.Q_shape = Q_shape
        self.shape = (1, 1)


class Prod(Expression):
    """Product of all elements -> (1, 1)."""
    def __init__(self, child):
        self.child = child
        self.shape = (1, 1)


class ProdAxisZero(Expression):
    """Product along axis 0 -> (1, d2)."""
    def __init__(self, child):
        self.child = child
        self.shape = (1, child.shape[1])


class ProdAxisOne(Expression):
    """Product along axis 1 -> (1, d1). C layer returns row vector."""
    def __init__(self, child):
        self.child = child
        self.shape = (1, child.shape[0])
