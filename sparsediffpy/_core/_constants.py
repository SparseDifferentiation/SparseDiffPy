"""Constant and SparseConstant expression nodes, plus _wrap_constant."""

import numpy as np
import scipy.sparse


class Constant:
    """A fixed dense constant in the expression tree.

    Stores values in column-major (Fortran) flat order to match the C layer.
    """

    __array_ufunc__ = None
    __array_priority__ = 20

    def __init__(self, value, shape):
        self.shape = shape
        self._value_flat = np.asarray(value, dtype=np.float64).ravel(order="F")
        expected_size = shape[0] * shape[1]
        if self._value_flat.size != expected_size:
            raise ValueError(
                f"Constant value has {self._value_flat.size} elements, "
                f"expected {expected_size} for shape {shape}"
            )


class SparseConstant:
    """A fixed sparse constant in the expression tree.

    Stores CSR arrays for use with make_left_matmul / make_right_matmul.
    """

    __array_ufunc__ = None
    __array_priority__ = 20

    def __init__(self, csr_matrix):
        csr = scipy.sparse.csr_matrix(csr_matrix)
        self.shape = (csr.shape[0], csr.shape[1])
        self._csr_data = np.asarray(csr.data, dtype=np.float64)
        self._csr_indices = np.asarray(csr.indices, dtype=np.int32)
        self._csr_indptr = np.asarray(csr.indptr, dtype=np.int32)

    def _to_dense_flat(self):
        """Convert to dense column-major flat array (for standalone use)."""
        dense = scipy.sparse.csr_matrix(
            (self._csr_data, self._csr_indices, self._csr_indptr),
            shape=self.shape,
        ).toarray()
        return dense.ravel(order="F").astype(np.float64)


def _wrap_constant(value):
    """Wrap a raw value into an expression node.

    - Expression subclass -> return as-is
    - int / float -> Constant with shape (1, 1)
    - np.ndarray 1D (n,) -> Constant with shape (n, 1) (column vector)
    - np.ndarray 2D (m, n) -> Constant with shape (m, n)
    - scipy.sparse -> SparseConstant
    """
    # Avoid circular import: check for Expression base by duck-typing
    # (has a .shape attribute and is from our module)
    if hasattr(value, "_is_sparsediff_expr"):
        return value

    if isinstance(value, (int, float)):
        return Constant(np.array([float(value)]), (1, 1))

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return Constant(np.array([value.item()]), (1, 1))
        if value.ndim == 1:
            return Constant(value, (value.shape[0], 1))
        if value.ndim == 2:
            return Constant(value, (value.shape[0], value.shape[1]))
        raise ValueError(f"Cannot wrap {value.ndim}D array as constant")

    if scipy.sparse.issparse(value):
        return SparseConstant(value)

    raise TypeError(f"Cannot convert {type(value).__name__} to expression")
