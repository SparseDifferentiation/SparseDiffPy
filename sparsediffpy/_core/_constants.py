"""Constant and SparseConstant expression nodes."""

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
