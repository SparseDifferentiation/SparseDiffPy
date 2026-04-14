"""Scope, Variable, and Parameter."""

import numpy as np

from sparsediffpy._core._expression import Expression
from sparsediffpy._core._shapes import validate_shape


class Variable(Expression):
    """A decision variable in the expression tree.

    Created by Scope.Variable(). Has a .value property that reads/writes
    into the scope's flat value buffer.
    """

    def __init__(self, scope, var_id, shape):
        self._scope = scope
        self._var_id = var_id
        self.shape = shape

    @property
    def value(self):
        size = self.shape[0] * self.shape[1]
        return self._scope._flat_values[self._var_id:self._var_id + size].copy()

    @value.setter
    def value(self, val):
        val = np.asarray(val, dtype=np.float64).ravel()
        size = self.shape[0] * self.shape[1]
        if val.size != size:
            raise ValueError(
                f"Expected {size} elements for Variable with shape {self.shape}, "
                f"got {val.size}"
            )
        self._scope._flat_values[self._var_id:self._var_id + size] = val


class Parameter(Expression):
    """An updatable parameter in the expression tree.

    Created by Scope.Parameter(). Values are stored on the parameter itself
    (not in the scope's flat buffer). Updated via .value property.
    """

    def __init__(self, scope, param_id, shape, value=None):
        self._scope = scope
        self._param_id = param_id
        self.shape = shape
        size = shape[0] * shape[1]
        if value is not None:
            self._value_flat = np.asarray(value, dtype=np.float64).ravel(order="F")
            if self._value_flat.size != size:
                raise ValueError(
                    f"Parameter value has {self._value_flat.size} elements, "
                    f"expected {size} for shape {shape}"
                )
        else:
            self._value_flat = np.zeros(size, dtype=np.float64)

    @property
    def value(self):
        return self._value_flat.copy()

    @value.setter
    def value(self, val):
        val = np.asarray(val, dtype=np.float64).ravel(order="F")
        size = self.shape[0] * self.shape[1]
        if val.size != size:
            raise ValueError(
                f"Expected {size} elements for Parameter with shape {self.shape}, "
                f"got {val.size}"
            )
        self._value_flat[:] = val


# Patch _is_param_like to recognize Parameter
# (already handled via lazy import in _expressions.py)


class Scope:
    """Owns the variable/parameter space and flat value buffer."""

    def __init__(self):
        self._variables = []
        self._parameters = []
        self._flat_values = np.zeros(0, dtype=np.float64)
        self._next_var_offset = 0
        self._next_param_offset = 0

    def Variable(self, d1, d2):
        """Create a new variable in this scope."""
        validate_shape(d1, d2)
        size = d1 * d2
        var_id = self._next_var_offset

        new_flat = np.zeros(self._next_var_offset + size, dtype=np.float64)
        if self._next_var_offset > 0:
            new_flat[:self._next_var_offset] = self._flat_values
        self._flat_values = new_flat
        self._next_var_offset += size

        var = Variable(self, var_id, (d1, d2))
        self._variables.append(var)
        return var

    def Parameter(self, d1, d2, value=None):
        """Create a new updatable parameter in this scope."""
        validate_shape(d1, d2)
        size = d1 * d2
        param_id = self._next_param_offset
        self._next_param_offset += size

        param = Parameter(self, param_id, (d1, d2), value)
        self._parameters.append(param)
        return param

    def set_values(self, flat_array):
        """Set all variable values at once from a flat array."""
        flat_array = np.asarray(flat_array, dtype=np.float64)
        if flat_array.size != self._flat_values.size:
            raise ValueError(
                f"Expected flat array of size {self._flat_values.size}, "
                f"got {flat_array.size}"
            )
        self._flat_values[:] = flat_array

    def get_values(self):
        """Return a copy of the flat value buffer."""
        return self._flat_values.copy()

    @property
    def total_var_size(self):
        return self._next_var_offset
