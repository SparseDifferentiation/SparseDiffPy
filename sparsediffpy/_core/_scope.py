"""Scope, Variable, and Parameter."""

import numpy as np

from sparsediffpy._core._expression import Expression
from sparsediffpy._core._shapes import validate_shape


class DimensionError(ValueError):
    """Raised when a value has the wrong number of elements."""
    pass


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
        return self._scope._flat_values[self._var_id:self._var_id + self.size].copy()

    @value.setter
    def value(self, val):
        val = np.asarray(val, dtype=np.float64).ravel()
        if val.size != self.size:
            raise DimensionError(f"expected {self.size} elements, got {val.size}")
        self._scope._flat_values[self._var_id:self._var_id + self.size] = val


class Parameter(Expression):
    """An updatable parameter in the expression tree.

    Created by Scope.Parameter(). Values must be set via .value before
    evaluating any expression that uses this parameter.
    """

    def __init__(self, scope, param_id, shape):
        self._scope = scope
        self._param_id = param_id
        self.shape = shape
        self._value_flat = None

    @property
    def value(self):
        if self._value_flat is None:
            return None
        return self._value_flat.copy()

    @value.setter
    def value(self, val):
        val = np.asarray(val, dtype=np.float64).ravel(order="F")
        if val.size != self.size:
            raise DimensionError(f"expected {self.size} elements, got {val.size}")
        self._value_flat = val.copy()


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

    def Parameter(self, d1, d2):
        """Create a new updatable parameter in this scope.

        Set its value via .value = ... before evaluating.
        """
        validate_shape(d1, d2)
        size = d1 * d2
        param_id = self._next_param_offset
        self._next_param_offset += size

        param = Parameter(self, param_id, (d1, d2))
        self._parameters.append(param)
        return param

    def set_values(self, array):
        """Set all variable values at once from a flat array."""
        array = np.asarray(array, dtype=np.float64)
        in_size = self._flat_values.size
        if array.size != in_size:
            raise DimensionError(f"expected {in_size} elements, got {array.size}")
        self._flat_values[:] = array

    def get_values(self):
        """Return a copy of the flat value buffer."""
        return self._flat_values.copy()

    @property
    def total_var_size(self):
        return self._next_var_offset
