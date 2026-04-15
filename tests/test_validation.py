"""Tests for error/validation: shape mismatches, wrong assignments, mixed scopes."""

import numpy as np
import pytest
import scipy.sparse
import sparsediffpy as sp


# ---------------------------------------------------------------------------
# Shape mismatch in binary operators
# ---------------------------------------------------------------------------

class TestShapeMismatch:
    def test_add_incompatible(self, scope):
        x = scope.Variable(3, 1)
        y = scope.Variable(2, 1)
        with pytest.raises(ValueError, match="Cannot broadcast"):
            x + y

    def test_add_incompatible_matrix(self, scope):
        X = scope.Variable(3, 2)
        Y = scope.Variable(2, 3)
        with pytest.raises(ValueError, match="Cannot broadcast"):
            X + Y

    def test_sub_incompatible(self, scope):
        x = scope.Variable(3, 1)
        y = scope.Variable(2, 1)
        with pytest.raises(ValueError, match="Cannot broadcast"):
            x - y

    def test_matmul_incompatible(self, scope):
        x = scope.Variable(3, 1)
        y = scope.Variable(2, 1)
        with pytest.raises(ValueError, match="Matmul shape mismatch"):
            x @ y

    def test_matmul_inner_dim_mismatch(self, scope):
        X = scope.Variable(3, 4)
        Y = scope.Variable(2, 3)
        with pytest.raises(ValueError, match="Matmul shape mismatch"):
            X @ Y

    def test_rel_entr_shape_mismatch(self, scope):
        x = scope.Variable(3, 1)
        y = scope.Variable(2, 1)
        with pytest.raises(ValueError, match="shapes must match or one must be scalar"):
            sp.rel_entr(x, y)

    def test_quad_over_lin_non_scalar_z(self, scope):
        x = scope.Variable(3, 1)
        z = scope.Variable(2, 1)
        with pytest.raises(ValueError, match="must be scalar"):
            sp.quad_over_lin(x, z)


# ---------------------------------------------------------------------------
# Atom-specific shape validation
# ---------------------------------------------------------------------------

class TestAtomValidation:
    def test_trace_non_square(self, scope):
        X = scope.Variable(3, 2)
        with pytest.raises(ValueError, match="square matrix"):
            sp.trace(X)

    def test_diag_vec_non_column(self, scope):
        X = scope.Variable(3, 2)
        with pytest.raises(ValueError, match="column vector"):
            sp.diag_vec(X)

    def test_diag_vec_row_vector(self, scope):
        r = scope.Variable(1, 3)
        with pytest.raises(ValueError, match="column vector"):
            sp.diag_vec(r)

    def test_reshape_size_mismatch(self, scope):
        x = scope.Variable(3, 1)
        with pytest.raises(ValueError, match="Cannot reshape"):
            sp.reshape(x, 2, 2)

    def test_quad_form_wrong_Q_size(self, scope):
        x = scope.Variable(3, 1)
        Q = np.eye(4)
        with pytest.raises(ValueError, match="need x"):
            sp.quad_form(x, Q)

    def test_quad_form_non_column(self, scope):
        x = scope.Variable(1, 3)
        Q = np.eye(3)
        with pytest.raises(ValueError, match="need x"):
            sp.quad_form(x, Q)

    def test_pow_non_numeric_exponent(self, scope):
        x = scope.Variable(3, 1)
        with pytest.raises(TypeError, match="constant number"):
            x ** "two"

    def test_hstack_empty(self):
        with pytest.raises(ValueError, match="empty argument"):
            sp.hstack([])

    def test_vstack_empty(self):
        with pytest.raises(ValueError, match="empty argument"):
            sp.vstack([])

    def test_hstack_mismatched_rows(self, scope):
        x = scope.Variable(3, 1)
        y = scope.Variable(2, 1)
        with pytest.raises(ValueError, match="row mismatch"):
            sp.hstack([x, y])

    def test_vstack_mismatched_cols(self, scope):
        X = scope.Variable(3, 2)
        Y = scope.Variable(3, 3)
        with pytest.raises(ValueError, match="column mismatch"):
            sp.vstack([X, Y])

    def test_restricted_domain_on_index(self, scope):
        x = scope.Variable(4, 1)
        with pytest.raises(ValueError, match="cannot be applied directly"):
            sp.log(x[1:3])

        with pytest.raises(ValueError, match="cannot be applied directly"):
            sp.tan(x[1:3])

        with pytest.raises(ValueError, match="cannot be applied directly"):
            sp.atanh(x[1:3])

        with pytest.raises(ValueError, match="cannot be applied directly"):
            sp.entr(x[1:3])

    def test_quad_over_lin_args_must_be_variables(self, scope):
        x = scope.Variable(3, 1)
        z = scope.Variable(1, 1)
        # This should work — both are plain variables
        sp.quad_over_lin(x, z)

        # x is a composition — fails
        with pytest.raises(ValueError, match="x.*must be a plain Variable"):
            sp.quad_over_lin(sp.sin(x), z)

        # z is a composition — fails
        with pytest.raises(ValueError, match="z.*must be a plain"):
            sp.quad_over_lin(x, sp.exp(z))

    def test_quad_over_lin_z_not_in_x(self, scope):
        z = scope.Variable(1, 1)
        # z used as both args
        with pytest.raises(ValueError, match="z must not appear in x"):
            sp.quad_over_lin(z, z)

    def test_prod_must_be_variable(self, scope):
        x = scope.Variable(3, 1)
        # This should work — x is a plain variable
        sp.prod(x)

        # This should fail — argument is a composition
        with pytest.raises(ValueError, match="plain Variable"):
            sp.prod(sp.sin(x))

    def test_prod_axis_must_be_variable(self, scope):
        X = scope.Variable(3, 2)
        sp.prod(X, axis=0)
        sp.prod(X, axis=1)

        with pytest.raises(ValueError, match="plain Variable"):
            sp.prod(sp.sin(X), axis=0)


# ---------------------------------------------------------------------------
# Wrong-size value assignment
# ---------------------------------------------------------------------------

class TestValueAssignment:
    def test_variable_wrong_size(self, scope):
        x = scope.Variable(3, 1)
        with pytest.raises(ValueError, match="expected 3 elements"):
            x.value = np.array([1.0, 2.0])

    def test_variable_too_many(self, scope):
        x = scope.Variable(3, 1)
        with pytest.raises(ValueError, match="expected 3 elements"):
            x.value = np.array([1.0, 2.0, 3.0, 4.0])

    def test_parameter_wrong_size(self, scope):
        p = scope.Parameter(2, 2)
        p.value = np.eye(2)
        with pytest.raises(ValueError, match="expected 4 elements"):
            p.value = np.array([1.0, 2.0])

    def test_scope_set_values_wrong_size(self, scope):
        x = scope.Variable(3, 1)
        with pytest.raises(ValueError, match="expected 3 elements"):
            scope.set_values(np.array([1.0, 2.0]))

    def test_parameter_unset_value_is_none(self, scope):
        p = scope.Parameter(2, 2)
        assert p.value is None

    def test_parameter_unset_raises_on_eval(self, scope):
        x = scope.Variable(3, 1)
        A = scope.Parameter(3, 3)
        f = A @ x
        fn = sp.compile(f)
        x.value = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="has no value set"):
            fn.forward()


# ---------------------------------------------------------------------------
# Mixed scopes
# ---------------------------------------------------------------------------

class TestMixedScopes:
    def test_mixed_scopes_raises(self):
        scope1 = sp.Scope()
        scope2 = sp.Scope()
        x = scope1.Variable(3, 1)
        y = scope2.Variable(3, 1)
        f = x + y
        with pytest.raises(ValueError, match="same Scope"):
            sp.compile(f)


# ---------------------------------------------------------------------------
# Invalid shape dimensions
# ---------------------------------------------------------------------------

class TestInvalidShapes:
    def test_variable_zero_dim(self, scope):
        with pytest.raises(ValueError, match="positive"):
            scope.Variable(0, 1)

    def test_variable_negative_dim(self, scope):
        with pytest.raises(ValueError, match="positive"):
            scope.Variable(-1, 3)

    def test_parameter_zero_dim(self, scope):
        with pytest.raises(ValueError, match="positive"):
            scope.Parameter(3, 0)


# ---------------------------------------------------------------------------
# Index out of bounds
# ---------------------------------------------------------------------------

class TestIndexOutOfBounds:
    def test_scalar_index_out_of_range(self, scope):
        x = scope.Variable(3, 1)
        with pytest.raises(IndexError, match="out of range"):
            x[5]

    def test_negative_index_out_of_range(self, scope):
        x = scope.Variable(3, 1)
        with pytest.raises(IndexError, match="out of range"):
            x[-4]

    def test_fancy_index_out_of_range(self, scope):
        x = scope.Variable(3, 1)
        with pytest.raises(IndexError, match="out of range"):
            x[[0, 5]]

    def test_matrix_index_out_of_range(self, scope):
        X = scope.Variable(3, 2)
        with pytest.raises(IndexError, match="out of range"):
            X[5, 0]

    def test_matrix_col_out_of_range(self, scope):
        X = scope.Variable(3, 2)
        with pytest.raises(IndexError, match="out of range"):
            X[0, 3]
