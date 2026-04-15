import numpy as np
import pytest
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_positive_point


# --- Vector-vector (both same shape) ---

def test_rel_entr_vector_jacobian(scope, rng):
    x = scope.Variable(3, 1)
    y = scope.Variable(3, 1)
    f = sp.rel_entr(x, y)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_positive_point(scope, rng))


def test_rel_entr_vector_hessian(scope, rng):
    x = scope.Variable(3, 1)
    y = scope.Variable(3, 1)
    f = sp.rel_entr(x, y)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    x0 = random_positive_point(scope, rng)
    checker.check_hessian(x0, rng.standard_normal(3))


def test_rel_entr_vector_forward(scope, rng):
    x = scope.Variable(3, 1)
    y = scope.Variable(3, 1)
    f = sp.rel_entr(x, y)
    fn = sp.compile(f)
    x0 = random_positive_point(scope, rng)
    x_val = x.value
    y_val = y.value
    expected = x_val * np.log(x_val / y_val)
    np.testing.assert_allclose(fn.forward(), expected)


# --- Scalar x, vector y ---

def test_rel_entr_scalar_vector_jacobian(scope, rng):
    x = scope.Variable(1, 1)
    y = scope.Variable(3, 1)
    f = sp.rel_entr(x, y)
    assert f.shape == (3, 1)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_positive_point(scope, rng))


def test_rel_entr_scalar_vector_hessian(scope, rng):
    x = scope.Variable(1, 1)
    y = scope.Variable(3, 1)
    f = sp.rel_entr(x, y)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    x0 = random_positive_point(scope, rng)
    checker.check_hessian(x0, rng.standard_normal(3))


def test_rel_entr_scalar_vector_forward(scope, rng):
    x = scope.Variable(1, 1)
    y = scope.Variable(3, 1)
    f = sp.rel_entr(x, y)
    fn = sp.compile(f)
    x0 = random_positive_point(scope, rng)
    x_val = x.value[0]
    y_val = y.value
    expected = x_val * np.log(x_val / y_val)
    np.testing.assert_allclose(fn.forward(), expected)


# --- Vector x, scalar y ---

def test_rel_entr_vector_scalar_jacobian(scope, rng):
    x = scope.Variable(3, 1)
    y = scope.Variable(1, 1)
    f = sp.rel_entr(x, y)
    assert f.shape == (3, 1)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_positive_point(scope, rng))


def test_rel_entr_vector_scalar_hessian(scope, rng):
    x = scope.Variable(3, 1)
    y = scope.Variable(1, 1)
    f = sp.rel_entr(x, y)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    x0 = random_positive_point(scope, rng)
    checker.check_hessian(x0, rng.standard_normal(3))


def test_rel_entr_vector_scalar_forward(scope, rng):
    x = scope.Variable(3, 1)
    y = scope.Variable(1, 1)
    f = sp.rel_entr(x, y)
    fn = sp.compile(f)
    x0 = random_positive_point(scope, rng)
    x_val = x.value
    y_val = y.value[0]
    expected = x_val * np.log(x_val / y_val)
    np.testing.assert_allclose(fn.forward(), expected)


# --- Shape mismatch ---

def test_rel_entr_incompatible_shapes(scope):
    x = scope.Variable(3, 1)
    y = scope.Variable(2, 1)
    with pytest.raises(ValueError, match="shapes must match or one must be scalar"):
        sp.rel_entr(x, y)
