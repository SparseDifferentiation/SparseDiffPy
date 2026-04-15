"""Tests for addition with all broadcast combinations."""

import numpy as np
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_point


# --- Scalar + matrix broadcast ---

def test_add_scalar_plus_matrix_jacobian(scope, rng):
    a = scope.Variable(1, 1)
    B = scope.Variable(3, 2)
    f = a + B
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_add_scalar_plus_matrix_hessian(scope, rng):
    a = scope.Variable(1, 1)
    B = scope.Variable(3, 2)
    f = a + B
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    x0 = random_point(scope, rng)
    checker.check_hessian(x0, rng.standard_normal(6))


# --- Column + matrix broadcast ---

def test_add_column_plus_matrix_jacobian(scope, rng):
    x = scope.Variable(3, 1)
    Y = scope.Variable(3, 2)
    f = x + Y
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_add_column_plus_matrix_hessian(scope, rng):
    x = scope.Variable(3, 1)
    Y = scope.Variable(3, 2)
    f = x + Y
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    x0 = random_point(scope, rng)
    checker.check_hessian(x0, rng.standard_normal(6))


# --- Row + matrix broadcast ---

def test_add_row_plus_matrix_jacobian(scope, rng):
    r = scope.Variable(1, 2)
    Y = scope.Variable(3, 2)
    f = r + Y
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_add_row_plus_matrix_hessian(scope, rng):
    r = scope.Variable(1, 2)
    Y = scope.Variable(3, 2)
    f = r + Y
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    x0 = random_point(scope, rng)
    checker.check_hessian(x0, rng.standard_normal(6))


# --- Same shape (no broadcast) ---

def test_add_same_shape_jacobian(scope, rng):
    X = scope.Variable(3, 2)
    Y = scope.Variable(3, 2)
    f = X + Y
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


# --- Scalar constant broadcast ---

def test_add_constant_scalar_jacobian(scope, rng):
    x = scope.Variable(3, 1)
    f = x + 1.0
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_add_constant_scalar_forward(scope, rng):
    x = scope.Variable(3, 1)
    f = x + 2.5
    fn = sp.compile(f)
    x0 = random_point(scope, rng)
    np.testing.assert_allclose(fn.forward(), x0 + 2.5)


# --- NumPy array constant broadcast ---

def test_add_numpy_array_jacobian(scope, rng):
    x = scope.Variable(3, 1)
    b = np.array([1.0, 2.0, 3.0])
    f = b + x
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_add_numpy_array_forward(scope, rng):
    x = scope.Variable(3, 1)
    b = np.array([10.0, 20.0, 30.0])
    f = x + b
    fn = sp.compile(f)
    x0 = random_point(scope, rng)
    np.testing.assert_allclose(fn.forward(), x0 + b)


# --- Column vectors ---

def test_add_column_vectors_jacobian(scope, rng):
    x = scope.Variable(4, 1)
    y = scope.Variable(4, 1)
    f = x + y
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))
