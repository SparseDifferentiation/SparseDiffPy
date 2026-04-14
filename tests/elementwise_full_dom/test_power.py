import numpy as np
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_positive_point


def test_power_squared_jacobian(scope, rng):
    x = scope.Variable(4, 1)
    f = x ** 2
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_positive_point(scope, rng))


def test_power_squared_hessian(scope, rng):
    x = scope.Variable(4, 1)
    f = x ** 2
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    x0 = random_positive_point(scope, rng)
    checker.check_hessian(x0, rng.standard_normal(4))


def test_power_cubed_jacobian(scope, rng):
    x = scope.Variable(3, 1)
    f = x ** 3
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_positive_point(scope, rng))


def test_power_half_jacobian(scope, rng):
    x = scope.Variable(3, 1)
    f = sp.power(x, 0.5)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_positive_point(scope, rng))


def test_power_half_hessian(scope, rng):
    x = scope.Variable(3, 1)
    f = sp.power(x, 0.5)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    x0 = random_positive_point(scope, rng)
    checker.check_hessian(x0, rng.standard_normal(3))
