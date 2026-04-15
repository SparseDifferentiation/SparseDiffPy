import numpy as np
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_point


def test_sum_all_jacobian(scope, rng):
    x = scope.Variable(3, 2)
    f = sp.sum(x)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_sum_axis0_jacobian(scope, rng):
    x = scope.Variable(3, 2)
    f = sp.sum(x, axis=0)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_sum_axis1_jacobian(scope, rng):
    x = scope.Variable(3, 2)
    f = sp.sum(x, axis=1)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_sum_vector_jacobian(scope, rng):
    x = scope.Variable(4, 1)
    f = sp.sum(x)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))
