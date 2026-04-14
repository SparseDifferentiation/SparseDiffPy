import numpy as np
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_point


def test_sinh_vector_jacobian(scope, rng):
    x = scope.Variable(4, 1)
    f = sp.sinh(x)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_sinh_vector_hessian(scope, rng):
    x = scope.Variable(4, 1)
    f = sp.sinh(x)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    x0 = random_point(scope, rng)
    checker.check_hessian(x0, rng.standard_normal(4))
