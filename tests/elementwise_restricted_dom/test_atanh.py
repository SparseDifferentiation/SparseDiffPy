import numpy as np
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_point


def test_atanh_vector_jacobian(scope, rng):
    x = scope.Variable(4, 1)
    f = sp.atanh(x)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    # Domain: (-1, 1)
    checker.check_jacobian(random_point(scope, rng, low=-0.8, high=0.8))


def test_atanh_vector_hessian(scope, rng):
    x = scope.Variable(4, 1)
    f = sp.atanh(x)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    x0 = random_point(scope, rng, low=-0.8, high=0.8)
    checker.check_hessian(x0, rng.standard_normal(4))
