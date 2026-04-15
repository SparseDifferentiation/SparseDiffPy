import numpy as np
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_point


def test_trace_jacobian(scope, rng):
    X = scope.Variable(3, 3)
    f = sp.trace(X)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_trace_hessian(scope, rng):
    X = scope.Variable(3, 3)
    f = sp.trace(X)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    x0 = random_point(scope, rng)
    checker.check_hessian(x0, np.array([1.0]))
