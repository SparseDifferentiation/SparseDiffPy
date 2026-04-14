import numpy as np
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_positive_point


def test_entr_vector_forward(scope, rng):
    x = scope.Variable(4, 1)
    f = sp.entr(x)
    fn = sp.compile(f)
    x0 = random_positive_point(scope, rng)
    np.testing.assert_allclose(fn.forward(), -x0 * np.log(x0))


def test_entr_vector_jacobian(scope, rng):
    x = scope.Variable(4, 1)
    f = sp.entr(x)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_positive_point(scope, rng))


def test_entr_vector_hessian(scope, rng):
    x = scope.Variable(4, 1)
    f = sp.entr(x)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    x0 = random_positive_point(scope, rng)
    checker.check_hessian(x0, rng.standard_normal(4))
