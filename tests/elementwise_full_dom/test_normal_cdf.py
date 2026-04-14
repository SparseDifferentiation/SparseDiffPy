import numpy as np
from scipy.stats import norm
import sparsediffpy as sp
from tests.utils import NumericalDerivativeChecker, random_point


def test_normal_cdf_vector_jacobian(scope, rng):
    x = scope.Variable(4, 1)
    f = sp.normal_cdf(x)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    checker.check_jacobian(random_point(scope, rng))


def test_normal_cdf_vector_hessian(scope, rng):
    x = scope.Variable(4, 1)
    f = sp.normal_cdf(x)
    fn = sp.compile(f)
    checker = NumericalDerivativeChecker(fn, scope)
    x0 = random_point(scope, rng)
    checker.check_hessian(x0, rng.standard_normal(4))


def test_normal_cdf_forward(scope, rng):
    x = scope.Variable(3, 1)
    f = sp.normal_cdf(x)
    fn = sp.compile(f)
    x0 = random_point(scope, rng)
    np.testing.assert_allclose(fn.forward(), norm.cdf(x0), rtol=1e-6)
