"""Elementwise named functions: sp.sin, sp.exp, sp.log, etc.

_UnaryOp.__init__ handles _wrap_constant, so these are one-liners.
"""

from sparsediffpy._core._nodes_elementwise import (
    Asinh, Atanh, Cos, Entr, Exp, Log, Logistic, NormalCdf, Power,
    Sin, Sinh, Tan, Tanh, Xexp,
)


def sin(x):
    return Sin(x)

def cos(x):
    return Cos(x)

def exp(x):
    return Exp(x)

def log(x):
    return Log(x)

def tan(x):
    return Tan(x)

def sinh(x):
    return Sinh(x)

def tanh(x):
    return Tanh(x)

def asinh(x):
    return Asinh(x)

def atanh(x):
    return Atanh(x)

def logistic(x):
    return Logistic(x)

def normal_cdf(x):
    return NormalCdf(x)

def entr(x):
    return Entr(x)

def xexp(x):
    return Xexp(x)

def power(x, p):
    return Power(x, float(p))
