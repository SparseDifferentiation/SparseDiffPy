"""
Copyright 2025, the SparseDiffPy developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# C extension (low-level, for advanced users)
from sparsediffpy import _sparsediffengine  # noqa: F401

# Core classes
from sparsediffpy._core._scope import Scope, Variable, Parameter, DimensionError  # noqa: F401
from sparsediffpy._core._expression import Expression  # noqa: F401

# Compile
from sparsediffpy._core._compile import compile  # noqa: F401

# Elementwise functions
from sparsediffpy._core._fn_elementwise import (  # noqa: F401
    sin, cos, exp, log, tan, sinh, tanh, asinh, atanh,
    logistic, normal_cdf, entr, xexp, power,
)

# Affine / structural functions
from sparsediffpy._core._fn_affine import (  # noqa: F401
    diag_vec, trace, reshape, sum, prod, hstack, vstack,
)

# Bivariate / special functions
from sparsediffpy._core._fn_bivariate import (  # noqa: F401
    quad_form, quad_over_lin, rel_entr,
)
