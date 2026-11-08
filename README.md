# SparseDiffPy

Sparse Jacobians and Hessians of nonlinear expressions, computed by
[SparseDiffEngine](https://github.com/SparseDifferentiation/SparseDiffEngine),
a C library that works out the sparsity pattern once and then only computes
the structural nonzeros. Results are `scipy.sparse.csr_matrix`.

## Installation

```bash
pip install sparsediffpy
```

From source: `git clone --recursive https://github.com/SparseDifferentiation/SparseDiffPy && pip install -e SparseDiffPy`.

## Quick start

```python
import numpy as np
import sparsediffpy as sp

scope = sp.Scope()             # owns the flat variable vector
x = scope.Variable(3, 1)       # shapes are always 2-D: (rows, cols)

f = sp.exp(x) + 2.0 * x        # build an expression with operators and sp.* atoms
fn = sp.compile(f)             # analyze sparsity once

x.value = [1.0, 2.0, 3.0]
print(fn.forward())            # [[4.718] [11.389] [26.086]], shaped like f
print(fn.jacobian().toarray()) # diag(exp(x) + 2), a 3x3 csr_matrix
print(fn.hessian(np.ones(3)))  # Hessian of w' f(x) for weights w
```

`forward()` has the shape of `f`, `jacobian()` has shape `(f.size, n_vars)`
and `hessian(w)` has shape `(n_vars, n_vars)`, where `n_vars` is the total
size of all variables in the scope, in declaration order. Flat vectors (the
rows of the Jacobian, `x.value = [...]`, the weights `w`) are column-major.

**Parameters** are data you can change without recompiling:

```python
import numpy as np
import sparsediffpy as sp

scope = sp.Scope()
x = scope.Variable(3, 1)
A = scope.Parameter(2, 3, value=np.ones((2, 3)))
fn = sp.compile(sp.sin(A @ x))

x.value = [0.1, 0.2, 0.3]
J1 = fn.jacobian()
A.value = np.arange(6.0).reshape(2, 3)   # no recompile needed
J2 = fn.jacobian()
```

## Using it in a solver

`sp.Problem(objective, constraints)` exposes the callbacks NLP solvers ask for:
objective value and gradient, constraint values and Jacobian, and the Lagrangian
Hessian `obj_factor * H_f + sum_i lagrange_i * H_gi`. A stage runs in the engine
only when its point changed: a forward called again with the same `u` (or with
unchanged scope values and parameters) returns the cached result, and
`gradient()`/`jacobian()` are reused until the matching forward moves. Returned
arrays are owned by the problem, so do not modify them in place. Here it drives SciPy:

```python
import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

import sparsediffpy as sp

# minimize  sum(exp(x) - c * x)   subject to   ||x||^2 <= 1
scope = sp.Scope()
x = scope.Variable(4, 1)
c = np.array([[1.0], [2.0], [3.0], [4.0]])   # constants are 2-D, like variables
prob = sp.Problem(sp.sum(sp.exp(x) - c * x), [sp.sum(x ** 2)])

def grad(u):
    prob.objective_forward(u)
    return prob.gradient()

def obj_hess(u):  # the Lagrangian Hessian needs both forwards at u (cached if done)
    prob.objective_forward(u); prob.constraint_forward(u)
    return prob.hessian(1.0, [0.0])

def con_jac(u):
    prob.constraint_forward(u)
    return prob.jacobian()

def con_hess(u, v):
    prob.objective_forward(u); prob.constraint_forward(u)
    return prob.hessian(0.0, v)

con = NonlinearConstraint(prob.constraint_forward, -np.inf, 1.0,
                          jac=con_jac, hess=con_hess)
res = minimize(prob.objective_forward, np.zeros(4), jac=grad, hess=obj_hess,
               method="trust-constr", constraints=con)
print(res.x)  # approx. [0. 0.289 0.549 0.784], on the boundary ||x|| = 1
```

## Supported operations

- **Operators:** `+ - * @ **`, `.T`, and NumPy-style indexing (`x[1:3]`, `X[0, :]`, `X[[0, 2], 1]`).
  Scalars, rows and columns broadcast. NumPy arrays and `scipy.sparse` matrices are accepted as
  constants; like every expression they must be 2-D, so pass `a[:, None]` or `a[None, :]` for a 1-D `a`.
- **Elementwise:** `sin cos tan exp log sinh tanh asinh atan atanh logistic normal_cdf entr xexp power atan2`.
- **Affine:** `sum prod reshape broadcast hstack vstack trace diag_vec index_flat`.
- **Other:** `quad_form quad_over_lin rel_entr`.

`log`, `tan`, `atanh`, `entr`, `atan2`, `rel_entr`, `quad_over_lin` and `prod`
currently need plain Variables as arguments (an engine limitation).

The low-level C bindings remain available as `sparsediffpy._sparsediffengine`.

## Free-threaded Python

Wheels are published for free-threaded CPython 3.14 (`cp314t`) as well as the
default builds. There is no `cp313t` wheel because NumPy ships none from 2.5
onwards, so that build would have to compile NumPy from source.

The extension declares that it does not need the GIL, so importing it leaves
free threading enabled. Distinct problems can be built and evaluated
concurrently from different threads: the engine keeps no mutable global state,
every problem and expression owns its own buffers, and all inputs and outputs
are copied at the boundary. This assumes a default build, since
`SP_TRACK_MEMORY` adds global allocation counters that are not thread-safe.

A single problem or expression capsule is **not** thread-safe. Do not call into
the same problem from two threads at once, and do not share expression capsules
between problems that are evaluated concurrently. Note that `expr::refcount` is
a plain `int`, so breaking this rule corrupts the heap rather than merely
interleaving calls, as it would under the GIL.

## License

Apache License 2.0
