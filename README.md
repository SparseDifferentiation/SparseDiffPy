# SparseDiffPy

Python library for computing **sparse Jacobians** and **Hessians** of nonlinear expressions via automatic differentiation.

SparseDiffPy wraps [SparseDiffEngine](https://github.com/SparseDifferentiation/SparseDiffEngine), a C library that exploits the sparsity structure of expression graphs to compute derivatives efficiently. Instead of building dense Jacobian and Hessian matrices, SparseDiffPy analyzes the expression graph at compile time to determine which entries are structurally nonzero, then computes only those entries. The results are returned as `scipy.sparse.csr_matrix` objects, ready for use in optimization solvers.

This matters for large-scale nonlinear optimization, where the Jacobian and Hessian are typically very sparse — a variable in one constraint rarely affects all other constraints. Computing full dense matrices wastes both time and memory.

## Installation

```bash
pip install sparsediffpy
```

Requires Python >= 3.11 and NumPy >= 2.0.0.

### Building from source

```bash
git clone https://github.com/SparseDifferentiation/SparseDiffPy.git
cd SparseDiffPy
git submodule update --init
pip install -e .
```

## Quick start

```python
import sparsediffpy as sp
import numpy as np

# 1. Create a scope (owns the variable space)
scope = sp.Scope()

# 2. Declare variables (always 2D: rows, cols)
x = scope.Variable(3, 1)   # column vector with 3 elements

# 3. Build an expression using operators and functions
f = sp.exp(x) + 2.0 * x

# 4. Compile (analyzes sparsity — do this once)
fn = sp.compile(f)

# 5. Set variable values and evaluate
x.value = np.array([1.0, 2.0, 3.0])

fn.forward()                        # array([4.71828183, 11.3890561, 26.08553692])
fn.jacobian()                       # 3x3 sparse CSR matrix
fn.hessian(weights=np.ones(3))      # 3x3 sparse CSR matrix
```

## Core concepts

### Scope, Variables, and Parameters

A **Scope** owns the flat variable buffer. All variables must belong to a scope.

```python
scope = sp.Scope()
x = scope.Variable(3, 1)       # 3x1 column vector
Y = scope.Variable(2, 3)       # 2x3 matrix
a = scope.Variable(1, 1)       # scalar
```

**Parameters** are fixed data that can be updated without recompiling:

```python
A = scope.Parameter(3, 3, value=np.eye(3))

f = A @ x
fn = sp.compile(f)

x.value = np.array([1.0, 2.0, 3.0])
fn.jacobian()       # uses A = eye(3)

A.value = 2 * np.eye(3)
fn.jacobian()       # uses A = 2*eye(3), no recompile needed
```

### Building expressions

Use Python operators and named functions. NumPy arrays and scalars auto-convert to constants:

```python
x = scope.Variable(3, 1)
A = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
b = np.array([1.0, 2.0, 3.0])

f = A @ x + b                # linear
g = sp.sin(x) + sp.exp(x)    # nonlinear elementwise
h = x.T @ x                  # quadratic (scalar output)
```

Broadcasting follows NumPy/CVXPY conventions:

```python
a = scope.Variable(1, 1)     # scalar
X = scope.Variable(3, 2)     # matrix
r = scope.Variable(1, 2)     # row vector
c = scope.Variable(3, 1)     # column vector

f = a * X + r + c            # scalar, row, and column all broadcast to (3, 2)
```

### Compiling and evaluating

`sp.compile(f)` analyzes the sparsity pattern of the expression (this is the expensive step — do it once). The returned object is cheap to evaluate repeatedly:

```python
fn = sp.compile(f)

x.value = some_values
fn.forward()                     # evaluate f(x)
fn.jacobian()                    # sparse Jacobian df/dx as csr_matrix
fn.hessian(weights=w)            # sparse Hessian of w^T f(x) as csr_matrix
```

For a vector-valued function `f: R^n -> R^m`, the **Jacobian** is the `m x n` matrix of partial derivatives. The **Hessian** requires a weight vector `w` of length `m` — it computes the `n x n` Hessian of the scalar function `w^T f(x)`.

### Solver integration

In a solver loop, use `scope.set_values()` to write the entire flat variable vector at once:

```python
scope = sp.Scope()
x = scope.Variable(3, 1)
f = sp.sin(x)
fn = sp.compile(f)

def eval_f(x_flat):
    scope.set_values(x_flat)
    return fn.forward()

def eval_jac(x_flat):
    scope.set_values(x_flat)
    return fn.jacobian()

def eval_hess(x_flat, weights):
    scope.set_values(x_flat)
    return fn.hessian(weights)

# These functions can be passed directly to scipy.optimize, IPOPT, etc.
```

With multiple variables, the flat vector concatenates them in declaration order:

```python
scope = sp.Scope()
x = scope.Variable(3, 1)    # occupies flat positions [0, 1, 2]
y = scope.Variable(2, 1)    # occupies flat positions [3, 4]

# In a solver callback:
def eval_jac(z_flat):
    scope.set_values(z_flat)   # z_flat has length 5
    return fn.jacobian()       # returns a sparse matrix with 5 columns
```

## Supported operations

### Arithmetic operators

| Operator | Description |
|---|---|
| `x + y` | Addition (with broadcasting) |
| `x - y` | Subtraction (with broadcasting) |
| `-x` | Negation |
| `x * y` | Elementwise multiplication (with broadcasting) |
| `x @ y` | Matrix multiplication |
| `x ** p` | Power (constant exponent) |
| `x[i]`, `x[0:3]`, `x[[0, 2]]` | Indexing and slicing |
| `x.T` | Transpose |

### Elementwise functions

| Function | Description | Domain |
|---|---|---|
| `sp.exp(x)` | Exponential | all reals |
| `sp.sin(x)` | Sine | all reals |
| `sp.cos(x)` | Cosine | all reals |
| `sp.tan(x)` | Tangent | x != pi/2 + k*pi |
| `sp.sinh(x)` | Hyperbolic sine | all reals |
| `sp.tanh(x)` | Hyperbolic tangent | all reals |
| `sp.asinh(x)` | Inverse hyperbolic sine | all reals |
| `sp.atanh(x)` | Inverse hyperbolic tangent | \|x\| < 1 |
| `sp.log(x)` | Natural logarithm | x > 0 |
| `sp.logistic(x)` | Softplus: log(1 + exp(x)) | all reals |
| `sp.normal_cdf(x)` | Standard normal CDF | all reals |
| `sp.entr(x)` | Entropy: -x log(x) | x > 0 |
| `sp.xexp(x)` | x exp(x) | all reals |
| `sp.power(x, p)` | Power with float exponent | depends on p |

### Reductions

| Function | Description |
|---|---|
| `sp.sum(x)` | Sum all elements |
| `sp.sum(x, axis=0)` | Sum along rows |
| `sp.sum(x, axis=1)` | Sum along columns |
| `sp.prod(x)` | Product of all elements |
| `sp.prod(x, axis=0)` | Product along rows |
| `sp.prod(x, axis=1)` | Product along columns |
| `sp.trace(X)` | Matrix trace (square matrices) |

### Structural operations

| Function | Description |
|---|---|
| `sp.hstack([a, b, c])` | Horizontal concatenation (same row count) |
| `sp.vstack([a, b, c])` | Vertical concatenation (same column count) |
| `sp.reshape(x, d1, d2)` | Reshape (preserves total size) |
| `sp.diag_vec(x)` | Diagonal matrix from column vector |

### Special functions

| Function | Description |
|---|---|
| `sp.quad_form(x, Q)` | Quadratic form: x^T Q x |
| `sp.quad_over_lin(x, z)` | sum(x^2) / z |
| `sp.rel_entr(x, y)` | Relative entropy: x log(x/y) |

## Shapes

All shapes are 2D tuples `(rows, cols)`, matching the underlying C engine. There is no 1D shorthand — use `Variable(3, 1)` for a column vector, `Variable(1, 3)` for a row vector:

```python
x = scope.Variable(3, 1)     # column vector
r = scope.Variable(1, 3)     # row vector
M = scope.Variable(3, 3)     # matrix
a = scope.Variable(1, 1)     # scalar
```

## License

Apache License 2.0
