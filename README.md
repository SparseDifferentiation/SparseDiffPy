# SparseDiffPy

Python bindings for [SparseDiffEngine](https://github.com/SparseDifferentiation/SparseDiffEngine), a C library for computing sparse Jacobians and Hessians.

## Installation

```bash
pip install sparsediffpy
```

## Usage

```python
from sparsediffpy import _sparsediffengine
```

## Free-threaded Python

Wheels are published for free-threaded CPython 3.14 (`cp314t`) as well as the
default builds. There is no `cp313t` wheel: NumPy ships none from 2.5 onwards
(2.4.6 is the last release with one), so a 3.13t build would have to compile
NumPy from source.

The extension declares that it does not need the GIL, so importing it leaves
free threading enabled. Distinct problems can be built and evaluated
concurrently from different threads: the engine keeps no mutable global state,
every problem and expression owns its own buffers, and all inputs and outputs
are copied at the boundary.

This depends on the engine being built without `SP_TRACK_MEMORY`, which is the
default. That option makes every allocation update the `g_allocated_bytes` and
`g_peak_bytes` process globals non-atomically, so two threads that merely
allocate race on them. It is a development switch; do not turn it on for a
wheel build.

A single problem or expression capsule is **not** thread-safe. Do not call into
the same problem from two threads at once, and do not share expression capsules
between problems that are evaluated concurrently.

Without the GIL, breaking that rule is worse than it used to be. `expr::refcount`
is a plain `int` that `expr_retain()` and `free_expr()` update non-atomically,
and capsule destructors run on whichever thread drops the last Python reference.
So sharing one capsule across threads corrupts the heap instead of merely
interleaving calls: an eight-thread loop building and dropping `exp()` nodes over
one shared variable node crashes on every run here, while the same loop is
harmless under the GIL. Making that reference count atomic upstream would turn
this back into an ordinary "unsupported usage" rather than a crash.

## License

Apache License 2.0
