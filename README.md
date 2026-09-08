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
