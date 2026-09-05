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
default builds. There is no 3.13t wheel because NumPy 2.5 and later ship no
`cp313t` wheels. The extension declares that it does not need the GIL:
the engine keeps no global mutable state, every problem and expression owns its
own buffers, and all inputs and outputs are copied at the boundary. Distinct
problems can therefore be built and evaluated concurrently from different
threads.

A single problem or expression capsule is not thread-safe. Do not call into the
same problem from two threads at once, and do not share expression capsules
between problems that are evaluated concurrently. This is the same contract as
under the GIL, which only serialized individual calls and never protected
against interleaved use of one object.

## License

Apache License 2.0
