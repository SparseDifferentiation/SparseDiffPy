"""Merge scalar terms written in a Python loop into vector expressions.

A loop such as ``[x[k] ** 2 + x[k + 1] - 1 for k in range(n - 1)]`` gives the
engine one small expression per term. The terms share a *template*: the same
operations, differing only in which entries of a variable and which scalar
constants they use. :func:`compact` rebuilds each template once as a vector
expression, with the varying parts gathered into index arrays and constants.
"""

import numpy as np

from sparsediffpy import _atoms
from sparsediffpy._expression import (
    Constant, Node, Parameter, Variable, _wrap, broadcast, index_flat,
)

# Unary atoms that accept any argument (restricted ones need plain Variables).
_UNARY = {"sin", "cos", "exp", "sinh", "tanh", "asinh", "atan", "logistic",
          "normal_cdf", "xexp"}


class _Unsupported(Exception):
    pass


def compact(terms):
    """Stack scalar expressions like :func:`vstack`, merging shared templates.

    Returns ``(expr, perm)``: ``expr`` has shape ``(len(terms), 1)`` and its
    row ``j`` is ``terms[perm[j]]``, or ``perm`` is ``None`` if no row moved.
    Terms using other operations (matmul, sum, ...) are kept as they are.
    """
    if not terms:
        raise ValueError("compact: need at least one term")
    leaves, groups = {}, {}  # leaves: id -> Variable/Parameter used by templates
    for i, t in enumerate(terms):
        t = _wrap(t)
        if t.shape != (1, 1):
            raise ValueError(f"compact: term {i} has shape {t.shape}, not (1, 1)")
        try:
            key, holes = _template(t, leaves)
        except _Unsupported:
            key, holes = ("single", i), None
        groups.setdefault(key, []).append((i, t, holes))

    parts, perm = [], []
    for key, members in groups.items():
        perm += [i for i, _, _ in members]
        parts.append(members[0][1] if len(members) == 1 else
                     _rebuild(key, [holes for _, _, holes in members], leaves))
    expr = parts[0] if len(parts) == 1 else _atoms.vstack(parts)
    perm = None if perm == list(range(len(perm))) else np.array(perm)
    return expr, perm


def _template(term, leaves):
    """Describe a scalar term as ``(template, holes)``.

    The template lists the term's nodes in post-order. Each entry names the
    operation and refers to its children by their position in the list, so a
    subexpression shared inside the term is part of the template. Holes are
    what varies between terms -- the entry index of ``x[i]`` and the value of a
    scalar constant -- in the order they are met.
    """
    entries, holes, seen = [], [], {}

    def visit(e):
        if id(e) in seen:
            return seen[id(e)]
        if e.shape != (1, 1):
            raise _Unsupported
        if isinstance(e, Constant):
            ent = ("const",)
            holes.append(float(e.value[0, 0]))
        elif isinstance(e, (Variable, Parameter)):
            leaves[id(e)] = e
            ent = ("leaf", id(e))
        elif not isinstance(e, Node):
            raise _Unsupported
        elif e.op == "index" and isinstance(e.args[0], Variable):
            leaves[id(e.args[0])] = e.args[0]
            ent = ("var", id(e.args[0]))
            holes.append(int(e.attrs["idx"][0]))
        elif e.op in ("add", "multiply", "scalar_mult"):
            ent = (e.op, visit(e.args[0]), visit(e.args[1]))
        elif e.op == "neg" or e.op in _UNARY:
            ent = (e.op, visit(e.args[0]))
        elif e.op == "power":
            ent = ("power", e.attrs["p"], visit(e.args[0]))
        else:
            raise _Unsupported
        seen[id(e)] = len(entries)
        entries.append(ent)
        return seen[id(e)]

    visit(term)
    return tuple(entries), holes


def _rebuild(template, holes, leaves):
    """Evaluate the template once on (K, 1) vectors, one row per term."""
    K = len(holes)
    columns = iter(zip(*holes))  # the K values of each hole, in template order
    vals = []
    for ent in template:
        op = ent[0]
        if op == "var":
            vals.append(index_flat(leaves[ent[1]], next(columns), (K, 1)))
        elif op == "const":
            c = np.array(next(columns))
            vals.append(Constant(c[0] if np.all(c == c[0]) else c.reshape(K, 1)))
        elif op == "leaf":
            vals.append(leaves[ent[1]])
        elif op == "add":
            vals.append(vals[ent[1]] + vals[ent[2]])
        elif op in ("multiply", "scalar_mult"):
            vals.append(vals[ent[1]] * vals[ent[2]])
        elif op == "neg":
            vals.append(-vals[ent[1]])
        elif op == "power":
            vals.append(vals[ent[2]] ** ent[1])
        else:
            vals.append(getattr(_atoms, op)(vals[ent[1]]))
    # a template without holes (every term is e.g. the scalar y) builds a scalar
    return broadcast(vals[-1], (K, 1))
