"""Merge scalar terms written in a Python loop into vector expressions.

A loop such as ``[x[k] ** 2 + x[k + 1] - 1 for k in range(n - 1)]`` gives the
engine one small expression per term. :func:`compact` vectorizes them in the
spirit of LLVM's SLP vectorizer: terms with the same *template* (isomorphic
trees) are bundled, each term is one *lane*, and the bundle is rebuilt once as
a vector expression. As in LLVM's MergeFunctions, the operands that differ
between terms -- the index of ``x[i]`` and scalar constants -- become the
template's *params*; each term supplies its own *args*, which are gathered into
an index array or a constant vector.
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
            key, args = _template(t, leaves)
        except _Unsupported:
            key, args = ("single", i), None
        groups.setdefault(key, []).append((i, t, args))

    parts, perm = [], []
    for key, members in groups.items():
        perm += [i for i, _, _ in members]
        parts.append(members[0][1] if len(members) == 1 else
                     _vectorize(key, [args for _, _, args in members], leaves))
    expr = parts[0] if len(parts) == 1 else _atoms.vstack(parts)
    perm = None if perm == list(range(len(perm))) else np.array(perm)
    return expr, perm


def _template(term, leaves):
    """Describe a scalar term as ``(template, args)``.

    The template lists the term's nodes in post-order, like instructions in
    SSA form: each entry names the operation and refers to its operands by
    their position in the list, so a subexpression shared inside the term is
    part of the template. ``("gather", id(x))`` and ``("const",)`` entries are
    params; ``args`` holds this term's value for each (the index ``i`` of
    ``x[i]``, the constant's value), in template order.
    """
    entries, args, seen = [], [], {}

    def visit(e):
        if id(e) in seen:
            return seen[id(e)]
        if e.shape != (1, 1):
            raise _Unsupported
        if isinstance(e, Constant):
            ent = ("const",)
            args.append(float(e.value[0, 0]))
        elif isinstance(e, (Variable, Parameter)):
            leaves[id(e)] = e
            ent = ("leaf", id(e))
        elif not isinstance(e, Node):
            raise _Unsupported
        elif e.op == "index" and isinstance(e.args[0], Variable):
            leaves[id(e.args[0])] = e.args[0]
            ent = ("gather", id(e.args[0]))
            args.append(int(e.attrs["idx"][0]))
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
    return tuple(entries), args


def _vectorize(template, args, leaves):
    """Evaluate the template once on (lanes, 1) vectors, one lane per term.

    ``args[lane]`` is that term's args; transposed, each param gets one value
    per lane.
    """
    lanes = len(args)
    params = iter(zip(*args))
    vals = []
    for ent in template:
        op = ent[0]
        if op == "gather":
            vals.append(index_flat(leaves[ent[1]], next(params), (lanes, 1)))
        elif op == "const":
            c = np.array(next(params))
            vals.append(Constant(c[0] if np.all(c == c[0])
                                 else c.reshape(lanes, 1)))
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
    # a template without params (every term is e.g. the scalar y) is a scalar
    return broadcast(vals[-1], (lanes, 1))
