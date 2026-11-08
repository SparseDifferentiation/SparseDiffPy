"""Merge scalar terms written in a Python loop into vector expressions.

A loop such as ``[x[k] ** 2 + x[k + 1] - 1 for k in range(n - 1)]`` gives the
engine one small expression per term. :func:`compact` lowers each term to a
short list of instructions, in the style of LLVM IR::

    %0 = load x        ; arg: the index k
    %1 = power 2.0 %0
    %2 = load x        ; arg: k + 1
    %3 = add %1 %2
    %4 = const         ; arg: 1.0
    %5 = neg %4
    %6 = add %3 %5

Terms whose instructions are identical differ only in their args, so each such
group is rebuilt once as a vector expression: a ``load`` becomes a gather of
``x`` at the args, a ``const`` becomes a constant vector.
"""

import numpy as np

from sparsediffpy import atoms
from sparsediffpy.expression import (
    Constant, Node, Parameter, Variable, cast_to_const, broadcast, index_flat,
)

# Unary atoms that accept any argument (restricted ones need plain Variables).
_UNARY = {"sin", "cos", "exp", "sinh", "tanh", "asinh", "atan", "logistic",
          "normal_cdf", "xexp"}


class _Unsupported(Exception):
    pass


def compact(terms):
    """Stack scalar expressions like :func:`vstack`, merging identical ones.

    Returns ``(expr, perm)``: ``expr`` has shape ``(len(terms), 1)`` and its
    row ``j`` is ``terms[perm[j]]``, or ``perm`` is ``None`` if no row moved.
    Terms using other operations (matmul, sum, ...) are kept as they are.
    """
    if not terms:
        raise ValueError("compact: need at least one term")
    values, groups = {}, {}  # values: id -> Variable/Parameter the terms use
    for i, t in enumerate(terms):
        t = cast_to_const(t)
        if t.shape != (1, 1):
            raise ValueError(f"compact: term {i} has shape {t.shape}, not (1, 1)")
        try:
            insts, args = _lower(t, values)
        except (_Unsupported, RecursionError):  # deep terms are kept as is
            insts, args = ("single", i), None
        groups.setdefault(insts, []).append((i, t, args))

    parts, perm = [], []
    for insts, members in groups.items():
        perm += [i for i, _, _ in members]
        parts.append(members[0][1] if len(members) == 1 else
                     _vectorize(insts, [args for _, _, args in members], values))
    expr = parts[0] if len(parts) == 1 else atoms.vstack(parts)
    perm = None if perm == list(range(len(perm))) else np.array(perm)
    return expr, perm


def _lower(term, values):
    """Lower a scalar term to ``(insts, args)``.

    ``insts`` lists the term's nodes in post-order as ``(opcode, *operands)``;
    operands are value numbers (positions in the list), so a subexpression
    shared inside the term is computed once. ``args`` holds the per-term data,
    in instruction order: the index of each ``load`` and the value of each
    ``const``.
    """
    insts, args, number = [], [], {}  # number: id(node) -> value number

    def visit(e):
        if id(e) in number:
            return number[id(e)]
        if e.shape != (1, 1):
            raise _Unsupported
        if isinstance(e, Constant):
            inst = ("const",)
            args.append(float(e.value[0, 0]))
        elif isinstance(e, (Variable, Parameter)):
            values[id(e)] = e
            inst = ("global", id(e))
        elif not isinstance(e, Node):
            raise _Unsupported
        elif e.opcode == "index" and isinstance(e.args[0], Variable):
            values[id(e.args[0])] = e.args[0]
            inst = ("load", id(e.args[0]))
            args.append(int(e.attrs["idx"][0]))
        elif e.opcode in ("add", "multiply", "scalar_mult"):
            inst = (e.opcode, visit(e.args[0]), visit(e.args[1]))
        elif e.opcode == "neg" or e.opcode in _UNARY:
            inst = (e.opcode, visit(e.args[0]))
        elif e.opcode == "power":
            inst = ("power", e.attrs["p"], visit(e.args[0]))
        else:
            raise _Unsupported
        number[id(e)] = len(insts)
        insts.append(inst)
        return number[id(e)]

    visit(term)
    return tuple(insts), args


def _vectorize(insts, args, values):
    """Run ``insts`` once on (lanes, 1) vectors, one lane per term.

    ``args[lane]`` is that term's args; transposed, each ``load``/``const``
    gets one value per lane.
    """
    lanes = len(args)
    lane_args = iter(zip(*args))
    vecs = []  # vecs[n]: the vector for value number n
    for inst in insts:
        opcode = inst[0]
        if opcode == "load":
            vecs.append(index_flat(values[inst[1]], next(lane_args), (lanes, 1)))
        elif opcode == "const":
            c = np.array(next(lane_args))
            vecs.append(Constant(c[0] if np.all(c == c[0])
                                 else c.reshape(lanes, 1)))
        elif opcode == "global":
            vecs.append(values[inst[1]])
        elif opcode == "add":
            vecs.append(vecs[inst[1]] + vecs[inst[2]])
        elif opcode in ("multiply", "scalar_mult"):
            vecs.append(vecs[inst[1]] * vecs[inst[2]])
        elif opcode == "neg":
            vecs.append(-vecs[inst[1]])
        elif opcode == "power":
            vecs.append(vecs[inst[2]] ** inst[1])
        else:
            vecs.append(getattr(atoms, opcode)(vecs[inst[1]]))
    # with no load/const (every term is e.g. the scalar y) the result is a scalar
    return broadcast(vecs[-1], (lanes, 1))
