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
    """Stack scalar terms into a column vector, merging terms of the same form.

    ``terms`` is typically the list of scalar objective or constraint terms
    built in a Python loop, such as
    ``[x[k] ** 2 + x[k + 1] for k in range(n - 1)]``. Terms that apply the
    same operations, and differ only in which entries of a Variable they index
    or which constants they use, are rebuilt as one vector expression. Terms
    using other operations (matmul, sum, ...) are kept as they are.

    Parameters
    ----------
    terms : list of Expression or float
        The scalar terms to stack; each must have shape ``(1, 1)``.

    Returns
    -------
    expr : Expression
        An expression of shape ``(len(terms), 1)`` holding one term per row.
        Merged terms are grouped together, so rows may be reordered.
    perm : numpy.ndarray of int or None
        Row ``j`` of ``expr`` is ``terms[perm[j]]``. ``None`` if every term
        kept its position, i.e. ``expr`` is ``vstack(terms)``.

    Raises
    ------
    ValueError
        If ``terms`` is empty or a term does not have shape ``(1, 1)``.

    Examples
    --------
    >>> x = sp.Scope().Variable(5, 1)
    >>> expr, perm = sp.compact([x[k] ** 2 + x[k + 1] for k in range(4)])
    >>> expr.shape, perm
    ((4, 1), None)
    >>> expr, perm = sp.compact([x[0] ** 2, sp.exp(x[1]), x[2] ** 2])
    >>> perm
    array([0, 2, 1])
    """
    if not terms:
        raise ValueError("compact: need at least one term")
    values, groups = {}, {}  # values: id -> Variable/Parameter the terms use
    for i, t in enumerate(terms):
        t = cast_to_const(t)
        if t.shape != (1, 1):
            raise ValueError(f"compact: term {i} has shape {t.shape}, not (1, 1)")
        try:
            instructions, args = _lower(t, values)
        except (_Unsupported, RecursionError):  # deep terms are kept as is
            instructions, args = ("single", i), None
        groups.setdefault(instructions, []).append((i, t, args))

    parts, perm = [], []
    for instructions, members in groups.items():
        perm += [i for i, _, _ in members]
        parts.append(members[0][1] if len(members) == 1 else
                     _vectorize(instructions,
                                [args for _, _, args in members], values))
    expr = parts[0] if len(parts) == 1 else atoms.vstack(parts)
    perm = None if perm == list(range(len(perm))) else np.array(perm)
    return expr, perm


def _lower(term, values):
    """Lower a scalar term to a list of instructions and its per-term args.

    Two terms with equal ``instructions`` apply the same operations and differ
    only in their ``args``, so they can be merged by :func:`_vectorize`.

    Parameters
    ----------
    term : Expression
        A scalar term of shape ``(1, 1)``.
    values : dict
        Maps ``id`` to each Variable or Parameter the terms use. Updated in
        place with the ones ``term`` uses.

    Returns
    -------
    instructions : tuple of tuple
        The term's nodes in post-order, each as ``(opcode, *operands)``.
        Operands are value numbers (positions in ``instructions``), so a
        subexpression shared inside the term is computed once.
    args : list
        The per-term data, in instruction order: the index of each ``load``
        and the value of each ``const``.

    Raises
    ------
    _Unsupported
        If ``term`` uses an operation that cannot be vectorized.
    """
    instructions, args, number = [], [], {}  # number: id(node) -> value number

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
        number[id(e)] = len(instructions)
        instructions.append(inst)
        return number[id(e)]

    visit(term)
    return tuple(instructions), args


def _vectorize(instructions, args, values):
    """Build one vector expression for a group of terms with equal instructions.

    The instructions are run once on column vectors with one row per term:
    each ``load`` becomes a gather of the Variable at every term's index, and
    each ``const`` a constant vector of every term's value.

    Parameters
    ----------
    instructions : tuple of tuple
        The instructions shared by the terms, as returned by :func:`_lower`.
    args : list of list
        ``args[i]`` is the args of term ``i``, as returned by :func:`_lower`.
    values : dict
        Maps ``id`` to each Variable or Parameter the instructions use.

    Returns
    -------
    Expression
        An expression of shape ``(len(args), 1)`` whose row ``i`` is term ``i``.
    """
    n_rows = len(args)
    arg_columns = iter(zip(*args))  # one load/const arg across all terms
    vecs = []  # vecs[n]: the vector for value number n
    for inst in instructions:
        opcode = inst[0]
        if opcode == "load":
            vecs.append(index_flat(values[inst[1]], next(arg_columns),
                                   (n_rows, 1)))
        elif opcode == "const":
            c = np.array(next(arg_columns))
            vecs.append(Constant(c[0] if np.all(c == c[0])
                                 else c.reshape(n_rows, 1)))
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
    return broadcast(vecs[-1], (n_rows, 1))
