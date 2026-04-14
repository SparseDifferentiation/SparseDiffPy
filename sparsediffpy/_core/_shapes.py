"""Shape validation, broadcasting, and matmul checks.

All shapes are 2-tuples (d1, d2) matching the C layer's convention.
Column-major flat storage: flat_index = row + col * d1.
"""


def validate_shape(d1, d2):
    """Check that d1 and d2 are positive integers."""
    if not isinstance(d1, int) or not isinstance(d2, int):
        raise TypeError(f"Shape dimensions must be integers, got ({type(d1).__name__}, {type(d2).__name__})")
    if d1 <= 0 or d2 <= 0:
        raise ValueError(f"Shape dimensions must be positive, got ({d1}, {d2})")


def is_scalar(shape):
    return shape == (1, 1)


def is_column(shape):
    return shape[1] == 1


def is_row(shape):
    return shape[0] == 1


def broadcast_shape(left_shape, right_shape):
    """Compute broadcast result shape for elementwise operations.

    Returns (result_shape, left_needs_broadcast, right_needs_broadcast).
    Raises ValueError if shapes are incompatible.

    Rules (CVXPY/NumPy convention):
      (1,1) + (m,n) -> (m,n)   broadcast scalar
      (m,1) + (m,n) -> (m,n)   broadcast column
      (1,n) + (m,n) -> (m,n)   broadcast row
      (m,n) + (m,n) -> (m,n)   no broadcast
    """
    ld1, ld2 = left_shape
    rd1, rd2 = right_shape

    if left_shape == right_shape:
        return left_shape, False, False

    # Broadcast each dimension independently: 1 matches anything
    if ld1 == rd1:
        out_d1 = ld1
    elif ld1 == 1:
        out_d1 = rd1
    elif rd1 == 1:
        out_d1 = ld1
    else:
        raise ValueError(
            f"Cannot broadcast shapes {left_shape} and {right_shape}: "
            f"d1 mismatch ({ld1} vs {rd1})"
        )

    if ld2 == rd2:
        out_d2 = ld2
    elif ld2 == 1:
        out_d2 = rd2
    elif rd2 == 1:
        out_d2 = ld2
    else:
        raise ValueError(
            f"Cannot broadcast shapes {left_shape} and {right_shape}: "
            f"d2 mismatch ({ld2} vs {rd2})"
        )

    result = (out_d1, out_d2)
    return result, left_shape != result, right_shape != result


def check_matmul_shapes(left_shape, right_shape):
    """Validate matmul dimensions and return result shape.

    Requires left_shape[1] == right_shape[0].
    Returns (left_shape[0], right_shape[1]).
    """
    if left_shape[1] != right_shape[0]:
        raise ValueError(
            f"Matmul shape mismatch: ({left_shape[0]}, {left_shape[1]}) @ "
            f"({right_shape[0]}, {right_shape[1]})"
        )
    return (left_shape[0], right_shape[1])
