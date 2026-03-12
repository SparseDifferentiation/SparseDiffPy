/*
 * Stub implementations for dense matrix functions on Windows,
 * where BLAS is not available. These satisfy the linker for
 * left_matmul.c which references them, but the dense matmul
 * bindings are excluded via #ifndef _MSC_VER so they are
 * never actually called.
 */
#ifdef _MSC_VER
#include "utils/matrix.h"
#include <stdlib.h>

Matrix *new_dense_matrix(int m, int n, const double *data)
{
    (void) m;
    (void) n;
    (void) data;
    return NULL;
}

Matrix *dense_matrix_trans(const Dense_Matrix *self)
{
    (void) self;
    return NULL;
}
#endif
