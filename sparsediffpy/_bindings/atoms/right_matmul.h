#ifndef ATOM_RIGHT_MATMUL_H
#define ATOM_RIGHT_MATMUL_H

#include "bivariate.h"
#include "common.h"

/* Right matrix multiplication: f(x) @ A where A is a constant or parameter
 * sparse matrix.
 *
 * Python signature:
 *   make_sparse_right_matmul(param_or_none, child, data, indices, indptr, m, n)
 *
 * - param_or_none: None for constant matrix, or a parameter capsule.
 * - child: the child expression capsule f(x).
 * - data, indices, indptr: CSR arrays for matrix A.
 * - m, n: dimensions of matrix A. */
static PyObject *py_make_sparse_right_matmul(PyObject *self, PyObject *args)
{
    PyObject *param_obj;
    PyObject *child_capsule;
    PyObject *data_obj, *indices_obj, *indptr_obj;
    int m, n;
    if (!PyArg_ParseTuple(args, "OOOOOii", &param_obj, &child_capsule,
                          &data_obj, &indices_obj, &indptr_obj, &m, &n))
    {
        return NULL;
    }

    expr *child = (expr *) PyCapsule_GetPointer(child_capsule, EXPR_CAPSULE_NAME);
    if (!child)
    {
        PyErr_SetString(PyExc_ValueError, "invalid child capsule");
        return NULL;
    }

    PyArrayObject *data_array =
        (PyArrayObject *) PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *indices_array = (PyArrayObject *) PyArray_FROM_OTF(
        indices_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *indptr_array = (PyArrayObject *) PyArray_FROM_OTF(
        indptr_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);

    if (!data_array || !indices_array || !indptr_array)
    {
        Py_XDECREF(data_array);
        Py_XDECREF(indices_array);
        Py_XDECREF(indptr_array);
        return NULL;
    }

    int nnz = (int) PyArray_SIZE(data_array);

    /* Build the parameter node: use provided capsule or create PARAM_FIXED */
    expr *param_node = NULL;
    if (param_obj == Py_None)
    {
        param_node = new_parameter(nnz, 1, PARAM_FIXED, child->n_vars,
                                   (const double *) PyArray_DATA(data_array));
        if (!param_node)
        {
            Py_DECREF(data_array);
            Py_DECREF(indices_array);
            Py_DECREF(indptr_array);
            PyErr_SetString(PyExc_RuntimeError,
                            "failed to create parameter node for matrix");
            return NULL;
        }
    }
    else
    {
        param_node =
            (expr *) PyCapsule_GetPointer(param_obj, EXPR_CAPSULE_NAME);
        if (!param_node)
        {
            Py_DECREF(data_array);
            Py_DECREF(indices_array);
            Py_DECREF(indptr_array);
            PyErr_SetString(PyExc_ValueError, "invalid parameter capsule");
            return NULL;
        }
    }

    CSR_Matrix *A = new_csr_matrix(m, n, nnz);
    memcpy(A->x, PyArray_DATA(data_array), nnz * sizeof(double));
    memcpy(A->i, PyArray_DATA(indices_array), nnz * sizeof(int));
    memcpy(A->p, PyArray_DATA(indptr_array), (m + 1) * sizeof(int));

    Py_DECREF(data_array);
    Py_DECREF(indices_array);
    Py_DECREF(indptr_array);

    expr *node = new_right_matmul(param_node, child, A);
    free_csr_matrix(A);

    if (!node)
    {
        if (param_obj == Py_None) free_expr(param_node);
        PyErr_SetString(PyExc_RuntimeError,
                        "failed to create right_matmul node");
        return NULL;
    }
    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_RIGHT_MATMUL_H */
