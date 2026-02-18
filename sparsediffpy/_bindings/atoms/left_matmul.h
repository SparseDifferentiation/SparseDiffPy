#ifndef ATOM_LEFT_MATMUL_H
#define ATOM_LEFT_MATMUL_H

#include "bivariate.h"
#include "common.h"
#include "subexpr.h"

/* Left matrix multiplication: A @ f(x).
 *
 * Unified binding for both fixed-constant and updatable-parameter cases.
 * Python signature:
 *   make_left_matmul(param_or_none, child, data, indices, indptr, m, n)
 *
 * - param_or_none: None for fixed constants (a PARAM_FIXED parameter is created
 *   internally), or an existing parameter capsule for updatable parameters.
 * - child: the child expression capsule f(x).
 * - data, indices, indptr, m, n: CSR arrays defining the sparsity pattern and
 *   initial values of the matrix A. */
static PyObject *py_make_left_matmul(PyObject *self, PyObject *args)
{
    PyObject *param_obj;
    PyObject *child_capsule;
    PyObject *data_obj, *indices_obj, *indptr_obj;
    int m, n;
    if (!PyArg_ParseTuple(args, "OOOOOii", &param_obj, &child_capsule, &data_obj,
                          &indices_obj, &indptr_obj, &m, &n))
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

    double *csr_data = (double *) PyArray_DATA(data_array);
    int *csr_indices = (int *) PyArray_DATA(indices_array);
    int *csr_indptr = (int *) PyArray_DATA(indptr_array);
    int nnz = csr_indptr[m];

    /* Build CSR matrix from Python arrays */
    CSR_Matrix *A = new_csr_matrix(m, n, nnz);
    memcpy(A->p, csr_indptr, (m + 1) * sizeof(int));
    memcpy(A->i, csr_indices, nnz * sizeof(int));
    memcpy(A->x, csr_data, nnz * sizeof(double));

    /* Determine param_node: use passed capsule, or create PARAM_FIXED internally */
    expr *param_node;
    if (param_obj == Py_None)
    {
        /* Fixed constant: pass CSR data directly (values are already in CSR order) */
        param_node = new_parameter(nnz, 1, PARAM_FIXED, child->n_vars, csr_data);

        if (!param_node)
        {
            free_csr_matrix(A);
            Py_DECREF(data_array);
            Py_DECREF(indices_array);
            Py_DECREF(indptr_array);
            PyErr_SetString(PyExc_RuntimeError,
                            "failed to create matrix parameter node");
            return NULL;
        }
    }
    else
    {
        param_node = (expr *) PyCapsule_GetPointer(param_obj, EXPR_CAPSULE_NAME);
        if (!param_node)
        {
            free_csr_matrix(A);
            Py_DECREF(data_array);
            Py_DECREF(indices_array);
            Py_DECREF(indptr_array);
            PyErr_SetString(PyExc_ValueError, "invalid param capsule");
            return NULL;
        }
    }

    Py_DECREF(data_array);
    Py_DECREF(indices_array);
    Py_DECREF(indptr_array);

    expr *node = new_left_matmul(param_node, child, A);
    free_csr_matrix(A); /* constructor copies it */

    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create left_matmul node");
        return NULL;
    }
    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_LEFT_MATMUL_H */
