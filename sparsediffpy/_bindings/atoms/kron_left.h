#ifndef ATOM_KRON_LEFT_H
#define ATOM_KRON_LEFT_H

#include "bivariate.h"
#include "common.h"

/*
 * Python signature:
 *   make_kron_left(child_capsule, C_data, C_indices, C_indptr, m, n, p, q)
 *
 * Creates kron(C, X) where C is (m x n) constant CSR matrix and X is the
 * child expression of shape (p x q).
 */
static PyObject *py_make_kron_left(PyObject *self, PyObject *args)
{
    (void) self;
    PyObject *child_capsule;
    PyObject *data_obj, *indices_obj, *indptr_obj;
    int m, n, p, q;

    if (!PyArg_ParseTuple(args, "OOOOiiii", &child_capsule, &data_obj,
                          &indices_obj, &indptr_obj, &m, &n, &p, &q))
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
    PyArrayObject *indices_array =
        (PyArrayObject *) PyArray_FROM_OTF(indices_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *indptr_array =
        (PyArrayObject *) PyArray_FROM_OTF(indptr_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);

    if (!data_array || !indices_array || !indptr_array)
    {
        Py_XDECREF(data_array);
        Py_XDECREF(indices_array);
        Py_XDECREF(indptr_array);
        return NULL;
    }

    int nnz = (int) PyArray_SIZE(data_array);
    CSR_Matrix *C = new_csr_matrix(m, n, nnz);
    memcpy(C->x, PyArray_DATA(data_array), (size_t) nnz * sizeof(double));
    memcpy(C->i, PyArray_DATA(indices_array), (size_t) nnz * sizeof(int));
    memcpy(C->p, PyArray_DATA(indptr_array), (size_t)(m + 1) * sizeof(int));

    Py_DECREF(data_array);
    Py_DECREF(indices_array);
    Py_DECREF(indptr_array);

    expr *node = new_kron_left(child, C, p, q);
    free_csr_matrix(C);

    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create kron_left node");
        return NULL;
    }

    expr_retain(node);
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_KRON_LEFT_H */
