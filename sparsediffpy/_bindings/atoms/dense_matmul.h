#ifndef ATOM_DENSE_MATMUL_H
#define ATOM_DENSE_MATMUL_H

#include "bivariate_full_dom.h"
#include "common.h"

/* Dense left matrix multiplication: A @ f(x) where A is a dense matrix.
 *
 * Python signature:
 *   make_dense_left_matmul(child, A_data_flat, m, n)
 *
 * - child: the child expression capsule f(x).
 * - A_data_flat: contiguous row-major numpy float64 array of size m*n.
 * - m, n: dimensions of matrix A. */
static PyObject *py_make_dense_left_matmul(PyObject *self, PyObject *args)
{
    PyObject *child_capsule;
    PyObject *data_obj;
    int m, n;
    if (!PyArg_ParseTuple(args, "OOii", &child_capsule, &data_obj, &m, &n))
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
    if (!data_array)
    {
        return NULL;
    }

    double *A_data = (double *) PyArray_DATA(data_array);

    expr *node = new_left_matmul_dense(child, m, n, A_data);
    Py_DECREF(data_array);

    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "failed to create dense_left_matmul node");
        return NULL;
    }
    expr_retain(node);
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

/* Dense right matrix multiplication: f(x) @ A where A is a dense matrix.
 *
 * Python signature:
 *   make_dense_right_matmul(child, A_data_flat, m, n)
 *
 * - child: the child expression capsule f(x).
 * - A_data_flat: contiguous row-major numpy float64 array of size m*n.
 * - m, n: dimensions of matrix A. */
static PyObject *py_make_dense_right_matmul(PyObject *self, PyObject *args)
{
    PyObject *child_capsule;
    PyObject *data_obj;
    int m, n;
    if (!PyArg_ParseTuple(args, "OOii", &child_capsule, &data_obj, &m, &n))
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
    if (!data_array)
    {
        return NULL;
    }

    double *A_data = (double *) PyArray_DATA(data_array);

    expr *node = new_right_matmul_dense(child, m, n, A_data);
    Py_DECREF(data_array);

    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "failed to create dense_right_matmul node");
        return NULL;
    }
    expr_retain(node);
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_DENSE_MATMUL_H */