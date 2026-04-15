#ifndef EXPR_HESSIAN_H
#define EXPR_HESSIAN_H

#include "../atoms/common.h"

static PyObject *py_expr_hessian(PyObject *self, PyObject *args)
{
    PyObject *expr_capsule;
    PyObject *weights_obj;

    if (!PyArg_ParseTuple(args, "OO", &expr_capsule, &weights_obj))
    {
        return NULL;
    }

    expr *node = (expr *) PyCapsule_GetPointer(expr_capsule, EXPR_CAPSULE_NAME);
    if (!node)
    {
        PyErr_SetString(PyExc_ValueError, "invalid expression capsule");
        return NULL;
    }

    PyArrayObject *weights_arr = (PyArrayObject *) PyArray_FROM_OTF(
        weights_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!weights_arr)
    {
        return NULL;
    }

    node->eval_wsum_hess(node, (const double *) PyArray_DATA(weights_arr));
    Py_DECREF(weights_arr);

    CSR_Matrix *H = node->wsum_hess;
    npy_intp nnz = H->nnz;
    npy_intp n_plus_1 = H->n + 1;

    PyObject *data = PyArray_SimpleNew(1, &nnz, NPY_DOUBLE);
    PyObject *indices = PyArray_SimpleNew(1, &nnz, NPY_INT32);
    PyObject *indptr = PyArray_SimpleNew(1, &n_plus_1, NPY_INT32);

    if (!data || !indices || !indptr)
    {
        Py_XDECREF(data);
        Py_XDECREF(indices);
        Py_XDECREF(indptr);
        return NULL;
    }

    memcpy(PyArray_DATA((PyArrayObject *) data), H->x, nnz * sizeof(double));
    memcpy(PyArray_DATA((PyArrayObject *) indices), H->i, nnz * sizeof(int));
    memcpy(PyArray_DATA((PyArrayObject *) indptr), H->p, n_plus_1 * sizeof(int));

    return Py_BuildValue("(OOO(ii))", data, indices, indptr, H->m, H->n);
}

#endif /* EXPR_HESSIAN_H */
