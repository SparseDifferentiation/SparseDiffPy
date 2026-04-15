#ifndef EXPR_JACOBIAN_H
#define EXPR_JACOBIAN_H

#include "../atoms/common.h"

static PyObject *py_expr_jacobian(PyObject *self, PyObject *args)
{
    PyObject *expr_capsule;

    if (!PyArg_ParseTuple(args, "O", &expr_capsule))
    {
        return NULL;
    }

    expr *node = (expr *) PyCapsule_GetPointer(expr_capsule, EXPR_CAPSULE_NAME);
    if (!node)
    {
        PyErr_SetString(PyExc_ValueError, "invalid expression capsule");
        return NULL;
    }

    node->eval_jacobian(node);

    CSR_Matrix *jac = node->jacobian;
    npy_intp nnz = jac->nnz;
    npy_intp m_plus_1 = jac->m + 1;

    PyObject *data = PyArray_SimpleNew(1, &nnz, NPY_DOUBLE);
    PyObject *indices = PyArray_SimpleNew(1, &nnz, NPY_INT32);
    PyObject *indptr = PyArray_SimpleNew(1, &m_plus_1, NPY_INT32);

    if (!data || !indices || !indptr)
    {
        Py_XDECREF(data);
        Py_XDECREF(indices);
        Py_XDECREF(indptr);
        return NULL;
    }

    memcpy(PyArray_DATA((PyArrayObject *) data), jac->x, nnz * sizeof(double));
    memcpy(PyArray_DATA((PyArrayObject *) indices), jac->i, nnz * sizeof(int));
    memcpy(PyArray_DATA((PyArrayObject *) indptr), jac->p, m_plus_1 * sizeof(int));

    return Py_BuildValue("(OOO(ii))", data, indices, indptr, jac->m, jac->n);
}

#endif /* EXPR_JACOBIAN_H */
