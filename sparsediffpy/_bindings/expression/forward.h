#ifndef EXPR_FORWARD_H
#define EXPR_FORWARD_H

#include "../atoms/common.h"

static PyObject *py_expr_forward(PyObject *self, PyObject *args)
{
    PyObject *expr_capsule;
    PyObject *u_obj;

    if (!PyArg_ParseTuple(args, "OO", &expr_capsule, &u_obj))
    {
        return NULL;
    }

    expr *node = (expr *) PyCapsule_GetPointer(expr_capsule, EXPR_CAPSULE_NAME);
    if (!node)
    {
        PyErr_SetString(PyExc_ValueError, "invalid expression capsule");
        return NULL;
    }

    PyArrayObject *u_array =
        (PyArrayObject *) PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!u_array)
    {
        return NULL;
    }

    node->forward(node, (const double *) PyArray_DATA(u_array));
    Py_DECREF(u_array);

    npy_intp size = node->size;
    PyObject *out = PyArray_SimpleNew(1, &size, NPY_DOUBLE);
    if (!out)
    {
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject *) out), node->value,
           size * sizeof(double));

    return out;
}

#endif /* EXPR_FORWARD_H */
