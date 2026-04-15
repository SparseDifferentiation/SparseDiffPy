#ifndef EXPR_INIT_DERIVATIVES_H
#define EXPR_INIT_DERIVATIVES_H

#include "../atoms/common.h"

static PyObject *py_expr_init_jacobian(PyObject *self, PyObject *args)
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

    jacobian_init(node);
    Py_RETURN_NONE;
}

static PyObject *py_expr_init_hessian(PyObject *self, PyObject *args)
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

    wsum_hess_init(node);
    Py_RETURN_NONE;
}

#endif /* EXPR_INIT_DERIVATIVES_H */
