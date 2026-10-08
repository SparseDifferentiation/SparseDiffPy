#ifndef ATOM_ATAN2_H
#define ATOM_ATAN2_H

#include "bivariate_restricted_dom.h"
#include "common.h"

/* atan2(y, x), C argument order. Both arguments must be distinct variables of
   the same shape; the engine returns NULL otherwise. */
static PyObject *py_make_atan2(PyObject *self, PyObject *args)
{
    PyObject *y_capsule;
    PyObject *x_capsule;
    if (!PyArg_ParseTuple(args, "OO", &y_capsule, &x_capsule))
    {
        return NULL;
    }
    expr *y = (expr *) PyCapsule_GetPointer(y_capsule, EXPR_CAPSULE_NAME);
    expr *x = (expr *) PyCapsule_GetPointer(x_capsule, EXPR_CAPSULE_NAME);
    if (!y || !x)
    {
        PyErr_SetString(PyExc_ValueError, "invalid child capsule");
        return NULL;
    }

    expr *node = new_atan2(y, x);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create atan2 node");
        return NULL;
    }
    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_ATAN2_H */
