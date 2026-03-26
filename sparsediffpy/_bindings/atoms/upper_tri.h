#ifndef ATOM_UPPER_TRI_H
#define ATOM_UPPER_TRI_H

#include "common.h"

static PyObject *py_make_upper_tri(PyObject *self, PyObject *args)
{
    PyObject *child_capsule;

    if (!PyArg_ParseTuple(args, "O", &child_capsule))
    {
        return NULL;
    }

    expr *child = (expr *) PyCapsule_GetPointer(child_capsule, EXPR_CAPSULE_NAME);
    if (!child)
    {
        return NULL;
    }

    expr *node = new_upper_tri(child);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create upper_tri node");
        return NULL;
    }

    expr_retain(node);
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_UPPER_TRI_H */
