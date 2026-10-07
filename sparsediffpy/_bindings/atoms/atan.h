#ifndef ATOM_ATAN_H
#define ATOM_ATAN_H

#include "common.h"
#include "elementwise_full_dom.h"

static PyObject *py_make_atan(PyObject *self, PyObject *args)
{
    PyObject *child_capsule;
    if (!PyArg_ParseTuple(args, "O", &child_capsule))
    {
        return NULL;
    }
    expr *child = (expr *) PyCapsule_GetPointer(child_capsule, EXPR_CAPSULE_NAME);
    if (!child)
    {
        PyErr_SetString(PyExc_ValueError, "invalid child capsule");
        return NULL;
    }

    expr *node = new_atan(child);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create atan node");
        return NULL;
    }
    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_ATAN_H */
