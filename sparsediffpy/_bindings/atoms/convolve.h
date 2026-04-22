#ifndef ATOM_CONVOLVE_H
#define ATOM_CONVOLVE_H

#include "common.h"

/* 1D full convolution: y = conv(a, child), where a is a length-m kernel
 * held by a parameter node (wrap constants with make_parameter and
 * param_id=-1 / PARAM_FIXED) and child is a length-n vector expression.
 * Output is length m + n - 1.
 *
 * Python signature:
 *   make_convolve(param_capsule, child_capsule) */
static PyObject *py_make_convolve(PyObject *self, PyObject *args)
{
    PyObject *param_capsule;
    PyObject *child_capsule;

    if (!PyArg_ParseTuple(args, "OO", &param_capsule, &child_capsule))
    {
        return NULL;
    }

    expr *param_node =
        (expr *) PyCapsule_GetPointer(param_capsule, EXPR_CAPSULE_NAME);
    if (!param_node)
    {
        PyErr_SetString(PyExc_ValueError, "invalid parameter capsule");
        return NULL;
    }

    expr *child =
        (expr *) PyCapsule_GetPointer(child_capsule, EXPR_CAPSULE_NAME);
    if (!child)
    {
        PyErr_SetString(PyExc_ValueError, "invalid child capsule");
        return NULL;
    }

    expr *node = new_convolve(param_node, child);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create convolve node");
        return NULL;
    }
    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_CONVOLVE_H */
