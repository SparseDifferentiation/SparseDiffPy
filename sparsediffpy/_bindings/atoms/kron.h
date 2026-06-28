#ifndef ATOM_KRON_H
#define ATOM_KRON_H

#include "common.h"

/* Kronecker product Z = kron(A, B). Exactly one operand is variable-free and is
 * passed as the parameter capsule (wrap constants with make_parameter and
 * param_id=-1 / PARAM_FIXED); the other carries the variables and is passed as
 * the child capsule.
 *
 * Python signature:
 *   make_kron(param_capsule, child_capsule, const_is_left, p, q, r, s)
 * - const_is_left: 1 -> A=param, B=child; 0 -> A=child, B=param.
 * - (p, q): A's dims; (r, s): B's dims. */
static PyObject *py_make_kron(PyObject *self, PyObject *args)
{
    PyObject *param_capsule;
    PyObject *child_capsule;
    int const_is_left, p, q, r, s;

    if (!PyArg_ParseTuple(args, "OOiiiii", &param_capsule, &child_capsule,
                          &const_is_left, &p, &q, &r, &s))
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

    expr *node = new_kron(param_node, child, const_is_left, p, q, r, s);
    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create kron node");
        return NULL;
    }

    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_KRON_H */
