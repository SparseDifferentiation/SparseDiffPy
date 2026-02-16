#ifndef ATOM_CONST_SCALAR_MULT_H
#define ATOM_CONST_SCALAR_MULT_H

#include "bivariate.h"
#include "common.h"
#include "subexpr.h"

/* Constant scalar multiplication: a * f(x) where a is a constant double.
 * Creates a fixed parameter node for the scalar and calls new_scalar_mult. */
static PyObject *py_make_const_scalar_mult(PyObject *self, PyObject *args)
{
    PyObject *child_capsule;
    double a;

    if (!PyArg_ParseTuple(args, "Od", &child_capsule, &a))
    {
        return NULL;
    }

    expr *child = (expr *) PyCapsule_GetPointer(child_capsule, EXPR_CAPSULE_NAME);
    if (!child)
    {
        PyErr_SetString(PyExc_ValueError, "invalid child capsule");
        return NULL;
    }

    /* Create a 1x1 fixed parameter for the scalar value */
    expr *a_node = new_parameter(1, 1, PARAM_FIXED, child->n_vars, &a);
    if (!a_node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create scalar parameter node");
        return NULL;
    }

    expr *node = new_scalar_mult(a_node, child);

    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "failed to create scalar_mult node");
        return NULL;
    }
    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_CONST_SCALAR_MULT_H */
