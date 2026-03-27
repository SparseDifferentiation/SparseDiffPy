#ifndef ATOM_CONST_SCALAR_MULT_H
#define ATOM_CONST_SCALAR_MULT_H

#include "bivariate.h"
#include "common.h"

/* Constant scalar multiplication: a * f(x) where a is a constant double */
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

    expr *a_node = new_parameter(1, 1, PARAM_FIXED, child->n_vars, &a);
    if (!a_node)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "failed to create parameter node for scalar");
        return NULL;
    }

    expr *node = new_scalar_mult(a_node, child);
    if (!node)
    {
        free_expr(a_node);
        PyErr_SetString(PyExc_RuntimeError,
                        "failed to create const_scalar_mult node");
        return NULL;
    }
    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_CONST_SCALAR_MULT_H */
