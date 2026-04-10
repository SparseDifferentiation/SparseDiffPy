#ifndef ATOM_PARAMETER_H
#define ATOM_PARAMETER_H

#include "common.h"

/* Create a parameter node.
 *
 * Python signature:
 *   make_parameter(d1, d2, param_id, n_vars[, values])
 *
 * - param_id >= 0: updatable parameter (values optional, NULL if omitted).
 * - param_id == -1 (PARAM_FIXED): constant node (values required). */
static PyObject *py_make_parameter(PyObject *self, PyObject *args)
{
    int d1, d2, param_id, n_vars;
    PyObject *values_obj = NULL;
    if (!PyArg_ParseTuple(args, "iiii|O", &d1, &d2, &param_id, &n_vars,
                          &values_obj))
    {
        return NULL;
    }

    const double *values = NULL;
    PyArrayObject *values_array = NULL;

    if (values_obj && values_obj != Py_None)
    {
        values_array = (PyArrayObject *) PyArray_FROM_OTF(
            values_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (!values_array)
        {
            return NULL;
        }
        values = (const double *) PyArray_DATA(values_array);
    }

    expr *node = new_parameter(d1, d2, param_id, n_vars, values);
    Py_XDECREF(values_array);

    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create parameter node");
        return NULL;
    }
    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

#endif /* ATOM_PARAMETER_H */
