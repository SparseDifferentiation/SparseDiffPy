#ifndef EXPR_UPDATE_PARAMS_H
#define EXPR_UPDATE_PARAMS_H

#include "../atoms/common.h"
#include "subexpr.h"

/*
 * py_expr_update_params(root_capsule, param_capsule_list, theta_array)
 *
 * Updates parameter values from theta and propagates the refresh flag
 * down the expression tree rooted at root_capsule.
 */
static PyObject *py_expr_update_params(PyObject *self, PyObject *args)
{
    PyObject *root_capsule;
    PyObject *param_list;
    PyObject *theta_obj;

    if (!PyArg_ParseTuple(args, "OOO", &root_capsule, &param_list, &theta_obj))
    {
        return NULL;
    }

    expr *root = (expr *) PyCapsule_GetPointer(root_capsule, EXPR_CAPSULE_NAME);
    if (!root)
    {
        PyErr_SetString(PyExc_ValueError, "invalid root expression capsule");
        return NULL;
    }

    if (!PyList_Check(param_list))
    {
        PyErr_SetString(PyExc_TypeError,
                        "second argument must be a list of parameter capsules");
        return NULL;
    }

    PyArrayObject *theta_arr = (PyArrayObject *) PyArray_FROM_OTF(
        theta_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!theta_arr)
    {
        return NULL;
    }

    const double *theta = (const double *) PyArray_DATA(theta_arr);
    Py_ssize_t n = PyList_Size(param_list);

    for (Py_ssize_t i = 0; i < n; i++)
    {
        PyObject *capsule = PyList_GetItem(param_list, i);
        expr *pnode = (expr *) PyCapsule_GetPointer(capsule, EXPR_CAPSULE_NAME);
        if (!pnode)
        {
            Py_DECREF(theta_arr);
            PyErr_SetString(PyExc_ValueError,
                            "invalid parameter capsule in list");
            return NULL;
        }

        parameter_expr *param = (parameter_expr *) pnode;
        int offset = param->param_id;
        memcpy(pnode->value, theta + offset, pnode->size * sizeof(double));
    }

    Py_DECREF(theta_arr);

    expr_set_needs_refresh(root);

    Py_RETURN_NONE;
}

#endif /* EXPR_UPDATE_PARAMS_H */
