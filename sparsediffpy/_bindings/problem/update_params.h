#ifndef PROBLEM_UPDATE_PARAMS_H
#define PROBLEM_UPDATE_PARAMS_H

#include "common.h"

static PyObject *py_problem_update_params(PyObject *self, PyObject *args)
{
    PyObject *problem_capsule;
    PyObject *theta_obj;

    if (!PyArg_ParseTuple(args, "OO", &problem_capsule, &theta_obj))
    {
        return NULL;
    }

    problem *prob = (problem *) PyCapsule_GetPointer(problem_capsule,
                                                     PROBLEM_CAPSULE_NAME);
    if (!prob)
    {
        PyErr_SetString(PyExc_ValueError, "invalid problem capsule");
        return NULL;
    }

    /* the engine calls exit(1) when no parameters are registered */
    if (prob->n_param_nodes == 0)
    {
        PyErr_SetString(PyExc_ValueError, "no parameters registered");
        return NULL;
    }

    /* each node copies theta[param_id : param_id + size] */
    npy_intp theta_size = 0;
    for (int i = 0; i < prob->n_param_nodes; i++)
    {
        expr *node = prob->param_nodes[i];
        npy_intp end = ((parameter_expr *) node)->param_id + node->size;
        theta_size = end > theta_size ? end : theta_size;
    }

    PyArrayObject *theta_array =
        double_array_of_size(theta_obj, theta_size, 1, "theta");
    if (!theta_array)
    {
        return NULL;
    }

    problem_update_params(prob, (const double *) PyArray_DATA(theta_array));
    Py_DECREF(theta_array);

    Py_RETURN_NONE;
}

#endif /* PROBLEM_UPDATE_PARAMS_H */
