#ifndef PROBLEM_UPDATE_PARAMS_H
#define PROBLEM_UPDATE_PARAMS_H

#include "problem/common.h"

/* Update parameter values.
 * Python: problem_update_params(problem_capsule, theta_array) */
static PyObject *py_problem_update_params(PyObject *self, PyObject *args)
{
    PyObject *prob_capsule;
    PyObject *theta_obj;
    if (!PyArg_ParseTuple(args, "OO", &prob_capsule, &theta_obj))
    {
        return NULL;
    }

    problem *prob =
        (problem *) PyCapsule_GetPointer(prob_capsule, PROBLEM_CAPSULE_NAME);
    if (!prob)
    {
        PyErr_SetString(PyExc_ValueError, "invalid problem capsule");
        return NULL;
    }

    PyArrayObject *theta_array = (PyArrayObject *) PyArray_FROM_OTF(
        theta_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!theta_array)
    {
        return NULL;
    }

    problem_update_params(prob, (const double *) PyArray_DATA(theta_array));
    Py_DECREF(theta_array);

    Py_RETURN_NONE;
}

#endif /* PROBLEM_UPDATE_PARAMS_H */
