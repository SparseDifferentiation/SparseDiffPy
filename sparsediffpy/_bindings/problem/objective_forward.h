#ifndef PROBLEM_OBJECTIVE_FORWARD_H
#define PROBLEM_OBJECTIVE_FORWARD_H

#include "common.h"

static PyObject *py_problem_objective_forward(PyObject *self, PyObject *args)
{
    PyObject *prob_capsule;
    PyObject *u_obj;
    if (!PyArg_ParseTuple(args, "OO", &prob_capsule, &u_obj))
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

    PyArrayObject *u_array = double_array_of_size(u_obj, prob->n_vars, 0, "u");
    if (!u_array)
    {
        return NULL;
    }

    double obj_val =
        problem_objective_forward(prob, (const double *) PyArray_DATA(u_array));

    Py_DECREF(u_array);
    return Py_BuildValue("d", obj_val);
}

#endif /* PROBLEM_OBJECTIVE_FORWARD_H */
