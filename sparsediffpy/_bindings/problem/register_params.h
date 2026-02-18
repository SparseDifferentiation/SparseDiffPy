#ifndef PROBLEM_REGISTER_PARAMS_H
#define PROBLEM_REGISTER_PARAMS_H

#include "atoms/common.h"
#include "problem/common.h"

/* Register parameter nodes with the problem.
 * Python: problem_register_params(problem_capsule, [param_capsule, ...]) */
static PyObject *py_problem_register_params(PyObject *self, PyObject *args)
{
    PyObject *prob_capsule;
    PyObject *param_list;
    if (!PyArg_ParseTuple(args, "OO", &prob_capsule, &param_list))
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

    if (!PyList_Check(param_list))
    {
        PyErr_SetString(PyExc_TypeError, "param_nodes must be a list");
        return NULL;
    }

    Py_ssize_t n = PyList_Size(param_list);
    expr **param_nodes = (expr **) malloc(n * sizeof(expr *));
    if (!param_nodes)
    {
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t i = 0; i < n; i++)
    {
        PyObject *cap = PyList_GetItem(param_list, i);
        param_nodes[i] = (expr *) PyCapsule_GetPointer(cap, EXPR_CAPSULE_NAME);
        if (!param_nodes[i])
        {
            free(param_nodes);
            PyErr_SetString(PyExc_ValueError, "invalid parameter capsule");
            return NULL;
        }
    }

    problem_register_params(prob, param_nodes, (int) n);
    free(param_nodes);

    Py_RETURN_NONE;
}

#endif /* PROBLEM_REGISTER_PARAMS_H */
