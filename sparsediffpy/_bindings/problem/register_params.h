#ifndef PROBLEM_REGISTER_PARAMS_H
#define PROBLEM_REGISTER_PARAMS_H

#include "common.h"

static PyObject *py_problem_register_params(PyObject *self, PyObject *args)
{
    PyObject *problem_capsule;
    PyObject *param_list;

    if (!PyArg_ParseTuple(args, "OO", &problem_capsule, &param_list))
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

    if (!PyList_Check(param_list))
    {
        PyErr_SetString(PyExc_TypeError,
                        "second argument must be a list of parameter capsules");
        return NULL;
    }

    Py_ssize_t n = PyList_Size(param_list);
    expr **param_nodes = (expr **) malloc(n * sizeof(expr *));
    if (!param_nodes)
    {
        PyErr_SetString(PyExc_MemoryError,
                        "failed to allocate parameter node array");
        return NULL;
    }

    for (Py_ssize_t i = 0; i < n; i++)
    {
        PyObject *capsule = PyList_GetItem(param_list, i);
        expr *node =
            (expr *) PyCapsule_GetPointer(capsule, EXPR_CAPSULE_NAME);
        if (!node)
        {
            free(param_nodes);
            PyErr_SetString(PyExc_ValueError,
                            "invalid parameter capsule in list");
            return NULL;
        }
        param_nodes[i] = node;
    }

    problem_register_params(prob, param_nodes, (int) n);
    free(param_nodes);

    Py_RETURN_NONE;
}

#endif /* PROBLEM_REGISTER_PARAMS_H */
