#ifndef PROBLEM_COMMON_H
#define PROBLEM_COMMON_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include "problem.h"
#include "subexpr.h"

/* Also need expr types for capsule handling */
#include "../atoms/common.h"

#define PROBLEM_CAPSULE_NAME "DNLP_PROBLEM"

static void problem_capsule_destructor(PyObject *capsule)
{
    problem *prob = (problem *) PyCapsule_GetPointer(capsule, PROBLEM_CAPSULE_NAME);
    if (prob)
    {
        free_problem(prob);
    }
}

/* Convert obj to a contiguous double array of exactly n entries (at least n if
 * at_least is set). Returns a new reference, or NULL with an exception set. The
 * engine reads these buffers without knowing their length, so a short array
 * would make it read past the end. */
static PyArrayObject *double_array_of_size(PyObject *obj, npy_intp n, int at_least,
                                           const char *name)
{
    PyArrayObject *arr =
        (PyArrayObject *) PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!arr)
    {
        return NULL;
    }
    npy_intp size = PyArray_SIZE(arr);
    if (at_least ? size < n : size != n)
    {
        PyErr_Format(PyExc_ValueError, "%s must have %s%zd entries, got %zd", name,
                     at_least ? "at least " : "", (Py_ssize_t) n,
                     (Py_ssize_t) size);
        Py_DECREF(arr);
        return NULL;
    }
    return arr;
}

#endif /* PROBLEM_COMMON_H */
