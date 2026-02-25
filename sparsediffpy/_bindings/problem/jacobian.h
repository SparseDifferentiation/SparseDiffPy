#ifndef PROBLEM_JACOBIAN_H
#define PROBLEM_JACOBIAN_H

#include "common.h"

static PyObject *py_problem_jacobian(PyObject *self, PyObject *args)
{
    PyObject *prob_capsule;
    if (!PyArg_ParseTuple(args, "O", &prob_capsule))
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

    if (prob->n_constraints == 0)
    {
        /* Return empty CSR components */
        npy_intp zero = 0;
        npy_intp one = 1;
        PyObject *data = PyArray_SimpleNew(1, &zero, NPY_DOUBLE);
        PyObject *indices = PyArray_SimpleNew(1, &zero, NPY_INT32);
        PyObject *indptr = PyArray_SimpleNew(1, &one, NPY_INT32);
        ((int *) PyArray_DATA((PyArrayObject *) indptr))[0] = 0;
        return Py_BuildValue("(OOO(ii))", data, indices, indptr, 0, prob->n_vars);
    }

    problem_jacobian(prob);

    CSR_Matrix *jac = prob->jacobian;
    npy_intp nnz = jac->nnz;
    npy_intp m_plus_1 = jac->m + 1;

    PyObject *data = PyArray_SimpleNew(1, &nnz, NPY_DOUBLE);
    PyObject *indices = PyArray_SimpleNew(1, &nnz, NPY_INT32);
    PyObject *indptr = PyArray_SimpleNew(1, &m_plus_1, NPY_INT32);

    if (!data || !indices || !indptr)
    {
        Py_XDECREF(data);
        Py_XDECREF(indices);
        Py_XDECREF(indptr);
        return NULL;
    }

    memcpy(PyArray_DATA((PyArrayObject *) data), jac->x, nnz * sizeof(double));
    memcpy(PyArray_DATA((PyArrayObject *) indices), jac->i, nnz * sizeof(int));
    memcpy(PyArray_DATA((PyArrayObject *) indptr), jac->p, m_plus_1 * sizeof(int));

    return Py_BuildValue("(OOO(ii))", data, indices, indptr, jac->m, jac->n);
}

static PyObject *py_get_jacobian(PyObject *self, PyObject *args)
{
    PyObject *prob_capsule;
    if (!PyArg_ParseTuple(args, "O", &prob_capsule))
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

    if (!prob->jacobian)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "jacobian not initialized - call problem_jacobian first");
        return NULL;
    }

    CSR_Matrix *jac = prob->jacobian;
    npy_intp nnz = jac->nnz;
    npy_intp m_plus_1 = jac->m + 1;

    PyObject *data = PyArray_SimpleNew(1, &nnz, NPY_DOUBLE);
    PyObject *indices = PyArray_SimpleNew(1, &nnz, NPY_INT32);
    PyObject *indptr = PyArray_SimpleNew(1, &m_plus_1, NPY_INT32);

    if (!data || !indices || !indptr)
    {
        Py_XDECREF(data);
        Py_XDECREF(indices);
        Py_XDECREF(indptr);
        return NULL;
    }

    memcpy(PyArray_DATA((PyArrayObject *) data), jac->x, nnz * sizeof(double));
    memcpy(PyArray_DATA((PyArrayObject *) indices), jac->i, nnz * sizeof(int));
    memcpy(PyArray_DATA((PyArrayObject *) indptr), jac->p, m_plus_1 * sizeof(int));

    return Py_BuildValue("(OOO(ii))", data, indices, indptr, jac->m, jac->n);
}

static PyObject *py_get_jacobian_sparsity_coo(PyObject *self, PyObject *args)
{
    PyObject *prob_capsule;
    if (!PyArg_ParseTuple(args, "O", &prob_capsule))
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

    if (!prob->jacobian_coo)
    {
        PyErr_SetString(
            PyExc_RuntimeError,
            "jacobian COO not initialized - call problem_init_jacobian_coo first");
        return NULL;
    }

    COO_Matrix *coo = prob->jacobian_coo;
    npy_intp nnz = coo->nnz;

    PyObject *rows = PyArray_SimpleNew(1, &nnz, NPY_INT32);
    PyObject *cols = PyArray_SimpleNew(1, &nnz, NPY_INT32);

    if (!rows || !cols)
    {
        Py_XDECREF(rows);
        Py_XDECREF(cols);
        return NULL;
    }

    memcpy(PyArray_DATA((PyArrayObject *) rows), coo->rows, nnz * sizeof(int));
    memcpy(PyArray_DATA((PyArrayObject *) cols), coo->cols, nnz * sizeof(int));

    return Py_BuildValue("(OO(ii))", rows, cols, coo->m, coo->n);
}

static PyObject *py_problem_eval_jacobian_vals(PyObject *self, PyObject *args)
{
    PyObject *prob_capsule;
    if (!PyArg_ParseTuple(args, "O", &prob_capsule))
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

    problem_jacobian(prob);

    CSR_Matrix *jac = prob->jacobian;
    npy_intp nnz = jac->nnz;

    PyObject *data = PyArray_SimpleNew(1, &nnz, NPY_DOUBLE);
    if (!data)
    {
        return NULL;
    }

    memcpy(PyArray_DATA((PyArrayObject *) data), jac->x, nnz * sizeof(double));

    return data;
}

#endif /* PROBLEM_JACOBIAN_H */
