#ifndef PROBLEM_CSR_VIEWS_H
#define PROBLEM_CSR_VIEWS_H

#include "common.h"

/* The Jacobian and the Lagrangian Hessian as read-only numpy views of the
 * engine's own CSR storage, plus evaluations that return nothing. The Python
 * layer wraps the views in one csr_matrix per problem, once; an evaluation then
 * only refreshes the values in place, with no copy and no new array.
 *
 * Each view holds a reference to the problem capsule, so the storage stays
 * alive for as long as any view does. The storage is allocated by
 * problem_init_derivatives, which must have run, and never moves afterwards. */

/* Read-only 1-D view of `len` items at `ptr`, kept alive by `owner`. */
static PyObject *csr_view(void *ptr, npy_intp len, int typenum, PyObject *owner)
{
    PyObject *arr;
    if (len == 0)
    {
        /* nothing to view, and ptr may be NULL, which numpy must not wrap */
        arr = PyArray_SimpleNew(1, &len, typenum);
        if (!arr) return NULL;
    }
    else
    {
        arr = PyArray_SimpleNewFromData(1, &len, typenum, ptr);
        if (!arr) return NULL;
        Py_INCREF(owner);
        if (PyArray_SetBaseObject((PyArrayObject *) arr, owner) < 0)
        {
            Py_DECREF(arr);
            return NULL;
        }
    }
    PyArray_CLEARFLAGS((PyArrayObject *) arr, NPY_ARRAY_WRITEABLE);
    return arr;
}

/* (data, indices, indptr, (m, n)) viewing A */
static PyObject *csr_views(CSR_matrix *A, PyObject *owner)
{
    PyObject *data = csr_view(A->x, A->nnz, NPY_DOUBLE, owner);
    PyObject *indices = csr_view(A->i, A->nnz, NPY_INT32, owner);
    PyObject *indptr = csr_view(A->p, (npy_intp) A->m + 1, NPY_INT32, owner);
    if (!data || !indices || !indptr)
    {
        Py_XDECREF(data);
        Py_XDECREF(indices);
        Py_XDECREF(indptr);
        return NULL;
    }
    return Py_BuildValue("(NNN(ii))", data, indices, indptr, A->m, A->n);
}

/* problem_jacobian_csr(prob): views of the constraint Jacobian */
static PyObject *py_problem_jacobian_csr(PyObject *self, PyObject *capsule)
{
    (void) self;
    problem *prob = (problem *) PyCapsule_GetPointer(capsule, PROBLEM_CAPSULE_NAME);
    if (!prob) return NULL;
    return csr_views(prob->jacobian, capsule);
}

/* problem_hessian_csr(prob): views of the Lagrangian Hessian */
static PyObject *py_problem_hessian_csr(PyObject *self, PyObject *capsule)
{
    (void) self;
    problem *prob = (problem *) PyCapsule_GetPointer(capsule, PROBLEM_CAPSULE_NAME);
    if (!prob) return NULL;
    return csr_views(prob->lagrange_hessian, capsule);
}

/* problem_jacobian_eval(prob): refresh the Jacobian values in place */
static PyObject *py_problem_jacobian_eval(PyObject *self, PyObject *capsule)
{
    (void) self;
    problem *prob = (problem *) PyCapsule_GetPointer(capsule, PROBLEM_CAPSULE_NAME);
    if (!prob) return NULL;
    problem_jacobian(prob);
    Py_RETURN_NONE;
}

/* problem_hessian_eval(prob, obj_factor, lagrange): refresh the Hessian values
 * in place. lagrange holds one multiplier per constraint row. */
static PyObject *py_problem_hessian_eval(PyObject *self, PyObject *args)
{
    (void) self;
    PyObject *capsule;
    double obj_factor;
    PyObject *lagrange_obj;
    if (!PyArg_ParseTuple(args, "OdO", &capsule, &obj_factor, &lagrange_obj))
    {
        return NULL;
    }
    problem *prob = (problem *) PyCapsule_GetPointer(capsule, PROBLEM_CAPSULE_NAME);
    if (!prob) return NULL;

    PyArrayObject *lagrange = (PyArrayObject *) PyArray_FROM_OTF(
        lagrange_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!lagrange) return NULL;
    problem_hessian(prob, obj_factor, (double *) PyArray_DATA(lagrange));
    Py_DECREF(lagrange);
    Py_RETURN_NONE;
}

#endif /* PROBLEM_CSR_VIEWS_H */
