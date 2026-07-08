#ifndef ATOM_KRON_H
#define ATOM_KRON_H

#include "common.h"

/* Kronecker product Z = kron(A, B), built sparse-only. One operand is the
 * variable-free constant/parameter (wrap constants with make_parameter and
 * param_id=-1 / PARAM_FIXED), the other carries the variables. active_blocks is
 * an int32 array of the constant operand's active (nonzero) column-major block
 * indices; only those output rows are materialized. For a parametric operand,
 * pass all block indices (dense, slow).
 *
 * Python signatures:
 *   make_left_kron(const_capsule, var_capsule, p, q, r, s, active_blocks)
 *   make_right_kron(const_capsule, var_capsule, p, q, r, s, active_blocks)
 * left_kron:  A = const (p x q), B = var (r x s); active_blocks index A.
 * right_kron: A = var (p x q),   B = const (r x s); active_blocks index B. */

/* is_left selects new_left_kron vs new_right_kron. */
static PyObject *py_make_kron_impl(PyObject *args, int is_left)
{
    PyObject *const_capsule, *var_capsule, *active_obj;
    int p, q, r, s;

    if (!PyArg_ParseTuple(args, "OOiiiiO", &const_capsule, &var_capsule, &p, &q,
                          &r, &s, &active_obj))
    {
        return NULL;
    }

    expr *param_node =
        (expr *) PyCapsule_GetPointer(const_capsule, EXPR_CAPSULE_NAME);
    if (!param_node)
    {
        PyErr_SetString(PyExc_ValueError, "invalid constant operand capsule");
        return NULL;
    }

    expr *child =
        (expr *) PyCapsule_GetPointer(var_capsule, EXPR_CAPSULE_NAME);
    if (!child)
    {
        PyErr_SetString(PyExc_ValueError, "invalid variable operand capsule");
        return NULL;
    }

    PyArrayObject *active_array = (PyArrayObject *) PyArray_FROM_OTF(
        active_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (!active_array)
    {
        return NULL;
    }
    int n_active = (int) PyArray_SIZE(active_array);
    const int *active_blocks = (const int *) PyArray_DATA(active_array);

    expr *node = is_left
                     ? new_left_kron(param_node, child, p, q, r, s,
                                     active_blocks, n_active)
                     : new_right_kron(param_node, child, p, q, r, s,
                                      active_blocks, n_active);

    Py_DECREF(active_array);

    if (!node)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create kron node");
        return NULL;
    }

    expr_retain(node); /* Capsule owns a reference */
    return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
}

static PyObject *py_make_left_kron(PyObject *self, PyObject *args)
{
    return py_make_kron_impl(args, 1);
}

static PyObject *py_make_right_kron(PyObject *self, PyObject *args)
{
    return py_make_kron_impl(args, 0);
}

#endif /* ATOM_KRON_H */
