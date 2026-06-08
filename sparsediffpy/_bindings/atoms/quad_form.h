#ifndef ATOM_QUAD_FORM_H
#define ATOM_QUAD_FORM_H

#include "common.h"
#include "non_elementwise_full_dom.h"

/* Quadratic form: y = x' Q x.
 *
 * Python signatures (mirroring make_left_matmul):
 *   make_quad_form(None,  child, "sparse", data, indices, indptr, m, n)
 *   make_quad_form(None,  child, "dense",  P_data_flat, n)
 *   make_quad_form(src,   child, "dense",  None,        n)
 *
 * - "sparse": Q is a constant CSR matrix (param must be None).
 * - "dense":  Q is a dense n x n symmetric matrix, either a constant buffer
 *             (src None, P_data given) or supplied by a "source" expression
 *             capsule (src given, P_data None) -- a parameter, or any
 *             matrix-valued expression of parameters (e.g. reshape(M @ theta)) --
 *             which is re-evaluated each solve.
 */
static PyObject *py_make_quad_form(PyObject *self, PyObject *args)
{
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 4)
    {
        PyErr_SetString(PyExc_TypeError,
                        "make_quad_form requires at least 4 arguments");
        return NULL;
    }

    PyObject *param_obj = PyTuple_GetItem(args, 0);
    PyObject *child_capsule = PyTuple_GetItem(args, 1);
    PyObject *fmt_obj = PyTuple_GetItem(args, 2);

    if (!PyUnicode_Check(fmt_obj))
    {
        PyErr_SetString(PyExc_TypeError,
                        "third argument must be 'sparse' or 'dense'");
        return NULL;
    }

    expr *child = (expr *) PyCapsule_GetPointer(child_capsule, EXPR_CAPSULE_NAME);
    if (!child)
    {
        PyErr_SetString(PyExc_ValueError, "invalid child capsule");
        return NULL;
    }

    const char *fmt = PyUnicode_AsUTF8(fmt_obj);

    if (strcmp(fmt, "sparse") == 0)
    {
        /* Parse: param_or_none, child, "sparse", data, indices, indptr, m, n */
        PyObject *data_obj, *indices_obj, *indptr_obj;
        int m, n;
        if (!PyArg_ParseTuple(args, "OOsOOOii", &param_obj, &child_capsule, &fmt,
                              &data_obj, &indices_obj, &indptr_obj, &m, &n))
        {
            return NULL;
        }
        if (param_obj != Py_None)
        {
            PyErr_SetString(PyExc_ValueError,
                            "parameter for a sparse quad_form is not supported");
            return NULL;
        }

        PyArrayObject *data_array = (PyArrayObject *) PyArray_FROM_OTF(
            data_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        PyArrayObject *indices_array = (PyArrayObject *) PyArray_FROM_OTF(
            indices_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
        PyArrayObject *indptr_array = (PyArrayObject *) PyArray_FROM_OTF(
            indptr_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);

        if (!data_array || !indices_array || !indptr_array)
        {
            Py_XDECREF(data_array);
            Py_XDECREF(indices_array);
            Py_XDECREF(indptr_array);
            return NULL;
        }

        int nnz = (int) PyArray_SIZE(data_array);
        CSR_matrix *Q = new_CSR_matrix(m, n, nnz);
        memcpy(Q->x, PyArray_DATA(data_array), nnz * sizeof(double));
        memcpy(Q->i, PyArray_DATA(indices_array), nnz * sizeof(int));
        memcpy(Q->p, PyArray_DATA(indptr_array), (m + 1) * sizeof(int));

        Py_DECREF(data_array);
        Py_DECREF(indices_array);
        Py_DECREF(indptr_array);

        expr *node = new_quad_form_sparse(child, Q);
        free_CSR_matrix(Q);

        if (!node)
        {
            PyErr_SetString(PyExc_RuntimeError, "failed to create quad_form node");
            return NULL;
        }
        expr_retain(node); /* Capsule owns a reference */
        return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
    }
    else if (strcmp(fmt, "dense") == 0)
    {
        /* Parse: param_or_none, child, "dense", P_data_or_none, n.
         * (param_node, P_data) are mutually exclusive, like left_matmul. */
        PyObject *data_obj;
        int n;
        if (!PyArg_ParseTuple(args, "OOsOi", &param_obj, &child_capsule, &fmt,
                              &data_obj, &n))
        {
            return NULL;
        }

        expr *node;
        if (param_obj == Py_None)
        {
            PyArrayObject *data_array = (PyArrayObject *) PyArray_FROM_OTF(
                data_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
            if (!data_array)
            {
                return NULL;
            }
            double *P_data = (double *) PyArray_DATA(data_array);
            node = new_quad_form_dense(child, n, P_data, NULL);
            Py_DECREF(data_array);
        }
        else
        {
            expr *param_node =
                (expr *) PyCapsule_GetPointer(param_obj, EXPR_CAPSULE_NAME);
            if (!param_node)
            {
                PyErr_SetString(PyExc_ValueError, "invalid parameter capsule");
                return NULL;
            }
            node = new_quad_form_dense(child, n, NULL, param_node);
        }

        if (!node)
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "failed to create dense quad_form node");
            return NULL;
        }
        expr_retain(node); /* Capsule owns a reference */
        return PyCapsule_New(node, EXPR_CAPSULE_NAME, expr_capsule_destructor);
    }
    else
    {
        PyErr_SetString(PyExc_ValueError, "format must be 'sparse' or 'dense'");
        return NULL;
    }
}

#endif /* ATOM_QUAD_FORM_H */
