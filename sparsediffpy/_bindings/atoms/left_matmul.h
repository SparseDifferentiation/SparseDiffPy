#ifndef ATOM_LEFT_MATMUL_H
#define ATOM_LEFT_MATMUL_H

#include "bivariate.h"
#include "common.h"

/* Left matrix multiplication: A @ f(x).
 *
 * Python signatures:
 *   make_left_matmul(param_or_none, child, "sparse", data, indices, indptr,
 *                    m, n)
 *   make_left_matmul(param_or_none, child, "dense", A_data_flat, m, n)
 *
 * - param_or_none: None for constant matrix, or a parameter capsule.
 * - child: the child expression capsule f(x). */
static PyObject *py_make_left_matmul(PyObject *self, PyObject *args)
{
    /* We need to parse the first 3 args to determine the format, then
     * parse the rest accordingly. Use a tuple to get the format string. */
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 4)
    {
        PyErr_SetString(PyExc_TypeError,
                        "make_left_matmul requires at least 4 arguments");
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

    expr *child =
        (expr *) PyCapsule_GetPointer(child_capsule, EXPR_CAPSULE_NAME);
    if (!child)
    {
        PyErr_SetString(PyExc_ValueError, "invalid child capsule");
        return NULL;
    }

    const char *fmt = PyUnicode_AsUTF8(fmt_obj);

    if (strcmp(fmt, "sparse") == 0)
    {
        /* Parse: param_or_none, child, "sparse", data, indices, indptr, m, n
         */
        PyObject *data_obj, *indices_obj, *indptr_obj;
        int m, n;
        if (!PyArg_ParseTuple(args, "OOsOOOii", &param_obj, &child_capsule,
                              &fmt, &data_obj, &indices_obj, &indptr_obj, &m,
                              &n))
        {
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

        expr *param_node = NULL;
        if (param_obj == Py_None)
        {
            param_node = new_parameter(
                nnz, 1, PARAM_FIXED, child->n_vars,
                (const double *) PyArray_DATA(data_array));
            if (!param_node)
            {
                Py_DECREF(data_array);
                Py_DECREF(indices_array);
                Py_DECREF(indptr_array);
                PyErr_SetString(
                    PyExc_RuntimeError,
                    "failed to create parameter node for matrix");
                return NULL;
            }
        }
        else
        {
            param_node = (expr *) PyCapsule_GetPointer(param_obj,
                                                       EXPR_CAPSULE_NAME);
            if (!param_node)
            {
                Py_DECREF(data_array);
                Py_DECREF(indices_array);
                Py_DECREF(indptr_array);
                PyErr_SetString(PyExc_ValueError,
                                "invalid parameter capsule");
                return NULL;
            }
        }

        CSR_Matrix *A = new_csr_matrix(m, n, nnz);
        memcpy(A->x, PyArray_DATA(data_array), nnz * sizeof(double));
        memcpy(A->i, PyArray_DATA(indices_array), nnz * sizeof(int));
        memcpy(A->p, PyArray_DATA(indptr_array), (m + 1) * sizeof(int));

        Py_DECREF(data_array);
        Py_DECREF(indices_array);
        Py_DECREF(indptr_array);

        expr *node = new_left_matmul(param_node, child, A);
        free_csr_matrix(A);

        if (!node)
        {
            if (param_obj == Py_None) free_expr(param_node);
            PyErr_SetString(PyExc_RuntimeError,
                            "failed to create left_matmul node");
            return NULL;
        }
        expr_retain(node);
        return PyCapsule_New(node, EXPR_CAPSULE_NAME,
                             expr_capsule_destructor);
    }
    else if (strcmp(fmt, "dense") == 0)
    {
        /* Parse: param_or_none, child, "dense", A_data_flat, m, n */
        PyObject *data_obj;
        int m, n;
        if (!PyArg_ParseTuple(args, "OOsOii", &param_obj, &child_capsule,
                              &fmt, &data_obj, &m, &n))
        {
            return NULL;
        }

        PyArrayObject *data_array = (PyArrayObject *) PyArray_FROM_OTF(
            data_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (!data_array)
        {
            return NULL;
        }

        double *A_data = (double *) PyArray_DATA(data_array);

        expr *param_node = NULL;
        if (param_obj == Py_None)
        {
            param_node = new_parameter(m * n, 1, PARAM_FIXED,
                                       child->n_vars, A_data);
            if (!param_node)
            {
                Py_DECREF(data_array);
                PyErr_SetString(PyExc_RuntimeError,
                                "failed to create parameter node");
                return NULL;
            }
        }
        else
        {
            param_node = (expr *) PyCapsule_GetPointer(param_obj,
                                                       EXPR_CAPSULE_NAME);
            if (!param_node)
            {
                Py_DECREF(data_array);
                PyErr_SetString(PyExc_ValueError,
                                "invalid parameter capsule");
                return NULL;
            }
        }

        expr *node =
            new_left_matmul_dense(param_node, child, m, n, A_data);
        Py_DECREF(data_array);

        if (!node)
        {
            if (param_obj == Py_None) free_expr(param_node);
            PyErr_SetString(PyExc_RuntimeError,
                            "failed to create dense left_matmul node");
            return NULL;
        }
        expr_retain(node);
        return PyCapsule_New(node, EXPR_CAPSULE_NAME,
                             expr_capsule_destructor);
    }
    else
    {
        PyErr_SetString(PyExc_ValueError,
                        "format must be 'sparse' or 'dense'");
        return NULL;
    }
}

#endif /* ATOM_LEFT_MATMUL_H */
