"""Compute integration of elements on unit sphere

The module mainly supports computing areas and integration of elements on sphere
with high accuracy. The public interface compute_sphere_quadrature is a hybrid method
which will use the best setting automatically to find the weights and coordinates of
quadrature points for cell integration and areas.

.. moduleauthor:: Xiangmin Jiao, <xiangmin.jiao@stonybrook.edu>
.. moduleauthor:: Yipeng Li, <yipeng.li@stonybrook.edu>

"""

# Copyright (C) 2022 NumGeom Group at Stony Brook University

import numpy as np
import numba
import quadrature_rule


# _TYPE_MAP = [("f4", "i4"), ("f8", "i4"), ("f4", "i8"), ("f8", "i8")]
_TYPE_MAP = [("f8", "i4"), ("f8", "i8")]
NB_OPTS = {"nogil": True}


def spherical_integration(xs, elems, fun_handle=lambda _: 1.0, deg=-1):

    # generate quadrature points
    # This part could be reused for multiple functions
    if deg > 0:
        # if degree is set up, use the given degree
        pnts, ws, offset = compute_sphere_quadrature(xs, elems, 100, deg)
    else:
        # if degree is not set up, use our configuration for adaptive ARPIST
        pnts, ws, offset = compute_sphere_quadrature(xs, elems)

    # evaluate function values on each element
    # This part could be modified to deal with multiple functions
    nf = elems.shape[0]
    fs = [0] * nf

    for fid in range(nf):
        for pid in range(offset[fid], offset[fid + 1]):
            fs[fid] += fun_handle(pnts[pid]) * ws[pid]

    return fs


def compute_sphere_quadrature(xs, elems, h1=0.004, deg1=4, h2=0.05, deg2=8):
    """Find cell integration for test function f on sphere of a mixed mesh.
    It is a hybrid method which will automatically use the best setting.

    Parameters
    ----------
    xs:             n-by-3 array single or double, coordinates of vertices
    elems:          n-by-m array integer, connectivity table
    f_D:            a function handle that takes a coordinate and a value

    Returns
    ----------
    cell_int:       n-by-1 array single or double, integration on elements
    areas:          n-by-1 array single or double, areas of elements
    """

    # radius of the sphere
    r = _compute_norm(xs[0])

    for vid in range(xs.shape[0]):
        assert abs(_compute_norm(xs[vid]) - r) < 2e-6, "The input mesh is not a sphere"

    # initialization
    index = 0
    nf = elems.shape[0]
    nv_surf = elems.shape[1]
    max_nv = max(1000000, nf * 6)

    pnts = np.array([[0.0] * 3 for pid in range(max_nv)])
    ws = np.array([0.0 for pid in range(max_nv)])
    offset = np.array([0 for fid in range(nf + 1)])

    # go through all the faces
    for fid in range(nf):
        offset[fid] = index

        nhe = nv_surf - 1
        while elems[fid, nhe] < 0:
            nhe -= 1
        if nhe < 2:
            continue

        # split each element into several spherical triangles
        for j in range(1, nhe):
            lvids = [0, j, j + 1]
            pnts_tri = xs[elems[fid, lvids]] / r
            h = _compute_max_edge_length(pnts_tri)

            # generate quadrature points
            if h < h1:
                index = _quadrature_sphere_tri(
                    pnts_tri, np.array([[0, 1, 2]]), deg1, pnts, ws, index
                )
            elif h < h2:
                index = _quadrature_sphere_tri(
                    pnts_tri, np.array([[0, 1, 2]]), deg2, pnts, ws, index
                )
            else:
                index = _quadrature_sphere_tri_split(
                    pnts_tri, np.array([[0, 1, 2]]), h2, deg2, pnts, ws, index
                )

    pnts = r * pnts[:index]
    ws = (r * r) * ws[:index]
    offset[nf] = index
    return pnts, ws, offset


@numba.njit(["{0}({0}[:])".format("f8")], **NB_OPTS)
def _compute_norm(vec):
    sqnorm = np.float64(0.0)
    for i in range(len(vec)):
        sqnorm += vec[i] * vec[i]
    sqnorm = np.sqrt(sqnorm)
    return sqnorm


@numba.njit(["{0}({0}[:], {0}[:])".format("f8")], **NB_OPTS)
def _compute_dot(vec1, vec2):
    dotprdt = 0
    for i in range(len(vec1)):
        dotprdt += vec1[i] * vec2[i]
    return dotprdt


@numba.njit(["{0}[:]({0}[:], {0}[:])".format("f8")], **NB_OPTS)
def _cross(a, b):
    r"""Cross product axb

    Parameters
    ----------
    a, b: np.ndarray
        nx3 coordinates

    Returns
    -------
    np.ndarray
        The cross product of :math:`\boldsymbol{a}\times\boldsymbol{b}`.
    """

    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


def _quadrature_sphere_tri(xs, elems, deg, pnts, ws, index):
    """Find cell integration for test function f on sphere of a mixed mesh.

    Parameters
    ----------
    xs:             n-by-3 array single or double, coordinates of vertices
    elems:          n-by-m array integer, connectivity table
    f_D:            a function handle that takes a coordinate and a value
    deg:            integer, degree

    Returns
    ----------
    cell_int:       n-by-1 array single or double, integration on elements
    areas:          n-by-1 array single or double, areas of elements
    """

    nf = elems.shape[0]
    pnts_q = np.zeros((1, 3), dtype=np.float64)
    ws0, cs = quadrature_rule.get_fe2_quadrule(deg)
    nqp = ws0.shape[0]

    # enlarge the size of quadrature points buffer if inadequate
    if index + nf * nqp > len(ws):
        n_new = 2 * len(ws) + nf * nqp
        ws.resize(n_new, refcheck=False)
        pnts.resize((n_new, 3), refcheck=False)

    for fid in range(nf):
        # absolute value of triple product of x1, x2, x3.
        tri_pro = abs(
            _compute_dot(
                xs[elems[fid, 0]],
                _cross(
                    xs[elems[fid, 1]] - xs[elems[fid, 0]],
                    xs[elems[fid, 2]] - xs[elems[fid, 0]],
                ),
            )
        )

        # global coordinate of quadrature points on triangle x1x2x3
        for q in range(nqp):
            pnts_q = (
                cs[q, 0] * xs[elems[fid, 0]]
                + cs[q, 1] * xs[elems[fid, 1]]
                + cs[q, 2] * xs[elems[fid, 2]]
            )

            nrm_q = _compute_norm(pnts_q)
            # project quadrature points on sphere
            pnts[index] = pnts_q / nrm_q
            # weights x Jacobi
            ws[index] = ws0[q] * tri_pro / (nrm_q**3)
            index = index + 1

    return index


def _quadrature_sphere_tri_split(xs, elems, tol, deg, pnts, ws, index):
    """Find cell integration for test function f on sphere of a mixed mesh.
    It will split the mesh until we can get to machine precision.

    Parameters
    ----------
    xs:             n-by-3 array single or double, coordinates of vertices
    elems:          n-by-m array integer, connectivity table
    f_D:            a function handle that takes a coordinate and a value
    deg:            integer, degree

    Returns
    ----------
    cell_int:       n-by-1 array single or double, integration on elements
    areas:          n-by-1 array single or double, areas of elements
    """

    nf = elems.shape[0]
    h = _compute_max_edge_length(xs[elems[0]])

    if h > tol:

        # split one element
        surf_fid = np.array([[0, 3, 5], [5, 3, 4], [4, 3, 1], [5, 4, 2]])
        pnts_vor = np.zeros((6, 3), dtype=xs.dtype)

        for fid in range(nf):
            pnts_vor[:3] = xs[elems[fid, :3]]

            # insert points
            for j in range(3):
                index_local = j + 3
                pnts_vor[index_local] = (pnts_vor[j] + pnts_vor[_next_leid(j, 3)]) / 2.0
                pnts_vor[index_local] = pnts_vor[index_local] / _compute_norm(
                    pnts_vor[index_local]
                )

            # recursive
            index = _quadrature_sphere_tri_split(
                pnts_vor, surf_fid, tol, deg, pnts, ws, index
            )
    else:
        index = _quadrature_sphere_tri(xs, elems, deg, pnts, ws, index)

    return index


@numba.njit(["{0}({0}[:,:])".format("f8")], **NB_OPTS)
def _compute_max_edge_length(xs):
    # compute maximum edge length of elements

    return max(
        _compute_norm(xs[0] - xs[1]),
        _compute_norm(xs[1] - xs[2]),
        _compute_norm(xs[2] - xs[0]),
    )


def _next_leid(i, n):
    i = i + 1
    if i == n:
        return 0
    else:
        return i
