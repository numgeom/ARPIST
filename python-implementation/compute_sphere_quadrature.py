"""Compute integration of elements on unit sphere

The module mainly supports computing areas and integration of elements on sphere
with high accuracy. The public interface compute_sphere_quadrature is a hybrid method
which will use the best setting automatically to find cell integration and areas.

.. moduleauthor:: Xiangmin Jiao, <xiangmin.jiao@stonybrook.edu>
.. moduleauthor:: Yipeng Li, <yipeng.li@stonybrook.edu>

"""

import math as mt
import numpy as np
import numba


# _TYPE_MAP = [("f4", "i4"), ("f8", "i4"), ("f4", "i8"), ("f8", "i8")]
_TYPE_MAP = [("f8", "i4"), ("f8", "i8")]
NB_OPTS = {"nogil": True}


def compute_sphere_int(xs, elems, f_D=lambda _: 1.0):
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

    nf = elems.shape[0]
    nv_surf = elems.shape[1]
    cell_int = np.zeros((nf, 1), dtype=xs.dtype)
    area_cell = np.zeros((nf, 1), dtype=xs.dtype)

    for fid in range(nf):
        nhe = nv_surf - 1
        while(elems[fid, nhe] < 0):
            nhe -= 1
        if(nhe < 2):
            continue
        h = _compute_max_edge_length(xs[elems[fid, :nhe + 1]])

        if(h < 0.004):
            cell_int[fid], area_cell[fid] = _cell_average_sphere_mix2(
                xs, elems[fid].reshape((1, -1)), f_D, 8)
        elif(h < 0.09):
            cell_int[fid], area_cell[fid] = _cell_average_sphere_mix2(
                xs, elems[fid].reshape((1, -1)), f_D, 8)
        else:
            cell_int[fid], area_cell[fid] = _cell_average_sphere_mix_split(
                xs, elems[fid].reshape((1, -1)), f_D, 8)

    return cell_int, area_cell


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


@numba.njit(["Tuple((f8[:],f8[:,:]))({0})".format(x)
            for x in ("i4", "i8")], **NB_OPTS)
def _fe2_quadrule(deg):
    """Quadrature rule used in this function

    Parameters
    ----------
    deg:            integer, degree

    Returns
    ----------
    ws:             n-by-1 array single or double, weights of quadrature points
    cs:             n-by-2 array single or double,
                    natural coordinates of quadrature points
    """
    if(deg <= 4):
        ws = np.array([0.0549758718276609338191631624501052,
                       0.0549758718276609338191631624501052,
                       0.0549758718276609338191631624501052,
                       0.111690794839005732847503504216561,
                       0.111690794839005732847503504216561,
                       0.111690794839005732847503504216561])
        cs = np.array([[0.8168475729804585130808570731956, 0.0915762135097707434595714634022015],
                       [0.0915762135097707434595714634022015,
                           0.8168475729804585130808570731956],
                       [0.0915762135097707434595714634022015,
                        0.0915762135097707434595714634022015],
                       [0.1081030181680702273633414922339,
                        0.445948490915964886318329253883051],
                       [0.445948490915964886318329253883051,
                        0.1081030181680702273633414922339],
                       [0.445948490915964886318329253883051, 0.445948490915964886318329253883051]])
    elif(deg <= 8):
        cs = np.array([[0.33333333333333333333333333333333, 0.33333333333333333333333333333333],
                       [0.1705693077517602066222935014994,
                           0.1705693077517602066222935014994],
                       [0.1705693077517602066222935014994,
                           0.65886138449647958675541299700121],
                       [0.65886138449647958675541299700121,
                        0.1705693077517602066222935014994],
                       [0.050547228317030975458423550596387,
                        0.050547228317030975458423550596387],
                       [0.050547228317030975458423550596387,
                        0.89890554336593804908315289880723],
                       [0.89890554336593804908315289880723,
                        0.050547228317030975458423550596387],
                       [0.45929258829272315602881551450124,
                        0.45929258829272315602881551450124],
                       [0.45929258829272315602881551450124,
                        0.081414823414553687942368970997513],
                       [0.081414823414553687942368970997513,
                        0.45929258829272315602881551450124],
                       [0.72849239295540428124100037918962,
                        0.26311282963463811342178578626121],
                       [0.26311282963463811342178578626121,
                        0.72849239295540428124100037918962],
                       [0.72849239295540428124100037918962,
                        0.0083947774099576053372138345491687],
                       [0.0083947774099576053372138345491687,
                        0.72849239295540428124100037918962],
                       [0.26311282963463811342178578626121,
                        0.0083947774099576053372138345491687],
                       [0.0083947774099576053372138345491687, 0.26311282963463811342178578626121]])
        ws = np.array([0.072157803838893584125545555249701,
                       0.051608685267359125140895775145648,
                       0.051608685267359125140895775145648,
                       0.051608685267359125140895775145648,
                       0.016229248811599040155462964170437,
                       0.016229248811599040155462964170437,
                       0.016229248811599040155462964170437,
                       0.047545817133642312396948052190887,
                       0.047545817133642312396948052190887,
                       0.047545817133642312396948052190887,
                       0.013615157087217497132422345038231,
                       0.013615157087217497132422345038231,
                       0.013615157087217497132422345038231,
                       0.013615157087217497132422345038231,
                       0.013615157087217497132422345038231,
                       0.013615157087217497132422345038231])
    else:
        cs = np.array([[0.48821738977380488256466173878598, 0.48821738977380488256466173878598],
                       [0.48821738977380488256466173878598,
                           0.023565220452390234870676522428033],
                       [0.023565220452390234870676522428033,
                        0.48821738977380488256466173878598],
                       [0.43972439229446027297973620450348,
                        0.43972439229446027297973620450348],
                       [0.43972439229446027297973620450348,
                        0.12055121541107945404052759099305],
                       [0.12055121541107945404052759099305,
                        0.43972439229446027297973620450348],
                       [0.27121038501211592234595160781199,
                        0.27121038501211592234595160781199],
                       [0.27121038501211592234595160781199,
                        0.45757922997576815530809678437601],
                       [0.45757922997576815530809678437601,
                        0.27121038501211592234595160781199],
                       [0.12757614554158592467389281696323,
                        0.12757614554158592467389281696323],
                       [0.12757614554158592467389281696323,
                        0.74484770891682815065221436607355],
                       [0.74484770891682815065221436607355,
                        0.12757614554158592467389281696323],
                       [0.021317350453210370246857737134961,
                        0.021317350453210370246857737134961],
                       [0.021317350453210370246857737134961,
                        0.95736529909357925950628452573008],
                       [0.95736529909357925950628452573008,
                        0.021317350453210370246857737134961],
                       [0.11534349453469799916901160654623,
                        0.2757132696855141939747907691782],
                       [0.2757132696855141939747907691782,
                           0.11534349453469799916901160654623],
                       [0.11534349453469799916901160654623,
                        0.60894323577978780685619762427557],
                       [0.60894323577978780685619762427557,
                        0.11534349453469799916901160654623],
                       [0.2757132696855141939747907691782,
                           0.60894323577978780685619762427557],
                       [0.60894323577978780685619762427557,
                        0.2757132696855141939747907691782],
                       [0.022838332222257029610233386418649,
                        0.28132558098993954824813282149259],
                       [0.28132558098993954824813282149259,
                        0.022838332222257029610233386418649],
                       [0.022838332222257029610233386418649,
                        0.69583608678780342214163379208876],
                       [0.69583608678780342214163379208876,
                        0.022838332222257029610233386418649],
                       [0.28132558098993954824813282149259,
                        0.69583608678780342214163379208876],
                       [0.69583608678780342214163379208876,
                        0.28132558098993954824813282149259],
                       [0.11625191590759714124135593566697,
                        0.025734050548330228168108745174704],
                       [0.025734050548330228168108745174704,
                        0.85801403354407263059053531915832],
                       [0.85801403354407263059053531915832,
                        0.025734050548330228168108745174704],
                       [0.11625191590759714124135593566697,
                        0.85801403354407263059053531915832],
                       [0.85801403354407263059053531915832, 0.11625191590759714124135593566697]])

        ws = np.array([0.012865533220227667708895587247731,
                       0.012865533220227667708895587247731,
                       0.012865533220227667708895587247731,
                       0.021846272269019201067729355264938,
                       0.021846272269019201067729355264938,
                       0.021846272269019201067729355264938,
                       0.031429112108942550177134995670765,
                       0.031429112108942550177134995670765,
                       0.031429112108942550177134995670765,
                       0.017398056465354471494663093004469,
                       0.017398056465354471494663093004469,
                       0.017398056465354471494663093004469,
                       0.0030831305257795086169334151704928,
                       0.0030831305257795086169334151704928,
                       0.0030831305257795086169334151704928,
                       0.020185778883190464758914841227262,
                       0.020185778883190464758914841227262,
                       0.020185778883190464758914841227262,
                       0.020185778883190464758914841227262,
                       0.020185778883190464758914841227262,
                       0.020185778883190464758914841227262,
                       0.011178386601151722855919352997536,
                       0.011178386601151722855919352997536,
                       0.011178386601151722855919352997536,
                       0.011178386601151722855919352997536,
                       0.011178386601151722855919352997536,
                       0.011178386601151722855919352997536,
                       0.0086581155543294461858209159291448,
                       0.0086581155543294461858209159291448,
                       0.0086581155543294461858209159291448,
                       0.0086581155543294461858209159291448,
                       0.0086581155543294461858209159291448,
                       0.0086581155543294461858209159291448])
    return ws, cs


@numba.njit(
    [
        "void({0}[:,:],{1}[:,:],{0}[:],{1})".format(
            t[0], t[1]
        )
        for t in _TYPE_MAP
    ],
    **NB_OPTS
)
def _area_average_sphere_mix(xs, elems, areas, deg=8):
    """Find areas of elements on sphere of a mixed mesh.

    Parameters
    ----------
    xs:             n-by-3 array single or double, coordinates of vertices
    elems:          n-by-m array int, connectivity table
    deg:            int, degree

    Returns
    ----------
    areas:          n-by-1 array single or double, the area of each element
    """

    nf = elems.shape[0]
    # areas = np.zeros(nf, dtype=xs.dtype)
    nv_surf = elems.shape[1]
    pnts_q = np.zeros(3)
    ws, cs0 = _fe2_quadrule(deg)
    nqp = ws.shape[0]
    # cs=[ones(nqp,1)-cs(:,1)-cs(:,2), cs];
    # cs = np.array([[1 - cs0[row1, 0] - cs0[row1, 1], cs0[row1, 0], cs0[row1, 1]] for row1 in range(nqp)])
    cs = np.zeros((nqp, 3))
    for qid in range(nqp):
        cs[qid, 0] = 1.0 - cs0[qid, 0] - cs0[qid, 1]
        cs[qid, 1] = cs0[qid, 0]
        cs[qid, 2] = cs0[qid, 1]

    for fid in range(nf):
        nhe = nv_surf - 1
        while(elems[fid, nhe] < 0):
            nhe -= 1
        if(nhe < 2):
            continue

        for j in range(1, nhe):
            # absolute value of triple product of x1, x2, x3.
            tri_pro = abs(_compute_dot(xs[elems[fid, 0]],
                                       _cross(xs[elems[fid, j]], xs[elems[fid, j + 1]])))

            # global coordinate of quadrature points on triangle x1x2x3
            for q in range(nqp):
                pnts_q = cs[q, 0] * xs[elems[fid, 0]] + \
                    cs[q, 1] * xs[elems[fid, j]] + \
                    cs[q, 2] * xs[elems[fid, j + 1]]

                nrm_q = _compute_norm(pnts_q)
                # weights x Jacobi
                w_j = ws[q] * tri_pro / (nrm_q**3)
                areas[fid] += w_j

    # return areas


def _cell_average_sphere_mix2(xs, elems, f_D, deg=8):
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

    import computeSphericalCartesianTransforms as sphcrt
    nf = elems.shape[0]
    cell_int = np.zeros((nf, 1), dtype=xs.dtype)
    areas = np.zeros((nf, 1), dtype=xs.dtype)
    nv_surf = elems.shape[1]
    pnts_q = np.zeros((1, 3))
    ws, cs0 = _fe2_quadrule(deg)
    nqp = ws.shape[0]
    cs = np.array([[1 - cs0[row1, 0] - cs0[row1, 1],
                  cs0[row1, 0], cs0[row1, 1]] for row1 in range(nqp)])
    # cs=[ones(nqp,1)-cs(:,1)-cs(:,2), cs];
    for fid in range(nf):
        nhe = nv_surf - 1
        while(elems[fid, nhe] < 0):
            nhe -= 1
        if(nhe < 2):
            continue

        for j in range(1, nhe):
            # absolute value of triple product of x1, x2, x3.
            tri_pro = abs(_compute_dot(xs[elems[fid, 0]],
                                       _cross(xs[elems[fid, j]], xs[elems[fid, j + 1]])))

            # global coordinate of quadrature points on triangle x1x2x3
            for q in range(nqp):
                pnts_q = cs[q, 0] * xs[elems[fid, 0]] + \
                    cs[q, 1] * xs[elems[fid, j]] + \
                    cs[q, 2] * xs[elems[fid, j + 1]]

                nrm_q = _compute_norm(pnts_q)
                # project quadrature points on sphere
                sph_q = pnts_q / nrm_q
                sph_llq = sphcrt.computePointCart2LL(sph_q)
                # weights x Jacobi
                w_j = ws[q] * tri_pro / (nrm_q**3)
                cell_int[fid] = cell_int[fid] + w_j * \
                    f_D(lon=sph_llq[0], lat=sph_llq[1])
                areas[fid] += w_j

    return cell_int, areas


def _cell_average_sphere_mix_split(xs, elems, f_D, deg=8):
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

    if(deg >= 8):
        tol = 0.05
    else:
        tol = 0.003

    nf = elems.shape[0]
    nv_surf = elems.shape[1]
    cell_int = np.zeros((nf, 1), dtype=xs.dtype)
    area_cell = np.zeros((nf, 1), dtype=xs.dtype)

    ws, cs0 = _fe2_quadrule(deg)
    nqp = ws.shape[0]
    cs = np.array([[1 - cs0[row1, 0] - cs0[row1, 1],
                  cs0[row1, 0], cs0[row1, 1]] for row1 in range(nqp)])

    # split one element
    surf_fid = np.zeros((16, 3), dtype='i4')
    pnts_vor = np.zeros((12, 3), dtype=xs.dtype)

    for fid in range(nf):
        nhe = nv_surf - 1
        while(elems[fid, nhe] < 0):
            nhe -= 1
        if(nhe < 2):
            continue

        h = _compute_max_edge_length(xs[elems[fid, :nhe + 1]])
        if(h > tol):
            pnts_vor[:nhe + 1] = xs[elems[fid, :nhe + 1]]

            # insert elements and points
            index = nhe + 1
            pnts_vor[index] = (pnts_vor[0] + pnts_vor[1]) / 2.0
            pnts_vor[index] = pnts_vor[index] / _compute_norm(pnts_vor[index])
            for j in range(1, nhe):
                # insert elements
                surf_fid[j * 4 - 4] = [0, index, index + 2]
                surf_fid[j * 4 - 3] = [index + 2, index, index + 1]
                surf_fid[j * 4 - 2] = [index + 1, index, j]
                surf_fid[j * 4 - 1] = [index + 2, index + 1, j + 1]

                # insert points
                index += 1
                pnts_vor[index] = (pnts_vor[j] + pnts_vor[j + 1]) / 2.0
                pnts_vor[index] = pnts_vor[index] / \
                    _compute_norm(pnts_vor[index])
                index += 1
                pnts_vor[index] = (pnts_vor[0] + pnts_vor[j + 1]) / 2.0
                pnts_vor[index] = pnts_vor[index] / \
                    _compute_norm(pnts_vor[index])

            # recursive
            cell_int_fid, area_cell_fid = _cell_average_sphere_mix_split(
                pnts_vor, surf_fid[:(nhe - 1) * 4], f_D, deg)
            area_cell[fid] = sum(area_cell_fid)
            cell_int[fid] = sum(cell_int_fid)
        else:
            cell_int[fid], area_cell[fid] = _cell_average_sphere_mix2(
                xs, elems[fid].reshape((1, -1)), f_D, deg)

    return cell_int, area_cell


@numba.njit(["{0}({0}[:,:])".format("f8")], **NB_OPTS)
def _compute_max_edge_length(xs):
    # compute maximum edge length of elements

    nv_surf = xs.shape[0]
    next_id = [col1 + 1 for col1 in range(nv_surf)]
    next_id[-1] = 0
    h = 0.0

    for leid in range(nv_surf):
        h = max(h, _compute_norm(xs[leid] - xs[next_id[leid]]))

    return h