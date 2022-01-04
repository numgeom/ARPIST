import compute_sphere_quadrature as csq


def spherical_integration(xs, elems, fun_handle=lambda _: 1.0, deg=-1):

    # generate quadrature points
    # This part could be reused for multiple functions
    if deg > 0:
        # if degree is set up, use the given degree
        pnts, ws, offset = csq.compute_sphere_quadrature(xs, elems, 100, deg)
    else:
        # if degree is not set up, use our configuration for adaptive ARPIST
        pnts, ws, offset = csq.compute_sphere_quadrature(xs, elems)

    # evaluate function values on each element
    # This part could be modified to deal with multiple functions
    nf = elems.shape[0]
    fs = [0] * nf

    for fid in range(nf):
        for pid in range(offset[fid], offset[fid + 1]):
            fs[fid] += fun_handle(pnts[pid]) * ws[pid]

    return fs

