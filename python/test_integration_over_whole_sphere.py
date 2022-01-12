import numpy as np
import scipy.io
import compute_sphere_quadrature as csq
import math


# compute norm of a vector
def _compute_norm(vec):
    sqnorm = np.float64(0.0)
    for i in range(len(vec)):
        sqnorm += vec[i] * vec[i]
    sqnorm = np.sqrt(sqnorm)
    return sqnorm


# test functions
def f1(x):
    x = x / _compute_norm(x)
    return (1 + math.tanh((x[2] - x[0] - x[1]) * 9)) / 9


def f3(x):
    x = x / _compute_norm(x)
    return 0.5 + math.atan(300 * (x[2] - 0.9999)) / math.pi


def sin_cos_exp(x):
    x = x / _compute_norm(x)
    return math.exp(x[0]) * (x[1] ** 2 + x[0] * math.sin(x[1])) + x[1] * math.cos(x[2])


# test degrees
degs = [-1, 2, 4, 6, 8]
degree_name = [
    "adaptive ARPIST",
    "degree-2 ARPIST",
    "degree-4 ARPIST",
    "degree-6 ARPIST",
    "degree-8 ARPIST",
]
num_deg = len(degs)

# test functions
function_handles = [f1, f3, sin_cos_exp]
num_fun = len(function_handles)
exact_integrations = [4 * math.pi / 9, 0.049629692928687, 4 * math.pi / math.exp(1)]

# test meshs
mesh_path = "../meshes/"
mesh_names = [
    "SphereMesh_N=64_r=1.mat",
    "SphereMesh_N=256_r=2.mat",
    "SphereMesh_N=1024_r=3000.mat",
]
num_mesh = len(mesh_names)

# radius for different meshes
r = [0] * 3

# number of elements for each mesh
num_elems = [0] * 3

# Spherical Integration
for mesh_id in range(num_mesh):
    mesh_mat = scipy.io.loadmat(mesh_path + mesh_names[mesh_id])
    # 1-based data to 0-based data
    xs = mesh_mat["xs"]
    elems = mesh_mat["surfs"] - 1
    r[mesh_id] = _compute_norm(xs[0])
    num_elems[mesh_id] = elems.shape[0]
    print("====================================")
    print("Get into mesh ", mesh_id)

    for ii in range(num_deg):
        deg = degs[ii]

        # Compute cell area by integrating constant 1
        areas = csq.spherical_integration(xs, elems, lambda _: 1, deg)
        sum_area = sum(areas)
        print("Area of sphere for ", degree_name[ii], ": ", sum_area)
        exact_area = 4 * r[mesh_id] ** 2 * math.pi
        print("Relative error: ", abs(sum_area - exact_area) / exact_area)

        # compute integration of function using ARPIST
        if deg > 0:
            pnts, ws, offset = csq.compute_sphere_quadrature(xs, elems, 100, deg)
        else:
            pnts, ws, offset = csq.compute_sphere_quadrature(xs, elems)

        # number of quadrature points
        nqp = pnts.shape[0]
        # function value at each quadrature points
        fs_qp = np.array(
            [
                [
                    function_handles[function_id](pnts[qpid])
                    for function_id in range(num_fun)
                ]
                for qpid in range(nqp)
            ]
        )

        # numerical integrations
        nints = np.matmul(ws, fs_qp)

        for function_id in range(num_fun):
            exact_int = exact_integrations[function_id] * r[mesh_id] ** 2
            err = abs(nints[function_id] - exact_int) / exact_int

            # print out relative error
            print(
                "Relative error for function ",
                function_id,
                " with ",
                degree_name[ii],
                ": ",
                err,
            )
