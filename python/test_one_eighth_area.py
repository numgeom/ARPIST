import numpy as np
import compute_sphere_quadrature as csq
import math


xs = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
elems = np.array([[0, 1, 2]])
ARPIST_int = csq.spherical_integration(xs, elems)
exact_int = math.pi/2

print("Numerical Integration: ", ARPIST_int)
print("Exact spherical integration: ", exact_int)
print("Relative error: ", abs(ARPIST_int[0] - exact_int)/exact_int)