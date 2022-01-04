import sys
import numpy as np
sys.path.insert(1, '/home/liyipeng/ARPIST/python-implementation')

import spherical_integration as si
import math


xs = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
elems = np.array([[0, 1, 2]])
ARPIST_int = si.spherical_integration(xs, elems)
exact_int = math.pi/2

print("Numerical Integration: ", ARPIST_int)
print("Exact spherical integration: ", exact_int)
print("Relative error: ", abs(ARPIST_int[0] - exact_int)/exact_int)