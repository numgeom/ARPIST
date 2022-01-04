import sys
import numpy as np
sys.path.insert(1, '/home/liyipeng/ARPIST/python-implementation')

import spherical_integration as si


xs = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
elems = np.array([[0, 1, 2]])

print(si.spherical_integration(xs, elems))