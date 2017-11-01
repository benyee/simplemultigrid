""" Sample 1D homogeneous problem with zero boundary conditions using the
    CC MG technique. 
"""

import sys
sys.path.insert(0, '..')

from py_simplemg import MultigridOptions, SmootherOptions, solve_multigrid, BC
import numpy as np

nx = 2**9
A = np.zeros((nx, nx))
b = np.zeros(nx)
for i in range(nx):
  if i > 0:
    A[i, i-1] = -1
  A[i, i] = 2.0
  if i < nx-1:
    A[i, i+1] = -1

##Reflective boundary conditions:
#A[0, 0] = 2.01
#A[0, 1] = -2
#A[-1, -1] = 2.01
#A[-1, -2] = -2
# Zero-flux dirichlet boundary conditions are present if you don't
#  explicitly do anything to the matrix to account for b.c.'s

x = np.zeros(nx)
x[0:nx//2] = 0.5

my_mg_opts = MultigridOptions(num_it=10,
                              num_level=2,
                              cycle='W',
                              geom_type='cc1D',
                              bcs=(BC.ZERO, BC.ZERO),
                              sparse=True)
my_smooth_opts = SmootherOptions(smoothdown=1,
                                 smoothup=0,
                                 omega=1.0,
                                 num_color=2,
                                 color_flip=False,
                                 sparse=True)
x = solve_multigrid(A, b, x, my_mg_opts, my_smooth_opts)
print("Final L2 error = ", np.linalg.norm(x, 2)/np.sqrt(len(x)))
print("Final Linf error = ", max(x))
