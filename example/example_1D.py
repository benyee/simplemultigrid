""" Sample 1D homogeneous problem with zero boundary conditions. """

import sys
sys.path.insert(0, '..')

from py_simplemg import MultigridOptions, SmootherOptions, solve_multigrid
import numpy as np

nx = 2**10+1
A = np.zeros((nx, nx))
b = np.zeros(nx)
for i in range(nx):
  if i > 0:
    A[i, i-1] = -1
  A[i, i] = 2.0001
  if i < nx-1:
    A[i, i+1] = -1

x = np.zeros(nx)
x[0:nx//2] = 0.5

my_mg_opts = MultigridOptions(num_it=1,
                              num_level=4,
                              cycle='W',
                              geom_type='1D',
                              sparse=True)
my_smooth_opts = SmootherOptions(smoothdown=1,
                                 smoothup=0,
                                 omega=1.0,
                                 num_color=2,
                                 color_flip=False,
                                 sparse=True)
print(solve_multigrid(A, b, x, my_mg_opts, my_smooth_opts))
