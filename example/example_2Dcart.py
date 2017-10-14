""" Sample 2D homogeneous problem with zero boundary conditions. """

import sys
sys.path.insert(0, '..')

from py_simplemg import MultigridOptions, SmootherOptions, solve_multigrid
import numpy as np

nx = 2**5+1
ny = nx
n = nx*ny
A = np.zeros((n, n))
b = np.zeros(n)
for ix in range(nx):
  for iy in range(ny):
    i = ix+iy*nx
    left  = ix-1+iy*nx
    right = ix+1+iy*nx
    south = ix+(iy-1)*nx
    north = ix+(iy+1)*nx

    if ix > 0:
      A[i, left] = -1
    if ix < nx-1:
      A[i, right] = -1
    if iy > 0:
      A[i, south] = -0.75
    if iy < ny-1:
      A[i, north] = -0.75
    A[i, i] = 3.7

x = np.zeros(n)
x[0:n//2] = 0.5

my_mg_opts = MultigridOptions(num_it=1,
                              num_level=4,
                              cycle='W',
                              geom_type='cart',
                              sparse=True)
my_mg_opts.dims = [nx, ny]
my_smooth_opts = SmootherOptions(smoothdown=1,
                                 smoothup=1,
                                 num_color=2,
                                 color_flip=False,
                                 sparse=True)
x = solve_multigrid(A, b, x, my_mg_opts, my_smooth_opts)
print("Final L2 error = ", np.linalg.norm(x, 2)/np.sqrt(len(x)))
print("Final Linf error = ", max(x))
