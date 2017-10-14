""" Sample 3D homogeneous problem with zero boundary conditions. """

import sys
sys.path.insert(0, '..')

from py_simplemg import MultigridOptions, SmootherOptions, solve_multigrid
import numpy as np

nx = 2**4+1
ny = nx
nz = (nx-1)//2+1
n = nx*ny*nz
A = np.zeros((n, n))
b = np.zeros(n)
for ix in range(nx):
  for iy in range(ny):
    for iz in range(nz):
      i       = ix+nx*(iy+ny*iz)
      left    = ix-1+nx*(iy+ny*iz)
      right   = ix+1+nx*(iy+ny*iz)
      south   = ix+nx*(iy-1+ny*iz)
      north   = ix+nx*(iy+1+ny*iz)
      bottom  = ix+nx*(iy+ny*(iz-1))
      top     = ix+nx*(iy+ny*(iz+1))

      if ix > 0:
        A[i, left] = -1
      if ix < nx-1:
        A[i, right] = -1
      if iy > 0:
        A[i, south] = -1
      if iy < ny-1:
        A[i, north] = -1
      if iz > 0:
        A[i, bottom] = -1
      if iz < nz-1:
        A[i, top] = -1
      A[i, i] = 6.01

x = np.zeros(n)
x[0:n//2] = 0.5

my_mg_opts = MultigridOptions(num_it=1,
                              num_level=4,
                              cycle='W',
                              geom_type='cart',
                              sparse=True)
my_mg_opts.dims = [nx, ny, nz]
my_smooth_opts = SmootherOptions(smoothdown=1,
                                 smoothup=0,
                                 num_color=2,
                                 sparse=True)
x = solve_multigrid(A, b, x, my_mg_opts, my_smooth_opts)
print("Final L2 error = ", np.linalg.norm(x, 2)/np.sqrt(len(x)))
print("Final Linf error = ", max(x))
