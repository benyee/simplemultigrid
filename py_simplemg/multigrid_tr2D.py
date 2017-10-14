""" This module defines the TR multigrid level class for 2D Cartesian problems.
    The TR 2D method is like the standard method for Cartesian grids, but there
    is an extra intermediate grid to maintain the coarsening factor of 2 fine
    cells per coarse cell.  This intermediate grid can be viewed as a rotated
    grid with the distance between points reduced by a factor of 1/sqrt(2).
"""

import numpy as np
import scipy.sparse as sp
from .multigrid_base import MultigridLevel_Base
from .smoothers import blk_jacobi

class MultigridLevel_TR2D(MultigridLevel_Base):
  """ Each instance represents a level of a multigrid solver for a 2D 
  Cartesian problem. Generation of the coarse grid is done via the TR method.

  To initialize a multigrid solver, initialize the finest level using the
  number of levels and the problem matrix.

  LEXICOGRAPHCIAL ORDERING IS ASSUMED.
  """
  def __init__(self, level, A, mg_opts, parent=None):
    self.dims = mg_opts.dims[:]
    self.num_dim = len(self.dims)

    if self.num_dim != 2:
      raise ValueError("This class can only be used for 2 dimensional"+\
                       "problems.")

    if parent == None or parent.is_rotated:
      self.is_rotated = False
    else:
      self.is_rotated = True

    tmpsize = self.dims[0]
    for dim in range(1, self.num_dim):
      tmpsize *= self.dims[dim]
    if (tmpsize != A.shape[0] and not self.is_rotated) or \
        ((tmpsize+1)//2 != A.shape[0] and self.is_rotated):
      raise ValueError("Provided domain dimensions inconsistent with " + \
                       "problem matrix size.")

    #Alter dims for the child class:
    if self.is_rotated:
      for dim in range(self.num_dim):
        mg_opts.dims[dim] = (mg_opts.dims[dim]-1)//2+1
    if level:
      self.child_dims = mg_opts.dims[:]

    super(self.__class__, self).__init__(level, A, mg_opts, parent)

    #Restore original dims so that the mg_opts object is the same as before:
    mg_opts.dims = self.dims

    self.has_color_list = False
    self.color_list = None

  def generate_interp(self):
    """ Defines the interpolation operator from level-1 to level.
        Restriction is assumed by default to be interp^T.

    """
    n = self.A.shape[0]
    if self.is_rotated:
      dims_coarse = self.dims[:]
      n_coarse = 1
      for dim in range(self.num_dim):
        dims_coarse[dim] = (dims_coarse[dim]-1)//2+1
        n_coarse = n_coarse*dims_coarse[dim]
    else:
      n_coarse = (n+1)//2

    if self.mg_opts.sparse:
      #Create matrix using lil_matrix, then convert to more
      #  computationally efficient csr_matrix.
      self.interpmat = sp.lil_matrix((n, n_coarse))
    else:
      self.interpmat = np.zeros((n, n_coarse))

    for i in range(n):
      coord = self.index_to_ijk(i)
      coarse_indices = self.get_coarse_neighbors(coord, not i%2)
      if i%2:
        self.interpmat[i, coarse_indices] = 1./(2*self.num_dim)
      else:
        self.interpmat[i, coarse_indices] = 1.

    if self.mg_opts.sparse:
      self.interpmat = sp.csr_matrix(self.interpmat)

    self.has_interp = True

  def get_coarse_neighbors(self, coord, isred):
    """ Given a coordinate on the fine grid, get the coarse grid global
        indices of the coarse grid cells used to determine its value.
    """
    new_coord = coord[:]
    #If it corresponds with a coarse-grid point/is a red point:
    if isred:
      if self.is_rotated:
        for dim in range(self.num_dim):
          new_coord[dim] = coord[dim]//2
      return self.ijk_to_index_child(new_coord)

    num_neigh = 2**self.num_dim
    list_of_neighs = ['']*num_neigh

    ineigh = 0
    if not self.is_rotated:
      for dim in range(self.num_dim):
        #Increment/decrement that index to find out the neighbors' neighbors
        for adjustment in (-1,1):
          new_coord[dim] = coord[dim]+adjustment
          list_of_neighs[ineigh] = self.get_coarse_neighbors(new_coord, True)
          if list_of_neighs[ineigh] != None:
            ineigh += 1
    else:
      adjustments = ((1,1),(1,-1),(-1,1),(-1,-1))
      for adjustment in adjustments:
        new_coord[0] = coord[0]+adjustment[0]
        new_coord[1] = coord[1]+adjustment[1]
        list_of_neighs[ineigh] = self.get_coarse_neighbors(new_coord, True)
        if list_of_neighs[ineigh] != None:
          ineigh += 1
    list_of_neighs = list_of_neighs[:ineigh]

    return list_of_neighs

  def index_to_ijk(self, i):
    """ Converts a 1-D index i to a [x_1,x_2,...x_n] list of indices where
        n is the number of dimensions.
    """
    tmp_i = i
    if self.is_rotated:
      tmp_i *= 2
    ijk = [0]*self.num_dim
    for dim in reversed(range(1, self.num_dim)):
      tmpsize = self.dims[0]
      for dim2 in range(1, dim):
        tmpsize *= self.dims[dim2]
      ijk[dim] = tmp_i//tmpsize
      tmp_i = tmp_i%tmpsize
    ijk[0] = tmp_i
    return ijk

  def ijk_to_index(self, ijk):
    """ Converts a [x_1,x_2,...x_n] list of indices to a 1-D index. """
    i = ijk[-1]
    for dim in reversed(range(self.num_dim-1)):
      i *= self.dims[dim]
      i += ijk[dim]
    if self.is_rotated:
      return i//2
    return i

  def ijk_to_index_child(self, ijk):
    """ Converts a [x_1,x_2,...x_n] list of indices to a 1-D index. """
    for dim in range(self.num_dim):
      if ijk[dim] >= self.child_dims[dim]:
        return None
      elif ijk[dim] < 0:
        return None

    i = ijk[-1]
    for dim in reversed(range(self.num_dim-1)):
      i *= self.child_dims[dim]
      i += ijk[dim]
    if not self.is_rotated:
      return i//2
    return i

  def smooth(self, x, b, smooth_opts):
    """ Performs a smoothing step. Right now, it is just RB Jacobi."""
    print("smoothing at level", self.level)
    x = blk_jacobi(self.A, x, b, smooth_opts)
    return x
