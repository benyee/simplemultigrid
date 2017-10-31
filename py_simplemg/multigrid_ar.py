""" This module defines the multigrid level class for the multigrid-AR method.

    In short, the multigrid-AR method collapses one dimension at a time.
"""

import warnings
import numpy as np
import scipy.sparse as sp
from .multigrid_base import MultigridLevel_Base
from .smoothers import blk_jacobi

class MultigridLevel_AR(MultigridLevel_Base):
  """ Each instance represents a level of a multigrid solver for a general
      n-dimensional Cartesian problems using the multigrid-AR method.

  To initialize a multigrid solver, initialize the finest level using the
  number of levels and the problem matrix.

  LEXICOGRAPHCIAL ORDERING IS ASSUMED.
  """
  def __init__(self, level, A, mg_opts, parent=None):
    if parent is None:
      self.dims = mg_opts.dims[:]
    else:
      self.dims = parent.child_dims
    self.num_dim = len(self.dims)

    if self.num_dim > 3:
      warnings.warn("Though technically supported, the performance of MG-AR"+ \
                    " with >3 dimensions is untested and possibly poor.")
    elif self.num_dim < 3:
      raise ValueError("This only works for 3-D problems.")

    #Determine number of levels dynamically:
    if level < 0:
      level = 0
      #Find out how many times we can coarsen the z-dimension:
      tmpint = self.dims[-1]
      while tmpint > 1:
        level += 1
        tmpint = (tmpint+1)//2
      ##Find out how many times we can coarsen in the x/y dimensions:
      #for dim in self.dims[:-1]:
      #  tmpint = dim
      #  while tmpint > 5:
      #    level += 1
      #    tmpint = (tmpint+1)//2
      if level == 0:
        raise ValueError("Domain is too small for multigrid.")
      self.level = level

    tmpsize = self.dims[0]
    for dim in range(1, self.num_dim):
      tmpsize *= self.dims[dim]
    if tmpsize != A.shape[0]:
      raise ValueError("Provided domain dimensions inconsistent with " + \
                       "problem matrix size.")

    super(self.__class__, self).__init__(level, A, mg_opts, parent)

    self.has_color_list = False
    self.color_list = None

  def generate_interp(self):
    """ Defines the interpolation operator from level-1 to level.
        Restriction is assumed by default to be interp^T.

        It first collapses the "z" direction (assumed to the last dimension).
        Then, it alternates between collapsing the other 2 (or more)
        directions.
!ZZZZ need changes here...
    """
    n = self.A.shape[0]
    self.child_dims = self.dims[:]
    n_coarse = 1
    if self.dims[-1] > 1: #ZZZZ
      dim_to_collapse = -1
    else:
      dim_to_collapse = self.dims.index(max(self.dims))
    print "dim_to_collapse = ", dim_to_collapse, self.dims
    self.child_dims[dim_to_collapse] = (self.child_dims[dim_to_collapse]-1)//2+1
    n_coarse = n/self.dims[dim_to_collapse]*self.child_dims[dim_to_collapse]

    if self.mg_opts.sparse:
      #Create matrix using lil_matrix, then convert to more
      #  computationally efficient csr_matrix.
      self.interpmat = sp.lil_matrix((n, n_coarse))
    else:
      self.interpmat = np.zeros((n, n_coarse))

    for i in range(n):
      coord = self.index_to_ijk(i)

      #Determine interpolation degree:
      interpdegree = 0
      if coord[dim_to_collapse]%2 or self.dims[dim_to_collapse] == 2:
        interpdegree = 1

      wt = 1./(2**interpdegree)
      coarse_indices = self.get_coarse_neighbors(coord,
                                                 interpdegree,
                                                 dim_to_collapse)
      self.interpmat[i, coarse_indices] = wt
    #endfor i in range(n)

    if self.mg_opts.sparse:
      self.interpmat = sp.csr_matrix(self.interpmat)

    self.has_interp = True

  def get_coarse_neighbors(self, coord, interpdegree, dim_to_collapse):
    """ Given a coordinate on the fine grid, get the coarse grid global
        indices of the coarse grid cells used to determine its value.
        #ZZZZ well this need to be changed...
    """
    new_coord = coord[:]
    if interpdegree > 1:
      raise ValueError("interpdegree should be 0 or 1")
    if interpdegree == 0 or self.dims[dim_to_collapse] == 2:
      new_coord[dim_to_collapse] = coord[dim_to_collapse]//2
      return [self.ijk_to_index_child(new_coord),]
    num_neigh = 2#**interpdegree
    list_of_neighs = ['']*num_neigh

    #Increment/decrement that index to find out the neighbors' neighbors
    new_coord[dim_to_collapse] = coord[dim_to_collapse]+1
    list_of_neighs[:1] = \
      self.get_coarse_neighbors(new_coord, interpdegree-1, dim_to_collapse)

    new_coord[dim_to_collapse] = coord[dim_to_collapse]-1
    list_of_neighs[1:] = \
      self.get_coarse_neighbors(new_coord, interpdegree-1, dim_to_collapse)

    return list_of_neighs

  def index_to_ijk(self, i):
    """ Converts a 1-D index i to a [x_1,x_2,...x_n] list of indices where
        n is the number of dimensions.
    """
    tmp_i = i
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
    return i

  def ijk_to_index_child(self, ijk):
    """ Converts a [x_1,x_2,...x_n] list of indices to a 1-D index. """
    i = ijk[-1]
    for dim in reversed(range(self.num_dim-1)):
      i *= self.child_dims[dim]
      i += ijk[dim]
    return i

  def __define_color_list(self, smooth_opts):
    #Define color list if it doesn't already exist:
    if not self.has_color_list:
      self.color_list = ['']*smooth_opts.num_color
      for color in range(smooth_opts.num_color):
        tmpcolor_list = ['']*self.A.shape[0]
        tmpint = 0
        for ind in range(self.A.shape[0]):
          ijk = self.index_to_ijk(ind)
          interpdegree = 0
          for ijk_ind in ijk:
            if ijk_ind%2:
              interpdegree += 1
          if interpdegree%smooth_opts.num_color == color:
            tmpcolor_list[tmpint] = ind
            tmpint += 1
        #endfor ind in range(self.A.shape[0])
        self.color_list[color] = tmpcolor_list[:tmpint]
      #endfor color in range(num_color)
      self.has_color_list = True

    smooth_opts.color_list = self.color_list

  def smooth(self, x, b, smooth_opts):
    """ Performs a smoothing step. Right now, it is just RB Jacobi."""
    self.__define_color_list(smooth_opts)

    print("smoothing at level", self.level)
    x = blk_jacobi(self.A, x, b, smooth_opts)
    return x
