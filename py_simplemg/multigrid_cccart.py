"""This module defines the multigrid level class for general CC_Cartesian problems."""

import numpy as np
import scipy.sparse as sp
from .multigrid_base import MultigridLevel_Base
from .smoothers import blk_jacobi

class MultigridLevel_ccCartesian(MultigridLevel_Base):
  """ Each instance represents a level of a multigrid solver for a general 
      n-dimensional CC_Cartesian problems.

  To initialize a multigrid solver, initialize the finest level using the
  number of levels and the problem matrix.

  LEXICOGRAPHCIAL ORDERING IS ASSUMED.
  """
  def __init__(self, level, A, mg_opts, parent=None):
    self.dims = mg_opts.dims[:]
    self.num_dim = len(self.dims)

    tmpsize = self.dims[0]
    for dim in range(1, self.num_dim):
      tmpsize *= self.dims[dim]
    if tmpsize != A.shape[0]:
      raise ValueError("Provided domain dimensions inconsistent with " + \
                       "problem matrix size.")

    #Check that the parent is collapsing in a manner consistent with how
    #  the generate_interp method is defined in this class:
    if parent != None:
      bool_list = [(parent_dim-1)//2+1 == child_dim \
                     for (parent_dim, child_dim) in zip(parent.dims, self.dims)]
      if not all(bool_list):
        raise ValueError("Inconsistent dimensions between parent and child.")

    #Alter dims for the child class:
    for dim in range(self.num_dim):
      mg_opts.dims[dim] = (mg_opts.dims[dim]+1)//2
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
    dims_coarse = self.dims[:]
    n_coarse = 1
    for dim in range(self.num_dim):
      dims_coarse[dim] = (dims_coarse[dim]+1)//2
      n_coarse = n_coarse*dims_coarse[dim]

    if self.mg_opts.sparse:
      #Create matrix using lil_matrix, then convert to more
      #  computationally efficient csr_matrix.
      self.interpmat = sp.lil_matrix((n, n_coarse))
    else:
      self.interpmat = np.zeros((n, n_coarse))

    num_indices = 0
    for i in range(n):
      coord = self.index_to_ijk(i)
      coarse_coord = coord[:]
      for dim in range(self.num_dim):
        coarse_coord[dim] = coord[dim]//2
      coarse_index = self.ijk_to_index_child(coarse_coord)

      wt = 1.
      for dim in range(self.num_dim):
        if self.dims[dim] > 1 and not (self.dims[dim] % 2 and i == n-1):
          wt /= 2.
      self.interpmat[i, coarse_index] = wt
    #endfor i in range(n)

    if self.mg_opts.sparse:
      self.interpmat = sp.csr_matrix(self.interpmat)

    self.has_interp = True

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
