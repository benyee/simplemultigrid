"""This module defines the multigrid level class for 1D problems."""

import numpy as np
import scipy.sparse as sp
from .multigrid_base import MultigridLevel_Base, BC
from .smoothers import blk_jacobi

class MultigridLevel_1D(MultigridLevel_Base):
  """ Each instance represents a level of a multigrid solver for a 1D problem

  To initialize a multigrid solver, initialize the finest level using the
  number of levels and the problem matrix.
  """
  def generate_interp(self):
    """ Defines the interpolation operator from level-1 to level.
        Restriction is assumed by default to be interp^T."""
    nx_fine = self.A.shape[0]
    n_coarse = (nx_fine-1)//2+1
    if self.mg_opts.sparse:
      #Create matrix using lil_matrix, then convert to more
      #  computationally efficient csr_matrix.
      self.interpmat = sp.lil_matrix((nx_fine, n_coarse))
    else:
      self.interpmat = np.zeros((nx_fine, n_coarse))
    for i in range(nx_fine):
      i_coarse = i//2
      if not i%2:
        self.interpmat[i, i_coarse] = 1.0
      else:
        if i > 0:
          self.interpmat[i, i_coarse] = 0.5
        if i < nx_fine-1:
          self.interpmat[i, i_coarse+1] = 0.5
        elif self.mg_opts.bcs[1] == BC.REFL:
          self.interpmat[i, i_coarse] += 0.5
    if self.mg_opts.sparse:
      self.interpmat = sp.csr_matrix(self.interpmat)

    self.has_interp = True

  def smooth(self, x, b, smooth_opts):
    """ Performs a smoothing step. Right now, it is just RB block Jacobi."""
    print("smoothing at level", self.level)
    x = blk_jacobi(self.A, x, b, smooth_opts)
    return x
