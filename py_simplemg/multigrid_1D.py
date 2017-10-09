"""This module defines the multigrid level class for 1D problems."""

import numpy as np
from .multigrid_base import MultigridLevel_Base
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
    self.interpmat = np.zeros((nx_fine, n_coarse))
    for i in range(nx_fine):
      i_coarse = (i-1)//2
      if i%2:
        self.interpmat[i, i_coarse] = 1.0
      else:
        self.interpmat[i, i_coarse] = 0.5
        self.interpmat[i, i_coarse+1] = 0.5
    self.has_interp = True

  def smooth(self, x, b, redblack=True):
    """ Performs a smoothing step. Right now, it is just RB block Jacobi."""
    print("smoothing at level", self.level)
    x = blk_jacobi(self.A, x, b, redblack)
    return x

  def iterate(self, x, b, smooth_opts):
    """ Performs one multigrid "cycle" (e.g., a V-cycle or a W-cycle). """
    if self.level == 0:
      return np.linalg.solve(self.A, b)
    for i in range(smooth_opts.smoothdown):
      x = self.smooth(x, b, smooth_opts.redblack)
    r = self.restrict(b-np.dot(self.A, x))
    xc = np.zeros(len(r))
    x = x+self.interp(self.child.iterate(xc, r, smooth_opts))
    #begin W-cycle
    if self.mg_opts.cycle == 'W':
      for i in range(smooth_opts.smoothdown):
        x = self.smooth(x, b, smooth_opts.redblack)
      r = self.restrict(b-np.dot(self.A, x))
      xc = np.zeros(len(r))
      x = x+self.interp(self.child.iterate(xc, r, smooth_opts))
    #end W-cycle
    for i in range(smooth_opts.smoothup):
      x = self.smooth(x, b, smooth_opts.redblack)
    return x

