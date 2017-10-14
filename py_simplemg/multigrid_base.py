""" Defines a wrapper function to solve a linear system using multigrid.
Future work will include converting this into a class.
"""

from enum import Enum
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import numpy as np

MAX_DIMS = 5

class MultigridLevel_Base(object):
  """ Each instance represents a level of the multigrid solver.

  To initialize a multigrid solver, initialize the finest level using the
  number of levels and the problem matrix.

  This is a base class and should be treated as an abstract class.
  """
  def __init__(self, level, A, mg_opts, parent=None):
    self.level = level
    self.mg_opts = mg_opts
    self.parent = parent
    self.has_interp = False

    if self.mg_opts.sparse:
      self.A = sp.csr_matrix(A)
    else:
      self.A = A

    if level:
      self.generate_interp()
      self.child = self.generate_coarser_level()
    else:
      self.child = None

  def generate_coarser_level(self):
    """ Creates a multigrid object representing the next coarser level.
        It uses a Galerkin triple product with restriction = interpolation^T to
        generate the coarse grid operator Ac.
        A pointer to the coarser level is returned."""
    Ac = self.interpmat.transpose().dot(self.A.dot(self.interpmat))
    return type(self)(self.level-1, Ac, self.mg_opts, self)

  def restrict(self, x):
    """ Restricts a vector x onto the child (coarser) grid."""
    if self.has_interp:
      return self.interpmat.transpose().dot(x)
    raise AttributeError("Interpolation operator not defined yet.")

  def interp(self, x):
    """ Interpolates a vector x from the child (coarser) grid to the current
        grid."""
    if self.has_interp:
      return self.interpmat.dot(x)
    raise AttributeError("Interpolation operator not defined yet.")

  def residual(self, x, b):
    """ Computes the residual b - Ax """
    return b - self.A.dot(x)

  def iterate(self, x, b, smooth_opts):
    """ Performs one multigrid "cycle" (e.g., a V-cycle or a W-cycle). """
    if self.level == 0:
      if self.mg_opts.sparse:
        return spsolve(self.A, b)
      return np.linalg.solve(self.A, b)
    for i in range(smooth_opts.smoothdown):
      x = self.smooth(x, b, smooth_opts)
    r = self.restrict(self.residual(x, b))
    xc = np.zeros(len(r))
    x = x+self.interp(self.child.iterate(xc, r, smooth_opts))
    #begin W-cycle
    if self.mg_opts.cycle == 'W':
      for i in range(smooth_opts.smoothdown):
        x = self.smooth(x, b, smooth_opts)
      r = self.restrict(self.residual(x, b))
      xc = np.zeros(len(r))
      x = x+self.interp(self.child.iterate(xc, r, smooth_opts))
    #end W-cycle
    for i in range(smooth_opts.smoothup):
      x = self.smooth(x, b, smooth_opts)
    return x

  #Begin "abstract" methods:
  def generate_interp(self):
    """ Dummy method to define the interpolation from level-1 to level."""
    raise NotImplementedError("generate_interp method not defined for base " + \
                              "abstract multigrid level class!")

  def smooth(self, x, b, smooth_opts):
    """ Dummy smooth method for the base class."""
    raise NotImplementedError("smooth method not defined for " + \
                              "base abstract multigrid level class!")
    return x
  #End "abstract" methods

class BC(Enum):
  """ Enumeration for boundary conditions. """
  ZERO = 0
  REFL = 1

class MultigridOptions(object):
  """ A structure to store multigrid solver options. """
  def __init__(self, num_it=10,
               num_level=4,
               cycle='W',
               geom_type='1D',
               bcs=(BC.ZERO, BC.ZERO),
               sparse=False):
    """ Inputs:

    num_it -- number of iterations (V/W cycles)
    num_level -- number of multigrid grids
    cycle -- type of cycle (V vs. W)
    geom_type -- Type of geometry for the multigrid grids.  Currently only 1D
                 and cart (general N-D Cartesian) are available.
    """
    self.num_it    = num_it
    self.num_level = num_level
    self.sparse    = sparse
    self.bcs       = bcs
    if cycle in ['W', 'V']:
      self.cycle = cycle
    else:
      raise ValueError("MultigridOptions cycle must be either V or W")
    if geom_type in ['1D', 'cart', 'tr-2D']:
      self.geom_type = geom_type
    else:
      raise ValueError("Only 1D and cart supported so far!")
