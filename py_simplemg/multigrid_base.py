""" Defines a wrapper function to solve a linear system using multigrid.
Future work will include converting this into a class.
"""

import numpy as np

class MultigridLevel_Base(object):
  """ Each instance represents a level of the multigrid solver.

  To initialize a multigrid solver, initialize the finest level using the
  number of levels and the problem matrix.

  This is a base class and should be treated as an abstract class.
  """
  def __init__(self, level, A, mg_opts, parent=None):
    self.level = level
    self.A = A
    self.mg_opts = mg_opts
    self.parent = parent
    self.has_interp = False
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
    Ac = np.dot(np.transpose(self.interpmat), np.dot(self.A, self.interpmat))
    return type(self)(self.level-1, Ac, self.mg_opts, self)

  def restrict(self, x):
    """ Restricts a vector x onto the child (coarser) grid."""
    if self.has_interp:
      return np.dot(np.transpose(self.interpmat), x)
    raise AttributeError("Interpolation operator not defined yet.")

  def interp(self, x):
    """ Interpolates a vector x from the child (coarser) grid to the current
        grid."""
    if self.has_interp:
      return np.dot(self.interpmat, x)
    raise AttributeError("Interpolation operator not defined yet.")

  #Begin "abstract" methods:
  def generate_interp(self):
    """ Dummy method to define the interpolation from level-1 to level."""
    raise NotImplementedError("generate_interp method not defined for base " + \
                              "abstract multigrid level class!")

  def smooth(self, x, b, redblack=True):
    """ Dummy smooth method for the base class."""
    raise NotImplementedError("smooth method not defined for " + \
                              "base abstract multigrid level class!")
    return x

  def iterate(self, x, b, smooth_opts):
    """ Dummy iterate method for the base class."""
    raise NotImplementedError("iterate method not defined for " + \
                              "base abstract multigrid level class!")
    return x
  #End "abstract" methods

class MultigridOptions(object):
  """ A structure to store multigrid solver options. """
  def __init__(self, num_its=10, num_levels=4, cycle='W', dim=1):
    self.num_its = num_its
    self.num_levels = num_levels
    if cycle in ['W', 'V']:
      self.cycle = cycle
    else:
      raise ValueError("MultigridOptions cycle must be either V or W")
    if dim == 1:
      self.dim = dim
    else:
      raise ValueError("Only 1D supported right now.")

class SmootherOptions(object):
  """ A structure to store smoother options. """
  def __init__(self, smoothdown=1, smoothup=1, redblack=True):
    self.smoothdown = smoothdown
    self.smoothup = smoothup
    self.redblack = redblack
