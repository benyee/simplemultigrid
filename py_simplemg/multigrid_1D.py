"""This module defines the classes and functions needed to build a 1D multigrid
solver."""

import numpy as np

class MultigridOptions(object):
  """ A structure to store multigrid solver options. """
  def __init__(self, num_its=10, num_levels=4, cycle='W'):
    self.num_its = num_its
    self.num_levels = num_levels
    if cycle in ['W', 'V']:
      self.cycle = cycle
    else:
      raise ValueError("MultigridOptions cycle must be either V or W")

class SmootherOptions(object):
  """ A structure to store smoother options. """
  def __init__(self, smoothdown=1, smoothup=1, redblack=True):
    self.smoothdown = smoothdown
    self.smoothup = smoothup
    self.redblack = redblack

class MultigridLevel(object):
  """ Each instance represents a level of the multigrid solver.

  To initialize a multigrid solver, initialize the finest level using the
  number of levels and the problem matrix.
  """
  def __init__(self, level, A, parent=None):
    self.level = level
    self.A = A
    self.parent = parent
    self.has_interp = False
    if level:
      self.__generate_interp()
      self.child = self.generate_coarser_level()
    else:
      self.child = None

  #interp from level-1 to level
  #restrict is always interp^T
  def __generate_interp(self):
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

  def generate_coarser_level(self):
    """ Creates a multigrid object representing the next coarser level.
        It uses a Galerkin triple product with restriction = interpolation^T to
        generate the coarse grid operator Ac.
        A pointer to the coarser level is returned."""
    Ac = np.dot(np.transpose(self.interpmat), np.dot(self.A, self.interpmat))
    return MultigridLevel(self.level-1, Ac, self)

  def smooth(self, x, b, redblack=True):
    """ Performs a smoothing step. Right now, it is just RB block Jacobi."""
    print("smoothing at level", self.level)
    x = blk_jacobi(self.A, x, b, redblack)
    return x

  def restrict(self, x):
    """ Restricts a vector x onto the child (coarser) grid."""
    return np.dot(np.transpose(self.interpmat), x)

  def interp(self, x):
    """ Interpolates a vector x from the child (coarser) grid to the current
        grid."""
    return np.dot(self.interpmat, x)

  def iterate(self, x, b, smooth_opts, mg_opts):
    """ Performs one multigrid "cycle" (e.g., a V-cycle or a W-cycle). """
    if self.level == 0:
      return np.linalg.solve(self.A, b)
    for i in range(smooth_opts.smoothdown):
      x = self.smooth(x, b, smooth_opts.redblack)
    r = self.restrict(b-np.dot(self.A, x))
    xc = np.zeros(len(r))
    x = x+self.interp(self.child.iterate(xc, r, smooth_opts, mg_opts))
    #begin W-cycle
    if mg_opts.cycle == 'W':
      for i in range(smooth_opts.smoothdown):
        x = self.smooth(x, b, smooth_opts.redblack)
      r = self.restrict(b-np.dot(self.A, x))
      xc = np.zeros(len(r))
      x = x+self.interp(self.child.iterate(xc, r, smooth_opts, mg_opts))
    #end W-cycle
    for i in range(smooth_opts.smoothup):
      x = self.smooth(x, b, smooth_opts.redblack)
    return x


def blk_jacobi(A, x, b, redblack=True):
  """ Performs one block Jacobi iteration.  Red-black ordering can be toggled
      on/off. """
  tmpdiag = np.diag(A)
  tmpdiag = 1./tmpdiag
  if redblack:
    tmpdiag[1::2] = 0.
  tmpdiag = np.diag(tmpdiag)

  x = x+np.dot(tmpdiag, b-np.dot(A, x))
  if not redblack:
    return x

  tmpdiag2 = np.diag(A)
  tmpdiag2 = 1./tmpdiag2
  tmpdiag2[0::2] = 0.
  tmpdiag2 = np.diag(tmpdiag2)

  x = x+np.dot(tmpdiag, b-np.dot(A, x))

  return x

def solve_multigrid(A, b, x0, mg_opts, smooth_opts):
  """ Wrapper function to solve a linear system using multigrid."""
  x = x0

  mymgsolver = MultigridLevel(mg_opts.num_levels-1, A)

  for iteration in range(mg_opts.num_its):
    x = mymgsolver.iterate(x, b, smooth_opts, mg_opts)

  return x
