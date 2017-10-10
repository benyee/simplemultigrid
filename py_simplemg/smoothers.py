"""This module houses all of the functionality for multigrid smoothers."""

import numpy as np
import scipy.sparse as sp

class SmootherOptions(object):
  """ A structure to store smoother options. """
  def __init__(self, smoothdown=1, smoothup=1, omega = 1.0, sparse = True,
               redblack=True):
    self.smoothdown = smoothdown
    self.smoothup = smoothup
    self.redblack = redblack
    self.omega    = omega
    self.sparse   = sparse

def blk_jacobi(A, x, b, smooth_opts):
  """ Performs one block Jacobi iteration.  Red-black ordering can be toggled
      on/off.  The "block" aspect has not been implemented yet, so it's
      really just red/black Jacobi.
  """
  x0 = x
  if smooth_opts.redblack:
    num_color = 2
  else:
    num_color = 1

  color = num_color
  while(color > 1):
    color -= 1

    if smooth_opts.sparse:
      tmpdiag = A.diagonal()
    else:
      tmpdiag = np.diag(A)
    tmpdiag = 1./tmpdiag

    if smooth_opts.redblack:
      tmpdiag[color::2] = 0.

    if smooth_opts.sparse:
      tmpdiag = sp.diags(tmpdiag)
    else:
      tmpdiag = np.diag(tmpdiag)

    x = x+tmpdiag.dot(b - A.dot(x))

    color = color + 1
  return smooth_opts.omega*x + (1-smooth_opts.omega)*x0
