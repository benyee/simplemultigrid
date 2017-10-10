"""This module houses all of the functionality for multigrid smoothers."""

import numpy as np
import scipy.sparse as sp

class SmootherOptions(object):
  """ A structure to store smoother options. """
  def __init__(self, smoothdown=1, smoothup=1, omega = 1.0, sparse = True,
               num_color=True):
    self.smoothdown = smoothdown
    self.smoothup   = smoothup
    self.num_color  = num_color
    self.omega      = omega
    self.sparse     = sparse

def blk_jacobi(A, x, b, smooth_opts):
  """ Performs one block Jacobi iteration.  Colored ordering can be toggled
      on/off via smooth_opts.  The "block" aspect has not been implemented
      yet, so it's really just red/black Jacobi.
  """
  x0 = x[:]

  if smooth_opts.sparse:
    diag = A.diagonal()
  else:
    diag = np.diag(A)

  color = smooth_opts.num_color
  while color:
    color -= 1

    diaginv = np.zeros(len(diag))
    diaginv[color::smooth_opts.num_color] = \
            1./diag[color::smooth_opts.num_color]

    if smooth_opts.sparse:
      diaginv = sp.diags(diaginv)
    else:
      diaginv = np.diag(diaginv)

    x += diaginv.dot(b - A.dot(x))

  return smooth_opts.omega*x + (1-smooth_opts.omega)*x0
