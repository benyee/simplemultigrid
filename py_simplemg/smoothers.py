"""This module houses all of the functionality for multigrid smoothers."""

import numpy as np
import scipy.sparse as sp

class SmootherOptions(object):
  """ A structure to store smoother options. """
  def __init__(self, smoothdown=1, smoothup=1, redblack=True):
    self.smoothdown = smoothdown
    self.smoothup = smoothup
    self.redblack = redblack

def blk_jacobi(A, x, b, redblack=True, sparse=False):
  """ Performs one block Jacobi iteration.  Red-black ordering can be toggled
      on/off. """

  if sparse:
    tmpdiag = A.diagonal()
  else:
    tmpdiag = np.diag(A)
  tmpdiag = 1./tmpdiag
  if redblack:
    tmpdiag[1::2] = 0.

  if sparse:
    tmpdiag = sp.diags(tmpdiag)
  else:
    tmpdiag = np.diag(tmpdiag)

  x = x+tmpdiag.dot(b - A.dot(x))

  if not redblack:
    return x

  if sparse:
    tmpdiag2 = A.diagonal()
  else:
    tmpdiag2 = np.diag(A)
  tmpdiag2 = 1./tmpdiag2
  tmpdiag2[0::2] = 0.

  if sparse:
    tmpdiag2 = sp.diags(tmpdiag2)
  else:
    tmpdiag2 = np.diag(tmpdiag2)

  x = x+tmpdiag2.dot(b - A.dot(x))

  return x
