"""This module houses all of the functionality for multigrid smoothers."""

import numpy as np

class SmootherOptions(object):
  """ A structure to store smoother options. """
  def __init__(self, smoothdown=1, smoothup=1, redblack=True):
    self.smoothdown = smoothdown
    self.smoothup = smoothup
    self.redblack = redblack

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
