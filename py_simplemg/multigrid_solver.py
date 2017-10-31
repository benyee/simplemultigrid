""" Defines a wrapper function to solve a linear system using multigrid.
Future work will include converting this into a class.
"""

from .multigrid_1D import MultigridLevel_1D
from .multigrid_cart import MultigridLevel_Cartesian
from .multigrid_ar import MultigridLevel_AR
from .multigrid_base import MultigridType

def solve_multigrid(A, b, x0, mg_opts, smooth_opts):
  """ Wrapper function to solve a linear system using multigrid.

  Note that the size of the problem in each dimension MUST be of the form 2**n+1
  for some positive integer n.

  Inputs:
  A -- the problem matrix
  b -- the right hand side
  x0 -- the initial guess
  mg_ops -- Options for the multigrid hierachy/solver
  smooth_ops -- Options for smoother.  In the future, this may be
               replaced with a direct pointer to a smoother object or function

  Outputs:
  x -- solution x after num_it iterations
  """
  x = x0

  if mg_opts.mg_type == MultigridType.MG_1D:
    mymgsolver = MultigridLevel_1D(mg_opts.num_level-1, A, mg_opts)
  elif mg_opts.mg_type == MultigridType.MG_Cart:
    mymgsolver = MultigridLevel_Cartesian(mg_opts.num_level-1, A, mg_opts)
  elif mg_opts.mg_type == MultigridType.MG_cc1D:
    mymgsolver = MultigridLevel_cc1D(mg_opts.num_level-1, A, mg_opts)
  elif mg_opts.mg_type == MultigridType.MG_ccCart:
    mymgsolver = MultigridLevel_ccCartesian(mg_opts.num_level-1, A, mg_opts)
  elif mg_opts.mg_type == MultigridType.MG_AR:
    mymgsolver = MultigridLevel_AR(mg_opts.num_level-1, A, mg_opts)
  else:
    raise ValueError("Only 1D, N-dimensional Cartesian, and 'AR' are"+\
                    " supported currently.")

  for iteration in range(mg_opts.num_it):
    x = mymgsolver.iterate(x, b, smooth_opts)

  return x
