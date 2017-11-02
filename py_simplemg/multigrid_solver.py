""" Defines a wrapper function to solve a linear system using multigrid.
Future work will include converting this into a class.
"""

from .multigrid_1D import MultigridLevel_1D
from .multigrid_cart import MultigridLevel_Cartesian
from .multigrid_cc1D import MultigridLevel_cc1D
from .multigrid_cccart import MultigridLevel_ccCartesian

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

  if mg_opts.geom_type == '1D':
    mymgsolver = MultigridLevel_1D(mg_opts.num_level-1, A, mg_opts)
  elif mg_opts.geom_type == 'cart':
    mymgsolver = MultigridLevel_Cartesian(mg_opts.num_level-1, A, mg_opts)
  elif mg_opts.geom_type == 'cc1D':
    mymgsolver = MultigridLevel_cc1D(mg_opts.num_level-1, A, mg_opts)
  elif mg_opts.geom_type == 'cccart':
    mymgsolver = MultigridLevel_ccCartesian(mg_opts.num_level-1, A, mg_opts)
  else:
    raise ValueError("Only 1D and general N-dimensional Cartesian are"+\
                    " supported currently.")

  for iteration in range(mg_opts.num_it):
    x = mymgsolver.iterate(x, b, smooth_opts)

  return x
