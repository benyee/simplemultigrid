""" Defines a wrapper function to solve a linear system using multigrid.
Future work will include converting this into a class.
"""

from .multigrid_1D import MultigridLevel_1D

def solve_multigrid(A, b, x0, mg_opts, smooth_opts):
  """ Wrapper function to solve a linear system using multigrid.

  Inputs:
  b -- the right hand side
  x0 -- the initial guess
  mg_ops -- Options for the multigrid hierachy/solver
  smooth_ops -- Options for smoother.  In the future, this may be
               replaced with a direct pointer to a smoother object or function

  Outputs:
  x -- solution x after num_its iterations
  """
  x = x0

  if mg_opts.dim == 1:
    mymgsolver = MultigridLevel_1D(mg_opts.num_levels-1, A, mg_opts)
  else:
    raise ValueError("No support currently for anything other than 1D.")

  for iteration in range(mg_opts.num_its):
    x = mymgsolver.iterate(x, b, smooth_opts)

  return x
