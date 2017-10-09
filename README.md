# Simple Multigrid Test Code

A simple 1D multigrid solver in Python for testing out ideas quickly.

It should be fairly easy to adjust various aspects of the code.  Here are some
suggestions of things that could be easily looked at with the current version
of the code:
* Adjusting the number of levels or number of multigrid V/W cycles.
* Turning the coloring of the smoother on/off or use more colors.
* Adjusting the number of smoothing steps per multigrid cycle.
* Changing between a V- and W- cycle.
* Making the interpolation operators problem-dependent.  (This will require
some effort to code up your own interpolation matrix.)

Some potential to-do's:
* Implement a 2D/3D version.
* Create a multi-equation version (i.e., multiple unknowns per spatial node)
* Make it easier to define custom restriction/interpolation operators
* More complex boundary conditions
* Use sparse matrices for the interpolation operators and problem operators.
