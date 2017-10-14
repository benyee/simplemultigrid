# Simple Multigrid Test Code

A simple multigrid solver in Python for testing out ideas quickly.

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
* Create a multi-equation (block) version (i.e., multiple unknowns per spatial node)
* Make it easier to define custom restriction/interpolation operators
* Add better support for dimensions not of size 2^N+1
* More complex boundary conditions
