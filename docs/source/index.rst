GeometricMD
===========


What is this project for?
-------------------------

GeometricMD is an implementation of the Birkhoff curve shortening procedure using a technique devised by Schwetlick-Zimmer to compute local geodesics. It is currently implemented in Python using multiprocessing to distribute the computing.


How do I get set up?
----------------------------

Either use PIP with package name 'geometricmd' or run 'python setup.py install'.

Who do I talk to?
-----------------------

* For questions about the project please contact Daniel Sutton (sutton.c.daniel@gmail.com)

* For scientific queries please contact Johannes Zimmer (zimmer@maths.bath.ac.uk)

Package Contents
-----------------

.. toctree::
   :maxdepth: 4

   animation

Aperiodic Systems
=================
   curve_shorten
   geometry

Periodic Systems
================
   curve_shorten_pbc
   geometry_pbc


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

References
----------

.. [Sutton2013] Microscopic Hamiltonian Systems and their Effective Description, Daniel C. Sutton, 2013.

