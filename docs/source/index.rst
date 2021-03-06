===========
GeometricMD
===========


What is this project for?
=========================

GeometricMD is an implementation of the Birkhoff curve shortening procedure using a technique devised by Schwetlick-Zimmer to compute local geodesics. It is currently implemented in Python using multiprocessing to distribute the computing.


How do I get set up?
=====================

Either use PIP with package name 'geometricmd' or run 'python setup.py install' after downloading the package from GitHub (suttond/GeometricMD).

Who do I talk to?
==================

* For questions about the project please contact Daniel Sutton (sutton.c.daniel@gmail.com)

* For scientific queries please contact Johannes Zimmer (zimmer@maths.bath.ac.uk)

Package Contents
=================

Utilities
----------

.. toctree::

   animation

Aperiodic Systems
------------------

.. toctree::

   curve_shorten
   geometry

Periodic Systems
------------------

.. toctree::
   curve_shorten_pbc
   geometry_pbc

Tutorials
-----------

.. toctree::
   butane_tutorial

Indices and tables
====================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
