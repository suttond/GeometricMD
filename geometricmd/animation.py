import os
import pickle
import math

from ase.io import write
import numpy as np
from scipy.integrate import quad

from geometricmd.geometry import convert_vector_to_atoms


def get_times(curve):

    t = [0.0]

    mass_matrix = np.diag(np.dstack((curve.configuration['molecule'].get_masses(),)
                                    * (len(curve.points[0]) /
                                       len(curve.configuration['molecule'].get_masses()))).flatten())

    l = 0.0

    # For all of the nodes, less the end node...
    for i in xrange(curve.number_of_nodes-1):

        def integrand(t, x_1, x_2):
            x = np.add(np.subtract(x_2, x_1)*t, x_1)
            curve.configuration['molecule'].set_positions(convert_vector_to_atoms(x))
            metric_cf = 1/math.sqrt(2*(curve.energy - curve.configuration['molecule'].get_potential_energy()))

            return metric_cf * math.sqrt(np.inner(np.subtract(x_2, x_1), mass_matrix.dot(np.subtract(x_2, x_1))))

    # Add the trapezoidal rule approximation of the length functional for a line segment
        try:
            l += quad(integrand, 0, 1, args=(curve.points[i+1], curve.points[i]))[0]
        except ValueError:
            l = np.inf
        t.append(l)

    return t


def write_xyz_animation(curve_pickle, filename):
    """ This function takes a curve object that has been pickled and writes out an XYZ animation file to use with JMol.

    Args:
      curve_pickle (str): The location of a pickled curve object.
      filename (str): The location of where to write the XYZ animation.

    """

    # Unpickle the curve object
    curve = pickle.load(open(curve_pickle, "rb"))

    # Extract the trajectories points
    trajectory = curve.get_points()

    # Reparameterise to determine physical times
    times = get_times(curve)

    # Index to determine correct time
    i = 0

    # Extract the curves molecular configuration as described by an ASE atoms object
    molecule = curve.configuration['molecule']

    # Create a new file for the animation to be stored
    animation_file = open(filename, 'w')

    # For each node along the curve...
    for state in trajectory:

        # Determine the molecular configuration in ASE
        molecule.set_positions(convert_vector_to_atoms(state))

        # Produce a snapshot XYZ file of the configuration at that point in time
        write('current_state.xyz',
              molecule,
              format='xyz',
              comment='T=' + str(times[i]) + '\tPotential Energy: ' + str(molecule.get_potential_energy()))

        # Increase index
        i += 1

        # Open the newly produced XYZ file
        current_state = open('current_state.xyz', 'r')

        # Append the snapshot XYZ file into the animation file
        for line in current_state:
            animation_file.write(line)
        current_state.close()

    # Once finished close the file so that other programs can access it
    animation_file.close()

    # Delete the temporary file used to store current snapshots
    os.remove('current_state.xyz')