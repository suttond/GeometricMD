import os
import pickle
from Geometry import convert_vector_to_atoms
from ase.io import write


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

    # Extract the curves molecular configuration as described by an ASE atoms object
    molecule = curve.configuration['molecule']

    # Create a new file for the animation to be stored
    animation_file = open(filename, 'w')

    # For each node along the curve...
    for state in trajectory:

        # Determine the molecular configuration in ASE
        molecule.set_positions(convert_vector_to_atoms(state))

        # Produce a snapshot XYZ file of the configuration at that point in time
        write('current_state.xyz', molecule, format='xyz')

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