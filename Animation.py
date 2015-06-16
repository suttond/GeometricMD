import os
import pickle
from Geometry import convert_vector_to_atoms
from ase.io import write


def write_xyz_animation(curve_pickle, filename):

    curve = pickle.load(open(curve_pickle, "rb"))
    trajectory = curve.get_points()
    molecule = curve.molecule
    animation_file = open(filename, 'w')

    for state in trajectory:
        molecule.set_positions(convert_vector_to_atoms(state))
        write('current_state.xyz', molecule, format='xyz')
        current_state = open('current_state.xyz', 'r')
        for line in current_state:
            animation_file.write(line)
        current_state.close()
    animation_file.close()
    os.remove('current_state.xyz')