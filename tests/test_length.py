import unittest
from ase.io import read
from ase.calculators.emt import EMT
from geometricmd.curve_shorten import convert_vector_to_atoms, length, get_rotation
import math
import numpy as np
from scipy.optimize import check_grad

# Check length can be computed
def check_length(total_number_of_points):

    try:

        energy = 100.0
        molecule = read('test_files/x0.xyz')
        molecule.set_calculator(EMT())

        start_point = read('test_files/x0.xyz')
        start_point.set_calculator(EMT())
        end_point = read('test_files/xN.xyz')

        start = start_point.get_positions().flatten()
        end = end_point.get_positions().flatten()

        dimension = len(molecule.get_positions().flatten())
        codimension = dimension - 1

        rotation_matrix = get_rotation(start, end, dimension)

        mass_matrix = np.diag(np.dstack((molecule.get_masses(),) * (dimension /
                                                                len(molecule.get_masses()))).flatten())

        def metric(point):
            molecule.set_positions(convert_vector_to_atoms(point))
            cf = math.sqrt(max([2*(energy - molecule.get_potential_energy()), 1E-9]))

            return [cf, molecule.get_forces().flatten()/cf]

        x = np.random.rand((total_number_of_points-2) * codimension)

        l = length(x, start, end, mass_matrix, rotation_matrix, total_number_of_points, codimension, metric)

        return True

    except:

        return False


# Check length can be computed
def check_length_gradient(total_number_of_points):

    try:

        energy = 100.0
        molecule = read('test_files/x0.xyz')
        molecule.set_calculator(EMT())

        start_point = read('test_files/x0.xyz')
        start_point.set_calculator(EMT())
        end_point = read('test_files/xN.xyz')

        start = start_point.get_positions().flatten()
        end = end_point.get_positions().flatten()

        dimension = len(molecule.get_positions().flatten())
        codimension = dimension - 1

        rotation_matrix = get_rotation(start, end, dimension)

        mass_matrix = np.diag(np.dstack((molecule.get_masses(),) * (dimension /
                                                                len(molecule.get_masses()))).flatten())

        def metric(point):
            molecule.set_positions(convert_vector_to_atoms(point))
            cf = math.sqrt(max([2*(energy - molecule.get_potential_energy()), 1E-9]))

            return [cf, molecule.get_forces().flatten()/cf]

        x = np.random.rand((total_number_of_points-2) * codimension)

        def L(x):
            return length(x, start, end, mass_matrix, rotation_matrix, total_number_of_points, codimension, metric)[0]

        def GL(x):
            return length(x, start, end, mass_matrix, rotation_matrix, total_number_of_points, codimension, metric)[1]

        err = check_grad(L,GL,x)

        if abs(err) < 1E-4:
            return True
        else:
            return False

    except:

        return False


# Compile class of unit tests.
class GeometryTests(unittest.TestCase):

    def testOne(self):
        self.failUnless(check_length(10))

    def testTwo(self):
        self.failUnless(check_length_gradient(10))

def main():
    unittest.main()

if __name__ == '__main__':
    main()