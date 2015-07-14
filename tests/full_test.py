import unittest
from geometricmd.curve_shorten import compute_trajectory
from geometricmd.geometry import Curve
from ase.io import read
from ase.calculators.emt import EMT
from multiprocessing import cpu_count


# Test GeometricMD on a single process.
def butane_single_process():

    try:
        start_point = read('test_files/x0.xyz')
        start_point.set_calculator(EMT())
        end_point = read('test_files/xN.xyz')

        traj = Curve(start_point, end_point, 12, 1E+03)

        compute_trajectory(traj, 9, 1E+03, 0.003, 'Butane', {'processes': 1})

        return True

    except:

        return False

# Test GeometricMD on all processes.
def butane_multi_process():

    try:
        start_point = read('test_files/x0.xyz')
        start_point.set_calculator(EMT())
        end_point = read('test_files/xN.xyz')

        traj = Curve(start_point, end_point, 12, 1E+03)

        compute_trajectory(traj, 9, 1E+03, 0.003, 'Butane', {'processes': cpu_count() - 1})

        return True

    except:

        return False

# Compile class of unit tests.
class ButaneFullTests(unittest.TestCase):

    def testOne(self):
        self.failUnless(butane_single_process())

    def testTwo(self):
        self.failUnless(butane_multi_process())


def main():
    unittest.main()

if __name__ == '__main__':
    main()