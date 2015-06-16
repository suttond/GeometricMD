# Load packages which form part of the Python 2.7 core
import sys
import pickle
import multiprocessing
import math

# Load packages which are a part of GMFMD
from Animation import write_xyz_animation
from Geometry import Curve, convert_vector_to_atoms, convert_atoms_to_vector
from Physics import define_physics

# Load additional packages and check if they are installed
try:
    import ase
except ImportError as e:
    print 'Atomistic Simulation Environment is not installed. Try to run *pip install python-ase*.'
    quit()

try:
    import numpy as np
except ImportError as e:
    print 'NumPy is not installed. Try to run *pip install numpy*.'
    quit()

try:
    import scipy
except ImportError as e:
    print 'SciPy is not installed. Try to run *pip install scipy*.'
    quit()


def length(x, start_point, end_point, number_of_inner_nodes, mass_vector, dimension, molecule, energy):
    """

    :param x:
    :param start_point:
    :param end_point:
    :param number_of_inner_nodes:
    :param mass_vector:
    :param dimension:
    :param molecule:
    :param energy:
    :return:
    """

    curve = np.vstack((start_point, np.reshape(x, (number_of_inner_nodes, dimension)), end_point))

    metric = []

    for point in curve:
        molecule.set_positions(convert_vector_to_atoms(point))
        metric_value = math.sqrt(2*(energy - molecule.get_potential_energy()))
        metric.append([metric_value, -convert_atoms_to_vector(molecule.get_forces())/metric_value])

    l = 0.0

    for i in xrange(number_of_inner_nodes + 1):
        l += math.sqrt(np.inner(np.subtract(curve[i+1], curve[i]),
                                      mass_vector.dot(np.subtract(curve[i+1], curve[i])))) * 0.5 \
             * (metric[i+1][0]+metric[i][0])

    g = []

    for i in xrange(1,number_of_inner_nodes + 1):
        n_1 = np.linalg.norm(np.subtract(curve[i+1], mass_matrix.dot(curve[i])))
        n_2 = np.linalg.norm(np.subtract(curve[i], mass_matrix.dot(curve[i-1])))
        t_1 = mass_matrix.dot(np.subtract(curve[i+1], curve[i])) / n_1
        t_2 = mass_matrix.dot(np.subtract(curve[i], curve[i-1])) / n_2

        g_component = metric[i][1] * (n_1 + n_2) - (metric[i+1][0] + metric[i][0]) * t_1 \
                      + (metric[i-1][0] + metric[i][0]) * t_2

        g.append(g_component)

    return l, 0.5*np.asarray(g).flatten()


def find_geodesic_midpoint(start_point, end_point, number_of_inner_points, dimension, mass_matrix, molecule, energy,
                           node_number):
    """

    :param start_point:
    :param end_point:
    :param number_of_inner_points:
    :param dimension:
    :param mass_matrix:
    :param molecule:
    :param energy:
    :param node_number:
    :return:
    """
    start_curve = Curve(start_point, end_point, number_of_inner_points+2,
                        number_of_inner_points+2, molecule).get_points()[1:-1].flatten()

    geodesic, f_min, detail = scipy.optimize.fmin_l_bfgs_b(func=length, x0=start_curve, args=(start_point, end_point,
                                                           number_of_inner_points, mass_matrix, dimension,
                                                           molecule, energy), pgtol=0.1)

    return [node_number, np.reshape(geodesic, (number_of_inner_points, dimension))[(number_of_inner_points + 1) / 2]]


def add_node_to_job_pool(node_number, local_number_of_nodes, dimension, mass_matrix, molecule, energy):
    """

    :param node_number:
    :param local_number_of_nodes:
    :param dimension:
    :param mass_matrix:
    :param molecule:
    :param energy:
    :return:
    """

    def update_curve(result):
        curve.set_node_position(result[0], result[1])

    pool.apply_async(func=find_geodesic_midpoint,
                     args=(curve.points[node_number - 1], curve.points[node_number + 1],
                           local_number_of_nodes, dimension, mass_matrix, molecule, energy, node_number,),
                     callback=update_curve)


if __name__ == '__main__':

    config_filename = sys.argv[1].split('.', 1)[0]

    f = open('conf/' + config_filename, 'r')

    for line in f.readlines():
        command = line[:2]
        value = line.replace(" ", "")[3:]

        if command == 'st':
            molecule = ase.io.read(value.strip())
            molecule.set_calculator(define_physics())
            start_point = convert_atoms_to_vector(molecule.get_positions())
            dimension = len(start_point)
            mass_matrix = np.diag(np.dstack((molecule.get_masses(),) * (dimension /
                                                                        len(molecule.get_masses()))).flatten())
        elif command == 'en':
            end_point = convert_atoms_to_vector(ase.io.read(value.strip()).get_positions())
        elif command == 'ln':
            local_num_nodes = int(value)
        elif command == 'gn':
            global_num_nodes = int(value)
        elif command == 'pa':
            metric_parameters = np.fromstring(value, dtype='f', sep=',')
            energy = metric_parameters[0]
        elif command == 'to':
            tol = float(value)
        elif command == 'ou':
            filename = str(value)

    curve = Curve(start_point, end_point, global_num_nodes, global_num_nodes * (local_num_nodes - 1) + 1, molecule)

    pool = multiprocessing.Pool()

    while True:

        for node in curve:
            add_node_to_job_pool(node, local_num_nodes, dimension, mass_matrix, molecule, energy)

        if curve.all_nodes_moved():

            print curve.movement

            pickle.dump(curve, open('out/'+filename+'.pkl', "wb"))
            write_xyz_animation('out/'+filename+'.pkl', 'out/'+filename+'.xyz')

            if curve.movement < tol:
                break
            curve.set_node_movable()

    pool.close()
    pool.join()
