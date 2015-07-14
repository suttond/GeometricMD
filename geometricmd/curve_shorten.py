# Load packages which form part of the Python 2.7 core
import pickle
import multiprocessing
import math
import logging

# Load packages which are a part of GeometricMD
from geometricmd.animation import write_xyz_animation
from geometricmd.geometry import convert_vector_to_atoms, convert_atoms_to_vector

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
    from scipy.optimize import fmin_l_bfgs_b, check_grad
except ImportError as e:
    print 'SciPy is not installed. Try to run *pip install scipy*.'
    quit()


def generate_points(x, start, end, rotation_matrix, total_number_of_points, codimension):

    tangent = np.subtract(end, start)/(total_number_of_points-1)

    points = []
    for i in xrange(total_number_of_points):
        points.append(np.add(start, float(i)*tangent))

    for i in xrange(0, len(x)/codimension):
        unrotated_shift = np.hstack((np.zeros(1), x[i*(codimension):(i+1)*codimension]))
        shift = rotation_matrix.dot(unrotated_shift)
        points[i+1] = np.add(points[i+1], shift)

    return points


def compute_metric(points, metric_function):

    metric = []

    for point in points:
        metric.append(metric_function(point))

    return metric


def norm(x, matrix):
    return math.sqrt(np.inner(x, matrix.dot(x)))


def gnorm(x, matrix):
    norm = 2 * math.sqrt(np.inner(x, matrix.dot(x)))
    A = (matrix + matrix.transpose())
    return A.dot(x) / norm


def length(x, start, end, mass_matrix, rotation_matrix, total_number_of_points, codimension, metric):

    points = generate_points(x, start, end, rotation_matrix, total_number_of_points, codimension)

    a = compute_metric(points, metric)

    n = np.subtract(points[1], points[0])
    b = norm(n, mass_matrix)
    c = gnorm(n, mass_matrix)
    u = (a[1][0]+a[0][0])

    l = u * b
    g = []

    for i in xrange(1, len(points)-1):

        n = np.subtract(points[i+1], points[i])
        d = norm(n, mass_matrix)
        e = gnorm(n, mass_matrix)
        v = (a[i+1][0]+a[i][0])

        l += v * norm(n, mass_matrix)

        g.append(rotation_matrix.transpose().dot(a[i][1] * (b + d) + u * c - v * e)[1:])

        b = d
        c = e
        u = v

    return 0.5 * l, 0.5 * np.asarray(g).flatten()


def get_rotation(start_point, end_point, dimension):

    tangent = np.subtract(end_point, start_point)

        # Set the first column of our output matrix as tangent
    mx = tangent

    # Find the first non-zero entry of the tangent vector (exists as start and endpoints are different)
    j = np.nonzero(mx)[0][0]

    # For the remaining dim - 1 columns choose unit basis vectors of the form (0,...,0,1,0,...,0) with the nonzero entry
    # not in position j.
    for i in xrange(1, dimension):
        if j != i:
            e = np.zeros(dimension)
            e[i] = 1
            mx = np.vstack((mx, e))

    mx = mx.transpose()

    # With the resulting matrix, perform the Gram-Schmidt orthonormalisation procedure on the transpose of the matrix
    # and return it.
    m, n = np.shape(mx)
    Q = np.zeros([m, n])
    R = np.zeros([n, n])
    v = np.zeros(m)

    for j in range(n):

        v[:] = mx[:,j]
        for i in range(j):
            r = np.dot(Q[:,i], mx[:,j]); R[i,j] = r
            v[:] = v[:] - r*Q[:,i]
        r = np.linalg.norm(v); R[j,j]= r
        Q[:,j] = v[:]/r

    return Q


def find_geodesic_midpoint(start_point, end_point, number_of_inner_points, dimension, mass_matrix, molecule, energy,
                           node_number, length_function):
    """ This function computes the local geodesic curve joining start_point to end_point using the L-BFGS method.

    Args:
      start_point (numpy.array): The first end point of the curve.
      end_point (numpy.array): The last end point of the curve.
      number_of_inner_points (int): The number of nodes along the curve, less the end points.
      dimension (int): The dimension of the problem. Computed from the atomistic simulation environment.
      mass_matrix (numpy.array): A diagonal NumPy array containing the masses of the molecular system as computed in the SimulationClient object.
      molecule (ase.atoms): The ASE atoms object corresponding to the molecule being simulated.
      energy (float): The total energy of the system.
      node_number (int): The node number for which we are calculating a new position for.

    Returns:
      numpy.array: The midpoint along the approximate local geodesic curve.

    """

    def metric(point):

        molecule.set_positions(convert_vector_to_atoms(point))

        cf = math.sqrt(max([2*(energy - molecule.get_potential_energy()), 1E-9]))

        return [cf, molecule.get_forces().flatten()/cf]

    Q = get_rotation(start_point, end_point, dimension)

    geodesic, f_min, detail = fmin_l_bfgs_b(func=length_function,
                                            x0=np.zeros(number_of_inner_points*(dimension-1)),
                                            args=(start_point,
                                                  end_point,
                                                  mass_matrix,
                                                  Q,
                                                  number_of_inner_points+2,
                                                  dimension-1,
                                                  metric))

    if detail['warnflag'] != 0:
        print 'BFGS Warning:' + detail['task']

    points = np.reshape(generate_points(geodesic, start_point, end_point, Q, number_of_inner_points+2, dimension-1),
                        (number_of_inner_points+2, dimension))

    if number_of_inner_points % 2 == 1:
        midpoint = points[(number_of_inner_points + 1) / 2]
    else:
        midpoint = 0.5 * (points[number_of_inner_points / 2] + points[(number_of_inner_points / 2) + 1])

    # Return the node number and new midpoint
    return [node_number, midpoint]


def compute_trajectory(trajectory, local_num_nodes, energy, tol, filename, configuration, length_function=length):
    """ This function creates a new task to compute a geodesic midpoint and submits it to the worker pool.

    Args:
      trajectory (curve): A GeometricMD curve object describing the initial trajectory between start and end configurations.
      local_num_nodes (int): The number of points to use when computing the local geodesics.
      energy (float): The total energy of the system.
      tol (float): The tolerance by which if the total curve movement falls below this number then the Birkhoff method stops.
      filename (str): The filename for the output files from the simulation.
      configuration (dict): A dictionary containing additional parameters for the simulation. Accepts: 'processes' - the number of processors to use (defaults to 1), 'write_to_log' - a boolean value, if true writes to a logfile, otherwise prints to console (defaults to False) and 'save_every' - an integer indicating the program will save after every 'save_every'th iteration of the Birkhoff algorithm (defaults to 1).
    """

    # Extract a copy of the ASE atoms object to determine forces
    molecule = trajectory.configuration['molecule']

    # Determine the dimension of the Hamiltonian system
    dimension = len(trajectory.get_points()[0])

    # Compute the mass matrix for the Hamiltonian system
    mass_matrix = np.diag(np.dstack((molecule.get_masses(),) * (dimension /
                                                                len(molecule.get_masses()))).flatten())

    # Set counter for saving
    i = 0

    # Attempt to extract additional parameters, otherwise set default behaviours
    try:
        processes = configuration['processes']
    except KeyError:
        processes = 1
    try:
        write_to_log = configuration['write_to_log']
    except KeyError:
        write_to_log = False
    try:
        save_frequency = configuration['save_every']
    except KeyError:
        save_frequency = 1

    # Initialise logging based on whether the user indicated whether they would like it printing to stout or not.
    if write_to_log:
        logging.basicConfig(format='%(asctime)s %(message)s', filename='out/'+filename+'.log', level=logging.INFO)
    else:
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    # If the user intends to use the algorithm on one core then...
    if processes == 1:

        # Main loop of the Birkhoff algorithm, continues until curve.movement < tol then breaks out
        while True:

            # Iterating over each node in the trajectory find a new position based on the geodesic midpoint
            # joining it's neighbours
            for node_number in trajectory:
                trajectory.set_node_position(node_number, find_geodesic_midpoint(trajectory.points[node_number - 1],
                                                                                 trajectory.points[node_number + 1],
                                                                                 local_num_nodes,
                                                                                 dimension,
                                                                                 mass_matrix,
                                                                                 molecule,
                                                                                 energy,
                                                                                 node_number,
                                                                                 length_function)[1])

            # Once all the nodes in the curve have been tested, print the node movement
            logging.info('Curve Movement: ' + str(trajectory.movement))

            # If the required save frequency has passed then have both a pickle of the current curve object and
            # write out the animation.
            if i % save_frequency == 0:
                pickle.dump(trajectory, open(filename+'.pkl', "wb"))
                write_xyz_animation(filename+'.pkl',filename+'.xyz')

            # If the movement of the curve is below the tol threshold then exit the main loop
            if trajectory.movement < tol:
                break

            # Indicate that the next iteration is to be completed
            trajectory.set_node_movable()

    # Otherwise the user has indicated they would like to perform a parallel computation...
    else:

        # Create a callback function that updates the node position once it is calculated
        def update_curve(result):
            trajectory.set_node_position(result[0], result[1])

        # Create a pool of worker processes to work in parallel
        pool = multiprocessing.Pool(processes=processes)

        # Main loop of the Birkhoff algorithm, continues until curve.movement < tol then breaks out
        while True:

            # Iterating over each node in the trajectory create a task to find a new position based on the
            # geodesic midpoint joining it's neighbours. Add this task to the pool queue.
            for node_number in trajectory:
                pool.apply_async(func=find_geodesic_midpoint,
                         args=(trajectory.points[node_number - 1], trajectory.points[node_number + 1], local_num_nodes,
                               dimension, mass_matrix, molecule, energy, node_number, length_function,),
                         callback=update_curve)

            # If all the nodes in the trajectory have been moved...
            if trajectory.all_nodes_moved():

                # Once all the nodes in the curve have been tested, print the node movement
                logging.info('Curve Movement: ' + str(trajectory.movement))

                # If the required save frequency has passed then have both a pickle of the current curve object and
                # write out the animation.
                if i % save_frequency == 0:
                    pickle.dump(trajectory, open(filename+'.pkl', "wb"))
                    write_xyz_animation(filename+'.pkl',filename+'.xyz')

                # If the movement of the curve is below the tol threshold then exit the main loop
                if trajectory.movement < tol:
                    break

                # Indicate that the next iteration is to be completed
                trajectory.set_node_movable()

        # Once the algorithm has executed close the pool
        pool.close()
        pool.join()

