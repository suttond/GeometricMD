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
    from scipy.optimize import fmin_l_bfgs_b
except ImportError as e:
    print 'SciPy is not installed. Try to run *pip install scipy*.'
    quit()


def length(x, start_point, end_point, number_of_inner_nodes, mass_matrix, dimension, molecule, energy):
    """ This function computes an approximation of the length functional for local geodesics. It also provides the
    gradient of this approximation.

    Args:
      x (numpy.array): The positions of the interior curve points.
      start_point (numpy.array): The first end point of the curve.
      end_point (numpy.array): The last end point of the curve.
      number_of_inner_points (int): The number of nodes along the curve, less the end points.
      mass_matrix (numpy.array): A diagonal NumPy array containing the masses of the molecular system as computed in the SimulationClient object.
      dimension (int): The dimension of the problem. Computed from the atomistic simulation environment.
      molecule (ase.atoms): The ASE atoms object corresponding to the molecule being simulated.
      energy (float): The total energy of the system.

    Returns:
      float, numpy.array: The approximate length of the curve and the corresponding gradient.

    """

    # Convert start_point, x and end_point into an array representing the curve.
    curve = np.vstack((start_point, np.reshape(x, (number_of_inner_nodes, dimension)), end_point))

    # Initialise the list to contain the metric values.
    metric = []

    # We pre-compute the required values of the metric and forces to minimise repeated calls
    for point in curve:
        # Update the positions in the molecule object
        molecule.set_positions(convert_vector_to_atoms(point))

        # Compute the Maupertuis metric co-efficient for this configuration
        if energy <= molecule.get_potential_energy():
            logging.error('Boundary of Configuration Space Reached: Replacing with Small Number.')
            metric_value = 1E-12
        else:
            metric_value = math.sqrt(2*(energy - molecule.get_potential_energy()))

        # Insert this value and the approximate value for the gradient of the Maupertuis metric into the metric list
        metric.append([metric_value, -convert_atoms_to_vector(molecule.get_forces())/metric_value])

    # Initialise the length variable
    l = 0.0

    # For all of the nodes, less the end node...
    for i in xrange(number_of_inner_nodes + 1):

        # Add the trapezoidal rule approximation of the length functional for a line segment
        l += math.sqrt(np.inner(np.subtract(curve[i+1], curve[i]),
                                      mass_matrix.dot(np.subtract(curve[i+1], curve[i])))) * 0.5 \
             * (metric[i+1][0]+metric[i][0])

    # Initialize the list to store the gradient approximation
    g = []

    # For all of the interior nodes...
    for i in xrange(1,number_of_inner_nodes + 1):

        # Pre-compute repeated quantities in the gradient expression
        n_1 = np.linalg.norm(np.subtract(curve[i+1], mass_matrix.dot(curve[i])))
        n_2 = np.linalg.norm(np.subtract(curve[i], mass_matrix.dot(curve[i-1])))
        t_1 = mass_matrix.dot(np.subtract(curve[i+1], curve[i])) / n_1
        t_2 = mass_matrix.dot(np.subtract(curve[i], curve[i-1])) / n_2

        # Compute the gradient component
        g_component = metric[i][1] * (n_1 + n_2) - (metric[i+1][0] + metric[i][0]) * t_1 \
                      + (metric[i-1][0] + metric[i][0]) * t_2

        # Append this component into the gradient list
        g.append(g_component)

    # Return the values of length and gradient - the gradient is flattened for dimensional consistency
    return l, 0.5*np.asarray(g).flatten()


def find_geodesic_midpoint(start_point, end_point, number_of_inner_points, dimension, mass_matrix, molecule, energy,
                           node_number):
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

    # Compute the tangent vector - the rescaled vector of the line joining the start and end points
    tangent = (1/(float(number_of_inner_points+1)))*np.subtract(end_point, start_point)

    # Compute the initial curve, the straight line joining the start point to the end point
    start_curve = np.asarray([np.add(start_point, tangent)], dtype='float64')
    for i in xrange(0, int(number_of_inner_points-1)):
        start_curve = np.concatenate((start_curve, [np.add(start_curve[i], tangent)]), axis=0)
    start_curve = start_curve.flatten()

    # Perform the L-BFGS method on the length functional
    geodesic, f_min, detail = fmin_l_bfgs_b(func=length, x0=start_curve, args=(start_point, end_point,
                                                           number_of_inner_points, mass_matrix, dimension,
                                                           molecule, energy), pgtol=0.1)

    # Return the node number and new midpoint
    return [node_number, np.reshape(geodesic, (number_of_inner_points, dimension))[(number_of_inner_points + 1) / 2]]


def compute_trajectory(trajectory, local_num_nodes, energy, tol, filename, configuration):
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
                                                                                 node_number)[1])

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
                               dimension, mass_matrix, molecule, energy, node_number,),
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

