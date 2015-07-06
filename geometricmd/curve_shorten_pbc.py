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
    from ase.optimize import BFGS
    from ase.constraints import StrainFilter
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


def length(x, start_point, end_point, start_cell, end_cell, number_of_inner_nodes, mass_matrix, dimension, molecule,
           energy, pressure, W, epsilon=1E-12):
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
      pressure (float): The pressure of the system.
      W (float): The stiffness of the unit cell.
      epsilon (optional, float): A small number used where the metric co-efficient is zero or not defined.

    Returns:
      float, numpy.array: The approximate length of the curve and the corresponding gradient.

    """

    # Convert start_point, x and end_point into an array representing the curve.
    eff_mass_matrix = np.vstack((np.hstack((mass_matrix, np.zeros((dimension, 9)))),
                                 np.hstack((np.zeros((9, dimension)), np.diag(np.asarray([W] * 9))))))

    x = np.hstack((start_point, x[:-number_of_inner_nodes*9], end_point, start_cell.flatten(),
                   x[-number_of_inner_nodes*9:], end_cell.flatten()))

    # Initialise the list to contain the metric values.
    metric = []

    for state_number in xrange(0,number_of_inner_nodes+2):
        # Update the positions in the molecule object
        cell_matrix = np.reshape(x[state_number*9+dimension*(number_of_inner_nodes+2):(state_number+1)*9+dimension*(number_of_inner_nodes+2)], (3, 3))
        cell_volume = pressure * abs(np.linalg.det(cell_matrix))

        molecule.set_positions(convert_vector_to_atoms(x[state_number*dimension:(state_number+1)*dimension]))
        molecule.set_cell(cell_matrix)

        val = molecule.get_potential_energy() + cell_volume

        # Compute the Maupertuis metric co-efficient for this configuration
        if energy <= val:
            logging.error('Boundary of Configuration Space Reached: Replacing with Small Number.')
            metric_value = epsilon
        else:
            metric_value = math.sqrt(2*(energy - val))

        # Compute gradient of metric co-efficient
        metric_gradient = -convert_atoms_to_vector(molecule.get_forces()) / metric_value

        grad_volume = -cell_volume * np.linalg.inv(cell_matrix).transpose().flatten() / metric_value

        # Insert this value and the approximate value for the gradient of the Maupertuis metric into the metric list
        metric.append([metric_value, np.hstack((metric_gradient, grad_volume))])

    def state(i):
        return np.hstack((x[(i)*dimension:(i+1)*dimension],
                          x[i*9+dimension*(number_of_inner_nodes+2):(i+1)*9+dimension*(number_of_inner_nodes+2)]))

    def mass_norm(vector):
        return math.sqrt(np.inner(vector, eff_mass_matrix.dot(vector)))

    # Initialise the length variable
    n_1 = mass_norm(np.subtract(state(1), state(0)))
    t_1 = eff_mass_matrix.dot(np.subtract(state(1), state(0))) / n_1
    m_1 = metric[1][0] + metric[0][0]
    l = n_1 * m_1


    # Initialize the list to store the gradient approximation
    g = np.empty((number_of_inner_nodes, dimension))
    c = np.empty((number_of_inner_nodes, 9))

    # For all of the interior nodes...
    for i in xrange(1, number_of_inner_nodes + 1):

        # Pre-compute repeated quantities in the gradient expression
        n_2 = mass_norm(np.subtract(state(i+1), state(i)))
        t_2 = eff_mass_matrix.dot(np.subtract(state(i+1), state(i))) / n_2
        m_2 = metric[i+1][0] + metric[i][0]

        # Compute length contribution
        l += n_2 * m_2

        # Compute the gradient component
        g_component = metric[i][1] * (n_2 + n_1) - m_2 * t_2 + m_1 * t_1

        # Append this component into the gradient list

        g[i-1] = g_component[:-9]
        c[i-1] = g_component[-9:]

        n_1 = n_2
        t_1 = t_2
        m_1 = m_2

    # Return the values of length and gradient - the gradient is flattened for dimensional consistency

    return 0.5*l, 0.5*np.hstack((g.flatten(), c.flatten()))


def find_geodesic_midpoint(start_point, end_point, start_cell, end_cell, number_of_inner_points, dimension,
                           mass_matrix, molecule, energy, node_number, pressure, W):
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
      pressure (float): The pressure of the system.
      W (float): The stiffness of the unit cell.

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

    # Compute the cell tangent vector - the rescaled vector of the line joining the start and end points
    cell_tangent = (1/(float(number_of_inner_points)-1))*np.subtract(end_cell, start_cell)

    # Compute the cells, the straight line joining the start point to the end point
    cells = [np.add(start_cell, cell_tangent)]
    for i in xrange(0, int(number_of_inner_points-1)):
        cells.append(np.add(cells[i], cell_tangent))
    cells = np.asarray(cells).flatten()

    x0 = np.hstack((start_curve, cells))

    # Perform the L-BFGS method on the length functional
    geodesic, f_min, detail = fmin_l_bfgs_b(func=length, x0=x0, args=(start_point, end_point, start_cell, end_cell,
                                                           number_of_inner_points, mass_matrix, dimension,
                                                           molecule, energy, pressure, W))

    flat_mid_cell = np.reshape(geodesic[-number_of_inner_points*9:],
                               (number_of_inner_points, 9))[(number_of_inner_points + 1) / 2]

    # Return the node number and new midpoint
    return [node_number, np.reshape(geodesic[:-number_of_inner_points*9],
                                             (number_of_inner_points, dimension))[(number_of_inner_points + 1) / 2],
            np.reshape(flat_mid_cell, (3, 3))]


def compute_trajectory(trajectory, local_num_nodes, energy, pressure, W, tol, filename, configuration):
    """ This function creates a new task to compute a geodesic midpoint and submits it to the worker pool.

    Args:
      trajectory (curve): A GeometricMD curve object describing the initial trajectory between start and end configurations.
      local_num_nodes (int): The number of points to use when computing the local geodesics.
      energy (float): The total energy of the system.
      pressure (float): The pressure of the system.
      W (float): The stiffness of the unit cell.
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
                result = find_geodesic_midpoint(trajectory.points[node_number - 1],
                                                                                 trajectory.points[node_number + 1],
                                                                                 trajectory.cells[node_number - 1],
                                                                                 trajectory.cells[node_number + 1],
                                                                                 local_num_nodes,
                                                                                 dimension,
                                                                                 mass_matrix,
                                                                                 molecule,
                                                                                 energy,
                                                                                 node_number,
                                                                                 pressure,
                                                                                 W)
                trajectory.set_node_position(result[0], result[1], result[2])

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
            trajectory.set_node_position(result[0], result[1], result[2])

        # Create a pool of worker processes to work in parallel
        pool = multiprocessing.Pool(processes=(processes-1))

        # Main loop of the Birkhoff algorithm, continues until curve.movement < tol then breaks out
        while True:

            # Iterating over each node in the trajectory create a task to find a new position based on the
            # geodesic midpoint joining it's neighbours. Add this task to the pool queue.
            for node_number in trajectory:
                pool.apply_async(func=find_geodesic_midpoint,
                         args=(trajectory.points[node_number - 1], trajectory.points[node_number + 1], trajectory.cells[node_number - 1],
                                                                                 trajectory.cells[node_number + 1], local_num_nodes,
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

