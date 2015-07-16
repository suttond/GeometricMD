# Load packages which form part of the Python 2.7 core
import pickle
import multiprocessing
import math
import logging

# Load packages which are a part of GeometricMD
from geometricmd.animation import write_xyz_animation
from geometricmd.geometry import convert_vector_to_atoms
from geometricmd.curve_shorten import generate_points, get_rotation, length

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


def find_geodesic_midpoint(start_point, end_point, start_cell, end_cell, number_of_inner_points, dimension, mass_matrix,
                           molecule, energy, node_number, length_function, W, pressure):
    """ This function computes the local geodesic curve joining start_point to end_point using the L-BFGS method.

    Args:
      start_point (numpy.array) :
          The first end point of the curve.
      end_point (numpy.array) :
          The last end point of the curve.
      start_cell (numpy.array) :
          The first cell of the curve.
      end_cell (numpy.array) :
          The last cell of the curve.
      number_of_inner_points (int) :
          The number of nodes along the curve, less the end points.
      dimension (int) :
          The dimension of the problem. Computed from the atomistic simulation environment.
      mass_matrix (numpy.array) :
          A diagonal NumPy array containing the masses of the molecular system as computed in the SimulationClient
          object.
      molecule (ase.atoms) :
          The ASE atoms object corresponding to the molecule being simulated.
      energy (float) :
          The total energy of the system.
      node_number (int) :
          The node number for which we are calculating a new position for.
      length_function (func) :
          A Python function that estimates the length of a curve and also returns it's gradient.
      W (float) :
          A parameter for NPT simulations, used to define how mobile the unit cell is.
      pressure (float) :
          The constant pressure for the simulation.

    Returns:
      int :
          The node number for which the returned midpoint corresponds to.
      numpy.array :
          The midpoint along the approximate local geodesic curve.
      numpy.array :
          The midpoint cell along the approximate local geodesic curve.

    """

    # Define a function that returns sqrt(2(E-V)) and it's gradient based on a given configuration
    def metric(point):

        # Update molecular configuration based on given configuration
        molecule.set_positions(convert_vector_to_atoms(point[:-9]))

        # Compute -grad(V)
        minus_grad_V = molecule.get_forces().flatten()

        # Extract cell information from the point
        cell = np.reshape(convert_vector_to_atoms(point[-9:]), (3, 3))

        # Compute the cell volume scaled by pressure
        cell_volume = pressure * abs(np.linalg.det(cell))

        # Compute gradient of the volume
        grad_cell_volume = -cell_volume * np.linalg.inv(cell).transpose().flatten()

        # Update the molecule's cell
        molecule.set_cell(cell)

        # Evaluate the value of sqrt(2(E-V)), replacing E-V with 1E-9 if V > E.
        cf = math.sqrt(max([2*(energy - molecule.get_potential_energy() - cell_volume), 1E-9]))

        return [cf, np.hstack((minus_grad_V, grad_cell_volume))/cf]

    # Determine a start and end configuration, incorporating the cell
    start = np.hstack((start_point, start_cell.flatten()))
    end = np.hstack((end_point, end_cell.flatten()))

    # Obtain the transformation from dimension dimensional space to the tangent space of the line
    # joining start_point to end_point.
    Q = get_rotation(start, end, dimension + 9)

    # Compute a new mass matrix, accommodating the additional cell information
    mass_matrix = np.vstack((np.hstack((mass_matrix, np.zeros((dimension, 9)))),
                                 np.hstack((np.zeros((9, dimension)), np.diag(np.asarray([W] * 9))))))

    # Perform L-BFGS optimisation on length_function, returning a new geodesic midpoint
    geodesic, f_min, detail = fmin_l_bfgs_b(func=length_function,
                                            x0=np.zeros(number_of_inner_points*(dimension+8)),
                                            args=(start,
                                                  end,
                                                  mass_matrix,
                                                  Q,
                                                  number_of_inner_points+2,
                                                  dimension+8,
                                                  metric))

    # If something went wrong with the L-BFGS algorithm print an error message for the end user
    if detail['warnflag'] != 0:
        print 'BFGS Warning:' + detail['task']

    # Convert the obtained geodesic from it's shift description to the full point description
    points = np.reshape(generate_points(geodesic, start, end, Q, number_of_inner_points+2, dimension+8),
                        (number_of_inner_points+2, dimension+9))

    # Compute the midpoint and corresponding cell
    if number_of_inner_points % 2 == 1:
        # If there is an odd number of inner points then return the middle element of the array
        midpoint = points[(number_of_inner_points + 1) / 2][:-9]
        midpoint_cell = np.reshape(points[(number_of_inner_points + 1) / 2][-9:], (3, 3))
    else:
        # If there is an even number of inner points return the midpoint of the two middle points - this prevents
        # artificial movement of the curve due to the algorithm.
        midpoint = 0.5 * (points[number_of_inner_points / 2] + points[(number_of_inner_points / 2) + 1])[:-9]
        midpoint_cell = np.reshape(0.5 * (points[number_of_inner_points / 2] +
                                          points[(number_of_inner_points / 2) + 1])[-9:], (3, 3))

    # Return the node number and new midpoint
    return [node_number, midpoint, midpoint_cell]


def compute_trajectory(trajectory, local_num_nodes, energy, tol, filename, configuration, length_function=length,
                       W=1.0, pressure=1.0):
    """ This function updates the trajectory object positions to represent the shortest curve.

    Args:
      trajectory (curve) :
          A GeometricMD curve object describing the initial trajectory between start and end configurations.
      local_num_nodes (int) :
          The number of points to use when computing the local geodesics.
      energy (float) :
          The total energy of the system.
      tol (float) :
          The tolerance by which if the total curve movement falls below this number then the Birkhoff method stops.
      filename (str) :
          The filename for the output files from the simulation.
      configuration (dict) :
          A dictionary containing additional parameters for the simulation. Accepts: 'processes' - the number of
          processors to use (defaults to 1), 'write_to_log' - a boolean value, if true writes to a logfile, otherwise prints to console (defaults to False) and 'save_every' - an integer indicating the program will save after every 'save_every'th iteration of the Birkhoff algorithm (defaults to 1).
      length_function (optional, func) :
          A Python function that approximates the length of a curve.
      W (optional, float) :
          A parameter for NPT simulations, used to define how mobile the unit cell is.
      pressure (optional, float) :
          The constant pressure for the simulation.

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
                                                length_function,
                                                W,
                                                pressure)

                trajectory.set_node_position(node_number, result[1], result[2])

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
        pool = multiprocessing.Pool(processes=processes)

        # Main loop of the Birkhoff algorithm, continues until curve.movement < tol then breaks out
        while True:

            # Iterating over each node in the trajectory create a task to find a new position based on the
            # geodesic midpoint joining it's neighbours. Add this task to the pool queue.
            for node_number in trajectory:
                pool.apply_async(func=find_geodesic_midpoint,
                         args=(trajectory.points[node_number - 1], trajectory.points[node_number + 1],
                               trajectory.cells[node_number - 1], trajectory.cells[node_number + 1], local_num_nodes,
                               dimension, mass_matrix, molecule, energy, node_number, length_function, W, pressure,),
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

