# Load packages which form part of the Python 2.7 core
import pickle
import multiprocessing
import math
import logging

# Load packages which are a part of GeometricMD
from geometricmd.animation import write_xyz_animation
from geometricmd.geometry import convert_vector_to_atoms

# Load additional packages and check if they are installed
import numpy as np
from scipy.optimize import fmin_l_bfgs_b


def generate_points(x, start_point, end_point, rotation_matrix, total_number_of_points, co_dimension):
    """ This function computes the local geodesic curve joining start_point to end_point using the L-BFGS method.

    Args:
      x (numpy.array) :
          Array of vectors in co-dimension dimensional space, stacked flat. This vector characterises points of the
          curve as translations from the line joining start_point to end_point.
      start_point (numpy.array) :
          The first end point of the curve.
      end_point (numpy.array) :
          The last end point of the curve.
      rotation_matrix (numpy.array) :
          A numpy.array describing the rotation from co_dimension + 1 dimensional space to the tangent space of the line
          joining start_point to end_point.
      total_number_of_points (int):
          The total number of points used in the local geodesic computation, including endpoints.
      co_dimension (int):
          The dimension of the configuration space, less one.

    Returns:
      numpy.array :
          The midpoint along the approximate local geodesic curve.

    """

    # Compute tangent direction of line joining start and end points
    tangent = np.subtract(end_point, start_point)/(total_number_of_points-1)

    # Initialise list to store points
    points = []

    # Generate points that are uniformly distributed along the initial line
    for i in xrange(total_number_of_points):
        points.append(np.add(start_point, float(i)*tangent))

    # Shift the points as encoded in x
    for i in xrange(0, len(x)/co_dimension):

        # Embed vector i into co_dimension + 1 dimensional space
        unrotated_shift = np.hstack((np.zeros(1), x[i*co_dimension:(i+1)*co_dimension]))

        # Convert vector, by rotation, from shift from e_1 basis direction to shift from tangent direction
        shift = rotation_matrix.dot(unrotated_shift)

        # Append point to list
        points[i+1] = np.add(points[i+1], shift)

    return points


def compute_metric(points, metric_function):
    """ Takes a list of NumPy arrays describing points molecular configurations, evaluates the metric at each point and
    returns a list of metric values at those points.

    Args:
      points (list) :
          A list of NumPy arrays describing molecular configurations.
      metric_function (func) :
          A Python function which gives the value of sqrt(2(E - V)) at a given point.


    Returns:
      list :
          A list of metric values at the corresponding points.

    """

    # Initialise the list to store metric values
    metric = []

    # For each point, compute the metric at that point
    for point in points:
        metric.append(metric_function(point))

    return metric


def norm(x, matrix):
    """ Computes the value of sqrt(<x, matrix*x>).

    Args:
      x (numpy.array) :
          A vector, stored as a NumPy array, to compute the norm for.
      matrix (numpy.array) :
          A matrix, stored as a NumPy array, used in the computation of <x, matrix*x>.


    Returns:
      float :
          The value of sqrt(<x, matrix*x>).

    """

    return math.sqrt(np.inner(x, matrix.dot(x)))


def norm_gradient(x, matrix):
    """ Computes the gradient of sqrt(<x, matrix*x>).

    Args:
      x (numpy.array) :
          A vector, stored as a NumPy array, to compute the norm for.
      matrix (numpy.array) :
          A matrix, stored as a NumPy array, used in the computation of <x, matrix*x>.


    Returns:
      numpy.array :
          The gradient of sqrt(<x, matrix*x>).

    """

    a = (matrix + matrix.transpose())

    return a.dot(x) / (2 * norm(x, matrix))


def length(x, start_point, end_point, mass_matrix, rotation_matrix, total_number_of_points, co_dimension, metric):
    """ This function computes the length of the local geodesic as a function of shifts from the line joining
    start_point to end_point. It also returns the gradient of this function for the L-BFGS method.

    Args:
      x (numpy.array) :
          Array of vectors in co-dimension dimensional space, stacked flat. This vector characterises points of the
          curve as translations from the line joining start_point to end_point.
      start_point (numpy.array) :
          The first end point of the curve.
      end_point (numpy.array) :
          The last end point of the curve.
      mass_matrix (numpy.array) :
          A numpy.array describing the mass matrix for the molecule as a dynamical system.
      rotation_matrix (numpy.array) :
          A numpy.array describing the rotation from co_dimension + 1 dimensional space to the tangent space of the line
          joining start_point to end_point.
      total_number_of_points (int) :
          The total number of points used in the local geodesic computation, including endpoints.
      co_dimension (int) :
          The dimension of the configuration space, less one.
      metric (func) :
          A Python function which when given a list of NumPy arrays, returns a list of metric values on those arrays.

    Returns:
      float :
          The approximate length of the geodesic.
      numpy.array :
          The gradient of the approximate length of the geodesic.

    """

    # Convert the shifts x into points in the full dimensional space
    points = generate_points(x, start_point, end_point, rotation_matrix, total_number_of_points, co_dimension)

    # Pre-compute the metric values to minimise repeated metric evaluations
    a = compute_metric(points, metric)

    # Compute quantities used to determine the length and gradient
    n = np.subtract(points[1], points[0])
    b = norm(n, mass_matrix)
    c = norm_gradient(n, mass_matrix)
    u = (a[1][0]+a[0][0])

    # Initialise the length with the trapezoidal approximation of the first line segments length
    l = u * b
    # Initialise a list to store the gradient
    g = []

    for i in xrange(1, len(points)-1):

        # Compute the quantities needed for the next trapezoidal rule approximation.
        n = np.subtract(points[i+1], points[i])
        d = norm(n, mass_matrix)
        e = norm_gradient(n, mass_matrix)
        v = (a[i+1][0]+a[i][0])

        # Add length of line segment to total length
        l += v * norm(n, mass_matrix)

        # Compute next gradient component and update gradient
        g.append(rotation_matrix.transpose().dot(a[i][1] * (b + d) + u * c - v * e)[1:])

        # Pass back calculated values for efficiency
        b = d
        c = e
        u = v

    return 0.5 * l, 0.5 * np.asarray(g).flatten()


def get_rotation(start_point, end_point, dimension):
    """ Computes the transformation from dimension dimensional space to the tangent space of the line
          joining start_point to end_point.

    Args:
      start_point (numpy.array) :
          The first end point of the curve.
      end_point (numpy.array) :
          The last end point of the curve.
      dimension (int) :
          The dimension of the configuration space.

    Returns:
      numpy.array :
          The matrix representing the linear transformation from dimension dimensional space to the tangent space of
          the line joining start_point to end_point.

    """

    # Compute tangent direction of line joining start and end points
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
      start_point (numpy.array) :
          The first end point of the curve.
      end_point (numpy.array) :
          The last end point of the curve.
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

    Returns:
      int :
          The node number for which the returned midpoint corresponds to.
      numpy.array :
          The midpoint along the approximate local geodesic curve.

    """

    # Define a function that returns sqrt(2(E-V)) and it's gradient based on a given configuration
    def metric(point):

        # Update molecular configuration based on given configuration
        molecule.set_positions(convert_vector_to_atoms(point))

        # Evaluate the value of sqrt(2(E-V)), replacing E-V with 1E-9 if V > E.
        cf = math.sqrt(max([2*(energy - molecule.get_potential_energy()), 1E-9]))

        # Return sqrt(2(E-V)) and it's gradient
        return [cf, molecule.get_forces().flatten()/cf]

    # Obtain the transformation from dimension dimensional space to the tangent space of the line
    # joining start_point to end_point.
    Q = get_rotation(start_point, end_point, dimension)

    # Perform L-BFGS optimisation on length_function, returning a new geodesic midpoint
    geodesic, f_min, detail = fmin_l_bfgs_b(func=length_function,
                                            x0=np.zeros(number_of_inner_points*(dimension-1)),
                                            args=(start_point,
                                                  end_point,
                                                  mass_matrix,
                                                  Q,
                                                  number_of_inner_points+2,
                                                  dimension-1,
                                                  metric))

    # If something went wrong with the L-BFGS algorithm print an error message for the end user
    if detail['warnflag'] != 0:
        print 'BFGS Warning:' + detail['task']

    # Convert the obtained geodesic from it's shift description to the full point description
    points = np.reshape(generate_points(geodesic, start_point, end_point, Q, number_of_inner_points+2, dimension-1),
                        (number_of_inner_points+2, dimension))

    # Compute the midpoint
    if number_of_inner_points % 2 == 1:
        # If there is an odd number of inner points then return the middle element of the array
        midpoint = points[(number_of_inner_points + 1) / 2]
    else:
        # If there is an even number of inner points return the midpoint of the two middle points - this prevents
        # artificial movement of the curve due to the algorithm.
        midpoint = 0.5 * (points[number_of_inner_points / 2] + points[(number_of_inner_points / 2) + 1])

    # Return the node number and new midpoint
    return [node_number, midpoint]


def compute_trajectory(trajectory, local_num_nodes, energy, tol, filename, configuration, length_function=length):
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
          processors to use (defaults to 1), 'write_to_log' - a boolean value, if true writes to a logfile, otherwise
          prints to console (defaults to False) and 'save_every' - an integer indicating the program will save after
          every 'save_every'th iteration of the Birkhoff algorithm (defaults to 1).
      length_function (func) :
          A Python function that approximates the length of a curve.

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

