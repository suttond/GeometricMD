import numpy as np
from numpy import linalg as la


def convert_atoms_to_vector(atoms):
    """

    :param atoms:
    :return:
    """
    vector = np.asarray([], dtype='float64')
    for atom in atoms:
        vector = np.hstack((vector, np.asarray(atom, dtype='float64')))
    return vector


def convert_vector_to_atoms(vector, dimension=3):
    """

    :param vector:
    :param dimension:
    :return:
    """
    return np.asarray(vector, dtype='float64').reshape((len(vector) / dimension, dimension))


class Curve:

    def __init__(self, start_point, end_point, number_of_nodes, total_number_of_nodes, molecule):

        self.start_point = np.asarray(start_point, dtype='float64')
        self.end_point = np.asarray(end_point, dtype='float64')
        self.number_of_nodes = int(number_of_nodes)
        self.total_number_of_nodes = float(total_number_of_nodes)

        self.tangent = (1/(float(self.number_of_nodes)-1))*np.subtract(self.end_point, self.start_point)

        self.points = np.asarray([self.start_point], dtype='float64')
        for i in xrange(0, int(self.number_of_nodes - 1)):
            self.points = np.concatenate((self.points, [np.add(self.points[i], self.tangent)]), axis=0)
        np.concatenate((self.points, [self.end_point]), axis=0)

        self.default_initial_state = np.zeros(self.number_of_nodes, dtype='int')
        for i in xrange(self.number_of_nodes - 1):
            if i % 2 != 0:
                self.default_initial_state[i] = 2

        self.movement = 0.0
        self.nodes_moved = np.ones(self.number_of_nodes, dtype='int')
        self.node_movable = np.copy(self.default_initial_state)
        self.number_of_distinct_nodes_moved = 0

        self.configuration = {}

        self.molecule = molecule

    def set_node_movable(self):
        self.movement = 0.0
        self.nodes_moved = np.ones(self.number_of_nodes, dtype='int')
        self.node_movable = np.copy(self.default_initial_state)
        self.number_of_distinct_nodes_moved = 0

    def set_node_position(self, node_number, new_position):
        self.movement += self.number_of_nodes * \
                         la.norm(np.subtract(new_position, self.points[node_number]))

        self.points[node_number] = new_position

        self.node_movable[node_number-1] += 1
        self.node_movable[node_number+1] += 1
        self.node_movable[0] = 0
        self.node_movable[-1] = 0
        if node_number == 2:
            self.node_movable[1] += 1
            self.node_movable[0] = 0
        if node_number == self.number_of_nodes - 3:
            self.node_movable[-2] += 1
            self.node_movable[-1] = 0

        self.nodes_moved[node_number] = 0
        self.number_of_distinct_nodes_moved = sum(self.nodes_moved)

    def next(self):

        try:
            next_movable_node = np.where(np.multiply(self.node_movable, self.nodes_moved) > 1)[0][0]

            if next_movable_node is None:
                next_movable_node = np.where(self.node_movable > 1)[0][0]

            if next_movable_node is None:

                raise StopIteration

            else:

                self.node_movable[next_movable_node] = 0

                return next_movable_node

        except IndexError:
            raise StopIteration

    def get_points(self):
        return self.points

    def all_nodes_moved(self):
        if self.number_of_distinct_nodes_moved == 2:
            return True
        else:
            return False

    def __iter__(self):
        return self