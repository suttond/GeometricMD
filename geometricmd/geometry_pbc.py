# Load packages which are a part of GeometricMD
import numpy as np
import geometricmd.geometry

class Curve(geometricmd.geometry.Curve):

    def __init__(self, start_point, end_point, number_of_nodes, energy):

        geometricmd.geometry.Curve.__init__(self, start_point, end_point, number_of_nodes, energy)

        self.start_cell = start_point.get_cell()
        self.end_cell = end_point.get_cell()

        # Compute the cell tangent vector - the rescaled vector of the line joining the start and end points
        self.cell_tangent = (1/(float(self.number_of_nodes)-1))*np.subtract(self.end_cell, self.start_cell)

        # Compute the cells, the straight line joining the start point to the end point
        self.cells = [self.start_cell]
        for i in xrange(0, int(self.number_of_nodes-1)):
            self.cells.append(np.add(self.cells[i], self.cell_tangent))


    def set_node_position(self, node_number, new_position, new_cell):
        """ Update the position of the node at node_number to new_position. This processes the logic for releasing
        neighbouring nodes for further computation.

        Arguments:
            node_number (int): The node number of the node whose position is to be updated.
            new_position (numpy.array): The new position of the node.

        """

        geometricmd.geometry.Curve.set_node_position(self, node_number, new_position)

        # Update new cell shape
        self.cells[node_number] = new_cell
