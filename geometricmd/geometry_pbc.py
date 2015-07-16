# Load packages which are a part of GeometricMD
import numpy as np
import geometricmd.geometry

class Curve(geometricmd.geometry.Curve):
    """
    The curve class that is part of the geometry_pbc module is a Child object of the curve class in the geometry module.
     It extends provides all of the functionality of its parent class, however has additional attributes to handle
    cell behaviour.

    The purpose of this object is to provide a Curve object with a custom iterator allowing for the Birkhoff algorithm
    to be applied.

    In addition to the attributes of the parent class the following additional attributes are provided.

    Attributes:
      start_cell (numpy.array) :
          A NumPy array describing the first point in the curve.
      end_cell (numpy.array) :
          A NumPy array describing the last point in the curve.
      cell_tangent (numpy.array) :
          A 'tangent' matrix allowing for the code to estimate the cell shapes for intermediate configurations.
      cells (list) :
          A list of cell configurations corresponding to each molecular configuration.

    """

    def __init__(self, start_point, end_point, number_of_nodes, energy):
        """The constructor for the Curve class.

        Args:
          start_point (ase.atoms) :
              An ASE atoms object describing the initial state. A calculator needs to be set on this object.
          end_point (ase.atoms) :
              An ASE atoms object describing the final state.
          number_of_nodes (int) :
              The number of nodes that the curve is to consist of, including the start and end points.
          energy (float) :
              The total Hamiltonian energy to be used in the simulation.

        """

        # Call initialiser from parent class
        geometricmd.geometry.Curve.__init__(self, start_point, end_point, number_of_nodes, energy)

        # Obtain and store initial and final cell shapes
        self.start_cell = start_point.get_cell()
        self.end_cell = end_point.get_cell()

        # Compute the cell tangent vector - the rescaled vector of the line joining the start and end points
        self.cell_tangent = (1/(float(self.number_of_nodes)-1))*np.subtract(self.end_cell, self.start_cell)

        # Compute the cells, the straight line joining the start point to the end point
        self.cells = [self.start_cell]
        for i in xrange(0, int(self.number_of_nodes-1)):
            self.cells.append(np.add(self.cells[i], self.cell_tangent))

    def set_node_position(self, node_number, new_position, new_cell):
        """ Override the set_node_position method of the parent class to handle updating new cell configurations.

        Arguments:
            node_number (int) :
                The node number of the node whose position is to be updated.
            new_position (numpy.array) :
                The new position of the node.
            new_cell (numpy.array) :
                The new cell shape for the node.

        """

        # Call method from parent class
        geometricmd.geometry.Curve.set_node_position(self, node_number, new_position)

        # Update new cell shape
        self.cells[node_number] = new_cell
