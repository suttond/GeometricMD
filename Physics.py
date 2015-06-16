from ase.calculators.emt import EMT


def define_physics():
    """
    A function wrapper used to configure the atomistic simulation environment calculator for the computation.

    :return: ase.calculator -- The ASE calculator.
    """

    return EMT()