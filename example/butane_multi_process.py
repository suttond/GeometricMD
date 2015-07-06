from geometricmd.curve_shorten import compute_trajectory
from geometricmd.geometry import Curve
from ase.io import read
from ase.calculators.emt import EMT
from multiprocessing import cpu_count

start_point = read('x0.xyz')
start_point.set_calculator(EMT())
end_point = read('xN.xyz')

traj = Curve(start_point, end_point, 12, 1E+03)

compute_trajectory(traj, 10, 1E+03, 0.001, 'Butane', {'processes': (cpu_count()-1)})