# add the root directory to the python path so that we can import modules from there
import sys
import pathlib
root_path = pathlib.Path(__file__).resolve().parent.parent.parent.absolute()
sys.path.insert(0, str(root_path))

# set the current working directory to the current file's directory
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from solvers.solver_pctsp.MILP import PCTSPSubPath, decode_subpath
from solvers.solver_pctsp import read_data_datepenalty, evaluate

# Read input data
inst_path = "../../instances/pctsp/size_10/data_4.txt"
prizes, penalties, distance_matrix, minprize = read_data_datepenalty(inst_path)
# prizes, penalties, distance_matrix, minprize, distances = read_data_datepenalty(inst_path, return_distances=True)

# Initialize solver
pctsp = PCTSPSubPath(minprize=minprize)
results = pctsp.solve(prizes, penalties, distance_matrix, k=1, n_threads=0)
# results = pctsp.solve(prizes, penalties, distance_matrix, k=1, n_threads=0, real_sol=[1,9,2,0,4], distances=distances)

print("============= Problem params =============")
print("Input file path:", inst_path)
print("nStops:", len(prizes))
print("Min Prize required:", minprize)
print("==========================================")

print("=============== Solution =================")
sol, outs = decode_subpath(results['store'][0])
print("Route found:", " -> ".join(map(str, sol)))
print("Unselected stops:", ", ".join([str(i) for i in outs[0].tolist()]))
print("Collected prize:", results['total_prizes'][0])
print("Transition cost:", results['total_travel'][0])
print("Total penalty:", results['total_penalties'][0])
print("Objective value:", results['total_travel'][0] + results['total_penalties'][0])
print("Runtime: ", round(results['runtime'][0], 2), "s", sep="")
print("==========================================")

evaluate("8-9-7-2", "-", prizes, penalties, distance_matrix)
evaluate("1-2-4-9-3-8", "-", prizes, penalties, distance_matrix)
print("==========================================")
evaluate("2-9", "-", prizes, penalties, distance_matrix)
