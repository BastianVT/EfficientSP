# add the root directory to the python path so that we can import modules from there
import sys
import pathlib
root_path = pathlib.Path(__file__).resolve().parent.parent.parent.absolute()
sys.path.insert(0, str(root_path))

# set the current working directory to the current file's directory
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# from learning import is_verbose

# from solvers.solver_pctsp.CP import PCTSPSubPath, decode_subpath
from solvers.solver_pctsp.MILP import PCTSPSubPath, decode_subpath
from solvers.solver_pctsp import read_data_datepenalty, evaluate

# Read input data
inst_path = "../../instances/pctsp/size_100/data_80.txt"
prizes, penalties, distance_matrix, minprize = read_data_datepenalty(inst_path, matrix_weights=[1., 1., 1., 1.], penalty_weight=1.)

# Initialize solver
pctsp = PCTSPSubPath(minprize=minprize, precision=1)
results = pctsp.solve(prizes, penalties, distance_matrix, n_threads=0)

print("============= Problem params =============")
print("Input file path:", inst_path)
print("nStops:", len(prizes))
print("Min Prize required:", minprize)
print("==========================================")

print("=============== Solution =================")
sol, outs = decode_subpath(results['store'])
print("Route found:", " -> ".join(map(str, sol)))
print("Unselected stops:", ", ".join([str(i) for i in outs.tolist()]))
print("Collected prize:", results['total_prizes'])
print("Transition cost:", results['total_travel'])
print("Total penalty:", results['total_penalties'])
print("Objective value:", results['total_travel'] + results['total_penalties'])
print("Runtime: ", round(results['runtime'], 2), "s", sep="")
print("==========================================")

evaluate("8-9-7-2", "-", prizes, penalties, distance_matrix)
evaluate("1-2-4-9-3-8", "-", prizes, penalties, distance_matrix)
