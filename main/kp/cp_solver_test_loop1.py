import numpy as np
import pathlib
from time import time
import sys
import os

# add the root directory to the python path so that we can import modules from there
root_path = pathlib.Path(__file__).resolve().parent.parent.parent.absolute()
sys.path.insert(0, str(root_path))

# set the current working directory to the current file's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from solvers.solver_kp import read_knapsack, read_knapsack_hard, read_knapsack_hard_new
# from solvers.solver_kp.CP import solve_knapsack
from solvers.solver_kp.MILP import solve_knapsack

# start_time = time()
# profits_, weights_, capacity_ = read_knapsack("../../instances/kp/low-dimensional/f10_l-d_kp_20_879")
# profits_, weights_, capacity_ = read_knapsack("../../instances/kp/multi_profits/inst_3_500_1000_1")
# profits_, weights_, capacity_ = read_knapsack_hard("/Users/aglin/Downloads/hardinstances_pisinger/knapPI_11_2000_1000.csv")
# profits_, weights_, capacity_ = read_knapsack_hard_new("/Users/aglin/Downloads/instances/Original_Instances/SpannerStrong062")
path = "/Users/aglin/Downloads/instances/Genetic_Instances"
files = sorted(os.listdir(path))
good_files = []
for index, file in enumerate(files):
    filepath = os.path.join(path, file)
    start_time = time()
    profits_, weights_, capacity_ = read_knapsack_hard_new(filepath)
    # parameters_ = np.random.rand(profits_.shape[1])
    parameters_ = np.ones(profits_.shape[1])
    s_items, t_weight, t_profit = solve_knapsack(parameters_, profits_, weights_, capacity_, time_limit=20)
    runtime = time() - start_time

    if runtime >= 15:
        good_files.append(filepath)
        print("good", len(good_files), filepath, runtime)
    print("{} over {}, {}, {}".format(index + 1, len(files), filepath, runtime))


for file in good_files:
    print(file)

# print("#items:", len(profits_))
# print("Capacity:", capacity_)
# print("Parameters:", parameters_)
# print("Selected items:", s_items)
# print("Number of selected items:", len(s_items))
# print("Total weight:", t_weight)
# print("Total profit:", t_profit)
# print("Runtime:", runtime)
