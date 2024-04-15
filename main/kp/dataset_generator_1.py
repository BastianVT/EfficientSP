import glob
import os
import random
import sys
import pathlib
import numpy as np
import cpmpy as cp

# add the root directory to the python path so that we can import modules from there
root_path = pathlib.Path(__file__).resolve().parent.parent.parent.absolute()
sys.path.insert(0, str(root_path))

# set the current working directory to the current file's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from solvers.solver_kp import read_knapsack, read_knapsack_hard_new
# from solvers.solver_kp.CP import solve_knapsack, compute_new_profits
from solvers.solver_kp.MILP import solve_knapsack


# define the number of instances for the dataset
n_instances = 200
# define the solver time limit
time_limits = [None]
# time_limits = [30, 60]
# define the real parameters to recover
true_parameters = np.array([0, 1, 2, 17])
# true_parameters = np.array([0, 1, 2, 17])

# all_profits = []
# with open("kp_run_sorted", "r") as file:
#     for i in range(n_instances):
#         instance_path = file.readline().split(",")[1]
#         origin_profits, weights, capacity = read_knapsack_hard_new(instance_path)
#         # print(origin_profits.tolist())
#         all_profits = all_profits + origin_profits.flatten().tolist()
# # print("All profits:", all_profits)
# all_profits = list(set(all_profits))
#
# model = cp.Model()
# p = cp.intvar(shape=4, lb=0, ub=1000)
# w1 = cp.intvar(shape=len(all_profits), lb=0, ub=1000)
# w2 = cp.intvar(shape=len(all_profits), lb=0, ub=1000)
# w3 = cp.intvar(shape=len(all_profits), lb=0, ub=1000)
# w4 = cp.intvar(shape=len(all_profits), lb=0, ub=1000)
#
# for i in range(len(all_profits)):
#     model += p[0] * w1[i] + p[1] * w2[i] + p[2] * w3[i] + p[3] * w4[i] == all_profits[i]
#
# model += cp.sum(p) >= 20
# model += cp.AllDifferent(p)
#
# model.solve()
#
# true_parameters = np.array([p[0].value(), p[1].value(), p[2].value(), p[3].value()])
# print("True parameters:", true_parameters)
#
# with open("kp_run_sorted", "r") as file:
#     for i in range(n_instances):
#         instance_path = file.readline().split(",")[1]
#         origin_profits, weights, capacity = read_knapsack_hard_new(instance_path)
#         n_items = len(origin_profits)
#
#         # convert the scalar profits into a vector
#         print("Converting scalar profits into a vector...")
#         profits = np.zeros((n_items, len(true_parameters))).astype(int)
#
#         print("Conversion for instance {}...".format(i))
#         for j in range(len(origin_profits)):
#             profits[j] = compute_new_profits(origin_profits[j][0], parameters=true_parameters)
#             # print("Item {} converted.".format(i))
#         print("Conversion finished for instance {}.\n".format(i))
#
#         os.makedirs("../../instances/kp/multi_profits_hard", exist_ok=True)
#         with open("../../instances/kp/multi_profits_hard/inst_n_{}_id_{}".format(n_items, i), 'w') as f:
#             f.write(str(n_items) + " " + str(capacity) + "\n")
#             for index in range(len(profits)):
#                 f.write(" ".join(map(str, profits[index])) + " " + str(weights[index]) + "\n")
#             f.flush()

n_items = 1000
# create the dataset for the learning task
print("Creation of learning dataset...\n")
os.makedirs("../../datasets/kp", exist_ok=True)
for time_limit in time_limits:
    with open("../../datasets/kp/datasethardeo_t{}_s{}.csv".format(time_limit, n_items), "w") as outfile:
        outfile.write("instance,selected_items,sol_weight,sol_profit\n")
        for ind, file in enumerate(sorted(glob.glob("../../instances/kp/multi_profits_hard/inst_n_{}_*".format(n_items)), key=lambda x: int(x.split("_")[-1]))):
            print("Instance {} of {}: {}. Time limit {}".format(ind+1, n_instances, os.path.basename(file), time_limit))
            profits, weights, capacity = read_knapsack(file)
            selected_items, total_weight, total_profit = solve_knapsack(true_parameters, profits, weights, capacity, time_limit)
            outfile.write(file + "," + "-".join(map(str, selected_items)) + "," + str(total_weight) + "," + str(total_profit) + "\n")
            outfile.flush()


