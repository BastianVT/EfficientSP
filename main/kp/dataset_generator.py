import glob
import os
import random
import sys
import pathlib
import numpy as np

# add the root directory to the python path so that we can import modules from there
root_path = pathlib.Path(__file__).resolve().parent.parent.parent.absolute()
sys.path.insert(0, str(root_path))

# set the current working directory to the current file's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from solvers.solver_kp import read_knapsack
from solvers.solver_kp.CP import solve_knapsack, compute_new_profits


# define the number of instances for the dataset
n_instances = 200
# define the number of items per instance of the dataset
n_items_per_instance = 750
# define the solver time limit
time_limit = 60
# define the real parameters to recover
true_parameters = np.array([7, 13, 4, 9])
# read the original profit that was a scalar
origin_profits, origin_weights, origin_capacity = read_knapsack("../../instances/kp/large_scale/knapPI_3_10000_1000_1")
origin_n_items = len(origin_profits)

# convert the scalar profits into a vector
print("Converting scalar profits into a vector...")
new_profits = np.zeros((origin_n_items, len(true_parameters))).astype(int)
for i in range(len(origin_profits)):
    new_profits[i] = compute_new_profits(origin_profits[i][0], parameters=true_parameters)
    print("Item {} converted.".format(i))
print("Conversion finished.\n")

# draw sample items from the large instance and create "mini" Knapsack problems
print("Creating the new instances...")
all_mini_item_indexes = []
n_instances_created = 0
while n_instances_created < n_instances:
    mini_item_indexes = random.sample(population=range(origin_n_items), k=n_items_per_instance)
    while mini_item_indexes in all_mini_item_indexes:
        mini_item_indexes = random.sample(population=range(origin_n_items), k=n_items_per_instance)
    all_mini_item_indexes.append(mini_item_indexes)
    n_instances_created += 1

    os.makedirs("../../instances/kp/multi_profits", exist_ok=True)
    with open("../../instances/kp/multi_profits/inst_3_{}_1000_{}".format(n_items_per_instance, n_instances_created), 'w') as f:
        sum_selected_weights = sum([origin_weights[index] for index in mini_item_indexes])
        f.write(str(n_items_per_instance) + " " + str(int(sum_selected_weights * random.uniform(0.2, 0.8))) + "\n")
        for index in mini_item_indexes:
            f.write(" ".join(map(str, new_profits[index])) + " " + str(origin_weights[index]) + "\n")
        f.flush()

# create the dataset for the learning task
print("Creation of learning dataset...\n")
os.makedirs("../../datasets/kp", exist_ok=True)
with open("../../datasets/kp/dataset_t{}_s{}.csv".format(time_limit, n_items_per_instance), "w") as outfile:
    outfile.write("instance,selected_items,sol_weight,sol_profit\n")
    for ind, file in enumerate(glob.glob("../../instances/kp/multi_profits/inst_3_{}_*".format(n_items_per_instance))):
        print("Instance {} of {}: {}".format(ind+1, n_instances, os.path.basename(file)))
        profits, weights, capacity = read_knapsack(file)
        selected_items, total_weight, total_profit = solve_knapsack(true_parameters, profits, weights, capacity, time_limit)
        outfile.write(file + "," + "-".join(map(str, selected_items)) + "," + str(total_weight) + "," + str(total_profit) + "\n")
        outfile.flush()


