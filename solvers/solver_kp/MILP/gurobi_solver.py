# add the root directory to the python path so that we can import modules from there
import sys
import pathlib
root_path = pathlib.Path(__file__).resolve().parent.parent.parent.parent.absolute()
sys.path.insert(0, str(root_path))

import learning

import gurobipy as gp
from gurobipy import GRB
import numpy as np


def selection_vector(selection, num_items):
    return [True if i in selection else False for i in range(num_items)]


def num_based_objective(items, real_sol):
    return np.sum(items[i] for i in range(len(items)) if i not in real_sol)


def phi_based_objective(items, real_sol, profits):
    phi = [gp.quicksum(profits[i][j] * items[i] for i in range(len(items))) for j in range(profits.shape[1])]
    phi_hat = [gp.quicksum(profits[i][j] for i in real_sol) for j in range(profits.shape[1])]
    return gp.quicksum((phi[j] - phi_hat[j]) ** 2 for j in range(profits.shape[1]))


def solve_knapsack(parameters, profits, weights, capacity, time_limit=None, n_jobs=0, init_obj=None, real_sol=None, slack=None, prediction=False):
    num_items = len(weights)

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()

        with gp.Model("KP", env=env) as model:

            # Create a binary variable for each item
            items = {}
            for i in range(num_items):
                items[i] = model.addVar(vtype=GRB.BINARY, name=f'item_{i}')

            # Add the constraint: sum of weights * items <= capacity
            model.addConstr(gp.quicksum(weights[i] * items[i] for i in range(num_items)) <= capacity, name="capacity")

            # Define the objective: maximize sum of profits * items
            objective = gp.quicksum(parameters.dot(profits[i]) * items[i] for i in range(num_items))

            tiny_val = 1e-10

            with_init_bound = not prediction and init_obj is not None

            # not loss-augmented
            if real_sol is None:
                if with_init_bound:
                    model.addConstr(objective, GRB.GREATER_EQUAL, init_obj + tiny_val, name="init_obj")
            else:
                objective -= num_based_objective(items, real_sol)
                if with_init_bound:
                    model.addConstr(objective, GRB.LESS_EQUAL, slack + init_obj - tiny_val, name="init_obj")

            model.setObjective(objective, GRB.MAXIMIZE)

            # Set time limit if provided
            if time_limit is not None:
                model.Params.TimeLimit = time_limit
            model.Params.Threads = n_jobs
            if with_init_bound:
                model.Params.SolutionLimit = 1

            model.optimize()

            if learning.is_verbose():
                print("solver status", model.status)

            if model.status == GRB.INFEASIBLE:
                return None

            # Retrieve the selected items
            selected_items = [i for i in range(num_items) if items[i].x == 1]

            # Calculate the total weight and total profit
            total_weight = sum(weights[i] for i in selected_items)
            total_profit = objective.getValue()

            return selected_items, total_weight, total_profit
