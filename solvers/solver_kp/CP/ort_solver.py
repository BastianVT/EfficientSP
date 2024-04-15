"""Code sample that solves a model and displays a small number of solutions."""
# add the root directory to the python path so that we can import modules from there
import sys
import pathlib
root_path = pathlib.Path(__file__).resolve().parent.parent.parent.parent.absolute()
sys.path.insert(0, str(root_path))

from config import is_verbose

from ortools.sat.python import cp_model
import numpy as np


class VarArraySolutionPrinterWithLimit(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables, limit):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.__solution_limit = limit

    def on_solution_callback(self):
        self.__solution_count += 1
        if self.__solution_count >= self.__solution_limit:
            self.StopSearch()

    def solution_count(self):
        return self.__solution_count


def selection_vector(selection, num_items):
    return [True if i in selection else False for i in range(num_items)]


def num_based_objective(items: list[cp_model.IntVar], real_sol: list = None):
    # return the number of elements in one parameter and not in the other
    return sum(items[i] != selection_vector(real_sol, len(items))[i] for i in range(len(items)))


# def phi_based_objective(items: NDVarArray, real_sol: list, profits: np.ndarray[np.float64, np.ndim[2]]):
#     phi = profits[items].sum(axis=0)
#     phi_hat = profits[real_sol].sum(axis=0)
#     return np.linalg.norm(phi - phi_hat, ord=2) ** 2


def phi_based_objective(items: list[cp_model.IntVar], real_sol: list, profits: np.ndarray):
    phi = [sum(profits[i][j] for i in range(len(items)) if items[i]) for j in range(profits.shape[1])]
    phi_hat = [sum(profits[i][j] for i in real_sol) for j in range(profits.shape[1])]
    return sum(phi[j] - phi_hat[j] for j in range(profits.shape[1])) ** 2


def solve_knapsack(parameters, profits, weights, capacity, time_limit=None, n_jobs=0, init_obj=None, real_sol=None, slack=None, prediction=False):
    num_items = len(weights)
    model = cp_model.CpModel()

    # Create a binary variable for each item
    # items = [model.NewBoolVar(f'item_{i}') for i in range(num_items)]
    items = [model.NewIntVar(0, 1, f'item_{i}') for i in range(num_items)]

    parameters = parameters.astype("int")

    # Add the constraint: sum of weights * items <= capacity
    total_weight = sum(weights[i] * items[i] for i in range(num_items))
    model.Add(total_weight <= capacity)

    # Define the objective: maximize sum of profits * items
    objective = sum(parameters.dot(profits[i]) * items[i] for i in range(num_items))

    # not loss-augmented
    if real_sol is None:
        if not prediction and init_obj is not None:
            y = model.NewIntVar(lb=int(init_obj), ub=99999999999, name="y")
            model.Add(objective == y)
            model.Add(objective > int(init_obj))
            # pass
    else:
        # objective -= phi_based_objective(items, real_sol, profits)
        objective -= num_based_objective(items, real_sol)
        if not prediction and init_obj is not None:
            model.Add(objective < slack + init_obj)

    # if prediction or init_obj is None:
    model.Maximize(objective)

    # Solve the model
    solver = cp_model.CpSolver()

    if time_limit is not None:
        solver.parameters.max_time_in_seconds = time_limit

    if init_obj is not None and not prediction:
        solution_printer = VarArraySolutionPrinterWithLimit([items], 1)
        solver.parameters.enumerate_all_solutions = True
        status = solver.Solve(model, solution_printer)
    else:
        status = solver.Solve(model)

    if is_verbose():
        print("solver status", solver.StatusName(status))

    if status == cp_model.INFEASIBLE or status == cp_model.UNKNOWN:
        return None

    # Retrieve the selected items
    selected_items = [i for i in range(num_items) if solver.Value(items[i]) == 1]

    # Calculate the total weight and total profit
    total_weight = sum(weights[i] for i in selected_items)
    total_profit = 0
    for i in selected_items:
        total_profit += parameters.dot(profits[i])

    return selected_items, total_weight, total_profit
