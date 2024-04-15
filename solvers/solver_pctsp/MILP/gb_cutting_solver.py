# add the root directory to the python path so that we can import modules from there
import sys
import pathlib
root_path = pathlib.Path(__file__).resolve().parent.parent.parent.parent.absolute()
sys.path.insert(0, str(root_path))

from config import is_verbose

from math import sqrt
import itertools
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from .utils import decode_subpath, phi_loss_grb, arc_hamming_loss_grb

# verbose=False
class PCTSPSubPath:
    """MILP model for PC-TSP using Gurobi.

    Args:
        minprize (float, optional): minimum total prize collected per tour, default None.
        precision (float, optional): precision when converting float to int, default 1e-4.
        time_limit (int, optional): maximum runtime in seconds, default 180.
    """

    def __init__(self, minprize=None, precision=1e-4, time_limit=180):
        self.minprize = minprize
        self.precision = precision
        self.time_limit = time_limit

    def solve(self, prize, penalty, distmatrix, minprize=None, distances=None, n_threads=0, init_obj=None, real_sol=None, slack=None, prediction=False):
        """Solve the PC-TSP problem."""
        if minprize is None:
            minprize = self.minprize

        n_vertex = len(prize)

        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()

            with gp.Model("PCTSP", env=env) as model:
                model._ub = float("inf") if init_obj is None else init_obj
                model._best_solution = None
                model._early_stop = False if init_obj is None else True
                model._prediction = prediction

                # Decision variables
                arcs_in = model.addVars(n_vertex + 1, n_vertex + 1, vtype=GRB.BINARY, name="arc")
                stops_outs = [arcs_in[i, i] for i in range(n_vertex + 1)]

                for i in range(n_vertex + 1):
                    # Each vertex has exactly one outgoing arc
                    model.addConstr(gp.quicksum(arcs_in[i, j] for j in range(n_vertex + 1) if j != i) == 1 - stops_outs[i], f"outgoing_{i}")
                    # Each vertex has exactly one incoming arc
                    model.addConstr(gp.quicksum(arcs_in[j, i] for j in range(n_vertex + 1) if j != i) == 1 - stops_outs[i], f"incoming_{i}")

                # Dummy node necessarily selected to break the only selected sub-tour into a sub-path
                model.addConstr(stops_outs[n_vertex] == 0, "dummy_stop")

                # Minimum cumulated prize
                model.addConstr(gp.quicksum((1 - stops_outs[i]) * prize[i] for i in range(n_vertex)) >= minprize, "min_prize")

                # Objective function
                cost_fun = (gp.quicksum(stops_outs[i] * penalty[i] for i in range(n_vertex)) + gp.quicksum(distmatrix[i, j] * arcs_in[i, j] for i in range(n_vertex) for j in range(n_vertex)))

                tiny_val = 1e-3

                with_init_bound = not prediction and init_obj is not None

                # For loss-augmented inference
                if real_sol is None:
                    if with_init_bound:
                        y = model.addVar(vtype=GRB.CONTINUOUS, name="y", ub=init_obj, lb=-GRB.INFINITY)
                        model.addConstr(cost_fun, gp.GRB.EQUAL, y, name="init_obj")  # violation condition
                        model.addConstr(cost_fun, gp.GRB.LESS_EQUAL, init_obj - tiny_val, name="init_obj")  # violation condition
                else:
                    if distances is not None:
                        cost_fun -= phi_loss_grb(arcs_in, distances, real_sol, n_vertex)
                    else:
                        cost_fun -= arc_hamming_loss_grb(arcs_in, real_sol, n_vertex, model)

                    if with_init_bound:
                        model.addConstr(cost_fun, gp.GRB.GREATER_EQUAL, slack + init_obj + tiny_val, name="init_obj")  # violation condition for the loss-augmented inference

                model.setObjective(cost_fun, GRB.MINIMIZE)

                # Set solver parameters
                model.Params.Threads = n_threads
                if self.time_limit is not None:
                    model.Params.TimeLimit = self.time_limit

                # Solve the problem
                model._vars = arcs_in
                model.Params.lazyConstraints = 1  # one cut per lazy constraint callback

                # set the callback function to be called for each new solution. if the solution violate a subtour
                # constraint, then the callback function will add the violated constraint to the model to forbid this solution
                model.optimize(subtour_elimination)

                if is_verbose():
                    print("model.status ", model.status)

                if model.status == GRB.INFEASIBLE:
                    return None

                if model.status == GRB.INTERRUPTED:
                    if model._best_solution is None:
                        return None

                if model.status == GRB.TIME_LIMIT:
                    if model._best_solution is None:
                        print("model interrupted, no sol found")
                        return None
                    else:
                        print("model interrupted, best sol ", model._best_solution)
                # Retrieve the solution
                store = np.zeros((n_vertex + 1, n_vertex + 1))
                for i in range(n_vertex + 1):
                    for j in range(n_vertex + 1):
                        if model.status == GRB.INTERRUPTED:
                            store[i, j] = round(model._best_solution[i, j])
                        else:
                            store[i, j] = arcs_in[i, j].x

                store_in = store[:-1, :-1]
                in_sol, out_sol = decode_subpath(store)  # format: list of int (stops)
                # convert the results into a boolean mask
                in_sol = np.array([1 if i in in_sol else 0 for i in range(n_vertex)])
                out_sol = np.array([1 if i in out_sol else 0 for i in range(n_vertex)])

                results = {
                    'store': store,
                    'in_solution': in_sol,
                    'total_prizes': in_sol @ prize,
                    'total_penalties': out_sol @ penalty,
                    'total_travel': np.sum(distmatrix * store_in),
                    'runtime': model.runtime,
                    'optimal': model.status == GRB.OPTIMAL
                }

                return results


def subtour_elimination(model, where):
    """Callback - use lazy constraints to eliminate sub-tours."""
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        arcs = model.cbGetSolution(model._vars)
        n = int(sqrt(len(arcs)) - 1)
        selected = gp.tuplelist((i, j) for i in range(n) for j in range(n) if i != j and arcs[i, j] > 0.5)

        # find the shortest cycle in the selected edge list
        tour = find_tour(selected, n)
        if tour == range(n+1):  # no subtour found
            if model._early_stop:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj_val < model._ub:  # If the current solution is better than the current upper bound
                    model._ub = obj_val  # Update the upper bound
                    model._best_solution = model.cbGetSolution(model._vars)  # Store the best solution
                    model.cbSetSolution(model._vars, model._best_solution)  # Set the current solution as the best solution
                    model.terminate()  # Terminate the optimization process

        # in case a subtour is found, add a subtour elimination constraint
        model.cbLazy(gp.quicksum(model._vars[i, j] for i, j in itertools.permutations(tour, 2)) <= len(tour) - 1)


def find_tour(edges, n):
    """Given a tuplelist of edges, find the shortest subtour."""
    unvisited = list(set([i for i, j in edges] + [j for i, j in edges]))
    cycle = range(n+1)
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        next = None
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*')]
            next = neighbors[0] if len(neighbors) > 0 else None
            neighbors = [j for j in neighbors if j in unvisited]
        if len(thiscycle) > 0:
            if next == thiscycle[0] and len(cycle) > len(thiscycle):
                cycle = thiscycle
    return cycle
