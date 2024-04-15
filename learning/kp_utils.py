from .base_utils import *

# add the root directory to the python path so that we can import modules from there
import sys
import pathlib
root_path = pathlib.Path(__file__).resolve().parent.parent.absolute()
sys.path.insert(0, str(root_path))

# from solvers.solver_kp.MILP import solve_knapsack
from solvers.solver_kp.CP import solve_knapsack
from solvers.solver_kp.Greedy import GreedyKP
from solvers.solver_kp.LS import solve_knapsack as solve_knapsack_ls


class KPUtils(ProblemUtils):

    def solve_opti(self, x, w, time_limit, n_jobs, init_obj=None, real_sol=None, slack=None, prediction=False):
        # it returns selected_items, total_weight, total_profit
        results = solve_knapsack(parameters=w, profits=x["profits"], weights=x["weights"], capacity=x["capacity"], time_limit=time_limit, n_jobs=n_jobs, init_obj=init_obj, real_sol=real_sol, slack=slack, prediction=prediction)
        if results is None:
            return None, None, None, 1
        else:
            return results + (1,)

    def solve_ls(self, x, w, time_limit, n_jobs, init_obj=None, real_sol=None, slack=None, prediction=False):
        results = solve_knapsack_ls(parameters=w, profits=x["profits"], weights=x["weights"], capacity=x["capacity"], time_limit=time_limit, init_sol=x["real_sol"], early_stopping=False, prediction=prediction)
        if results is None:
            return None, None, None, 1
        else:
            return results + (1,)

    def solve_ls_violation(self, x, w, time_limit, init_sol):
        results = solve_knapsack_ls(parameters=w, profits=x["profits"], weights=x["weights"], capacity=x["capacity"], time_limit=time_limit, init_sol=x["real_sol"], early_stopping=True, prediction=False)
        if results is None:
            return None, None, None, 1
        else:
            return results + (1,)

    def solve_greedy(self, x, w):
        kp = GreedyKP(parameters=w, profits=x["profits"], weights=x["weights"], capacity=x["capacity"])
        return kp.solve() + (1,)

    def is_obj1_better(self, obj1, obj2, percentage=0.0):
        epsilon = percentage * obj2 if obj2 != self.worst_error() else 0
        return -1 if obj1 < obj2 - epsilon else (0 if obj1 == obj2 - epsilon else 1)

    def worst_error(self):
        return -float("inf")
