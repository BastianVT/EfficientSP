"""
Create a class for the solver that have a method `solve` and an enum for the solver type.
"""

from enum import Enum
from time import perf_counter

import sys
import pathlib
root_path = pathlib.Path(__file__).resolve().parent.parent.absolute()
sys.path.insert(0, str(root_path))

from config import is_verbose

from .base_caching_strategy import CachingStrategy
from .base_problem import Problem
from .base_utils import ProblemUtils
from .base_solution import Solution


class Solver(Enum):
    OPTI = 1
    LS = 2
    LS_VIOLATION = 3
    GREEDY = 4


class Inference:

    def __init__(self, problem_name, solver, loss_fun, n_jobs, time_limit, caching_strategy: CachingStrategy, utils: ProblemUtils):
        self.solver = solver
        self.problem_name = problem_name
        self.loss_fun = loss_fun
        self.n_jobs = n_jobs
        self.caching_strategy = caching_strategy
        self.time_limit = time_limit
        self.utils = utils
        self.prediction_triggered = False
        self.inference_calls = 0
        self.inference_time = 0
        self.inference_index = 0
        self.problem_size = None
        self.n_train = None

    def solve(self, x, w, init_obj=None, slack=None, enforced_time_limit=None) -> (Solution, list[int], float):
        time_limit = self.time_limit if enforced_time_limit is None else enforced_time_limit
        if self.solver == Solver.LS_VIOLATION:
            y, out, obj, _ = self.utils.solve_ls_violation(x, w, time_limit, x["real_sol"])
            if is_verbose():
                print("ls viol sol obj", obj)
            if obj is not None and self.utils.is_obj1_better(obj, init_obj) < 1:
                y, out, obj, _ = None, None, None, None
            if is_verbose():
                print("Found Obj ls", obj)
        elif self.solver == Solver.OPTI:
            y, out, obj, _ = self.utils.solve_opti(x, w, time_limit, self.n_jobs, init_obj=init_obj, real_sol=x["real_sol"] if slack is not None else None, slack=slack, prediction=False)
            if is_verbose():
                print("Found Obj opti", obj)
        elif self.solver == Solver.GREEDY:
            y, out, obj, _  = self.utils.solve_greedy(x, w)
            if is_verbose():
                print("Found Obj greedy", obj)
        else:  # LS
            print('what the hell')    
            y, out, obj, _ = self.utils.solve_ls(x, w, time_limit, self.n_jobs, init_obj=init_obj, real_sol=x["real_sol"] if slack is not None else None, slack=slack, prediction=False)
            if is_verbose():
                print("Found Obj ls", obj)
        solution = Solution(x, y, self.loss_fun, self.solver, self.utils)
        
        return solution, out, obj, _
    

    def inference(self, x: Problem, w, use_cache, classic, slack=None, enforced_time_limit=None) -> (Solution, bool):
        # compute the initial bound to be used for the search
        if use_cache:  #caching enabled. Check if a relevant solution is already in the cache
            # the best solution in the cache is computed returned if it is relevant
            relevant_solution, init_bound = self.caching_strategy.should_solve(w, self.inference_calls, self.inference_index, self.n_train, x, x["real_sol"])
        else:  # caching disabled. The objective of the real solution is used as initial bound
            relevant_solution, init_bound = None, x.get_preferred_solution().compute_objective_w_phi(w)

        # if cache is disabled or no relevant solution is in the cache, then call the solver
        if relevant_solution is None:            
            inf_start = perf_counter()
            if not classic:
                print('SAT - solving')
                solution, out, obj,_ = self.solve(x, w, init_obj=init_bound, slack=slack, enforced_time_limit=enforced_time_limit)
            else: 
                print('OPT - solving')
                solution, out, obj,_ = self.solve(x, w, enforced_time_limit=enforced_time_limit)
            
            # count inference time and number of calls
            self.inference_time += perf_counter() - inf_start
            self.inference_calls += 1

            # if caching is enabled and a new solution is found, then add it to the cache
            if use_cache and solution.y is not None:
                #if len(solution.y) != 0:
                    print('storing solution in cache')
                    self.caching_strategy.cache.add(w.copy(), solution, obj)
     
            self.inference_index += 1
            return solution, _ , 'No_Cache '  # the boolean value can be used to count the number of reuses from the cache

        else:  # if cache is enabled and a solution is found in the cache, then use it for update
            self.inference_index += 1
            return relevant_solution, False, 'Cache '  # the boolean is to indicate that the solution is found in the cache