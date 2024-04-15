from .base_inference import *
from .pctsp_utils import PCTSPUtils
from .pctsp_solution import SolutionPCTSP


class InferencePCTSP(Inference):

    def __init__(self, problem_name, solver, loss_fun, n_jobs, time_limit, caching_strategy: CachingStrategy, utils: PCTSPUtils):
        super().__init__(problem_name, solver, loss_fun, n_jobs, time_limit, caching_strategy, utils)

    def solve(self, x, w, init_obj=None, slack=None, enforced_time_limit=None) -> (SolutionPCTSP, list[int], float):
        solution, out, obj, _ = super().solve(x, w, init_obj, slack=slack)
        return SolutionPCTSP(x, solution.y, self.loss_fun, self.solver, self.utils), out, obj, _
