from .base_inference import *
from .kp_utils import KPUtils
from .kp_solution import SolutionKP


class InferenceKP(Inference):

    def __int__(self, problem_name, solver, loss_fun, n_jobs, time_limit, caching_strategy: CachingStrategy, utils: KPUtils):
        super().__init__(problem_name, solver, loss_fun, n_jobs, time_limit, caching_strategy, utils)

    def solve(self, x, w, init_obj=None, slack=None, enforced_time_limit=None) -> (SolutionKP, list[int], float):
        solution, out, obj, _ = super().solve(x, w, init_obj, slack=slack)
        return SolutionKP(x, solution.y, self.loss_fun, self.solver, self.utils), out, obj, _
