from abc import abstractmethod
import numpy as np


class Solution:
    def __init__(self, x, sequence, loss=None, solver_type=None, utils=None):
        self.x = x
        self.y = sequence
        self.loss = loss
        self.solver_type = solver_type
        self.utils = utils

    def __str__(self):
        return str(self.y)

    def extend_solution(self, loss=None, solver_type=None, utils=None):
        self.loss = loss
        self.solver_type = solver_type
        self.utils = utils

    @abstractmethod
    def get_phi(self):
        pass

    @abstractmethod
    def get_loss(self):
        pass

    @abstractmethod
    def get_max_loss(self):
        pass

    @abstractmethod
    def get_score(self):
        pass

    # @abstractmethod
    # def _get_loss(self, w, time_limit, solver, n_jobs, ls_bin_dir):
    #     pass
    #
    # @abstractmethod
    # def _get_max_loss(self):
    #     pass
    #
    # @abstractmethod
    # def get_score(self, w, time_limit, algo, n_jobs, ls_bin_dir, print_errors=False):
    #     pass

    @abstractmethod
    def get_cost_vector(self, w):
        pass

    @abstractmethod
    def _convert_solution(self):
        pass

    def compute_objective_c_x(self, w):
        return np.dot(self.get_cost_vector(w), self._convert_solution())

    @abstractmethod
    def compute_objective_w_phi(self, w):
        pass
