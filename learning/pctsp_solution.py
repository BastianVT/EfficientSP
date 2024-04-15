from .base_solution import *
from .utils import LossFunction, node_hamming_distance, arc_hamming_distance, flatten_adjacency_matrix

from itertools import chain


class SolutionPCTSP(Solution):
    def __init__(self, x, y, loss=None, solver_type=None, utils=None):
        super().__init__(x, y, loss, solver_type, utils)

    def get_phi(self):
        n_dist_weights = len(self.x["raw_distances"])
        n_pen_weights = 1
        n_weights = n_dist_weights + n_pen_weights
        phi = np.zeros(n_weights).astype(float)
        for arc in zip(self.y[:-1], self.y[1:]):
            for i in range(n_dist_weights):
                phi[i] -= self.x["raw_distances"][i][arc]
        for stop in range(len(self.x["prizes"])):
            if stop not in self.y:
                phi[4] -= self.x["raw_penalties"][stop]
        return phi

    def get_loss(self):
        if self.loss == LossFunction.NODE_HAMMING:
            return node_hamming_distance(self.x.get_preferred_solution().y, self.y)
        else:  # LossFunction.ARC_HAMMING
            return arc_hamming_distance(self.x.get_preferred_solution().y, self.y)

    def get_max_loss(self):
        if self.loss == LossFunction.NODE_HAMMING:
            return len(self.x["prizes"])
        else:
            return (len(self.x["prizes"]) - 1) + (len(self.y) - 1)

    # def _get_loss(self, w, time_limit, solver, n_jobs, ls_bin_dir):
    #     if solver == Solver.CP:
    #         pred_y, _, _, _ = self.utils.solve_cp(self.x, w, time_limit, n_jobs)
    #     else:
    #         pred_y, _, _, _ = self.utils.solve_ls(self.x, w, ls_bin_dir, time_limit, n_jobs)
    #     return node_hamming_distance(self.y, pred_y) if self.loss == LossFunction.NODE_HAMMING else arc_hamming_distance(self.y, pred_y)
    #
    # def _get_max_loss(self):
    #     return len(self.x["prizes"]) if self.loss == LossFunction.NODE_HAMMING else (len(self.x["prizes"]) - 1) + (len(self.y) - 1)

    # def get_score(self, X, Y, w, time_limit, solver, n_jobs, ls_bin_dir, loss_fun, print_errors=False):
    #     errors = np.zeros((len(X), 2)).astype(float)
    #     for i, (x, y) in enumerate(zip(X, Y)):
    #         loss = self._get_loss(x, y, w, time_limit, solver, n_jobs, ls_bin_dir, loss_fun)
    #         max_loss = self._get_max_loss(x, y, loss_fun)
    #         errors[i][0] = loss
    #         errors[i][1] = max_loss
    #         if print_errors:
    #             print("Instance", i+1, "error ", [float(format(x, '.2f')) for x in errors[i]], flush=True)
    #     return 1 - np.sum(errors[:, 0]) / np.sum(errors[:, 1])

    # def get_score(self, w, time_limit, solver, n_jobs, ls_bin_dir, print_errors=False):
    #     errors = [0, 0]
    #     loss = self._get_loss(w, time_limit, solver, n_jobs, ls_bin_dir)
    #     max_loss = self._get_max_loss()
    #     errors[0] = loss
    #     errors[1] = max_loss
    #     return 1 - errors[0] / errors[1]

    def get_score(self):
        return 1 - self.get_loss() / self.get_max_loss()

    def get_cost_vector(self, w):
        vector = np.zeros(self.x["raw_distances"].size + len(self.x["prizes"])).astype(float)
        index = 0
        for i in range(len(self.x["raw_distances"])):
            for j in range(len(self.x["prizes"])):
                for k in range(len(self.x["prizes"])):
                    vector[index] = w[i] * self.x["raw_distances"][i][j][k]
                    index += 1
        for i in range(len(self.x["prizes"])):
            vector[index] = w[-1] * self.x["raw_penalties"][i]
            index += 1
        return vector

    def _convert_solution(self):
        mat = list(chain.from_iterable([flatten_adjacency_matrix(self.y, len(self.x["prizes"])) for _ in range(len(self.x["raw_distances"]))]))
        unselected = [1 if i not in self.y else 0 for i in range(len(self.x["prizes"]))]
        return np.concatenate((mat, unselected))

    def compute_objective_w_phi(self, w):
        return np.dot(-self.get_phi(), w)
    
   
