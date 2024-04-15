from .base_solution import *


class SolutionKP(Solution):
    def __init__(self, x, y, loss=None, solver_type=None, utils=None):
        super().__init__(x, y, loss, solver_type, utils)

    def get_phi(self):
        return self.x["profits"][list(self.y)].sum(axis=0)

    def get_loss(self):
        return len(set(self.x.get_preferred_solution().y).difference(set(self.y)))

    def get_max_loss(self):
        return self.x["size"]

    def get_score(self):
        return 1 - self.get_loss() / self.get_max_loss()

    def get_cost_vector(self, w):
        vector = np.zeros(self.x["profits"].size).astype(float)
        index = 0
        for i in range(self.x["profits"].shape[1]):
            for j in range(self.x["profits"].shape[0]):
                vector[index] = w[i] * self.x["profits"][j][i]
                index += 1
        return vector

    def _convert_solution(self):
        transformed_solution = np.zeros(self.x["profits"].size).astype(int)
        for i in range(self.x["profits"].shape[1]):
            for j in range(self.x["profits"].shape[0]):
                transformed_solution[i * self.x["profits"].shape[0] + j] = 1 if j in self.y else 0
        return transformed_solution

    def compute_objective_w_phi(self, w):
        w = w.astype(int)
        return np.dot(self.get_phi(), w)
