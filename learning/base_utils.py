from abc import abstractmethod
import numpy as np
import pulp

from .base_solution import Solution


class ProblemUtils:

    def __init__(self):
        pass

    @abstractmethod
    def solve_opti(self, x, weights, time_limit, n_jobs, init_obj=None, real_sol=None, slack=None, prediction=False) -> Solution:
        pass

    @abstractmethod
    def solve_ls(self, x, weights, time_limit, n_jobs, init_obj=None, real_sol=None, slack=None, prediction=False) -> Solution:
        pass

    @abstractmethod
    def solve_ls_violation(self, x, weights, time_limit, init_sol) -> Solution:
        pass

    @abstractmethod
    def solve_greedy(self, x, weights) -> Solution:
        pass

    @abstractmethod
    def is_obj1_better(self, obj1, obj2, percentage=0.0):
        pass

    @abstractmethod
    def worst_error(self):
        pass

    def match_theorem_1(self, cost_new, cost_old, solution_old, epsilon=0):
        for i in range(len(solution_old)):
            if (2 * solution_old[i] - 1) * (cost_new[i] - cost_old[i]) + epsilon < 0:
                return False
        return True

    def match_theorem_2(self, cost_new, cost_olds):
        # print("theorem 2")
        # using linear programming, check if there exists a solution to the following problem:
        # cost_new is a numpy 1d vector of shape (m,)
        # cost_olds is a numpy 2d  matrix of shape (n, m)
        # find a vector x such that:
        #   cost_new  = x_1 * cost_olds[0] + x_2 * cost_olds[1] + ... + x_n * cost_olds[n]
        #   each x_i >= 0

        n = cost_olds.shape[0]
        m = cost_new.shape[0]

        # Create the LP problem
        problem = pulp.LpProblem("LP_Problem2", pulp.LpMinimize)

        # Create the decision variables
        x = [pulp.LpVariable(f"x_{i}", lowBound=0) for i in range(n)]

        # Set the objective function
        problem += 0

        # Add the constraints
        for j in range(m):
            constraint = pulp.lpSum(x[i] * cost_olds[i][j] for i in range(n)) == cost_new[j]
            problem += constraint

        # Disable solver output
        pulp.LpSolverDefault.msg = 0

        # Solve the problem
        problem.solve()

        # Check if a solution exists
        return problem.status == pulp.LpStatusOptimal

    def match_theorem_3(self, cost_new, cost_olds, common_old_solution, epsilon=0):
        # using linear programming, check if there exists a solution to the following problem:
        # cost_new is a numpy 1d vector of shape (m,)
        # cost_olds is a numpy 2d  matrix of shape (n, m)
        # common_old_solution is a numpy 2d vector of shape (m,)
        # find a vector x such that:
        #   delta_c(x) = cost_new  - x_1 * cost_olds[0] - x_2 * cost_olds[1] - ... - x_n * cost_olds[n]
        #   (2 * common_old_solution[i] - 1) * delta_c(x)[i] + epsilon >= 0
        #   each x_i >= 0

        n = cost_olds.shape[0]
        m = cost_new.shape[0]

        # Create the LP problem
        problem = pulp.LpProblem("LP_Problem3", pulp.LpMinimize)

        # Create the decision variables
        x = [pulp.LpVariable(f"x_{i}", lowBound=0) for i in range(n)]

        # Create the delta_c(x) vector
        delta_c = cost_new - np.dot(cost_olds.T, x)

        # Add the constraints
        for i in range(m):
            constraint = (2 * common_old_solution[i] - 1) * delta_c[i] + epsilon >= 0
            problem += constraint

        # Disable solver output
        pulp.LpSolverDefault.msg = 0

        # Solve the problem
        problem.solve()

        # Check if a solution exists
        return problem.status == pulp.LpStatusOptimal
