import itertools
import numpy as np
import cpmpy as cp

from cpmpy.solvers.solver_interface import ExitStatus

from .utils import decode_subpath, phi_loss_cp, arc_hamming_loss_cp

class PCTSPSubPath:
    """CSP model for PC-TSP using cpmpy.

    Args:
        minprize (float, optional): minimum total prize collected per tour, default None.
        precision (float, optional): precision when converting float to int, default 1e-4.
        time_limit (int, optional): maximum runtime in seconds, default 180.
    """

    def __init__(self, minprize=None, precision=1e-4, time_limit=180):
        self.minprize = minprize
        self.precision = precision
        self.time_limit = time_limit

    def solve(self, prize, penalty, distmatrix, k=1, minprize=None, distances=None, n_threads=0, real_sol=None):
        """Solve the PC-TSP problem."""
        if minprize is None:
            minprize = self.minprize

        n_vertex = len(prize)

        # Create binary variables (0 or 1) using integers in cpmpy
        arcs_in = cp.boolvar(shape=(n_vertex + 1, n_vertex + 1), name="arc")
        stops_outs = np.diag(arcs_in)

        # Create the cpmpy model
        model = cp.Model()
        # model = cp.SolverLookup.get('ortools', model=model)

        # Constraints
        # Subtour elimination constraints unless the dummy node is selected
        for vertices_size in range(2, n_vertex):
            all_vertices = range(n_vertex)
            for selected_vertices in itertools.combinations(all_vertices, vertices_size):
                model += cp.sum(arcs_in[i, j] for i in selected_vertices for j in selected_vertices if i != j) <= vertices_size - 1

        for i in range(n_vertex + 1):
            # Each vertex has exactly one outgoing arc
            model += cp.sum(arcs_in[i, j] for j in range(n_vertex + 1) if j != i) == 1 - stops_outs[i]
            # Each vertex has exactly one incoming arc
            model += cp.sum(arcs_in[j, i] for j in range(n_vertex + 1) if j != i) == 1 - stops_outs[i]

        # Dummy node necessarily selected to break the only selected sub-tour into a sub-path
        model += stops_outs[n_vertex] == 0

        # Minimum cumulated prize
        model += cp.sum((1 - stops_outs[i]) * prize[i] for i in range(n_vertex)) >= minprize  # , "min_prize"

        # Objective function
        cost_fun = (cp.sum(stops_outs[i] * penalty[i] for i in range(n_vertex)) + cp.sum(distmatrix[i, j] * arcs_in[i, j] for i in range(n_vertex) for j in range(n_vertex)))

        # For loss-augmented inference
        if real_sol is not None:
            if distances is not None:
                cost_fun -= phi_loss_cp(arcs_in[:-1, :-1], distances, real_sol)
            else:
                cost_fun -= arc_hamming_loss_cp(arcs_in[:-1, :-1], real_sol)

        model.minimize(cost_fun)

        # Solve the problem
        model.solve(time_limit=self.time_limit)
        # model.solve(time_limit=self.time_limit, num_workers=n_threads)

        # Retrieve the solution
        store = arcs_in.value().astype(int)
        store_in = store[:-1, :-1]
        in_sol, out_sol = decode_subpath(store)
        in_sol = np.array([1 if i in in_sol else 0 for i in range(n_vertex)])
        out_sol = np.array([1 if i in out_sol[0] else 0 for i in range(n_vertex)])

        results = {
            'store': store,
            'in_solution': in_sol,
            'total_prizes': in_sol @ prize,
            'total_penalties': out_sol @ penalty,
            'total_travel': np.sum(distmatrix * store_in),
            'runtime': model.status().runtime,
            'optimal': model.status().exitstatus == ExitStatus.OPTIMAL
        }

        return results
