# add the root directory to the python path so that we can import modules from there
import sys
import pathlib
root_path = pathlib.Path(__file__).resolve().parent.parent.parent.parent.absolute()
sys.path.insert(0, str(root_path))

from config import is_verbose

import numpy as np
import cpmpy as cp
from cpmpy.solvers.solver_interface import ExitStatus
from .utils import phi_loss, arc_hamming_loss

class PCTSPSubPath:
    """CPMpy model for PC-TSP.

    Args:
        minprize (float, optional): minimum total prize collected per tours,  default None.
        time_limit (int, optional): maximum runtime in seconds. default 180.
        precision (float, optional): precision when converting float to int, default 1e-4. 
    """

    def __init__(self, minprize=None, time_limit=180, precision=1e-4) -> None:
        self.minprize = minprize
        self.time_limit = time_limit
        self.precision = precision

    @staticmethod
    def post_subpath_ortools(model: cp.Model, arcs_in: cp.boolvar):
        """This assumes arcs_in is an NxN boolvar

        Args:
            model (cp.Model): CP model.
            arcs_in (boolvar[N,N]): array of arcs present/absent

        Returns:
            cp.Model
        """
        if not isinstance(model, cp.SolverLookup.lookup('ortools')):
            model = cp.SolverLookup.get('ortools', model=model)

        N = arcs_in.shape[0]

        # create arcs (list[tuple(i,j, lit)])
        # providing self arcs (i->i) makes ortools circuit behave as subcircuit
        arcs = [(i, j, model.solver_var(b)) for (i, j), b in np.ndenumerate(arcs_in)]

        # add extra dummy row (arcs from dummy node)
        arcs += [(N, i, model.solver_var(cp.boolvar())) for i in range(N)]
        # add extra dummy column (arcs to dummy node)
        arcs += [(i, N, model.solver_var(cp.boolvar())) for i in range(N+1)]
        # force dummy node to be selected (set dummy self-loop arc to 0) to make it a subpath
        model.ort_model.Add(arcs[-1][2] == 0)
        # # add extra dummy column (arcs to dummy node). The self-loop var is not created, so it cannot be out
        # arcs += [(i, N, model.solver_var(cp.boolvar())) for i in range(N)]

        model.ort_model.AddCircuit(arcs)

        return model, arcs

    def get_model(self, prize):
        N = len(prize)
        arcs_in = cp.boolvar(shape=(N, N), name='arc')

        model, arcs = PCTSPSubPath.post_subpath_ortools(cp.Model(), arcs_in)
        model += cp.any(arcs_in)  # force all variables in?
        return model, arcs_in, arcs

    def solve(self, prize, penalty, distance, minprize=None, n_threads=0, real_sol=None, distances=None):
        """

        Args:
            prize (iterable): vector of prizes.
            distance (iterable): distance matrix.
            penalty (iterable): vector of penalties.
            k (int, optional): Number of tours. Defaults to 1.
            minprize (int, optional): minimum cumulated prize of the tour.
            

        Returns:
            dict:`store` is a list of tours, 
                `total_prizes` is a list of cumulated prizes (int) per tour,
                `total_penalties` is a list of cumulated penalties (int) per tour,
                `total_travel` is a list of total distance (int) per tour,
                `runtime` is a list of runtimes (s)
        """
        if minprize is None:
            minprize = self.minprize

        distmatrix = (distance / self.precision).astype(int)
        prize = (prize / self.precision).astype(int)
        penalty = (penalty / self.precision).astype(int)

        # distmatrix = (distance / self.precision).astype(float)
        # prize = (prize / self.precision).astype(float)
        # penalty = (penalty / self.precision).astype(float)

        # minprize = np.ceil(minprize / self.precision).astype(int)

        # model
        m, arcs_in, arcs = self.get_model(prize)
        # diagonal tells whether node is in
        outs = np.diag(arcs_in)
        m += [np.sum((~outs) * prize) >= minprize]
        # incurs a penality when NOT selecting the stop
        cost_fun = np.sum(outs * penalty) + np.sum(distmatrix * arcs_in)

        if real_sol is not None:
            if distances is not None:
                cost_fun -= phi_loss(arcs_in, distances, real_sol)
            else:
                cost_fun -= arc_hamming_loss(arcs_in, real_sol)
        m.minimize(cost_fun)

        solution_found = m.solve(time_limit=self.time_limit, num_workers=n_threads)

        if is_verbose():
            print("solver status", m.status().exitstatus)

        if solution_found:
            results = {
                'store': arcs_in.value(),
                'in_solution': 1 - outs.value(),
                'total_prizes': np.sum((1 - outs.value()) * prize),
                'total_penalties': np.sum(outs.value() * penalty),
                'total_travel': np.sum(distmatrix * arcs_in.value()),
                'runtime': m.status().runtime,
                'optimal': m.status().exitstatus == ExitStatus.OPTIMAL
            }
            return results

        return None
