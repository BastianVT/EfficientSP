import shlex
import subprocess

from .base_utils import *
from .utils import MAX_ITER, MAX_NO_IMPROV, MULTI_PRIZES, CLASS_DOUBLEBRIDGE, INTENSITY

# add the root directory to the python path so that we can import modules from there
import sys
import os
import pathlib
root_path = pathlib.Path(__file__).resolve().parent.parent.absolute()
sys.path.insert(0, str(root_path))

from solvers.solver_pctsp.MILP import PCTSPSubPath, decode_subpath
from solvers.solver_pctsp.Greedy import GreedyPCTSP


class PCTSPUtils(ProblemUtils):

    def solve_opti(self, x, weights, time_limit, n_jobs, init_obj=None, real_sol=None, slack=None, prediction=False):
        distances = np.sum(x["raw_distances"] * np.array(weights[:-1])[:, None, None], axis=0)
        penalties = x["raw_penalties"] * weights[-1]
        pctsp = PCTSPSubPath(minprize=x["min_prize"], precision=1, time_limit=time_limit)
        results = pctsp.solve(x["prizes"], penalties, distances, n_threads=n_jobs, init_obj=init_obj, real_sol=real_sol, slack=slack, prediction=prediction)
        if results is None:
            return None, None, None, 1
        route, out = decode_subpath(results['store'])
        out = out.tolist()
        obj = results['total_travel'] + results['total_penalties']
        timeout = 1 if not results['optimal'] else 0
        return np.array(route), out, obj, timeout

    def solve_ls(self, x, weights, time_limit, n_jobs, init_obj=None, real_sol=None, slack=None, prediction=False):
        old_cwd = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        bin_path = "../solvers/solver_pctsp/LS/bin/pctsp_ls"
        cmd = "{} -f {} -m {} -n {} -i {} -l".format(bin_path, x["file_path"][3:], MAX_ITER, MAX_NO_IMPROV, INTENSITY) + (" -w {}".format(" ".join([format(x, 'f') for x in weights])) if weights is not None else "") + (" -t {}".format(time_limit) if time_limit is not None else "") + (" -p" if MULTI_PRIZES else "") + (" -d" if CLASS_DOUBLEBRIDGE else "") + (" -r {}".format(" ".join([str(x) for x in real_sol.tolist()])) if real_sol is not None else "")
        proc_ls = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        out, err = proc_ls.communicate()
        output = out.decode()
        outputs = output.split("\n")
        route = [int(x) for x in outputs[7].split(": ")[1].strip().split(" -> ")]
        out = outputs[8].split(": ")[1].strip()
        out = [] if out == "" else [int(out)] if "," not in out else [int(x) for x in out.split(", ")]
        obj = float(outputs[12].split(": ")[1].strip())
        timeout = 1 if outputs[13].split(": ")[1].strip().lower() == "true" else 0
        os.chdir(old_cwd)
        return np.array(route), out, obj, timeout

    def solve_ls_violation(self, x, weights, time_limit, init_sol):
        old_cwd = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        bin_path = "../solvers/solver_pctsp/LS/bin/pctsp_ls"
        cmd = "{} -f {} -m {} -n {} -i {} -l".format(bin_path, x["file_path"][3:], MAX_ITER, MAX_NO_IMPROV, INTENSITY) + (" -w {}".format(" ".join([format(x, 'f') for x in weights])) if weights is not None else "") + (" -t {}".format(time_limit) if time_limit is not None else "") + (" -p" if MULTI_PRIZES else "") + (" -d" if CLASS_DOUBLEBRIDGE else "") + (" -s {}".format(" ".join([str(x) for x in init_sol])) if init_sol is not None else "") + " -e"
        proc_ls = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        out, err = proc_ls.communicate()
        output = out.decode()
        outputs = output.split("\n")
        route = [int(x) for x in outputs[7].split(": ")[1].strip().split(" -> ")]
        out = outputs[8].split(": ")[1].strip()
        out = [] if out == "" else [int(out)] if "," not in out else [int(x) for x in out.split(", ")]
        obj = float(outputs[12].split(": ")[1].strip())
        timeout = 1 if outputs[13].split(": ")[1].strip().lower() == "true" else 0
        os.chdir(old_cwd)
        return np.array(route), out, obj, timeout

    def solve_greedy(self, x, weights):
        distances = np.sum(x["raw_distances"] * np.array(weights[:-1])[:, None, None], axis=0)
        penalties = x["raw_penalties"] * weights[-1]
        pctsp = GreedyPCTSP(distmatrix=distances, prize=x["prizes"], penalty=penalties, minprize=x["min_prize"])
        return pctsp.solve()

    def is_obj1_better(self, obj1, obj2, percentage=0.0):
        epsilon = percentage * obj2 if obj2 != self.worst_error() else 0
        return 1 if obj1 < obj2 - epsilon else (0 if obj1 == obj2 - epsilon else -1)

    def worst_error(self):
        return float("inf")
