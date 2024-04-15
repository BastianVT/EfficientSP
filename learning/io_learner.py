import signal

import numpy as np
from scipy.spatial.distance import cosine

import sys
import pathlib
import os

root_path = pathlib.Path(__file__).resolve().parent.parent.absolute()
sys.path.insert(0, str(root_path))
from config import is_verbose

from .base_learner import *
from .utils import compute_opposite_vector
from .base_inference import Inference
from .base_problem import Problem
from .base_solution import Solution

import time
from enum import Enum

import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum


class Noise(Enum):
    ZERO = 1
    UNIFORM = 2
    CUSTOM = 3


class LearnerIO(Learner):

    def __init__(self, result_dir, inference_model: Inference, use_cache, save_results, noise: Noise, noise_rate=0.8, lambdas=[100,100], v_count=10,
                 reset_dataset=False, epsilon=0, max_runtime=300, classic=False, real_weights=None,
                 prediction_time_limit=None):
        additional_args = {
            "noise": noise,
            "noise_rate": noise_rate,
            "lambdas": lambdas,
            "v_count": v_count,
            "classic": classic,
            "reset_dataset": reset_dataset,
            "epsilon": epsilon,
            "real_weights": real_weights,
            "c0": None,
        }

        super().__init__(result_dir, inference_model, max_runtime, prediction_time_limit, use_cache, save_results,
                         **additional_args)
        self.ctrl_c_stop = False
        signal.signal(signal.SIGINT, self.handler)

    def get_cosine(self):
        if self.kwargs["real_weights"] is not None:
            return cosine(self.w, self.kwargs["real_weights"])
        else:
            return None

    def handler(self, signum, frame):
        if is_verbose():
            print("Stopping learning. Last inference will be completed.", flush=True)
        self.ctrl_c_stop = True

    @staticmethod
    def noise_weight_matrix(Matrix, noise):
        c0 = [0 for i in Matrix]
        for i in range(len(Matrix)):
            c0[i] = Matrix[i] * (1+np.random.uniform(-noise, noise))
        return c0

    def MP_IO_PSI(self, X, Xtong, y, Lambda):

        c0 = self.kwargs["c0"]
        D = len(y)
        L = len(c0)

        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()

            with gp.Model("MP", env=env) as m:

                m.setParam('TimeLimit', 5*60)

                # variables
                c = m.addVars(L, name="c", lb=-GRB.INFINITY)
                cd = m.addVars(D, L, name="cd", lb=-GRB.INFINITY)
                Xi = m.addVars(D, name="Xi")

                z = m.addMVar(L, name="z", lb=-GRB.INFINITY)
                zf = m.addVars(D, L, name="zf", lb=-GRB.INFINITY)

                absdif = m.addVars(L, name="absdif")
                absdiff = m.addVars(D, L, name="absdiff")

                for j in range(0, L):
                    m.addConstr(c0[j]-c[j] == z[j])
                    m.addConstr(z[j] <= absdif[j])
                    m.addConstr(-absdif[j] <= z[j])

                for i in range(0, D):
                    for j in range(0, L):
                        m.addConstr(zf[i, j] == c[j]-cd[i, j])
                        m.addConstr(zf[i, j] <= absdiff[i, j])
                        m.addConstr(-absdiff[i, j] <= zf[i, j])

                m.addConstr(quicksum(c) == quicksum(c0))

                ODMatrix = X[0]["raw_distances"]
                # ODMatrix = X[0][11]

                def out_of_seq(sol: Solution):
                    # print("sol.y", sol.y)
                    # print("sol.x['size']", sol.x["size"])
                    return [i for i in range(sol.x["size"]) if i not in sol.y]

                for d in range(0, D):
                    if len(Xtong[d]) != 0:
                        for minixhat in Xtong[d]:
                            m.addConstr(quicksum(cd[d, ODM]*ODMatrix[ODM][O][De] for O, De in zip(y[d].y, y[d].y[1:]) for ODM in range(0, len(ODMatrix))) + cd[d, 4]*quicksum(X[0]["raw_penalties"][ii] for ii in out_of_seq(y[d])) - quicksum(
                                cd[d, ODM]*ODMatrix[ODM][O][De] for O, De in zip(minixhat.y, minixhat.y[1:]) for ODM in range(0, len(ODMatrix))) - cd[d, 4]*quicksum(X[0]["raw_penalties"][ii] for ii in out_of_seq(minixhat)) - Xi[d] <= 0)

                m.setObjective(absdif.sum()+Lambda[0]/D*absdiff.sum() + Lambda[1]*Xi.sum()/D, GRB.MINIMIZE)
                m.optimize()

                ctong = [c[t].X for t in range(0, L)]
                cdtong = [[cd[i, j].X for j in range(0, L)] for i in range(0, D)]

                return ctong, cdtong, [Xi[i].X for i in range(0, D)]

    def fit(self, X: list[Problem], Y: list[Solution]):
        start_time = time.perf_counter()

        if self.kwargs["real_weights"] is not None:
            initial_w = compute_opposite_vector(self.kwargs["real_weights"], magnitude=100)
        else:
            initial_w = np.ones(X[0]["n_params"])

        if is_verbose():
            print("Init weights:", initial_w, flush=True)

        if self.use_cache:
            self.inference_model.caching_strategy.cache.clean()

        if self.kwargs["noise"] == Noise.ZERO:
            self.kwargs["c0"] = [0 for _ in range(X[0]["n_params"])]
        elif self.kwargs["noise"] == Noise.UNIFORM:
            print(self.inference_model.problem_size)
            self.kwargs["c0"] = [0.01 for _ in range(X[0]["n_params"])]
        elif self.kwargs["noise"] == Noise.CUSTOM:
            self.kwargs["c0"] = LearnerIO.noise_weight_matrix(self.kwargs["real_weights"], self.kwargs["noise_rate"])

        D = len(Y)
        counte = 0
        countv = 0
        Xt = [[] for i in range(0, D)]
        rep = [0 for i in range(0, D)]
        ctong, cdtong = self.kwargs["c0"], [self.kwargs["c0"] for _ in range(D)]
        # ctong, cdtong = X[0][12], [X[0][12] for d in range(D)]
        Xi = [0 for i in range(0, D)]
        EvolLoss = []
        cutcounter = []
        TimeMP = []
        TimeFP = []
        TimeEval = []
        FP_call = 0
        MP_call = 0
        At = []
        AAt = []
        init_c = ctong.copy()
        if is_verbose():
            print("initial weights: ", init_c)

        # It should be double-checked
        structuredloss = 0

        # 2/3/4

        if self.inference_model.time_limit is not None:
            inf_max_time = self.inference_model.time_limit
        else:  # no time limit set for the inference
            if self.max_runtime is None:  # run until the required solution is found
                inf_max_time = None
            else:  # use the remaining time
                inf_max_time = self.max_runtime - (time.perf_counter() - start_time)

        n_iter = 0
        while counte < D and self.max_runtime > time.perf_counter() - start_time:
            if is_verbose():
                print()
                print()
                print("restart checking the new weights against the training instances for the ", n_iter+1, "th time")
            counte = 0
            countv = 0
            count2 = 0
            # 7 for each instance in the dataset
            for d in range(0, D):
                if is_verbose():
                    print()
                # 8: Solve subroutine (Solving the forward problem and delivering another potential route with lower cost for that c)
                timeFP_d = time.time()
                FP_call = FP_call + 1
                if is_verbose():
                    print('training instance: ', d+1, 'of: ', D)
                yhat = self.inference_model.solve(X[d], cdtong[d], enforced_time_limit=inf_max_time)[0]
                if is_verbose():
                    print("data_name: ", os.path.basename(X[d]["file_path"]), "yhat: ", yhat)
                aftertimeFP_d = time.time()
                TimeFP.append((aftertimeFP_d - timeFP_d) / 60)
                loss_gap = Y[d].compute_objective_w_phi(cdtong[d]) - yhat.compute_objective_w_phi(cdtong[d]) - Xi[d]
                if type(structuredloss) != int:
                    loss_gap += structuredloss(Y[d], yhat)
                if is_verbose():
                    print("found yhat: ", yhat.y)
                    print("real y: ", Y[d].y)
                    print("loss_gap: ", loss_gap)
                # 12: If we have found we add it as a plane
                if loss_gap > self.kwargs["epsilon"] + 0.000001:
                    countv = countv + 1
                    rep[d] = rep[d] + 1
                    Xt[d].append(yhat)
                    if all(not np.array_equal(yhat.y, arr) for arr in At):
                        At.append(yhat.y)
                # 9: If we havent found a route with a better objective for this cost, skip
                else:
                    counte = counte + 1
                    count2 = count2 + 1

                if is_verbose():
                    print("countv: ", countv, "over: ", self.kwargs["v_count"], "count2+countv: ", count2+countv, "over: ", D)
                # when we identify Vcount possible cuts
                if countv == self.kwargs["v_count"] or count2 + countv == D:
                    # 5: Solve MP and make ctong and cdtong its optimal solution
                    timeMP_d = time.time()
                    MP_call = MP_call + 1
                    if is_verbose():
                        print("Solving the Master Problem to find new weights")
                        print("old ctong: ", ctong)
                    ctong, cdtong, Xi = self.MP_IO_PSI(X, Xt, Y, self.kwargs["lambdas"])
                    if is_verbose():
                        print("new ctong: ", ctong)
                    # ctong, cdtong, Xi = f_MP(X, Xt, Y, Lambda, structuredloss)
                    aftertimeMP_d = time.time()
                    TimeMP.append((aftertimeMP_d - timeMP_d) / 60)
                    if self.kwargs["reset_dataset"]:
                        print("resetting dataset")
                        break
                    else:
                        count2 = count2 + countv
                        countv = 0

                if time.perf_counter() - start_time > inf_max_time:
                    print("Time limit reached")
                    break

            n_iter += 1
            if is_verbose():
                print("counte: ", counte)

        if time.perf_counter() - start_time > self.max_runtime:
            if is_verbose():
                print("Time Limit reached or exceeded")

        if is_verbose():
            print("final weights: ", ctong)
            print("initial weights: ", init_c)

        self.w = np.array(ctong)

    def predict(self, X: list[Problem]):
        Y_pred = []
        for x in X:
            assert isinstance(x, Problem)
            y, out, obj, _ = self.inference_model.utils.solve_opti(x, self.w, self.pred_inf_time_limit,
                                                                   self.inference_model.n_jobs, prediction=True)
            Y_pred.append(x.build_solution(y))
            if is_verbose():
                print("Predicted sol of instance", len(Y_pred), "out of", len(X), ":", y, flush=True)
        return Y_pred

    def score(self, Y: list[Solution], Y_pred: list[Solution]):
        scores = []
        for y_pred in Y_pred:
            if y_pred.y is None:
                scores.append(0)
            else:
                scores.append(y_pred.get_score())
            if is_verbose():
                print("Instance", len(scores), "out of", len(Y_pred), "score", scores[-1], flush=True)
        return np.mean(scores)

