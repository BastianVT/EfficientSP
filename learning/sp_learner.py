import signal

import numpy as np
from scipy.spatial.distance import cosine

import sys
import pathlib
root_path = pathlib.Path(__file__).resolve().parent.parent.absolute()
sys.path.insert(0, str(root_path))
from config import is_verbose

from .base_learner import *
from .utils import compute_opposite_vector
from .base_inference import Inference
from .base_problem import Problem
from .base_solution import Solution
from .base_cache import Cache

import time


class LearnerSP(Learner):

    def __init__(self, result_dir, inference_model: Inference, use_cache, save_results, n_epochs, learning_rate, max_runtime=300, max_updates=None, classic=False, real_weights=None, prediction_time_limit=None, preload_cache=False,problem_type =False):
        additional_args = {
            "n_epochs": n_epochs,
            "learning_rate": learning_rate,
            "max_updates": max_updates,
            "classic": classic,
            "real_weights": real_weights,
            "preload_cache": preload_cache,
        }
        super().__init__(result_dir, inference_model, max_runtime, prediction_time_limit, use_cache, save_results, **additional_args)
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

    def fit(self, X: list[Problem], Y: list[Solution]):
        start_time = time.perf_counter()
        
        #computing initial weights
        if self.kwargs["real_weights"] is not None:
            initial_w = compute_opposite_vector(self.kwargs["real_weights"], magnitude=100)
        else:
            initial_w = np.ones(X[0]["n_params"])
        
        
        #initial_w = np.ones(X[0]["n_params"])
        if is_verbose():
            print("Init weights:", initial_w, flush=True)
        
        self.w = initial_w.copy()

        if self.use_cache:
            self.inference_model.caching_strategy.cache.clean() 
          #  if self.kwargs["preload_cache"]:               
           #     for ind, (x, y) in enumerate(zip(X, Y)):
            #          self.inference_model.caching_strategy.cache.add(self.w,x.get_preferred_solution(),x.get_preferred_solution().compute_objective_w_phi(self.w))
        
        stop_learning = False
        n_iterations, n_updates, n_reuses = 0, 0, 0
        cosine_val = self.get_cosine()
        w_list, cosines, update_times = [self.w.copy()], [cosine_val], [0]
        iter_updated = [0]
        iter_updateTrue = [0]
        iterator = range(self.kwargs["n_epochs"]) if self.kwargs["n_epochs"] is not None else range(10000)
        
        for epoch in iterator:
            if stop_learning:
                break
            epoch_time = time.perf_counter()
            if is_verbose():
                print("EPOCH", epoch, flush=True)
            for ind, (x, y) in enumerate(zip(X, Y)):                
                assert isinstance(x, Problem)
                if self.ctrl_c_stop or \
                        (self.max_runtime is not None and time.perf_counter() - start_time > self.max_runtime) or \
                        (self.kwargs["max_updates"] is not None and n_updates >= self.kwargs["max_updates"]):
                        stop_learning = True
                        break
        
                #recheck
                '''
                if (self.inference_model.caching_strategy is not None):
                    if (self.inference_model.caching_strategy.max_inference is not None):
                        if self.inference_model.inference_calls >= self.inference_model.caching_strategy.max_inference != -1:
                            stop_learning = True
                            break
                '''
                if is_verbose():
                    print("\nepoch:", epoch+1, "instance:", ind+1, "\nw:", self.w)
                    print("cosine dist:", self.get_cosine(), flush=True)
                    print(x["file_path"], flush=True)

                if self.inference_model.time_limit is not None:
                    inf_max_time = self.inference_model.time_limit
                else:  # no time limit set for the inference
                    if self.max_runtime is None:  # run until the required solution is found
                        inf_max_time = None
                    else:  # use the remaining time
                        inf_max_time = self.max_runtime - (time.perf_counter() - start_time)
                
                if is_verbose():
                        print("classic: real sol score:", x.get_preferred_solution().compute_objective_w_phi(self.w))
                learning_solution, _ , reuse = self.inference_model.inference(x, self.w, self.use_cache, self.kwargs["classic"], enforced_time_limit=inf_max_time)
                
                assert isinstance(learning_solution, Solution)
                if reuse == 'Cache ':
                    n_reuses += 1
                n_iterations += 1

                if not np.array_equal(learning_solution.y, y) and learning_solution.y is not None:
                    learning_phi = learning_solution.get_phi()
                    real_phi = x.get_preferred_solution().get_phi()
                    self.w += self.kwargs["learning_rate"] * (real_phi - learning_phi)
                    n_updates += 1
                    iter_updateTrue.append(str(reuse)+'update')
                else:
                    iter_updateTrue.append(str(reuse)+'no_update')
            
                if self.inference_model.problem_name == 'KP':
                    try:
                        iter_updated.append(int(x["file_path"].split('_')[5]))
                    except ValueError:
                        iter_updated.append(int(x["file_path"].split('_')[6]))
                else:
                    iter_updated.append(int(x["file_path"].split('data_')[1].split('.')[0]))
                cosines.append(self.get_cosine())
                w_list.append(self.w.copy())
                update_times.append(time.perf_counter() - start_time)

            if is_verbose():
                print("epoch time:", time.perf_counter() - epoch_time, flush=True)
                
        
        if self.inference_model.problem_name == 'KP':
            try:
                iter_updated.append(int(x["file_path"].split('_')[5]))
            except ValueError:
                iter_updated.append(int(x["file_path"].split('_')[6]))
        else:
            iter_updated.append(int(x["file_path"].split('data_')[1].split('.')[0]))
        cosines.append(self.get_cosine())
        
        w_list.append(self.w.copy())
        update_times.append(time.perf_counter() - start_time)
        
        self.runtime = round(time.perf_counter() - start_time, 2)
                
        if is_verbose():
            print("\nThe inference algorithm is: ", self.inference_model.solver.name)
            print("The number of epochs is: ", epoch)
            print("The number of inferences is: ", self.inference_model.inference_calls)
            print("The total inference time is: %.2f" % self.inference_model.inference_time)
            print("Total training time is: %.2f" % self.runtime)
            print("The learning rate is: ", self.kwargs["learning_rate"])
            print("Predicted weights: ", self.w)
            print("Cosine distance with real weights: ", self.get_cosine())
      
        if self.save_results:
            if self.use_cache:
                cache_name = 'Cache'
            else:
                cache_name = 'NoCache'
            if self.kwargs["classic"]:
                name_model = 'SOP'
            else:
                name_model = 'SAT'
            if self.inference_model.problem_name == 'PCTSP':
                with open("{}/H3_{}_{}_{}_t{}_{}_{}".format(self.result_dir,self.inference_model.problem_name,  self.inference_model.solver.name, name_model, inf_max_time,x['file_path'].split('/')[-2:][0],cache_name), "w") as file:
                    file.write("update,update_time,weights,cosine,iter_updated,iter_updateTrue\n")
                    for ind, (w, t, cosine_val,upd_val,updTrue_val) in enumerate(zip(w_list, update_times, cosines,iter_updated,iter_updateTrue)):
                        file.write("{},{},{},{},{},{}\n".format(ind, t, "_".join(map(str, w)), cosine_val, upd_val,updTrue_val))
                   
            else:
                with open("{}/H3_{}_{}_{}_t{}_{}_{}".format(self.result_dir,self.inference_model.problem_name, name_model,  self.inference_model.solver.name, inf_max_time,'_'.join(x['file_path'].split('/')[-1:][0].split('_')[:-1]),cache_name), "w") as file:
                    file.write("update,update_time,weights,cosine,iter_updated,iter_updateTrue\n")
                    for ind, (w, t, cosine_val,upd_val,updTrue_val) in enumerate(zip(w_list, update_times, cosines,iter_updated,iter_updateTrue)):
                        file.write("{},{},{},{},{},{}\n".format(ind, t, "_".join(map(str, w.tolist())), cosine_val, upd_val,updTrue_val))
                    
                
                
    def predict(self, X: list[Problem]):
        Y_pred = []
        for x in X:
            assert isinstance(x, Problem)
            y, out, obj, _ = self.inference_model.utils.solve_opti(x, self.w , self.pred_inf_time_limit, self.inference_model.n_jobs, prediction=True)
            Y_pred.append(x.build_solution(y))
            if is_verbose():
                print("Predicted sol of instance", len(Y_pred), "out of", len(X), ":", y, flush=True)
        print(x["file_path"])
        print('classic:',self.kwargs["classic"])
        print('usecache:',self.use_cache)
        return Y_pred

    def predict_truew(self, X: list[Problem],w):
        Y_pred = []
        for x in X:
            assert isinstance(x, Problem)
            y, out, obj, _ = self.inference_model.utils.solve_opti(x, w, self.pred_inf_time_limit, self.inference_model.n_jobs, prediction=True)
            Y_pred.append(x.build_solution(y))
            if is_verbose():
                print("Predicted sol of instance", len(Y_pred), "out of", len(X), ":", y, flush=True)
        return Y_pred


    def score(self, X:list[Problem], Y: list[Solution], Y_pred: list[Solution]):
        scores = []
        inf_max_time = self.inference_model.time_limit
        for y_pred in Y_pred:
            if y_pred.y is None:
                scores.append(0)
            else:
                scores.append(y_pred.get_score())
            if is_verbose():
                print("Instance", len(scores), "out of", len(Y_pred), "score", scores[-1], flush=True)
        filepath = X[0]["file_path"]
        if self.use_cache:
            cache_name = 'Cache'
        else:
            cache_name = 'NoCache'
        if self.kwargs["classic"]:
            name_model = 'SOP'
        else:
            name_model = 'SAT'
        if self.inference_model.problem_name == 'PCTSP':
            with open("{}/H3_{}_{}_{}_t{}_{}_{}".format(self.result_dir,self.inference_model.problem_name,  self.inference_model.solver.name, name_model, inf_max_time,filepath.split('/')[-2:][0],cache_name), "a") as file:
                file.write(str(np.mean(scores)))
        else:
            with open("{}/H3_{}_{}_{}_t{}_{}_{}".format(self.result_dir,self.inference_model.problem_name, name_model,  self.inference_model.solver.name, inf_max_time,'_'.join(filepath.split('/')[-1:][0].split('_')[:-1]),cache_name), "a") as file:
                file.write(str(np.mean(scores)))
        
        return np.mean(scores)

    def val_score(self, X:list[Problem], Y: list[Solution], Y_pred: list[Solution]):
        df = pd.read_cvs(X)
        df2 = df[['']]

def eval_sol(Y,w):
        val = 0
        for y in Y:
            val = val + y.compute_objective_w_phi(w)
        return val


        # if self.save_results:
        #     os.makedirs(self.result_dir, exist_ok=True)
        #     file_name = "{}/results_{}.csv".format(self.result_dir, datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d_%H%M%S"))
        #     file_exists = os.path.isfile(file_name)
        #     with open("{}/results_{}.csv".format(self.result_dir, datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d_%H%M%S")), "a") as file:
        #         if not file_exists:
        #             file.write("problem,solver,classic,remaining_time,n_epochs,n_inferences,inference_time,training_time,learning_rate,cosine,weights\n")
        #         file.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(self.inference_model.problem_name, self.inference_model.solver.name, self.kwargs["classic"], self.kwargs["remaining_time"], self.kwargs["n_epochs"], self.inference_model.inference_calls, self.inference_model.inference_time, runtime, self.kwargs["learning_rate"], self.get_cosine(), "_".join(map(str, self.w.tolist()))))


