# add the root directory to the python path so that we can import modules from there
import numpy as np
import sys
import pathlib
root_path = pathlib.Path(__file__).resolve().parent.parent.absolute()
sys.path.insert(0, str(root_path))

# set the current working directory to the current file's directory
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from learning import LearnerIO, Noise, LossFunction, Solver, ProblemType
from learning import Cache, AmortizedCache, EvaluateCompareCaching, ArmotizedCaching, CachingStrategyType
from learning import InferencePCTSP, PCTSPUtils, ProblemPCTSP
from learning import InferenceKP, KPUtils, ProblemKP
from config import set_verbose, is_verbose

from sklearn.model_selection import train_test_split
from datetime import datetime
import argparse


parser = argparse.ArgumentParser('Experiments on synthetic pctsp datasets')
parser.add_argument('-p', '--problem_type', type=lambda input: ProblemType[input], default=ProblemType.PCTSP, choices=list(ProblemType), help="Problem that will be solved (PCTSP or KP)")
parser.add_argument('-i', '--dataset_path', type=str, default="../datasets/pctsp/nthreads_0/cp_training_t30_s10.csv", help="dataset path")
parser.add_argument('-c', '--results_dir', type=str, default="results", help="results directory")
parser.add_argument('-r', '--solver', type=lambda input: Solver[input], default=Solver.OPTI, choices=list(Solver), help="Solver to use (OPTI, LS, etc.) for inference")
parser.add_argument('-z', '--loss', type=lambda input: LossFunction[input], default=LossFunction.ARC_HAMMING, choices=list(LossFunction), help="Loss function used to evaluate the model")
parser.add_argument('-b', '--caching_strategy', type=lambda input: CachingStrategyType[input], default=CachingStrategyType.EVAL_COMP, choices=list(CachingStrategyType), help="Decide caching strategy")
parser.add_argument('-n', '--n_inf_jobs', type=int, default=1, help="number of threads for the inference solver")
parser.add_argument('-e', '--noise', type=lambda input: Noise[input], default=Noise.CUSTOM, choices=list(ProblemType), help="number of epochs for the SP. If None, the SP will run until another stopping criterion is met")
parser.add_argument('-l', '--noise_rate', type=float, default=0.8, help="learning rate for the structured perceptron")
parser.add_argument('-m', '--training_percentage', type=float, default=0.8, help="percentage of the dataset used for training")
parser.add_argument('-v', '--verbose', action='store_true', default=True, help="Print learning info")
parser.add_argument('-q', '--save', type=str, nargs='?', const="results_" + datetime.now().strftime("%Y%m%d"), help="save the results. Add a filename (without extension) if wanted")
parser.add_argument('-f', '--max_inference', type=int, default=60, help="Max number of total inferences for the structured perceptron")
parser.add_argument('-k', '--inf_time_limit', type=int, default=3, help="time limit for the pctsp instances")
parser.add_argument('-ki', '--pred_inf_time_limit', type=int, default=30, help="time limit for the pctsp instances")
parser.add_argument('-ko', '--overall_time_limit', type=int, default=60, help="time limit for the pctsp instances")
parser.add_argument('-fu', '--lambdas', type=list, default=[100,100], help="Max number of total inferences for the structured perceptron")
parser.add_argument('-fo', '--v_count', type=int, default=10, help="Max number of total inferences for the structured perceptron")
parser.add_argument('-fz', '--reset_dataset', action="store_true", default=False, help="Max number of total inferences for the structured perceptron")
parser.add_argument('-xx', '--classic', action='store_true', default=False, help="use classic SP")
parser.add_argument('-u', '--theorem', type=int, default=3, help="theorem used for amortized caching")
parser.add_argument('-j', '--epsilon', type=float, default=0.0, help="error gap parameter for amortized caching")
args = parser.parse_args()

set_verbose(args.verbose)

if args.problem_type == ProblemType.PCTSP:
    X, Y = ProblemPCTSP.read_data(args.dataset_path)
    utils = PCTSPUtils()
else:  # KP
    X, Y = ProblemKP.read_data(args.dataset_path)
    utils = KPUtils()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=args.training_percentage, random_state=42)

if args.caching_strategy == CachingStrategyType.NONE:
    use_cache = False
    cache = None
    caching_strategy = None
else:
    use_cache = True
    if args.caching_strategy == CachingStrategyType.EVAL_COMP:
        cache = Cache(utils)
        caching_strategy = EvaluateCompareCaching(cache, args.max_inference)
    else:  # amortized
        cache = AmortizedCache(utils, args.theorem)
        caching_strategy = ArmotizedCaching(cache, args.max_inference, args.epsilon)

if args.problem_type == ProblemType.PCTSP:
    inference_model = InferencePCTSP(args.problem_type.name, args.solver, args.loss, args.n_inf_jobs, args.inf_time_limit, caching_strategy, utils)
else:  # KP
    inference_model = InferenceKP(args.problem_type.name, args.solver, args.loss, args.n_inf_jobs, args.inf_time_limit, caching_strategy, utils)

learner = LearnerIO(args.results_dir, inference_model, use_cache, args.save, args.noise, args.noise_rate, args.lambdas, args.v_count, args.reset_dataset, args.epsilon, args.overall_time_limit, args.classic, X[0]["real_weights"], args.pred_inf_time_limit)
learner.fit(X_train, Y_train)
Y_pred = learner.predict(X_test)

train_score = learner.score(Y_train, learner.predict(X_train))
test_score = learner.score(Y_test, Y_pred)

print("Train score:", train_score)
print("Test score:", test_score)
print("Weights:", learner.w)



# if self.save_results:
#     os.makedirs(self.result_dir, exist_ok=True)
#     file_name = "{}/results_{}.csv".format(self.result_dir, datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d_%H%M%S"))
#     file_exists = os.path.isfile(file_name)
#     with open("{}/results_{}.csv".format(self.result_dir, datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d_%H%M%S")), "a") as file:
#         if not file_exists:
#             file.write("problem,solver,classic,remaining_time,n_epochs,n_inferences,inference_time,training_time,learning_rate,cosine,weights\n")
#         file.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(self.inference_model.problem_name, self.inference_model.solver.name, self.kwargs["classic"], self.kwargs["remaining_time"], self.kwargs["n_epochs"], self.inference_model.inference_calls, self.inference_model.inference_time, runtime, self.kwargs["learning_rate"], self.get_cosine(), "_".join(map(str, self.w.tolist()))))
