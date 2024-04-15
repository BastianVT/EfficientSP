from abc import abstractmethod

from .base_inference import Inference


class Learner:

    def __init__(self, result_dir, inference_model: Inference, max_runtime, pred_inf_time_limit, use_cache, save_results, **kwargs):
        self.result_dir = result_dir
        self.inference_model = inference_model
        self.max_runtime = max_runtime
        self.pred_inf_time_limit = pred_inf_time_limit
        self.use_cache = use_cache
        self.save_results = save_results
        self.kwargs = kwargs
        self.w = None
        self.runtime = None

    @abstractmethod
    def fit(self, X, Y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def score(self, Y, Y_pred):
        pass
