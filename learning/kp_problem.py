import numpy as np
import pandas as pd
import os
from .base_problem import *
from .kp_solution import SolutionKP

# add the root directory to the python path so that we can import modules from there
import sys
import pathlib
root_path = pathlib.Path(__file__).resolve().parent.parent.absolute()
sys.path.insert(0, str(root_path))

from solvers.solver_kp import read_knapsack


class ProblemKP(Problem):

    def __init__(self, container: dict):
        super().__init__(container)

    def get_preferred_solution(self):
        return SolutionKP(self, self.container["real_sol"])

    def build_solution(self, y):
        return SolutionKP(self, y)

    @staticmethod
    def get_vals_from_dataset(df, instance):
        file_path = df.iloc[instance]["instance"]
        real_set = [int(x) for x in df.iloc[instance]["selected_items"].split("-")]
        real_weight = int(df.iloc[instance]["sol_weight"])
        real_obj = int(df.iloc[instance]["sol_profit"])
        return file_path, real_set, real_obj, real_weight

    @staticmethod
    def read_data(dataset_path):
        """
        :param dataset_path: the path of the dataset
        :return:
        X: list of dictionaries, each dictionary contains the following keys:
            "file_path": PCTSP instance path
            "size": size of the PCTSP instance
            "prizes": prizes vector
            "min_prize": min_prize
            "real_raw_distances": four raw distance matrices
            "real_raw_penalties": raw penalties vector
            "real_weights": weights (distances and penalty)
            "real_obj": real objective value
            "real_out": real out vector
            "dataset_time_limit": time limit set for the real solution calculation
            "n_data_jobs": number of threads set for the real solution calculation
            "n_params": number of parameters to learn (number of distances + penalty)
            "index": index of the instance in the dataset
        Y: list of solutions, each solution is a list of nodes

        """
        print(dataset_path)
        time_str = os.path.basename(dataset_path).split("_")[1][1:]
        time_limit = int(time_str) if time_str.lower() != "none" else None
        size = int(os.path.basename(dataset_path).split("_")[2].split(".")[0][1:])
        X, Y = [], []
        df = pd.read_csv(dataset_path)
        for instance in range(len(df)):
            file_path, real_set, real_obj, real_weight = ProblemKP.get_vals_from_dataset(df, instance)
            profits, weights, capacity = read_knapsack(file_path[3:])
            problem_desc = {
                "file_path": file_path,
                "size": size,
                "profits": profits,
                "weights": weights,
                "capacity": capacity,
                "dataset_time_limit": time_limit,
                "n_params": profits.shape[1],
                "index": instance,
                "real_sol": real_set,
                "real_weights": np.array([0, 1, 2, 17])}
            x = ProblemKP(problem_desc)
            X.append(x)
            Y.append(x.build_solution(np.array(real_set)))
        return X, Y
