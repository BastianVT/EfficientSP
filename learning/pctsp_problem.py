import numpy as np
import pandas as pd
from .base_problem import *
from .pctsp_solution import SolutionPCTSP

# add the root directory to the python path so that we can import modules from there
import sys
import pathlib
root_path = pathlib.Path(__file__).resolve().parent.parent.absolute()
sys.path.insert(0, str(root_path))

from solvers.solver_pctsp import read_data_datepenalty


class ProblemPCTSP(Problem):

    def __init__(self, container: dict):
        super().__init__(container)

    def get_preferred_solution(self):
        return SolutionPCTSP(self, self.container["real_sol"])

    def build_solution(self, y):
        return SolutionPCTSP(self, y)

    @staticmethod
    def get_vals_from_dataset(df, instance):
        file_path = df.iloc[instance]["instance"]
        real_route = [int(x) for x in df.iloc[instance]["in_seq"].split("-")]
        real_out = [] if pd.isna(df.iloc[instance]["out_seq"]) else [int(df.iloc[instance]["out_seq"])] if "-" not in df.iloc[instance]["out_seq"] else [int(x) for x in df.iloc[instance]["out_seq"].split("-")]
        real_obj = df.iloc[instance]["sol_cost"] + df.iloc[instance]["sol_penalty"]
        return file_path, real_route, real_out, real_obj

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
        time_limit = int(dataset_path.split("_")[3][1:])
        size = int(dataset_path.split("_")[4].split(".")[0][1:])
        # n_data_jobs = int(dataset_path.split("/")[1].split("_")[-1]) if "thr" in dataset_path.split("/")[1] else 0
        X, Y = [], []
        import os
        print("cwd", os.getcwd())
        df = pd.read_csv(dataset_path)
        for instance in range(len(df)):
            file_path, real_route, real_out, real_obj = ProblemPCTSP.get_vals_from_dataset(df, instance)
            prizes, real_mult_penalties, real_mult_distances, min_prize, real_p_weight, real_m_weights, real_raw_distances, real_raw_penalties = read_data_datepenalty(file_path[3:], return_weights=True, return_distances=True, return_penalties=True)
            problem_desc = {
                "file_path": file_path,
                "size": size,
                "prizes": prizes,
                "min_prize": min_prize,
                "raw_distances": real_raw_distances,
                "raw_penalties": real_raw_penalties,
                "dataset_time_limit": time_limit,
                "n_params": len(real_raw_distances) + 1,
                "index": instance,
                "real_sol": real_route,
                "real_weights": np.array(real_m_weights + [real_p_weight])}
            x = ProblemPCTSP(problem_desc)
            X.append(x)
            Y.append(x.build_solution(np.array(real_route)))
        return X, Y
