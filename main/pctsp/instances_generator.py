from math import fabs, log10
import numpy as np
from enum import Enum
import pandas as pd
import random
from pathlib import Path
import inspect
import networkx as nx


class Distance(Enum):
    Distance1 = 1
    Distance2 = 2
    Distance3 = 3
    Distance4 = 4


class Direction(Enum):
    UP = 1
    DOWN = 2
    SAME = 3


class Material:

    def __init__(self, width, thickness):
        self.width = width
        self.thickness = thickness

    def __eq__(self, other):
        assert isinstance(other, Material)
        return self.width == other.width and self.thickness == other.thickness

    def get_distance_width1(self, material):
        return fabs(self.width - material.width), Direction.UP if self.width < material.width else Direction.DOWN if self.width > material.width else Direction.SAME

    def get_distance_width2(self, material):
        return round(self.width / material.width, 2), Direction.UP if self.width < material.width else Direction.DOWN if self.width > material.width else Direction.SAME

    def get_distance_width4(self, material):
        return round(log10((self.width / material.width) + 1), 2), Direction.UP if self.width < material.width else Direction.DOWN if self.width > material.width else Direction.SAME

    def get_distance_width3(self, material):
        return log10(fabs(self.width - material.width) + 1), Direction.UP if self.width < material.width else Direction.DOWN if self.width > material.width else Direction.SAME

    def get_distance_thickness1(self, material):
        return fabs(self.thickness - material.thickness), Direction.UP if self.thickness < material.thickness else Direction.DOWN if self.thickness > material.thickness else Direction.SAME

    def get_distance_thickness2(self, material):
        return round(self.thickness / material.thickness, 2), Direction.UP if self.thickness < material.thickness else Direction.DOWN if self.thickness > material.thickness else Direction.SAME

    def get_distance_thickness4(self, material):
        return round(log10((self.thickness / material.thickness) + 1), 2), Direction.UP if self.thickness < material.thickness else Direction.DOWN if self.thickness > material.thickness else Direction.SAME

    def get_distance_thickness3(self, material):
        return log10(fabs(self.thickness - material.thickness) + 1), Direction.UP if self.thickness < material.thickness else Direction.DOWN if self.thickness > material.thickness else Direction.SAME


# convert numpy array to string
def np2str(array, delimiter="\t"):
    return "\n".join(delimiter.join("%d" % val for val in line) for line in array)


# add transition cost to transition matrix based on the transition direction
def set_distance(direction_, matrix_up, matrix_down, i, j, val):
    if direction_ == Direction.UP:
        matrix_up[i][j] = val
        matrix_down[i][j] = 0
    elif direction_ == Direction.DOWN:
        matrix_up[i][j] = 0
        matrix_down[i][j] = val
    else:
        matrix_up[i][j] = matrix_down[i][j] = 0


def generate_penalties_prizes(distance_matrix, p=-1, penalty_range=(-1, -1), prize_range=(-1, -1)):
    n = distance_matrix.shape[0]
    penalties = np.zeros(n)
    prizes = np.zeros(n)
    max_distance = np.max(distance_matrix)
    min_distance = np.min(distance_matrix)
    distance_range = max_distance - min_distance
    
    # the values 0.05 and 0.3 are chosen by hand to define the bounds of the penalties based on the maximum distance
    penalty_range = (0.05 * max_distance, 0.3 * max_distance) if penalty_range == (-1, -1) else penalty_range
    prize_range = (50, 2000) if prize_range == (-1, -1) else prize_range
    p = n - 1 if p == -1 else p

    # Compute the shortest paths
    G = nx.from_numpy_array(distance_matrix)
    shortest_path_distance = nx.floyd_warshall_numpy(G)
    # shortest_path_distance = distance_matrix  # if distance_matrix is symmetric and respects the triangular inequality
    for i in range(n):
        # find the p closest cities
        closest_cities = np.argsort(shortest_path_distance[i])[1:p+1]
        # calculate the average distance to the closest cities
        avg_distance = np.mean(shortest_path_distance[i][closest_cities])
        # assign a penalty based on the average distance
        penalties[i] = penalty_range[0] + (penalty_range[1] - penalty_range[0]) * (avg_distance - min_distance) / distance_range
        # assign a prize based on the average distance
        prizes[i] = prize_range[0] + (prize_range[1] - prize_range[0]) * (max_distance - avg_distance) / distance_range
    return penalties, prizes


if __name__ == "__main__":

    # read the Excel file
    df = pd.read_excel("../../raw/LSchedulerTemplate.xlsx", sheet_name="Main", index_col="Combination Name")
    df = df.tail(-6)
    df = df.drop(['Types', 'Identifier', 'Due Date'], axis=1)

    # get the output width values and compute the probability distribution
    out_width = df['Output Width'].to_list()
    out_width_dist = [(val, out_width.count(val) / len(out_width)) for val in set(out_width)]

    # get the output thickness values and compute the probability distribution
    out_thick = df['Output Thickness'].to_list()
    out_thick_dist = [(val, out_thick.count(val) / len(out_thick)) for val in set(out_thick)]

    # define parameters
    n_datasets = 200

    # get with distance methods
    distance_methods = methods = inspect.getmembers(Material, predicate=inspect.isfunction)
    width_dist_methods = [method[1] for method in sorted(distance_methods, key=lambda x: x[0]) if "width" in method[0]]
    thick_dist_methods = [method[1] for method in sorted(distance_methods, key=lambda x: x[0]) if "thickness" in method[0]]
    width_dist_methods += [random.choice(width_dist_methods)]
    thick_dist_methods += [random.choice(thick_dist_methods)]
    names = ["abs", "div", "logabs", "logdiv", "random"]

    # distance metrics
    for ind, (width_meth, thick_meth) in enumerate(zip(width_dist_methods, thick_dist_methods)):

        # for n_materials in [10, 20, 50, 75, 100, 125, 150, 200, 300]:
        for n_materials in [150]:
            matrix_weights = list(map(lambda x: int(round(x, 0)), (np.random.dirichlet(np.ones(4), size=1) * 100).tolist()[0]))
            while np.any(np.array(matrix_weights) < 5) or sum(matrix_weights) != 100:
                matrix_weights = list(map(lambda x: int(round(x, 0)), (np.random.dirichlet(np.ones(4), size=1) * 100).tolist()[0]))
            penalty_weight = np.random.randint(1, 100 + 1)
            
            data_ind = 1
            while data_ind <= n_datasets:

                # the material characteristic values are generated below. The distributions learnt from the Excel are used
                widths = np.random.choice(list(zip(*out_width_dist))[0], size=n_materials, replace=True, p=list(zip(*out_width_dist))[1])
                thicks = np.random.choice(list(zip(*out_thick_dist))[0], size=n_materials, replace=True, p=list(zip(*out_thick_dist))[1])

                # create material objects
                if names[ind] == "abs":
                    materials = [Material(widths[i], int(round(thicks[i] * 100, 2))) for i in range(n_materials)]
                else:
                    materials = [Material(widths[i], thicks[i]) for i in range(n_materials)]

                # initialize matrices that will store the transition costs
                width_up_matrix, width_down_matrix = np.zeros((n_materials, n_materials)), np.zeros((n_materials, n_materials))
                thickness_up_matrix, thickness_down_matrix = np.zeros((n_materials, n_materials)), np.zeros((n_materials, n_materials))

                # compute the transition costs between each pair of materials given the distance metric
                for i in range(len(materials)):
                    for j in range(len(materials)):
                        # compute the transition cost given material's width
                        distance, direction = width_meth(materials[i], materials[j])
                        if names[ind] != "abs":
                            distance = int(round(distance, 2) * 100)  # transform values to avoid float issue with ORTools
                        set_distance(direction, width_up_matrix, width_down_matrix, i, j, distance)

                        # compute the transition cost given material's thickness
                        distance, direction = thick_meth(materials[i], materials[j])
                        if names[ind] != "abs":
                            distance = int(round(distance, 2) * 100)
                        set_distance(direction, thickness_up_matrix, thickness_down_matrix, i, j, distance)
                noise = 0.01
                matrix_weights_noise = [matrix_weights[i]+random.uniform(-noise,noise) for i in range(len(matrix_weights))]
                
                distance_matrix = (matrix_weights_noise[0] * width_up_matrix) + (matrix_weights_noise[1] * width_down_matrix) + (matrix_weights_noise[2] * thickness_up_matrix) + (matrix_weights_noise[3] * thickness_down_matrix)
                penalties, prizes = generate_penalties_prizes(distance_matrix)
                penalties = penalties.astype(int)
                prizes = prizes.astype(int)
                
                print("../instances/pctsp/size_n{}_{}_".format(str(noise).split('.')[1],n_materials)+str(data_ind))
                # write the output file
                p = Path("C:/Users/basti/OneDrive - KU Leuven/Documents/GitHub/PreferenceDrivenOptimization/instances/pctsp/noisy/size_n{}_{}".format(str(noise).split('.')[1],n_materials))
                p.mkdir(parents=True, exist_ok=True)
                with (p / "data_{}.txt".format(data_ind)).open("w") as file:
                    penalties = np.rint(penalties.astype(float) / penalty_weight).astype(int)
                    file.write(" ".join([str(x) for x in prizes.tolist()]) + "\n\n")
                    file.write(str(penalty_weight) + "\n\n")
                    file.write(" ".join([str(round(x, 2)) for x in penalties.tolist()]) + "\n\n")
                    matrices = [width_up_matrix, width_down_matrix, thickness_up_matrix, thickness_down_matrix]
                    file.write(" ".join([str(x) for x in matrix_weights_noise]) + "\n\n")
                    min_prize = round(np.sum(prizes[:len(prizes) // 4 + 1]),0)
                    file.write(str(min_prize) + "\n\n")
                    file.write("\n\n".join([np2str(matrix) for matrix in matrices]))
                    file.flush()

                data_ind += 1

# import os
# print("cwd:", os.getcwd())
