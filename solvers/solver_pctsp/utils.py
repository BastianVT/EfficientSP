import numpy as np
from enum import Enum


class InputFormat(Enum):
    MultiPrizes = 1
    DateAsPenalty = 2
    Downloaded = 3

    def __str__(self):
        return "MultiPrizes" if self.value == 1 else "DateAsPenalty" if self.value == 2 else "Github"


def read_data_downloaded(filepath, return_weights=False, matrix_weights=None, penalty_weight=None):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        prizes = np.fromstring(lines[1].strip(), sep=' ').astype(int)
        penalties = np.fromstring(lines[4].strip(), sep=' ').astype(int)
        distmatrix = np.vstack([np.fromstring(line.strip(), sep=' ') for line in lines[7:]]).astype(int)
    if return_weights:
        return prizes, penalties, distmatrix, int(np.mean(prizes)), 1, [1, 0, 0, 0]
    else:
        return prizes, penalties, distmatrix, int(np.mean(prizes))


def read_data_multiprizes(filepath, return_weights=False, matrix_weights=None, penalty_weight=None):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        cur_line = 0

        prize_weights = [int(x) for x in lines[cur_line].strip().split(" ")]

        for i, pweight in enumerate(prize_weights):
            cur_line = 2 * (i + 1)
            if i == 0:
                prizes = (pweight * np.fromstring(lines[cur_line].strip(), sep=' ')).astype(int)
            else:
                prizes += (pweight * np.fromstring(lines[cur_line].strip(), sep=' ')).astype(int)
        cur_line += 2

        penalty_weight = int(lines[cur_line].strip()) if penalty_weight is None else penalty_weight
        cur_line += 2

        penalties = np.rint(np.fromstring(lines[cur_line], sep=' ') * penalty_weight).astype(int)
        cur_line += 2

        matrix_weights = [float(x) for x in lines[cur_line].strip().split(" ")] if matrix_weights is None else matrix_weights
        cur_line += 2

        min_prize = int(lines[cur_line].strip())
        cur_line += 2

        distmatrix = np.zeros((len(prizes), len(prizes))).astype(int)
        for ind, weight in enumerate(matrix_weights):
            for j in range(len(prizes)):
                distmatrix[j] += (weight * np.fromstring(lines[cur_line], sep=' ')).astype(int)
                cur_line += 1
            cur_line += 1
    if return_weights:
        return prizes, penalties, distmatrix, min_prize, penalty_weight, matrix_weights
    else:
        return prizes, penalties, distmatrix, min_prize


def read_data_datepenalty(filepath, return_weights=False, return_distances=False, return_penalties=False, matrix_weights=None, penalty_weight=None):
    with open(filepath, 'r') as f:
        lines = f.readlines()

        prizes = np.fromstring(lines[0], sep=' ').astype(int)

        penalty_weight = int(lines[2].strip()) if penalty_weight is None else penalty_weight

        penalties = np.rint(np.fromstring(lines[4], sep=' ')).astype(int)

        matrix_weights = [float(x) for x in lines[6].strip().split(" ")] if matrix_weights is None else matrix_weights

        minprize = float(lines[8].strip())

        distances = np.zeros((len(matrix_weights), len(prizes), len(prizes))).astype(int)

        cur_line = 10
        for i in range(len(matrix_weights)):
            for j in range(len(prizes)):
                distances[i][j] += np.fromstring(lines[cur_line], sep=' ').astype(int)
                cur_line += 1
            cur_line += 1

        distmatrix = np.sum(distances * np.array(matrix_weights)[:, None, None], axis=0)

    out = prizes, np.rint(penalties * penalty_weight).astype(int), distmatrix, minprize
    if return_weights:
        out += penalty_weight, matrix_weights
    if return_distances:
        out += distances,
    if return_penalties:
        out += penalties,

    return out


def evaluate(solution_str, delimiter, prizes, penalties, matrix):
    selected = [int(x) for x in solution_str.split(delimiter)]
    unselected = list(set(range(len(prizes))) - set(selected))
    prize = sum([prizes[i] for i in selected])
    penalty = sum([penalties[i] for i in unselected])
    cost = sum([matrix[selected[i], selected[i+1]] for i in range(len(selected)-1)])
    print("-> prize:", prize, "\t -> penalty:", penalty, "\t -> cost:", cost, "\t -> objective:", penalty + cost)
    return prize, penalty, cost, penalty + cost
