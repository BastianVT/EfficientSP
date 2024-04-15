import numpy as np


def read_knapsack(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        n_items, capacity = list(map(int, lines.pop(0).split()))

        n_parameters = len(lines[0].split()) - 1  # the last column is the weight

        profits = np.zeros((n_items, n_parameters)).astype(int)
        weights = np.zeros(n_items).astype(int)

        for index in range(n_items):
            line = list(map(int, lines[index].split()))
            profits[index] = line[:n_parameters]
            weights[index] = line[n_parameters]

        return profits, weights, capacity


def read_knapsack_hard(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
        lines.pop(0)  # remove the file name
        # print(repr(lines[0]))
        # print(repr(lines[0].split()[1]))
        n_items = int(lines.pop(0).split()[1])  # number of items
        capacity = int(lines.pop(0).split()[1])  # capacity of the knapsack
        objective = int(lines.pop(0).split()[1])  # objective value of the optimal solution
        runtime = float(lines.pop(0).split()[1])  # runtime of the optimal solution
        solution = []

        n_parameters = 1

        profits = np.zeros((n_items, n_parameters)).astype(int)
        weights = np.zeros(n_items).astype(int)

        for index in range(n_items):
            line = list(map(int, lines[index][:-1].split(",")))
            profits[index] = line[1]
            weights[index] = line[2]
            solution.append(line[3])

        print("Objective:", objective)
        print("Solution:", solution)

        return profits, weights, capacity


def read_knapsack_hard_new(filepath):
    with open(filepath) as file:
        lines = file.readlines()
        lines.pop(0)  # remove the file name
        lines.pop(0)  # remove the instance class
        capacity = int(lines.pop(0))  # capacity of the knapsack

        profits = np.zeros((len(lines), 1)).astype(int)
        weights = np.zeros(len(lines)).astype(int)
        for index, line in enumerate(lines):
            line = list(map(int, line.split()))
            weights[index] = line[0]
            profits[index] = line[1]

        return profits, weights, capacity


