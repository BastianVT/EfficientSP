import numpy as np


class GreedyKP:
    def __init__(self, parameters: np.ndarray, profits: np.ndarray, weights: np.ndarray, capacity):
        self.profits = np.sum(profits * parameters, axis=1)
        self.weights = weights
        self.capacity = capacity

    def compute_ranking(self, item):
        # Compute the ranking value for the stop based on the sum of outgoing distances divided by penalty
        return self.profits[item] / self.weights[item]

    def compute_all_rankings(self, selected_items):
        rankings = []
        for stop in range(len(self.weights)):
            if stop not in selected_items:
                ranking = self.compute_ranking(stop)
                rankings.append((stop, ranking))
        # Sort the rankings in ascending order
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def solve(self):
        selected_items = []  # List to store the selected stops
        total_profit = 0
        total_weight = 0

        # While the minimum reward is not achieved
        while total_weight < self.capacity or not selected_items:
            rankings = self.compute_all_rankings(selected_items)
            improved = False
            for item, ranking in rankings:
                if total_weight + self.weights[item] <= self.capacity:
                    total_profit += self.profits[item]
                    total_weight += self.weights[item]
                    selected_items.append(item)
                    improved = True
                    break
            if not improved:
                break

        return total_profit

        # return selected_items, total_weight, total_profit
