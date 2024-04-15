import numpy as np


class GreedyPCTSP:
    def __init__(self, distmatrix, prize, penalty, minprize=None):
        self.distmatrix = distmatrix
        self.prize = prize
        self.penalty = penalty
        self.minprize = minprize

    def compute_ranking(self, last, stop):
        # Compute the ranking value for the stop based on the sum of outgoing distances divided by penalty
        return (self.distmatrix[last, stop] if last is not None else 1) / self.penalty[stop] * np.sum(self.distmatrix[stop])
        # outgoing_distances = np.sum(self.distmatrix[stop, selected])
        # return outgoing_distances / self.penalty[stop]

    def compute_all_rankings(self, selected):
        rankings = []
        for stop in range(len(self.prize)):
            if stop not in selected:
                ranking = self.compute_ranking(selected[-1] if selected else None, stop)
                rankings.append((stop, ranking))
        # Sort the rankings in ascending order
        rankings.sort(key=lambda x: x[1])
        return rankings

    def solve(self):
        n_vertex = len(self.prize)
        selected_stops = []  # List to store the selected stops
        total_prize = 0
        total_distance = 0
        total_penalties = np.sum(self.penalty)
        objective = 0

        # While the minimum reward is not achieved
        while total_prize < self.minprize or not selected_stops:
            rankings = self.compute_all_rankings(selected_stops)
            best_stop = rankings[0][0]
            total_prize += self.prize[best_stop]
            total_distance += self.distmatrix[selected_stops[-1], best_stop] if selected_stops else 0
            total_penalties -= self.penalty[best_stop]
            objective += total_distance + total_penalties
            selected_stops.append(best_stop)
        # While the objective function is improved
        while True:
            improved = False
            rankings = self.compute_all_rankings(selected_stops)
            # If there are no more rankings, stop
            if not rankings:
                break
            # For each stop in the ranking, check if the objective function is improved
            for stop, ranking in rankings:
                distance = self.distmatrix[selected_stops[-1], stop] if selected_stops else 0
                penalty = self.penalty[stop]
                if objective + distance - penalty < objective:
                    objective += distance - penalty
                    total_distance += distance
                    total_penalties -= penalty                    
                    total_prize += self.prize[stop]
                    selected_stops.append(stop)
                    improved = True
                    break
            if not improved:
                break

        return selected_stops,False, objective,   False

        # # Create the solution arrays
        # in_solution = np.zeros(n_vertex, dtype=int)
        # in_solution[selected_stops] = 1
        # out_solution = np.ones(n_vertex, dtype=int)
        # out_solution[selected_stops] = 0
        #
        # store = np.zeros((n_vertex, n_vertex))
        # for i in range(n_vertex - 1):
        #     store[selected_stops[i], selected_stops[i + 1]] = 1
        #
        # # Create the results dictionary
        # results = {
        #     'store': store,
        #     'in_solution': in_solution,
        #     'total_prizes': total_prize,
        #     'total_penalties': total_penalties,
        #     'total_travel': total_distance,
        #     'objective': objective,
        #     'optimal': False  # The greedy algorithm always finds a feasible solution
        # }
        #
        # return results
