# Simulated annealing applied to a random instance of the knapsack problem

import pdb
import time
from random import Random
import numpy as np
from random import Random

# import common_functions as cf
# import problem_instance as pi


seed = 5113
myPRNG = Random(seed)


# function to evaluate a solution x
def evaluate(x, value, weights, maxWeight):
    # print("x", x)
    # print("value", value)
    totalValue = np.dot(x, value)  # compute the value of the knapsack selection
    totalWeight = np.dot(x, weights)  # compute the weight value of the knapsack selection

    if totalWeight > maxWeight:
        # set total value to be -inf
        totalValue = -float("inf")

    return [totalValue, totalWeight]  # returns a list of both total value and total weight


def random_subset(size, subset_size):
    return np.random.choice(size, subset_size, replace=False)


def neighborhood(x, n, max_subset_size=1):
    nbrhood = []

    for i in range(n):
        for subset_size in range(1, max_subset_size):
            nbr = np.copy(x)
            flip_indices = random_subset(n, subset_size)
            nbr[flip_indices] = 1 - nbr[flip_indices]
            nbrhood.append(nbr)
    #         print("add")
    # print("nbrhood", nbrhood)
    return np.array(nbrhood)


# create the initial solution
def initial_solution(value, weights, maxWeight, n):
    sorted_w = np.sort(weights)

    temp_w = 0  # weight tracker
    i = len(weights) - 1  # counter that's going to count down
    num_ones = 0  # number of 1s I need in my solution

    # A while loop to ensure that the initial solution is not going to be infeasible
    while temp_w <= maxWeight:
        temp_w += sorted_w[i]
        i -= 1
        num_ones += 1

    x = np.zeros(n, dtype=int)  # initializing solution array
    best_val_ind = np.argsort(value)[-num_ones:]  # indices of the first few (=num_ones) highest values
    x[best_val_ind] = 1  # taking some highest value items

    return x


def solve_knapsack(parameters, profits, weights, capacity, time_limit=None, n_jobs=0, init_sol=None, real_sol=None, slack=None, prediction=False, early_stopping=False):
    # print("time limit", time_limit)
    start_time = time.perf_counter()

    init_t = 100000  # setting an initial temperature
    t = init_t  # current temperature
    M = 500  # number of iterations at each temperature
    k = 0  # counter to keep track of main loop

    # sum_profit: for each profit vector, compute the dot of profits and parameters
    sum_profit = np.zeros(len(profits))
    for i in range(len(profits)):
        sum_profit[i] = np.dot(profits[i], parameters)

    x_init = initial_solution(sum_profit, weights, capacity, len(profits)) if init_sol is None else np.array([1 if i in init_sol else 0 for i in range(len(profits))])  # The very first or the initial solution
    f_init = evaluate(x_init, sum_profit, weights, capacity)  # evaluation of x_init
    x_curr = np.copy(x_init)  # Current solution. Starts out with x_init
    f_curr = evaluate(x_curr, sum_profit, weights, capacity)  # f_curr will hold the evaluation of the current soluton
    # print("initial solution with profit {} and weight {}".format(f_init[0], f_init[1]))

    # storing information for the feasible solutions
    f_value = []
    f_weight = []
    f_solution = []

    # variable to record the number of solutions evaluated
    solutionsChecked = 0

    # begin local search overall logic ----------------
    done = 0

    while done == 0:
        if time_limit is not None and time.perf_counter() - start_time >= time_limit:
            break

        # stopping criterion
        if t < 1:
            done = 1

        m = 0
        while m < M:
            if time_limit is not None and time.perf_counter() - start_time >= time_limit:
                break
            solutionsChecked += 1
            # print("k = {}, m = {}, s = {} \n".format(k,m,solutionsChecked))
            # print("duration", time.perf_counter() - start_time)

            # print("len(profits)", len(profits))
            N = neighborhood(x_curr, len(profits), max_subset_size=2)  # create a list of all neighbors in the neighborhood of x_curr
            # print('len(N)', len(N))
            s = N[myPRNG.randint(0, len(N) - 1)]  # A randomly selected neighbor

            if early_stopping:
                # s should be the best neighbor
                s = N[np.argmax([evaluate(sol, sum_profit, weights, capacity)[0] for sol in N])]

            # check for feasibility of this solution
            try:
                eval_s = evaluate(s, sum_profit, weights, capacity)
            except:
                continue

            if early_stopping and eval_s[1] <= capacity and eval_s[0] > f_init[0]:
                selected_items = [i for i in range(len(profits)) if s[i] == 1]
                # print("found a better solution with profit {} and weight {}".format(eval_s[0], eval_s[1]))
                return selected_items, eval_s[1], eval_s[0]

            # If this random neighbor is an improving move, accept it immediately
            # else accept it a probability distribution
            if eval_s[0] >= f_curr[0]:
                x_curr = np.copy(s)
                f_curr = np.copy(eval_s)

                f_solution.append(x_curr)
                f_value.append(f_curr[0])
                f_weight.append(f_curr[1])
            else:
                p = np.exp(-(f_curr[0] - eval_s[0]) / t)
                test_p = myPRNG.uniform(0, 1)

                if test_p <= p:
                    x_curr = np.copy(s)
                    f_curr = np.copy(eval_s)

                    f_solution.append(x_curr)
                    f_value.append(f_curr[0])
                    f_weight.append(f_curr[1])
            m += 1
            # print("current solution: {} \n".format(f_curr[0]))

        # incrementing k and updating functions of k
        k += 1
        # t = 0.8 * t  # cauchy cooling function
        t = init_t * np.exp(-(k * M + m) / 1)  # exponential cooling function

    # print("\nFinal number of solutions checked: ", solutionsChecked)
    # print("Best value found: ", np.nanmax(f_value))
    # print("Weight is: ", f_weight[np.nanargmax(f_value)])
    # print("Total number of items selected: ", np.sum(x_curr))
    # print("Best solution: ", x_curr)

    if early_stopping:
        if f_curr[1] <= capacity and f_curr[0] > f_init[0]:
            selected_items = [i for i in range(len(profits)) if x_curr[i] == 1]
            # print("Found solution with profit {} and weight {}".format(f_curr[0], f_curr[1]))
            return selected_items, f_curr[1], f_curr[0]
        else:
            # print("No solution found")
            return None
    else:
        if f_curr[1] <= capacity:
            selected_items = [i for i in range(len(profits)) if x_curr[i] == 1]
            # print("Found solution with profit {} and weight {}".format(f_curr[0], f_curr[1]))
            return selected_items, f_curr[1], f_curr[0]
        else:
            # print("No solution found")
            return None
