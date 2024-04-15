import cpmpy as cp


def compute_new_profits(old_profit, parameters):
    profits = cp.intvar(shape=len(parameters), lb=0, ub=1000)
    model = cp.Model()
    model += (sum(parameters[i] * profits[i] for i in range(len(parameters))) == old_profit)
    model.minimize(sum(abs(profits[i] - profits[j]) for i in range(len(parameters)) for j in range(len(parameters))))
    model.solve()
    return profits.value().astype(int)
