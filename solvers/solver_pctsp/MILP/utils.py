import numpy as np
from pulp import lpSum
from gurobipy import quicksum, abs_, Model
import cpmpy as cp


def decode_subpath(arcs):
    arcs = np.rint(arcs)
    path = [np.where(arcs[-1, :] == 1)[0][0]]
    next_ = np.where(arcs[path[-1], :] == 1)[0][0]
    while next_ != len(arcs) - 1:
        path.append(next_)
        next_ = np.where(arcs[path[-1], :] == 1)[0][0]
    outs = np.where(np.diag(arcs) == 1)[0]
    return path, outs


def adjacency_matrix(s, n):
    s = np.array(s)
    m = np.zeros((n, n))
    for i in range(n):
        # if stop `i` is not in the solution
        if i not in s:
            m[i][i] = 1
            continue  # if stop `i` is not in the solution, then it is its own successor using the subpath constraint
        # if stop `i` is the last stop in the solution, then there is no successor
        if i == s[-1]:
            continue
        # if stop `j` is the successor of stop `i`, then m[i][j] = True
        for j in range(n):
            if s[np.where(s == i)[0] + 1] == j:
                m[i][j] = 1
                break
    return m


def adjacency_matrix_cp(s, n):
    s = np.array(s)
    m = np.zeros((n, n)).astype(bool)
    for i in range(n):
        # if stop `i` is not in the solution
        if i not in s:
            m[i][i] = True
            continue  # if stop `i` is not in the solution, then it is its own successor using the subpath constraint
        # if stop `i` is the last stop in the solution, then there is no successor
        if i == s[-1]:
            continue
        # if stop `j` is the successor of stop `i`, then m[i][j] = True
        for j in range(n):
            if s[np.where(s == i)[0] + 1] == j:
                m[i][j] = True
                break
    return m


def arc_hamming_loss_pulp(arcs_in, real_sol):
    return lpSum((1 if arcs_in[i, j] != adjacency_matrix(real_sol, len(arcs_in))[i][j] else 0) for i in range(len(arcs_in)) for j in range(len(arcs_in)))


def arc_hamming_loss_cp(arcs_in, real_sol):
    return cp.sum((1 if arcs_in[i, j] != adjacency_matrix_cp(real_sol, len(arcs_in))[i][j] else 0) for i in range(len(arcs_in)) for j in range(len(arcs_in)))


def arc_hamming_loss_grb(arcs_in, real_sol, n_vertex, model: Model):
    matching_vars = model.addVars(n_vertex, n_vertex, vtype='B', name='matching_vars')
    adj_mat = adjacency_matrix(real_sol, n_vertex)
    for i in range(n_vertex):
        for j in range(n_vertex):
            model.addConstr(matching_vars[i, j] <= adj_mat[i, j], name='matching_vars_1')
            model.addConstr(matching_vars[i, j] <= arcs_in[i, j], name='matching_vars_2')
            model.addConstr(matching_vars[i, j] >= adj_mat[i, j] + arcs_in[i, j] - 1, name='matching_vars_3')
    return quicksum(1 - matching_vars[i, j] for i in range(n_vertex) for j in range(n_vertex))


def phi_loss_pulp(arcs_in, distances, real_sol):
    real_matrix = adjacency_matrix(real_sol, len(arcs_in))
    phi = np.zeros(len(distances), dtype=int)
    phi_hat = np.zeros(len(distances), dtype='object')
    for i in range(len(distances)):
        phi[i] = np.sum(distances[i] * real_matrix)
        phi_hat[i] = np.sum(distances[i] * arcs_in)
    # return lpSum(abs(phi - phi_hat)) ** 2
    # return np.linalg.norm(phi - phi_hat, ord=2) ** 2
    delta = [phi[i] - phi_hat[i] if phi[i] - phi_hat[i] >= 0 else phi_hat[i] - phi[i] for i in range(len(phi))]
    return lpSum(delta)


def phi_loss_cp(arcs_in, distances, real_sol):
    real_matrix = adjacency_matrix(real_sol, len(arcs_in))
    phi = np.zeros(len(distances), dtype=int)
    phi_hat = np.zeros(len(distances), dtype='object')
    for i in range(len(distances)):
        phi[i] = np.sum(distances[i] * real_matrix)
        phi_hat[i] = np.sum(distances[i] * arcs_in)
    # return cp.sum(abs(phi - phi_hat)) ** 2
    # return np.linalg.norm(phi - phi_hat, ord=2) ** 2
    delta = [phi[i] - phi_hat[i] if phi[i] - phi_hat[i] >= 0 else phi_hat[i] - phi[i] for i in range(len(phi))]
    return cp.sum(delta)


def phi_loss_grb(arcs_in, distances, real_sol, n_vertex):
    real_matrix = adjacency_matrix(real_sol, n_vertex)
    phi = np.zeros(len(distances), dtype=int)
    phi_hat = np.zeros(len(distances), dtype='object')
    for i in range(len(distances)):
        phi[i] = np.sum(distances[i] * real_matrix)
        phi_hat[i] = quicksum(distances[i][j, k] * arcs_in[j, k] for j in range(n_vertex) for k in range(n_vertex))
        # phi_hat[i] = np.sum(distances[i] * arcs_in)
    # return lpSum(abs(phi - phi_hat)) ** 2
    # return np.linalg.norm(phi - phi_hat, ord=2) ** 2
    # delta = [phi[i] - phi_hat[i] if phi[i] - phi_hat[i] >= 0 else phi_hat[i] - phi[i] for i in range(len(phi))]
    delta = [phi_hat[i] - phi[i] for i in range(len(phi))]
    return quicksum(delta)
    # return quicksum(abs_(phi[i] - phi_hat[i]) for i in range(len(phi)))
