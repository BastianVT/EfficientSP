import numpy as np
import cpmpy as cp


def decode_subpath(arcs_in):
    try:
        arcs = arcs_in.value()
    except: 
        arcs = arcs_in
    n = arcs.shape[0]

    outs = np.where(np.diag(arcs))[0]

    if len(outs) == n:
        return [], outs  # empty

    # start is column that has no entry
    cur = np.argmin(np.sum(arcs, axis=0))
    seq = [cur]
    while True:
        if np.max(arcs[cur]) == 0:
            break  # none, points to dummy for path
        cur = np.argmax(arcs[cur])  # next: first true one
        seq.append(cur)

    return seq, outs


def adjacency_matrix(s, n):
    s = np.array(s)
    m = [[False for _ in range(n)] for _ in range(n)]
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
                break  # only one successor
    return m


def arc_hamming_loss(arcs_in, real_sol):
    return np.sum(arcs_in != adjacency_matrix(real_sol, len(arcs_in)))


def phi_loss(arcs_in, distances, real_sol):
    real_matrix = adjacency_matrix(real_sol, len(arcs_in))
    phi = np.zeros(len(distances), dtype=int)
    phi_hat = np.zeros(len(distances), dtype='object')
    for i in range(len(distances)):
        phi[i] = cp.sum(distances[i] * real_matrix)
        phi_hat[i] = cp.sum(distances[i] * arcs_in)
    return cp.sum(phi - phi_hat) ** 2
    # return np.linalg.norm(phi - phi_hat, ord=2) ** 2
