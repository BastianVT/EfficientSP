//
// Created by Gael Aglin on 19/02/2023.
//

#include "Solution_WL.h"


Solution_WL::Solution_WL(int n, bool early_stop) : Solution(n, early_stop) {
    loss = -1;
}

Solution_WL::Solution_WL(const Solution_WL& other) : Solution(other) {
    loss = other.loss;
}

Solution_WL& Solution_WL::operator=(const Solution_WL& other) {
    Solution::operator=(other);
    loss = other.loss;
    return *this;
}

vector<vector<bool>> Solution_WL::successor_matrix(const vector<int>& route, int n_stops) {
    vector<vector<bool>> matrix(n_stops, vector<bool>(n_stops, false));
    for (int i = 0; i < n_stops; i++) {
        // if stop i is not in the solution
        if (std::find(route.begin(), route.end(), i) == route.end()) {
            matrix[i][i] = true;
            continue; // if stop i is not in the solution, then it is its own successor using the subpath constraint
        }
        // if stop i is the last stop in the solution, then there is no successor
        if (i == route.back()) {
            continue;
        }
        // if stop j is the successor of stop i, then m[i][j] = true
        for (int j = 0; j < n_stops; j++) {
            auto it = find(route.begin(), route.end(), i);
            if (it != route.end() && *(it + 1) == j) {
                matrix[i][j] = true;
                break; // only one successor
            }
        }
    }
    return matrix;
}

void Solution_WL::compute_loss(const Params* params) {
    if (((Params_WL*)params)->real_route.empty()) {
        loss = 0;
        return;
    }

    loss = 0;
    vector<vector<bool>> matrix = successor_matrix(route, ((Params_WL*)params)->prizes.size());
    vector<vector<bool>> real_matrix = successor_matrix(((Params_WL*)params)->real_route, ((Params_WL*)params)->prizes.size());
    for (int i = 0; i < ((Params_WL*)params)->prizes.size(); ++i) {
        for (int j = 0; j < ((Params_WL*)params)->prizes.size(); ++j) {
            if (matrix[i][j] != real_matrix[i][j]) loss++;
        }
    }
}

/*
void Solution_WL::compute_loss(const Params* params) {
    if (((Params_WL*)params)->real_route.empty()) {
        loss = 0;
        return;
    }
    loss = 0;
    for (int i = 0; i < min(route.size(), ((Params_WL*)params)->real_route.size()); ++i) {
        if (route[i] != ((Params_WL*)params)->real_route[i]) loss++;
    }
    loss += abs((int)route.size() - (int)((Params_WL*)params)->real_route.size());
}*/
