//
// Created by Gael Aglin on 19/02/2023.
//

#include "Solution.h"

Solution::Solution(int n, bool early_stop) {
    cost = numeric_limits<float>::infinity();
    penalty = numeric_limits<float>::infinity();
    prize = 0;
    for (int i = 0; i < n; ++i) visited.push_back(false);
    early_stopping = early_stop;
}

Solution::Solution(const Solution& other) {
    cost = other.cost;
    penalty = other.penalty;
    prize = other.prize;
    route = other.route;
    visited = other.visited;
}

void Solution::print() {
    cout << "Route found: ";
    for (int i = 0; i < route.size() - 1; ++i) cout << route[i] << " -> ";
    cout << route.end()[-1] << endl;
    cout << "Unselected stops: ";
    bool first = true;
    for (int i = 0; i < visited.size(); ++i) {
        if (find(route.begin(), route.end(), i) == route.end()) {
            if (first) {
                cout << i;
                first = false;
            }
            else cout << ", " << i;
        }
    }
    cout << endl << "Collected prize: " << prize << endl;
    cout << "Transition cost: " << cost << endl;
    cout << "Total penalty: " << penalty << endl;
    cout << "Objective value: " << cost + penalty << endl;
}

Solution& Solution::operator=(const Solution& other) {
    cost = other.cost;
    penalty = other.penalty;
    prize = other.prize;
    route = other.route;
    visited = other.visited;
    return *this;
}

void Solution::setInitSolution(Params* params, vector<int>& init_sol) {

    if (init_sol.empty()) {

        // cost and penalty initialized at \infty. prize is 0 and route is empty

        // decide the first stop. For each stop, compute a score = penalty / (in_cost + all out_costs)
        // the best stop is the one maximizing this value because its penalty is too high (compared to cost) to afford it
        vector<float> scores;
        scores.reserve(params->penalties.size());
        for (int i = 0; i < params->penalties.size(); ++i) {
            scores.push_back(params->penalties[i] / get_in_out_cost(params, i));
        }
        int best_stop = max_element(scores.begin(), scores.end()) - scores.begin();
        // initialize the solution variables after finding the starting point
        // the penalty is not computed yet. It will be computed once after knowing not visited stops
        cost = 0; // not cost because there is no move before a second stop
        route.push_back(best_stop);
        prize = params->prizes[best_stop];
        visited[best_stop] = true;

        // decide the remaining stops using the same process as above, until a stopping criterion
        while (prize < params->minPrize) {  // the minimum prize is reached
            // while(solution->route.size() < params->penalties.size()) {  // we use all the stops
            scores = {};
            for (int i = 0; i < params->penalties.size(); ++i) {
                if (visited[i]) scores.push_back(-numeric_limits<float>::infinity());
                else scores.push_back(params->penalties[i] / get_in_out_cost(params, i));
            }
            best_stop = max_element(scores.begin(), scores.end()) - scores.begin();
            cost += params->distanceMatrix[route.back()][best_stop];
            route.push_back(best_stop);
            prize += params->prizes[best_stop];
            visited[best_stop] = true;
        }

        // compute the penalty based on "not visited" stops
        penalty = 0; // initialize the value
        for (int i = 0; i < visited.size(); ++i) {
            if (!visited[i]) penalty += params->penalties[i];
        }
        init_obj = cost + penalty;
    } else {
        cost = 0; // not cost because there is no move before a second stop
        route.push_back(init_sol[0]);
        prize = params->prizes[init_sol[0]];
        visited[init_sol[0]] = true;
        penalty = std::accumulate(params->penalties.begin(), params->penalties.end(), 0.0f);
        for (int i = 1; i < init_sol.size(); ++i) {
            cost += params->distanceMatrix[route.back()][init_sol[i]];
            route.push_back(init_sol[i]);
            prize += params->prizes[init_sol[i]];
            visited[init_sol[i]] = true;
            penalty -= params->penalties[init_sol[i]];
        }
        init_obj = cost + penalty;
    }

    if (params->verbose) {
        cout << "Init route: ";
        for (int i : route) cout << i << ", ";
        cout << endl;
        cout << "visited stops: ";
        for (int i = 0; i < visited.size(); ++i) if (visited[i]) cout << i << ", ";
        cout << endl;
        cout << "Init Result: " << cost + penalty << " (Cost: " << cost << "; Penalty: " << penalty << ")" << endl;
        cout << "Init prize: " << prize << endl << endl;
    }

}

float Solution::get_in_out_cost(Params* params, int current_stop) {
    float cost_out = 0;
    if (!route.empty()) cost_out += params->distanceMatrix[route.back()][current_stop];
    for (int i = 0; i < params->prizes.size(); ++i) {
        if (i != current_stop and !visited[i]) {
            cost_out += params->distanceMatrix[current_stop][i];
        }
    }
    return cost_out;
}