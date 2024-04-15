//
// Created by Gael Aglin on 19/02/2023.
//

#include "ILS.h"


bool ILS::addNode(Params *params, Solution *solution) {
    if (params->is_timeout()) return false;
    auto *bestSolution = new Solution(*solution);
    float modifiedCost, modifiedPenalty, modifiedPrize;
    int iBest = -1, jBest = -1;

    // find the stop "i" to place before the stop at index "j"
    for (int i = 0; i < params->prizes.size(); i++) {

        if (solution->visited[i]) continue;

        // does the stop "i" produce a better result when it is placed before a visited stop at index j ?
        for (int j = 1; j < solution->route.size(); j++) {
            modifiedCost = solution->cost
                           - params->distanceMatrix[solution->route[j - 1]][solution->route[j]]
                           + params->distanceMatrix[solution->route[j - 1]][i]
                           + params->distanceMatrix[i][solution->route[j]];
            modifiedPenalty = solution->penalty - params->penalties[i];
            modifiedPrize = solution->prize + params->prizes[i];

            if (modifiedPrize >= params->minPrize and (modifiedCost + modifiedPenalty) < (bestSolution->cost + bestSolution->penalty)) {
                iBest = i;
                jBest = j;
                bestSolution->cost = modifiedCost;
                bestSolution->penalty = modifiedPenalty;
                // the prize, route and visited variables can remain unchanged as they are not important to decide the best solution
            }
        }

        // assess what happens when "i" is place at the beginning or at end
        // additional cost and index of "i" when it is placed at beginning
        pair<float,int> cost_ind_begin = make_pair(params->distanceMatrix[i][solution->route[0]], 0);
        // additional cost and index of "i" when it is placed at beginning
        pair<float,int> cost_ind_end = make_pair(params->distanceMatrix[solution->route.back()][i], solution->route.size());
        for (auto& cost_ind : {cost_ind_begin, cost_ind_end}) {
            modifiedCost = solution->cost + cost_ind.first;
            modifiedPenalty = solution->penalty - params->penalties[i];
            modifiedPrize = solution->prize + params->prizes[i];
            if (modifiedPrize >= params->minPrize and (modifiedCost + modifiedPenalty) < (bestSolution->cost + bestSolution->penalty)) {
                iBest = i;
                jBest = cost_ind.second;
                bestSolution->cost = modifiedCost;
                bestSolution->penalty = modifiedPenalty;
            }
        }
    }
    if (iBest != -1) {
        // the correct best cost and the best penalty are copied
        solution->cost = bestSolution->cost;
        solution->penalty = bestSolution->penalty;
        solution->prize += params->prizes[iBest];
        solution->route.insert(solution->route.begin() + jBest, iBest);
        solution->visited[iBest] = true;
        if (params->verbose) {
            cout << "add stop: " << iBest << " at index: " << jBest << endl;
            cout << "new route: ";
            for (auto s : solution->route) cout << s << ", ";
            cout << endl;
            cout << "new visited stops: ";
            for (int i = 0; i < solution->visited.size(); ++i) if (solution->visited[i]) cout << i << ", ";
            cout << endl;
            cout << "new value: " << solution->cost + solution->penalty << " (Cost: " << solution->cost << "; Penalty: " << solution->penalty << ")" << endl;
            cout << "---------------" << endl;
        }
        delete bestSolution;
        return true;
    }
    delete bestSolution;
    return false;
}

bool ILS::removeNode(Params *params, Solution *solution) {
    if (params->is_timeout()) return false;
    auto *bestSolution = new Solution(*solution);
    float modifiedCost, modifiedPenalty, modifiedPrize;
    int iBest = -1;

    for (int i = 1; i < solution->route.size() - 1; i++) {
        modifiedCost = solution->cost
                       - params->distanceMatrix[solution->route[i - 1]][solution->route[i]]
                       - params->distanceMatrix[solution->route[i]][solution->route[i + 1]]
                       + params->distanceMatrix[solution->route[i - 1]][solution->route[i + 1]];
        modifiedPenalty = solution->penalty + params->penalties[solution->route[i]];
        modifiedPrize = solution->prize - params->prizes[solution->route[i]];

        if ((modifiedPrize >= params->minPrize) && (modifiedCost + modifiedPenalty < bestSolution->cost + bestSolution->penalty)) {
            iBest = i;
            bestSolution->cost = modifiedCost;
            bestSolution->penalty = modifiedPenalty;
        }
    }

    if (solution->route.size() >= 2) {
        // assess what happens when "i" is place at the beginning or at end
        // additional cost and index of "i" when it is placed at beginning
        pair<float,int> cost_ind_begin = make_pair(params->distanceMatrix[solution->route[0]][solution->route[1]], 0);
        // additional cost and index of "i" when it is placed at beginning
        pair<float,int> cost_ind_end = make_pair(params->distanceMatrix[solution->route.end()[-2]][solution->route.end()[-1]], solution->route.size() - 1);
        for (auto& cost_ind : {cost_ind_begin, cost_ind_end}) {
            modifiedCost = solution->cost - cost_ind.first;
            modifiedPenalty = solution->penalty + params->penalties[solution->route[cost_ind.second]];
            modifiedPrize = solution->prize - params->prizes[solution->route[cost_ind.second]];
            if ((modifiedPrize >= params->minPrize) && (modifiedCost + modifiedPenalty < bestSolution->cost + bestSolution->penalty)) {
                iBest = cost_ind.second;
                bestSolution->cost = modifiedCost;
                bestSolution->penalty = modifiedPenalty;
            }
        }
    }

    if (iBest != -1) {
        // the correct best cost and the best penalty are copied
        solution->cost = bestSolution->cost;
        solution->penalty = bestSolution->penalty;
        solution->prize -= params->prizes[solution->route[iBest]];
        if (params->verbose) cout << "remove stop at index: " << iBest << "(" << solution->route[iBest] << ")" << endl;
        solution->visited[solution->route[iBest]] = false;
        solution->route.erase(solution->route.begin() + iBest);
        if (params->verbose) {
            cout << "new route: ";
            for (auto s : solution->route) cout << s << ", ";
            cout << endl;
            cout << "new visited stops: ";
            for (int i = 0; i < solution->visited.size(); ++i) if (solution->visited[i]) cout << i << ", ";
            cout << endl;
            cout << "new value: " << solution->cost + solution->penalty << " (Cost: " << solution->cost << "; Penalty: " << solution->penalty << ")" << endl;
            cout << "---------------" << endl;
        }
        delete bestSolution;
        return true;
    }
    delete bestSolution;
    return false;
}

bool ILS::swapNodes(Params *params, Solution *solution) {
    if (params->is_timeout()) return false;
    auto *bestSolution = new Solution(*solution);
    float modifiedCost;
    int iBest = -1, jBest = -1;

    for (int i = 0; i < solution->route.size() - 1; i++) {
        for (int j = i + 1; j < solution->route.size(); j++) {

            modifiedCost = solution->cost
                           - ((i >= 1) ? params->distanceMatrix[solution->route[i - 1]][solution->route[i]] : 0)
                           - ((j < solution->route.size() - 1) ? params->distanceMatrix[solution->route[j]][solution->route[j + 1]] : 0)
                           + ((i >= 1) ? params->distanceMatrix[solution->route[i - 1]][solution->route[j]] : 0)
                           + ((j < solution->route.size() - 1) ? params->distanceMatrix[solution->route[i]][solution->route[j + 1]] : 0);

            if (j != i + 1)
                modifiedCost = modifiedCost
                               - params->distanceMatrix[solution->route[i]][solution->route[i + 1]]
                               - params->distanceMatrix[solution->route[j - 1]][solution->route[j]]
                               + params->distanceMatrix[solution->route[j]][solution->route[i + 1]]
                               + params->distanceMatrix[solution->route[j - 1]][solution->route[i]];
            else // asymmetric problem since cost up is different from cost down and there are even timed by different weights
                modifiedCost = modifiedCost
                               - params->distanceMatrix[solution->route[i]][solution->route[j]]
                               + params->distanceMatrix[solution->route[j]][solution->route[i]];

            if (modifiedCost < bestSolution->cost) {
                iBest = i;
                jBest = j;
                bestSolution->cost = modifiedCost;
            }
        }
    }

    if (iBest != -1) {
        if (params->verbose) cout << "swap stop at index: " << iBest << "(" << solution->route[iBest] << ")" << " with stop at index: " << jBest << "(" << solution->route[jBest] << ")" << endl;
        solution->cost = bestSolution->cost;
        swap(solution->route[iBest], solution->route[jBest]);
        if (params->verbose) {
            cout << "new route: ";
            for (auto s : solution->route) cout << s << ", ";
            cout << endl;
            cout << "new visited stops: ";
            for (int i = 0; i < solution->visited.size(); ++i) if (solution->visited[i]) cout << i << ", ";
            cout << endl;
            cout << "new value: " << solution->cost + solution->penalty << " (Cost: " << solution->cost << "; Penalty: " << solution->penalty << ")" << endl;
            cout << "---------------" << endl;
        }
        delete bestSolution;
        return true;
    }
    delete bestSolution;
    return false;
}

bool ILS::twoOpt(Params *params, Solution *solution) {
    if (solution->route.size() < 3 or params->is_timeout()) return false;
    auto *bestSolution = new Solution(*solution);
    float modifiedCost;
    int iBest = -1, jBest = -1;

    for (int i = 0; i < solution->route.size() - 2; i++) {
        for (int j = i + 2; j < solution->route.size(); j++) {

            modifiedCost = solution->cost;
            for (int k = i-1; k <= j; ++k) modifiedCost -= ((k >= 0 and k < solution->route.size() - 1) ? params->distanceMatrix[solution->route[k]][solution->route[k+1]] : 0);

            for (int k = j; k > i; --k) modifiedCost += params->distanceMatrix[solution->route[k]][solution->route[k-1]];
            modifiedCost += ((i > 0) ? params->distanceMatrix[solution->route[i - 1]][solution->route[j]] : 0);
            modifiedCost += ((j < solution->route.size() - 1) ? params->distanceMatrix[solution->route[i]][solution->route[j+1]] : 0);

            if (modifiedCost < bestSolution->cost) {
                iBest = i;
                jBest = j;
                bestSolution->cost = modifiedCost;
            }
        }
    }

    if (iBest != -1) {
        if (params->verbose) cout << "2opt stop at index: " << iBest << "(" << solution->route[iBest] << ")" << " with stop at index: " << jBest << "(" << solution->route[jBest] << ")" << endl;
        solution->cost = bestSolution->cost;
        reverse(solution->route.begin() + iBest, solution->route.begin() + jBest + 1);
        if (params->verbose) {
            cout << "new route: ";
            for (auto s : solution->route) cout << s << ", ";
            cout << endl;
            cout << "new visited stops: ";
            for (int i = 0; i < solution->visited.size(); ++i) if (solution->visited[i]) cout << i << ", ";
            cout << endl;
            cout << "new value: " << solution->cost + solution->penalty << " (Cost: " << solution->cost << "; Penalty: " << solution->penalty << ")" << endl;
            cout << "---------------" << endl;
        }
        delete bestSolution;
        return true;
    }
    delete bestSolution;
    return false;
}

void ILS::oneLocalSearch(Params *params, Solution *solution) {
    // variables to keep whether the moves tested have produced better result or not
    bool foundBetter1, foundBetter2, foundBetter3, foundBetter4;

    while (true) {
        foundBetter1 = addNode(params, solution);
        if (solution->early_stopping and (solution->cost + solution->penalty) < solution->init_obj) break;
        foundBetter2 = swapNodes(params, solution);
        if (solution->early_stopping and (solution->cost + solution->penalty) < solution->init_obj) break;
        foundBetter3 = removeNode(params, solution);
        if (solution->early_stopping and (solution->cost + solution->penalty) < solution->init_obj) break;
        foundBetter4 = twoOpt(params, solution);
        if (solution->early_stopping and (solution->cost + solution->penalty) < solution->init_obj) break;

        if ((not foundBetter1 and not foundBetter2 and not foundBetter3 and not foundBetter4) or params->is_timeout()) break;

        if (params->verbose) {
            cout << "next route: ";
            for (int i : solution->route) cout << i << ", ";
            cout << endl;
            cout << "visited stops: ";
            for (int i = 0; i < solution->visited.size(); ++i) if (solution->visited[i]) cout << i << ", ";
            cout << endl;
            cout << "new Result: " << solution->cost + solution->penalty << " (Cost: " << solution->cost << "; Penalty: " << solution->penalty << ")" << endl;
            cout << "new prize: " << solution->prize << endl;
            cout << foundBetter1 << " " << foundBetter2 << " " << foundBetter3 << " " << foundBetter4 << endl << endl;
        }
    }
}

void ILS::doubleBridge(Params *params, Solution *solutionCandidate) {
    int sol_size = solutionCandidate->route.size();

    if (sol_size < 4) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::uniform_int_distribution<> dis(0, sol_size - 1);
        int i1 = dis(g), i2 = -1;
        do { i2 = dis(g); } while (i2 == i1);
        std::swap(solutionCandidate->route[i1], solutionCandidate->route[i2]);
    }
    else {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, solutionCandidate->route.size() - 1);

        int pos1 = dis(gen), pos2 = dis(gen), pos3 = dis(gen);
        while (pos1 == pos2) pos2 = dis(gen);
        while (pos1 == pos3 || pos2 == pos3) pos3 = dis(gen);

        if (pos1 > pos2) swap(pos1, pos2);
        if (pos2 > pos3) swap(pos2, pos3);
        if (pos1 > pos2) swap(pos1, pos2);

        vector<int> newRoute;
        if (params->classic_double) {
            newRoute.insert(newRoute.end(), solutionCandidate->route.begin(), solutionCandidate->route.begin() + pos1);
            newRoute.insert(newRoute.end(), solutionCandidate->route.begin() + pos3, solutionCandidate->route.end());
            newRoute.insert(newRoute.end(), solutionCandidate->route.begin() + pos2, solutionCandidate->route.begin() + pos3);
            newRoute.insert(newRoute.end(), solutionCandidate->route.begin() + pos1, solutionCandidate->route.begin() + pos2);
        }
        else {
            vector<pair<int,int>> bounds = {make_pair(0, pos1), make_pair(pos1, pos2), make_pair(pos2, pos3), make_pair(pos3, solutionCandidate->route.size())};
            shuffle(bounds.begin(), bounds.end(), gen);
            newRoute.insert(newRoute.end(), solutionCandidate->route.begin() + bounds[0].first, solutionCandidate->route.begin() + bounds[0].second);
            newRoute.insert(newRoute.end(), solutionCandidate->route.begin() + bounds[1].first, solutionCandidate->route.begin() + bounds[1].second);
            newRoute.insert(newRoute.end(), solutionCandidate->route.begin() + bounds[2].first, solutionCandidate->route.begin() + bounds[2].second);
            newRoute.insert(newRoute.end(), solutionCandidate->route.begin() + bounds[3].first, solutionCandidate->route.begin() + bounds[3].second);
        }
        solutionCandidate->route = newRoute;
    }

    solutionCandidate->cost = 0;
    for (int i = 0; i < solutionCandidate->route.size() - 1; i++)
        solutionCandidate->cost += params->distanceMatrix[solutionCandidate->route[i]][solutionCandidate->route[i + 1]];
}


void ILS::perturbation(Params *params, Solution *solutionCandidate) {
    for (int i = 0; i < params->intensity; i++) {
        doubleBridge(params, solutionCandidate);
    }
}

void ILS::runILS(Params *params, Solution *solution) {
    auto *modifiedSolution = new Solution(*solution);
    int iterations = 0;
    int noImprov = 0;
    while (noImprov < params->maxNoImprov and iterations < params->maxIter and not params->is_timeout()) {
        perturbation(params, modifiedSolution);
        oneLocalSearch(params, modifiedSolution);
        iterations++;
        if (modifiedSolution->prize >= params->minPrize and modifiedSolution->cost + modifiedSolution->penalty < solution->cost + solution->penalty) {
            *solution = *modifiedSolution;
            noImprov = 0;
        }
        else {
            *modifiedSolution = *solution;
            noImprov++;
        }
    }
    delete modifiedSolution;
}

