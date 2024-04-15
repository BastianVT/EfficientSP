//
// Created by Gael Aglin on 19/02/2023.
//

#include "ILS_WL.h"

bool ILS_WL::addNode(Params *params, Solution *solution) {
    if (params->is_timeout()) return false;
    auto *currentBestSolution = new Solution_WL(*((Solution_WL*)solution));
    int iBest = -1, jBest = -1;

    // find the stop "i" to place before the stop at index "j"
    for (int i = 0; i < params->prizes.size(); i++) {

        if (solution->visited[i]) continue;

        // does the stop "i" produce a better result when it is placed before a visited stop at index j ?
        for (int j = 1; j < solution->route.size(); j++) {
            auto *currentSolution = new Solution_WL(*((Solution_WL*)solution));
            currentSolution->cost = currentSolution->cost
                                    - params->distanceMatrix[currentSolution->route[j - 1]][currentSolution->route[j]]
                                    + params->distanceMatrix[currentSolution->route[j - 1]][i]
                                    + params->distanceMatrix[i][currentSolution->route[j]];
            currentSolution->penalty = currentSolution->penalty - params->penalties[i];
            currentSolution->prize = currentSolution->prize + params->prizes[i];
            currentSolution->route.insert(currentSolution->route.begin() + j, i);
            currentSolution->visited[i] = true;
            ((Solution_WL*)currentSolution)->compute_loss(params);
            if (currentSolution->prize >= params->minPrize and (currentSolution->cost + currentSolution->penalty - ((Solution_WL*)currentSolution)->loss) < (currentBestSolution->cost + currentBestSolution->penalty - ((Solution_WL*)currentSolution)->loss)) {
                *currentBestSolution = *currentSolution;
                iBest = i;
                jBest = j;
            }
        }

        // assess what happens when "i" is place at the beginning or at end
        // additional cost and index of "i" when it is placed at beginning
        pair<float,int> cost_ind_begin = make_pair(params->distanceMatrix[i][solution->route[0]], 0);
        // additional cost and index of "i" when it is placed at beginning
        pair<float,int> cost_ind_end = make_pair(params->distanceMatrix[solution->route.back()][i], solution->route.size());
        for (auto& cost_ind : {cost_ind_begin, cost_ind_end}) {
            auto *currentSolution = new Solution_WL(*((Solution_WL*)solution));
            currentSolution->cost = currentSolution->cost + cost_ind.first;
            currentSolution->penalty = currentSolution->penalty - params->penalties[i];
            currentSolution->prize = currentSolution->prize + params->prizes[i];
            currentSolution->route.insert(currentSolution->route.begin() + cost_ind.second, i);
            currentSolution->visited[i] = true;
            ((Solution_WL*)currentSolution)->compute_loss(params);
            if (currentSolution->prize >= params->minPrize and (currentSolution->cost + currentSolution->penalty - ((Solution_WL*)currentSolution)->loss) < (currentBestSolution->cost + currentBestSolution->penalty - ((Solution_WL*)currentSolution)->loss)) {
                *currentBestSolution = *currentSolution;
                iBest = i;
                jBest = cost_ind.second;
            }
            delete currentSolution;
        }
    }

    // if a better solution is found, update the current solution
    if (iBest != -1) {
        *((Solution_WL*)solution) = *currentBestSolution;
        if (params->verbose) {
            cout << "add stop: " << iBest << " at index: " << jBest << endl;
            cout << "new route: ";
            for (auto s : solution->route) cout << s << ", ";
            cout << endl;
            cout << "new visited stops: ";
            for (int i = 0; i < solution->visited.size(); ++i) if (solution->visited[i]) cout << i << ", ";
            cout << endl;
            cout << "new value: " << solution->cost + solution->penalty - ((Solution_WL*)solution)->loss << " (Cost: " << solution->cost << "; Penalty: " << solution->penalty << "; Loss: " << ((Solution_WL*)solution)->loss << ")" << endl;
            cout << "---------------" << endl;
        }
        delete currentBestSolution;
        return true;
    }
    delete currentBestSolution;
    return false;
}


bool ILS_WL::removeNode(Params *params, Solution *solution) {
    if (params->is_timeout()) return false;
    auto *currentBestSolution = new Solution_WL(*((Solution_WL*)solution));
    int iBest = -1;

    for (int i = 1; i < solution->route.size() - 1; i++) {
        auto *currentSolution = new Solution_WL(*((Solution_WL*)solution));
        currentSolution->cost = currentSolution->cost
                                - params->distanceMatrix[currentSolution->route[i - 1]][currentSolution->route[i]]
                                - params->distanceMatrix[currentSolution->route[i]][currentSolution->route[i + 1]]
                                + params->distanceMatrix[currentSolution->route[i - 1]][currentSolution->route[i + 1]];
        currentSolution->penalty = currentSolution->penalty + params->penalties[currentSolution->route[i]];
        currentSolution->prize = currentSolution->prize - params->prizes[currentSolution->route[i]];
        currentSolution->visited[currentSolution->route[i]] = false;
        currentSolution->route.erase(currentSolution->route.begin() + i);
        ((Solution_WL*)currentSolution)->compute_loss(params);
        if (currentSolution->prize >= params->minPrize and (currentSolution->cost + currentSolution->penalty - ((Solution_WL*)currentSolution)->loss) < (currentBestSolution->cost + currentBestSolution->penalty - ((Solution_WL*)currentSolution)->loss)) {
            *currentBestSolution = *currentSolution;
            iBest = i;
        }
        delete currentSolution;
    }

    if (solution->route.size() >= 2) {
        // assess what happens when "i" is place at the beginning or at end
        // additional cost and index of "i" when it is placed at beginning
        pair<float,int> cost_ind_begin = make_pair(params->distanceMatrix[solution->route[0]][solution->route[1]], 0);
        // additional cost and index of "i" when it is placed at beginning
        pair<float,int> cost_ind_end = make_pair(params->distanceMatrix[solution->route.end()[-2]][solution->route.end()[-1]], solution->route.size() - 1);
        for (auto& cost_ind : {cost_ind_begin, cost_ind_end}) {
            auto *currentSolution = new Solution_WL(*((Solution_WL*)solution));
            currentSolution->cost = currentSolution->cost - cost_ind.first;
            currentSolution->penalty = currentSolution->penalty + params->penalties[currentSolution->route[cost_ind.second]];
            currentSolution->prize = currentSolution->prize - params->prizes[currentSolution->route[cost_ind.second]];
            currentSolution->visited[currentSolution->route[cost_ind.second]] = false;
            currentSolution->route.erase(currentSolution->route.begin() + cost_ind.second);
            ((Solution_WL*)currentSolution)->compute_loss(params);
            if (currentSolution->prize >= params->minPrize and (currentSolution->cost + currentSolution->penalty - ((Solution_WL*)currentSolution)->loss) < (currentBestSolution->cost + currentBestSolution->penalty - ((Solution_WL*)currentSolution)->loss)) {
                *currentBestSolution = *currentSolution;
                iBest = cost_ind.second;
            }
            delete currentSolution;
        }
    }

    if (iBest != -1) {
        if (params->verbose) cout << "remove stop at index: " << iBest << "(" << solution->route[iBest] << ")" << endl;
        *((Solution_WL*)solution) = *currentBestSolution;
        if (params->verbose) {
            cout << "new route: ";
            for (auto s : solution->route) cout << s << ", ";
            cout << endl;
            cout << "new visited stops: ";
            for (int i = 0; i < solution->visited.size(); ++i) if (solution->visited[i]) cout << i << ", ";
            cout << endl;
            cout << "new value: " << solution->cost + solution->penalty - ((Solution_WL*)solution)->loss << " (Cost: " << solution->cost << "; Penalty: " << solution->penalty << "; Loss: " << ((Solution_WL*)solution)->loss << ")" << endl;
            cout << "---------------" << endl;
        }
        delete currentBestSolution;
        return true;
    }
    delete currentBestSolution;
    return false;
}


bool ILS_WL::swapNodes(Params *params, Solution *solution) {
    if (params->is_timeout()) return false;
    auto *currentBestSolution = new Solution_WL(*((Solution_WL*)solution));
    int iBest = -1, jBest = -1;

    for (int i = 0; i < solution->route.size() - 1; i++) {
        for (int j = i + 1; j < solution->route.size(); j++) {

            auto *currentSolution = new Solution_WL(*((Solution_WL*)solution));
            currentSolution->cost = currentSolution->cost
                                    - ((i >= 1) ? params->distanceMatrix[currentSolution->route[i - 1]][currentSolution->route[i]] : 0)
                                    - ((j < currentSolution->route.size() - 1) ? params->distanceMatrix[currentSolution->route[j]][currentSolution->route[j + 1]] : 0)
                                    + ((i >= 1) ? params->distanceMatrix[currentSolution->route[i - 1]][currentSolution->route[j]] : 0)
                                    + ((j < currentSolution->route.size() - 1) ? params->distanceMatrix[currentSolution->route[i]][currentSolution->route[j + 1]] : 0);

            if (j != i + 1)
                currentSolution->cost = currentSolution->cost
                                        - params->distanceMatrix[solution->route[i]][solution->route[i + 1]]
                                        - params->distanceMatrix[solution->route[j - 1]][solution->route[j]]
                                        + params->distanceMatrix[solution->route[j]][solution->route[i + 1]]
                                        + params->distanceMatrix[solution->route[j - 1]][solution->route[i]];
            else // asymmetric problem since cost up is different from cost down and there are even timed by different weights
                currentSolution->cost = currentSolution->cost
                                        - params->distanceMatrix[solution->route[i]][solution->route[j]]
                                        + params->distanceMatrix[solution->route[j]][solution->route[i]];

            swap(currentSolution->route[i], currentSolution->route[j]);
            ((Solution_WL*)currentSolution)->compute_loss(params);


            if (currentSolution->cost - ((Solution_WL*)currentSolution)->loss < currentBestSolution->cost - ((Solution_WL*)currentSolution)->loss) {
                *currentBestSolution = *currentSolution;
                iBest = i;
                jBest = j;
            }
            delete currentSolution;
        }
    }

    if (iBest != -1) {
        if (params->verbose) cout << "swap stop at index: " << iBest << "(" << solution->route[iBest] << ")" << " with stop at index: " << jBest << "(" << solution->route[jBest] << ")" << endl;
        *((Solution_WL*)solution) = *currentBestSolution;
        if (params->verbose) {
            cout << "new route: ";
            for (auto s : solution->route) cout << s << ", ";
            cout << endl;
            cout << "new visited stops: ";
            for (int i = 0; i < solution->visited.size(); ++i) if (solution->visited[i]) cout << i << ", ";
            cout << endl;
            cout << "new value: " << solution->cost + solution->penalty - ((Solution_WL*)solution)->loss << " (Cost: " << solution->cost << "; Penalty: " << solution->penalty << "; Loss: " << ((Solution_WL*)solution)->loss << ")" << endl;
            cout << "---------------" << endl;
        }
        delete currentBestSolution;
        return true;
    }
    delete currentBestSolution;
    return false;
}

bool ILS_WL::twoOpt(Params *params, Solution *solution) {
    if (solution->route.size() < 3 or params->is_timeout()) return false;
    auto *currentBestSolution = new Solution_WL(*((Solution_WL*)solution));
    int iBest = -1, jBest = -1;

    for (int i = 0; i < solution->route.size() - 2; i++) {
        for (int j = i + 2; j < solution->route.size(); j++) {
            auto *currentSolution = new Solution_WL(*((Solution_WL*)solution));

            for (int k = i-1; k <= j; ++k) currentSolution->cost -= ((k >= 0 and k < currentSolution->route.size() - 1) ? params->distanceMatrix[currentSolution->route[k]][currentSolution->route[k+1]] : 0);

            for (int k = j; k > i; --k) currentSolution->cost += params->distanceMatrix[currentSolution->route[k]][currentSolution->route[k-1]];
            currentSolution->cost += ((i > 0) ? params->distanceMatrix[currentSolution->route[i - 1]][currentSolution->route[j]] : 0);
            currentSolution->cost += ((j < currentSolution->route.size() - 1) ? params->distanceMatrix[currentSolution->route[i]][currentSolution->route[j+1]] : 0);

            reverse(currentSolution->route.begin() + i, currentSolution->route.begin() + j + 1);
            ((Solution_WL*)currentSolution)->compute_loss(params);

            if (currentSolution->cost - ((Solution_WL*)currentSolution)->loss < currentBestSolution->cost - ((Solution_WL*)currentSolution)->loss) {
                *currentBestSolution = *currentSolution;
                iBest = i;
                jBest = j;
            }
            delete currentSolution;
        }
    }

    if (iBest != -1) {
        if (params->verbose) cout << "2opt stop at index: " << iBest << "(" << solution->route[iBest] << ")" << " with stop at index: " << jBest << "(" << solution->route[jBest] << ")" << endl;
        *((Solution_WL*)solution) = *currentBestSolution;
        if (params->verbose) {
            cout << "new route: ";
            for (auto s : solution->route) cout << s << ", ";
            cout << endl;
            cout << "new visited stops: ";
            for (int i = 0; i < solution->visited.size(); ++i) if (solution->visited[i]) cout << i << ", ";
            cout << endl;
            cout << "new value: " << solution->cost + solution->penalty - ((Solution_WL*)solution)->loss << " (Cost: " << solution->cost << "; Penalty: " << solution->penalty << "; Loss: " << ((Solution_WL*)solution)->loss << ")" << endl;
            cout << "---------------" << endl;
        }
        delete currentBestSolution;
        return true;
    }
    delete currentBestSolution;
    return false;
}


void ILS_WL::runILS(Params *params, Solution *solution) {
    auto *modifiedSolution = new Solution_WL(*((Solution_WL*)solution));
    int iterations = 0;
    int noImprov = 0;
    while (noImprov < params->maxNoImprov and iterations < params->maxIter and not params->is_timeout()) {
        perturbation(params, modifiedSolution);
        oneLocalSearch(params, modifiedSolution);
        iterations++;
        if (modifiedSolution->prize >= params->minPrize and modifiedSolution->cost + modifiedSolution->penalty - ((Solution_WL*)modifiedSolution)->loss < solution->cost + solution->penalty - ((Solution_WL*)solution)->loss) {
            *((Solution_WL*)solution) = *modifiedSolution;
            noImprov = 0;
        }
        else {
            *modifiedSolution = *((Solution_WL*)solution);
            noImprov++;
        }
    }
    delete modifiedSolution;
}


/*
void ILS_WL::doubleBridge(Params *params, Solution *solutionCandidate) {
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
}*/
