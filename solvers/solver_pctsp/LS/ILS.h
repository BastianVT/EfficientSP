//
// Created by Gael Aglin on 19/02/2023.
//

#ifndef PCTSP_ILS_H
#define PCTSP_ILS_H


#include <random>
#include <iterator>
#include <algorithm>

#include "Solution.h"


class ILS {

    bool virtual addNode(Params *params, Solution *solution);
    bool virtual swapNodes(Params *params, Solution *solution);
    bool virtual removeNode(Params *params, Solution *solution);
    bool virtual twoOpt(Params *params, Solution *solution);

public:
    explicit ILS() = default;
    void oneLocalSearch(Params *params, Solution *solution);
    virtual void runILS(Params *params, Solution *solution);
    void perturbation(Params *params, Solution *solutionCandidate);
    void doubleBridge(Params *params, Solution *solutionCandidate);

};


#endif //PCTSP_ILS_H
