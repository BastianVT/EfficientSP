//
// Created by Gael Aglin on 19/02/2023.
//

#ifndef PCTSP_ILS_WL_H
#define PCTSP_ILS_WL_H

#include "ILS.h"
#include "Solution_WL.h"

class ILS_WL : public ILS {

public:
    bool addNode(Params *params, Solution *solution) override;
    bool swapNodes(Params *params, Solution *solution) override;
    bool removeNode(Params *params, Solution *solution) override;
    bool twoOpt(Params *params, Solution *solution) override;

    explicit ILS_WL() = default;
    void runILS(Params *params, Solution *solution) override;
//    void doubleBridge(Params *params, Solution *solutionCandidate);
};


#endif //PCTSP_ILS_WL_H
