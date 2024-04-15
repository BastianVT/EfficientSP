//
// Created by Gael Aglin on 19/02/2023.
//

#ifndef PCTSP_SOLUTION_WL_H
#define PCTSP_SOLUTION_WL_H


#include <iterator>
#include <algorithm>

#include "Solution.h"
#include "Params_WL.h"


class Solution_WL : public Solution {

public:
    int loss;

    explicit Solution_WL(int n, bool early_stopping = false);

    Solution_WL(const Solution_WL& other);

    Solution_WL& operator=(const Solution_WL& other);

    void compute_loss(const Params* params);

    static vector<vector<bool>> successor_matrix(const vector<int>& s, int n);
};


#endif //PCTSP_SOLUTION_WL_H
