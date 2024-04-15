//
// Created by Gael Aglin on 19/02/2023.
//

#ifndef PCTSP_SOLUTION_H
#define PCTSP_SOLUTION_H

#include <iostream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <numeric>

#include "Params.h"


class Solution {

public:
    vector<int> route;
    float cost;
    float penalty;
    float prize;
    vector<bool> visited; // whether each stop is visited or not
    float init_obj;
    bool early_stopping;

    explicit Solution(int n, bool early_stopping = false);

    Solution(const Solution& other);

    Solution& operator=(const Solution& other);

    void setInitSolution(Params* params, vector<int>& init_sol);

    float get_in_out_cost(Params* params, int current_stop);


    void print();
};


#endif //PCTSP_SOLUTION_H
