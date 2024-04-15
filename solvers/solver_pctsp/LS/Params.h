//
// Created by Gael Aglin on 19/02/2023.
//

#ifndef PCTSP_PARAMS_H
#define PCTSP_PARAMS_H

#include <vector>
#include <chrono>

using namespace std;
using namespace chrono;

class Params {

public:
    float minPrize{};
    vector<float> prizes; // contains nstops vectors where each vector contains 2 values (prize, penalty)
    vector<float> penalties; // contains nstops vectors where each vector contains 2 values (prize, penalty)
    vector<vector<float>> distanceMatrix;
    bool verbose;
    int maxIter;
    int maxNoImprov;
    int intensity;
    bool classic_double;
    int timelimit;
    time_point<high_resolution_clock> start_time;
    bool timeout;

    Params(bool v, int mi, int mni, int i, bool cd, int t = numeric_limits<int>::infinity());

    bool is_timeout();
};


#endif //PCTSP_PARAMS_H
