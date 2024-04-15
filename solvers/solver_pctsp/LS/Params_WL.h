//
// Created by Gael Aglin on 19/02/2023.
//

#ifndef PCTSP_PARAMS_WL_H
#define PCTSP_PARAMS_WL_H

#include "Params.h"

class Params_WL : public Params {

public:
    vector<int> real_route;

    Params_WL(bool v, int mi, int mni, int i, bool cd, const vector<int>& rr, int t = numeric_limits<int>::infinity());
};


#endif //PCTSP_PARAMS_WL_H
