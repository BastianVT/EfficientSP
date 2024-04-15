//
// Created by Gael Aglin on 19/02/2023.
//

#include "Params_WL.h"


Params_WL::Params_WL(bool v, int mi, int mni, int i, bool cd, const vector<int>& rr, int t) : Params(v, mi, mni, i, cd, t) {
    real_route = rr;
}