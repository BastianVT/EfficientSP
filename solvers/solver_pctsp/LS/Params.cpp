//
// Created by Gael Aglin on 19/02/2023.
//

#include "Params.h"


Params::Params(bool v, int mi, int mni, int i, bool cd, int t) {
    verbose = v;
    maxIter = mi;
    maxNoImprov = mni;
    intensity = i;
    classic_double = cd;
    timelimit = t;
    start_time = high_resolution_clock::now();
    timeout = false;
}

bool Params::is_timeout() {
    if (timelimit == numeric_limits<int>::infinity()) return false;
    if (timeout) return true;
    else if (duration_cast<seconds>(high_resolution_clock::now() - start_time).count() >= timelimit) timeout = true;
    return timeout;
}