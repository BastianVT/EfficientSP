//
// Created by Gael Aglin on 19/02/2023.
//

#ifndef PCTSP_UTILS_H
#define PCTSP_UTILS_H

#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iterator>
#include <algorithm>
#include <iostream>

#include "Params.h"

vector<string> readFile(const string& filepath);
void buildInputsMultiPrizes(const vector<string>& lines, Params* params, vector<float>& ex_weights, bool use_floor);
void buildInputsDateAsPenalty(const vector<string>& lines, Params* params, vector<float>& ex_weights, bool use_floor);
void buildInputsDateAsPenalty(const string& filepath, Params* params, const vector<float>& ex_weights, bool use_floor);

#endif //PCTSP_UTILS_H
