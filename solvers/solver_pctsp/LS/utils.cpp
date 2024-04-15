//
// Created by Gael Aglin on 19/02/2023.
//

#include "utils.h"


vector<string> readFile(const string& filepath) {
    vector<string> lines;
    ifstream inFile(filepath);

    if (not inFile) return {}; // file not found

    if (inFile.good()) {
        string sLine;

        int j = 0;
        while (getline(inFile, sLine)) {
            if (sLine.length() != 0 && sLine != "\r") {
                lines.push_back(sLine);
                j++;
            }
        }
    }
    inFile.close();
    return lines;
}

void buildInputsMultiPrizes(const vector<string> &lines, Params *params, vector<float>& ex_weights, bool use_floor) {
    istringstream iss0(lines[0]);
    vector<string> prof_weights{(istream_iterator<string>(iss0)), istream_iterator<string>() };

    // first prize
    istringstream iss(lines[1]);
    vector<string> prizes{(istream_iterator<string>(iss)), istream_iterator<string>() };
    for (auto & prize : prizes) {
        params->prizes.push_back(stof(prize) * stof(prof_weights[0]));
    }

    // other prizes
    for (int i = 1; i < prof_weights.size(); ++i) {
        istringstream iss1(lines[i+1]);
        vector<string> date_prizes{(istream_iterator<string>(iss1)), istream_iterator<string>() };
        for (int j = 0; j < date_prizes.size(); ++j) {
            params->prizes[j] += stof(date_prizes[j]) * stof(prof_weights[i]);
        }
    }

    float penalty_weight = (ex_weights[0] == -1) ? stof(lines[prof_weights.size() + 1]) : ex_weights.end()[-1];

    istringstream iss2(lines[prof_weights.size() + 2]);
    vector<string> penalties{(istream_iterator<string>(iss2)), istream_iterator<string>() };
    for (auto & penalty : penalties) {
        params->penalties.push_back(round(stof(penalty) * penalty_weight));
    }

    istringstream iss3(lines[prof_weights.size() + 3]);
    vector<string> weights{(istream_iterator<string>(iss3)), istream_iterator<string>() };

    vector<float> weights_float;
    if (ex_weights[0] != -1) weights_float = {ex_weights.begin(), ex_weights.begin() + 4};
    else transform(weights.begin(), weights.end(), std::back_inserter(weights_float),
                   [](const string& str) { return stof(str); });

    params->minPrize = stof(lines[prof_weights.size() + 4]);

    int init = prof_weights.size() + 5;
    for (int i = 0; i < weights_float.size(); ++i) { // every matrix
        for (int j = 0; j < params->prizes.size(); j++) { // every line of every matrix
            if (i == 0) params->distanceMatrix.emplace_back();
            istringstream iss4(lines[init]);
            vector<string> distances{(istream_iterator<string>(iss4)), istream_iterator<string>() };
            for (int k = 0; k < distances.size(); k++) {
                if (i == 0) params->distanceMatrix[j].push_back(weights_float[i] * stof(distances[k]));
                else params->distanceMatrix[j][k] += weights_float[i] *  stof(distances[k]);
            }
            init++;
        }

    }

    if (use_floor) {
        for (auto & row : params->distanceMatrix) for (auto & cell : row) cell = floor(cell);
        for (auto & val: params->penalties) val = floor(val);
    }
}

void buildInputsDateAsPenalty(const vector<string> &lines, Params *params, vector<float>& ex_weights, bool use_floor) {

    // native prize
    istringstream iss(lines[0]);
    vector<string> prizes{(istream_iterator<string>(iss)), istream_iterator<string>() };
    for (auto & prize : prizes) {
        params->prizes.push_back(stof(prize));
    }

    float penalty_weight = (ex_weights[0] == -1) ? stof(lines[1]) : ex_weights.end()[-1];

    // date prize as penalty
    istringstream iss1(lines[2]);
    vector<string> penalties{(istream_iterator<string>(iss1)), istream_iterator<string>() };
    for (auto & penalty : penalties) {
        params->penalties.push_back(round(stof(penalty) * penalty_weight));
    }

    istringstream iss2(lines[3]);
    vector<string> weights{(istream_iterator<string>(iss2)), istream_iterator<string>() };

    vector<float> weights_float;
    if (ex_weights[0] != -1) weights_float = {ex_weights.begin(), ex_weights.begin() + 4};
    else transform(weights.begin(), weights.end(), std::back_inserter(weights_float),
                   [](const string& str) { return stof(str); });

    params->minPrize = stof(lines[4]);

    int init = 5;
    for (int i = 0; i < weights_float.size(); ++i) { // every matrix
        for (int j = 0; j < params->prizes.size(); j++) { // every line of every matrix
            if (i == 0) params->distanceMatrix.emplace_back();
            istringstream iss4(lines[init]);
            vector<string> distances{(istream_iterator<string>(iss4)), istream_iterator<string>() };
            for (int k = 0; k < distances.size(); k++) {
                if (i == 0) params->distanceMatrix[j].push_back(weights_float[i] * stof(distances[k]));
                else params->distanceMatrix[j][k] += weights_float[i] *  stof(distances[k]);
            }
            init++;
        }
    }

    if (use_floor) {
        for (auto & row : params->distanceMatrix) for (auto & cell : row) cell = floor(cell);
        for (auto & val: params->penalties) val = floor(val);
    }
}


void buildInputsDateAsPenalty(const string& filepath, Params* params, const vector<float>& ex_weights, bool use_floor) {
    ifstream inputFile(filepath);
    if (!inputFile.is_open()) {
        cerr << "Error opening file " << filepath << endl;
        exit(1);
    }

    string line;
    float value;

    // Read the prizes vector
    getline(inputFile, line);
    istringstream ss(line);
    while (ss >> value) {
        params->prizes.push_back(value);
    }
    getline(inputFile, line);  // Skip the empty line

    // Read the penalty weight
    getline(inputFile, line);
    float penalty_weight = (ex_weights[0] == -1) ? stof(line) : ex_weights.end()[-1];
    getline(inputFile, line);  // Skip the empty line

    // Read the penalties vector
    getline(inputFile, line);
    ss = istringstream(line);
    while (ss >> value) {
        params->penalties.push_back(value * penalty_weight);
    }
    getline(inputFile, line);  // Skip the empty line

    // Read the weights vector
    vector<float> distance_weights;
    getline(inputFile, line);
    if (ex_weights[0] != -1) {
        distance_weights = {ex_weights.begin(), ex_weights.begin() + 4};
    } else {
        ss = istringstream(line);
        while (ss >> value) {
            distance_weights.push_back(value);
        }
    }
    getline(inputFile, line);  // Skip the empty line

    // Read the minPrize value
    getline(inputFile, line);
    params->minPrize = stof(line);
    getline(inputFile, line);  // Skip the empty line

    // Read the distance matrices
    vector<vector<float>> matrices(params->prizes.size(), vector<float>(params->prizes.size(), 0.0));
    for (int n = 0; n < 4; n++) {
        int i = 0;
        while (getline(inputFile, line) and not line.empty()) {
            ss = istringstream(line);
            int j = 0;
            while (ss >> value) {
                matrices[i][j] += distance_weights[n] * value;
                j++;
            }
            i++;
        }
    }
    params->distanceMatrix = matrices;

    if (use_floor) {
        for (auto & row : params->distanceMatrix) for (auto & cell : row) cell = floor(cell);
        for (auto & val: params->penalties) val = floor(val);
    }
}
