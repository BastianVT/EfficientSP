//
// Created by Gael Aglin on 19/02/2023.
//
#include "argparse.cpp"
#include "ILS.h"
#include "ILS_WL.h"
#include "utils.h"


int main(int argc, char** argv) {

    argparse::ArgumentParser program("pctsp");

//    program.add_argument("-f", "--file").help("Input file").default_value(string{"../../../data_v2/dist_abs/size_50/data_2.txt"});
    program.add_argument("-f", "--file").help("Input file").default_value(string{"../../PCTSP_instances/size_20/data_3.txt"});
    program.add_argument("-p", "--multiprizes").help("Whether the input file format contains multiple prizes").default_value(false).implicit_value(true);
    program.add_argument("-m", "--maxiter").help("Maximum iteration for the ILS_WL").default_value(400).scan<'d', int>();
    program.add_argument("-n", "--maxnoimprov").help("Maximum iteration after which to stop the ILS_WL if no improvement").default_value(200).scan<'d', int>();
    program.add_argument("-i", "--intensity").help("Intensity degree of the perturbation_WL").default_value(1).scan<'d', int>();
    program.add_argument("-t", "--timelimit").help("The maximum running time of the algorithm").default_value(numeric_limits<int>::infinity()).scan<'d', int>();
    program.add_argument("-v", "--verbose").help("Flag used to print details on the search").default_value(false).implicit_value(true);
    program.add_argument("-e", "--earlystop").help("Early stopping").default_value(false).implicit_value(true);
    program.add_argument("-d", "--classic").help("Use classic double bridge").default_value(false).implicit_value(true);
    program.add_argument("-l", "--floor").help("Use floor of float values").default_value(false).implicit_value(true);
//    program.add_argument("-w", "--weights").help("External weights for the problem").nargs(5).default_value(std::vector<float>{0.005033, 0.002467, -0.001174, -0.002074, 0.011888}).scan<'f', float>();
    program.add_argument("-w", "--weights").help("External weights for the problem").nargs(5).default_value(std::vector<float>{-1.0, -1.0, -1.0, -1.0, -1.0}).scan<'f', float>();
//    program.add_argument("-r", "--real").help("External weights for the problem").nargs(argparse::nargs_pattern::any).default_value(std::vector<int>{1, 2, 4, 9, 3, 8}).scan<'d', int>();
//    program.add_argument("-r", "--real").help("External weights for the problem").nargs(argparse::nargs_pattern::any).default_value(std::vector<int>{3, 18, 14, 16, 7, 19, 8, 10, 4, 11, 5, 15, 2, 0, 12, 1, 9}).scan<'d', int>();
    program.add_argument("-r", "--real").help("External weights for the problem").nargs(argparse::nargs_pattern::any).default_value(std::vector<int>{}).scan<'d', int>();
    program.add_argument("-s", "--initsol").help("Init solution").nargs(argparse::nargs_pattern::any).default_value(std::vector<int>{}).scan<'d', int>();
    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    auto filepath = program.get<string>("file");
    int maxIter = program.get<int>("maxiter");
    int maxNoImprov = program.get<int>("maxnoimprov");
    int intensity = program.get<int>("intensity");
    int timelimit = program.get<int>("timelimit");
    bool verbose = program.get<bool>("verbose");
    bool early_stopping = program.get<bool>("earlystop");
//    verbose = true;
    bool multiPrizes = program.get<bool>("multiprizes");
    bool classic_double = program.get<bool>("classic");
    bool use_floor = program.get<bool>("floor");
    auto ex_weights = program.get<vector<float>>("weights");
    auto real_route = program.get<vector<int>>("real");
    auto init_sol = program.get<vector<int>>("initsol");

    Params *params;
    if (real_route.empty()) params = new Params(verbose, maxIter, maxNoImprov, intensity, classic_double, timelimit);
    else params = new Params_WL(verbose, maxIter, maxNoImprov, intensity, classic_double, real_route, timelimit);


    if (multiPrizes) {
        vector<string> lines = readFile(filepath);
        if (lines.empty()) {
            cout << "File not found:" << filepath << endl;
            return 0;
        }
        buildInputsMultiPrizes(lines, params, ex_weights, use_floor); // the input format contains 2 prizes (e.g. data_In_multi_*.txt file)
    }
    else buildInputsDateAsPenalty(filepath, params, ex_weights, use_floor); // the input format contains 1 prize and 1 due_date penalty (e.g. data_In_datepenalty_*.txt file)

    Solution *solution = (real_route.empty()) ? new Solution(params->prizes.size(), early_stopping) : new Solution_WL(params->prizes.size(), early_stopping);
    solution->setInitSolution(params, init_sol);

    ILS *ils = (real_route.empty()) ? new ILS() : new ILS_WL();
    ils->oneLocalSearch(params, solution);
    ils->runILS(params, solution);


    duration<float> rtime = high_resolution_clock::now() - params->start_time;

    cout << "============= Problem params =============" << endl;
    cout << "Input file path: " << filepath << endl;
    cout << "nStops: " << params->prizes.size() << endl;
    cout << "Min prize required: " << params->minPrize << endl;
    if (multiPrizes) cout << "Input file format: MultiPrizes" << endl;
    else cout << "Input file format: DateAsPenalty" << endl;
    cout << "==========================================" << endl;
    cout << "=============== Solution_WL =================" << endl;
    solution->print();
    cout << "Timeout: " << (params->timeout ? "True" : "False") << endl;
    cout << "Runtime: " << (rtime).count() << "s" << endl;
    cout << "==========================================" << endl;
//    cout << "Runtime in Milliseconds: " << chrono::duration_cast<chrono::milliseconds>(rtime).count() << "ms" << endl;


    if (real_route.empty()) {
        delete (Solution*)solution;
        delete (Params*)params;
        delete (ILS*)ils;
    }
    else {
        delete (Solution_WL*)solution;
        delete (Params_WL*)params;
        delete (ILS_WL*)ils;
    }
    return 0;
}