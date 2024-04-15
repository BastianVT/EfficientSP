import shlex
import argparse
import platform
import traceback
import subprocess
import numpy as np
import pandas as pd
from distutils import util

# add the root directory to the python path so that we can import modules from there
import sys
import pathlib
root_path = pathlib.Path(__file__).resolve().parent.parent.parent.absolute()

sys.path.insert(0, str(root_path))

# set the current working directory to the current file's directory
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# from solvers.solver_pctsp.CP import PCTSPSubPath, decode_subpath

from solvers.solver_pctsp.MILP import PCTSPSubPath, decode_subpath
from solvers.solver_pctsp import InputFormat, read_data_downloaded, read_data_multiprizes, read_data_datepenalty, evaluate


def get_compiler():
    # Determine the operating system
    operating_system = platform.system()
    # Set the list of C++ compiler executables to search for based on the operating system
    compilers = ['g++.exe', 'clang++.exe', 'cl.exe'] if operating_system == 'Windows' else ['g++', 'clang++']
    # Use the `which` or `where` command to search for each compiler executable
    for compiler in compilers:
        try:
            result = subprocess.run(['where' if operating_system == 'Windows' else 'which', compiler], check=True, stdout=subprocess.PIPE)
            return result.stdout.decode().strip().split(".")[0]
        except subprocess.CalledProcessError:
            continue
    # if no compiler was found
    return -1


def solution_exist(file_path):
    df = pd.read_csv(f_path)
    result = df.loc[(df['instance'] == file_path.replace("../../instances", "../../instances"))
                    & (df['size'] == size)
                    & (df['timelimit'] == time_limit)
                    & (df['pref_w1'] == tweights[0])
                    & (df['pref_w2'] == tweights[1])
                    & (df['pref_w3'] == tweights[2])
                    & (df['pref_w4'] == tweights[3])
                    & (df['pref_p'] == penalty)
                    & (df['sol_cost'] == int(cp_cost))
                    & (df['sol_prize'] == int(cp_prize))
                    & (df['sol_penalty'] == int(cp_penalty))]
    return len(result) > 0
    # return result.empty


parser = argparse.ArgumentParser()
parser.add_argument('--data_generation', action='store_true', help='Enable data generation')
parser.add_argument('--algo', choices=['CP', 'LS', None], default='CP', help='Algorithm name')
parser.add_argument('--time_limits', type=int, nargs='+', default=[300], help='Time limits')  # [60, 180, 300]
parser.add_argument('--sizes', type=int, nargs='+', default=[50], help='Sizes')
parser.add_argument('--n_threads', type=int, default=0, help='Number of threads')
parser.add_argument('--noise_level', type=float, default=0, help='Noise level')

# local search parameters
parser.add_argument('--multi_prizes', action='store_true', help='Whether the input instances use several prizes')
parser.add_argument('--max_iter', type=int, default=400, help='Number of iterations for the iterated local search')
parser.add_argument('--max_no_improv', type=int, default=200, help='Number of iterations after which to stop if no improvement')
parser.add_argument('--intensity', type=int, default=1, help='Intensity degree of perturbations')
parser.add_argument('--classic_doublebridge', action='store_true', help='Perform a classic double bridge perturbation')
args = parser.parse_args()

# data_generation = args.data_generation
data_generation = True
sizes = args.sizes
time_limits = args.time_limits
algo = args.algo
n_threads = args.n_threads
noise_level = args.noise_level

multi_prizes = args.multi_prizes
max_iter = args.max_iter
max_no_improv = args.max_no_improv
intensity = args.intensity
classic_doublebridge = args.classic_doublebridge


for time_limit in time_limits:
    for size in sizes:

        init_weights = list(map(lambda x: int(round(x, 0)), (np.random.dirichlet(np.ones(4), size=1) * 100).tolist()[0]))
        while sum(init_weights) != 100:
            init_weights = list(map(lambda x: int(round(x, 0)), (np.random.dirichlet(np.ones(4), size=1) * 100).tolist()[0]))
        
        init_w1, init_w2, init_w3, init_w4 = init_weights
        init_p = np.random.randint(1, 100 + 1)
        
        file_dir = "../../instances/pctsp/noisy/size_n01_{}".format(size)
        for data_index in range(1, len(os.listdir(file_dir)) + 1):
            file_path = file_dir + "/data_{}.txt".format(data_index)
            file_format = InputFormat.DateAsPenalty
            filereader = read_data_downloaded if file_format == InputFormat.Downloaded else read_data_multiprizes if file_format == InputFormat.MultiPrizes else read_data_datepenalty

            # Read input data
            if data_generation:
                prizes, penalties, distance_matrix, minprize, penalty, tweights = filereader(file_path, return_weights=True)
                if noise_level > 0:
                    tweights = (np.array(tweights) + np.random.uniform(-noise_level, noise_level, len(tweights))).tolist()
                    penalty = penalty + np.random.uniform(-noise_level, noise_level)
                    prizes, penalties, distance_matrix, minprize, penalty, tweights = filereader(file_path, return_weights=True, matrix_weights=tweights, penalty_weight=penalty)
            else:
                prizes, penalties, distance_matrix, minprize = filereader(file_path)

            # Print problem's params
            print("============= Problem params ================")
            print("Input file path:", file_path)
            print("nStops:", len(prizes))
            print("Min Prize required:", minprize)
            # print("Input file format:", file_format)
            print("=============================================")

            if algo in ['CP', None]:
                # Run the CP solver and get stats
                try:
                    # raise Exception()  # in case the CP solver maybe too long, and you want to skip its run

                    # Initialize solver
                    pctsp = PCTSPSubPath(minprize=minprize, precision=1, time_limit=time_limit)
                    k=1
                    results = pctsp.solve(prizes, penalties, distance_matrix, k, n_threads=n_threads)

                    print("=============== Solution CP =================")
                    sol, outs = decode_subpath(results['store'])
                    cp_route = " -> ".join(map(str, sol))
                    # cp_out = ", ".join([str(i) for i, j in enumerate(outs.tolist()) if j is True])
                    cp_out = ", ".join([str(i) for i in outs.tolist()])
                    print("Route found:", cp_route)
                    print("Unselected stops:", cp_out)
                    print("Collected prize:", results['total_prizes'])
                    print("Transition cost:", results['total_travel'])
                    print("Total penalty:", results['total_penalties'])
                    print("Objective value:", results['total_travel'] + results['total_penalties'])
                    print("Timeout: ", not results['optimal'], sep="")
                    print("Runtime: ", round(results['runtime'], 2), "s", sep="")
                    print("=============================================")
                    # get statistics from an independent function instead of trusting the algorithm's output
                    cp_prize, cp_penalty, cp_cost, cp_obj = evaluate(cp_route, "->", prizes, penalties, distance_matrix)
                    vals = iter([cp_obj, cp_prize, cp_cost, cp_penalty, cp_route])

                    if data_generation:
                        pathlib.Path("../../datasets/pctsp/nthreads_{}".format(n_threads)).mkdir(parents=True, exist_ok=True)
                        f_path = "../../datasets/pctsp/nthreads_{}/cp_training_n01_t{}_s{}{}.csv".format(n_threads, time_limit, size, "_noise_{}".format(noise_level) if noise_level > 0 else "")
                        file_exists = os.path.exists(f_path)
                        if not file_exists or not solution_exist(f_path):
                            with open(f_path, "a") as outfile:
                                if not file_exists:
                                    outfile.write("instance,size,init_w1,init_w2,init_w3,init_w4,init_p,sol_cost,sol_prize,sol_penalty,timelimit,runtime,timeout,in_seq,out_seq,pref_w1,pref_w2,pref_w3,pref_w4,pref_p\n")
                                write = [file_path.replace("../../instances", "../../instances"), str(size), str(init_w1), str(init_w2), str(init_w3), str(init_w4), str(init_p), str(cp_cost), str(cp_prize), str(cp_penalty), str(time_limit), str(round(results['runtime'], 2)), str(int(not results['optimal']))] + [cp_route.replace(" -> ", "-"), cp_out.replace(", ", "-")] + [str(x) for x in tweights] + [str(penalty)]
                                outfile.write(",".join(write) + "\n")
                                outfile.flush()

                except Exception as e:
                    cp_route = "error"
                    print("CP solving has produced the following error:")
                    print(str(e))
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    # uncomment this line to see the error stack trace
                    traceback.print_exc()

            if algo in ['LS', None]:
                # if the C++ binary for the local search does not exist, compile it if a C++ compiler exists on the computer
                if not os.path.exists("../../solvers/solver_pctsp/LS/bin/pctsp_ls"):
                    compiler_ = get_compiler()
                    if compiler_ == -1:  # In case no compiler exists
                        print("No compiler found or it has not been added to the PATH variable")
                        continue
                    else:  # compile the local search code
                        print("LS compilation started with:", compiler_)
                        pathlib.Path("../../solvers/solver_pctsp/LS/bin").mkdir(parents=True, exist_ok=True)
                        src_files = ["main.cpp", "argparse.cpp", "Params.cpp", "Params_WL.cpp", "Solution.cpp", "Solution_WL.cpp", "ILS.cpp", "ILS_WL.cpp", "utils.cpp"]
                        compile_cmd = "{bin} -std=c++17 -O3 -march=native -fomit-frame-pointer -o ../../solvers/solver_pctsp/LS/bin/pctsp_ls {files}".format(bin=compiler_, files=" ".join(["../../LS/" + f for f in src_files]))
                        proc = subprocess.Popen(shlex.split(compile_cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        proc.communicate()

                # Run the LS algo and get stats
                cmd = "../../solvers/solver_pctsp/LS/bin/pctsp_ls -f {} -m {} -n {} -i {}".format(file_path, max_iter, max_no_improv, intensity) + (" -t {}".format(time_limit) if time_limit is not None else "") + (" -p" if multi_prizes else "") + (" -d" if classic_doublebridge else "")
                proc_ls = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
                out, err = proc_ls.communicate()
                output = out.decode()
                ls_route = output.split("\n")[7].split(": ")[1].strip()
                ls_out = output.split("\n")[8].split(": ")[1].strip()

                # print the local search solution
                to_print = output.split("\n")[6:]
                to_print[0] = to_print[0].replace("Solution", "Solution LS")
                to_print[-2] = "===" + to_print[-2]
                print("\n".join(to_print), sep="", end="")

                # get tle local search statistics from an independent function instead of trusting the algorithm's output
                ls_prize, ls_penalty, ls_cost, ls_obj = evaluate(ls_route, "->", prizes, penalties, distance_matrix)
                vals = iter([ls_obj, ls_prize, ls_cost, ls_penalty, ls_route])

                if data_generation:
                    pathlib.Path("../../datasets/pctsp/nthreads_{}".format(n_threads)).mkdir(parents=True, exist_ok=True)
                    f_path = "../../datasets/pctsp/nthreads_{}/ls_training_t{}_s{}{}.csv".format(n_threads, time_limit, size, "_noise_{}".format(noise_level) if noise_level > 0 else "")
                    file_exists = os.path.exists(f_path)
                    if not file_exists or not solution_exist(f_path):
                        with open(f_path, "a") as outfile:
                            if not file_exists:
                                outfile.write("instance,size,init_w1,init_w2,init_w3,init_w4,init_p,sol_cost,sol_prize,sol_penalty,timelimit,runtime,timeout,in_seq,out_seq,pref_w1,pref_w2,pref_w3,pref_w4,pref_p\n")
                            write = [file_path.replace("../../instances", "../../instances"), str(size), str(init_w1), str(init_w2), str(init_w3), str(init_w4), str(init_p), str(ls_cost), str(ls_prize), str(ls_penalty), str(time_limit), str(output.split("\n")[14].split(": ")[1][:-1]), str(int(bool(util.strtobool(output.split("\n")[13].split(": ")[1]))))] + [ls_route.replace(" -> ", "-"), ls_out.replace(", ", "-")] + [str(x) for x in tweights] + [str(penalty)]
                            outfile.write(",".join(write) + "\n")
                            outfile.flush()

            # print recap of both algorithms
            print("Recap")
            if algo in ['CP', None]:
                if cp_route != "error":
                    print("\t- CP evaluation:", end=" ")
                    print("Objective:", cp_obj, "--", "Transition cost:", cp_cost, "--", "Penalty:", cp_penalty, "--", "Prize:", cp_prize)
            if algo in ['LS', None]:
                print("\t- LS evaluation:", end=" ")
                print("Objective:", ls_obj, "--", "Transition cost:", ls_cost, "--", "Penalty:", ls_penalty, "--", "Prize:", ls_prize)

            print("\n\n")
