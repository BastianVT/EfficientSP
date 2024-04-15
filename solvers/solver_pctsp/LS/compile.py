import os
import platform
import shlex
import subprocess


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


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    ls_src_dir = "./"
    bindir = "./bin/"
    binpath = os.path.join(bindir, "pctsp_ls")
    os.makedirs(bindir, exist_ok=True)
    if not os.path.exists(binpath):
        compiler_ = get_compiler()
        if compiler_ == -1:  # In case no compiler exists
            print("No compiler found or it has not been added to the PATH variable")
        else:  # compile the local search code
            print("LS compilation started with:", compiler_)
            src_files = ["main.cpp", "argparse.cpp", "Params.cpp", "Params_WL.cpp", "Solution.cpp", "Solution_WL.cpp", "ILS.cpp", "ILS_WL.cpp", "utils.cpp"]
            compile_cmd = "{bin} -std=c++17 -O3 -march=native -fomit-frame-pointer -o {binpath} {files}".format(bin=compiler_, binpath=binpath, files=" ".join([(ls_src_dir + x) for x in src_files]))
            proc = subprocess.Popen(shlex.split(compile_cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc.communicate()
            print(out.decode(), err.decode())
            print("LS compilation finished")