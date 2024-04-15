
verbose = [False]


def set_verbose(v):
    global verbose
    verbose[0] = v
    # print("Verbose mode set to", verbose)


def is_verbose():
    global verbose
    return verbose[0]

