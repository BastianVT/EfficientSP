# run the file `run_struct_perc.py` for different values of parameters `--theorem` and `--caching_strategy`
# perform each run 10 times

# set the number of runs
n_runs=10

# set the list of theorems
theorems=(1 2 3)


for (( run=1; run<=n_runs; run++ ))
do
    echo "Run $run for theorem 0 and no caching strategy"
    python3 run_struct_perc.py --caching_strategy NONE
    # add empty separator line
    echo ""
    echo ""

    for theorem in "${theorems[@]}"
    do
        echo "Run $run for theorem $theorem and amortized caching strategy"
        python3 run_struct_perc.py --theorem $theorem --caching_strategy AMORTIZED
        echo ""
        echo ""
    done
done