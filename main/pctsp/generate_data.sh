#!/bin/bash

n_threads=0 # set to 0 to use all available threads
out_dir="../logs_nthreads_$n_threads"
mkdir $out_dir

#sizes=(10 20 50 75 100 125)
sizes=(75 100)
times=(30 60)
algos=(CP LS)

# set --jobs to 0 to run as many jobs as possible in parallel
parallel --jobs 30 --line-buffer "
echo python dataset_generator.py {1} {2} {3} $n_threads to $out_dir/generate_out_{1}_t{2}_s{3}.log
python dataset_generator.py --algo {1} --time_limits {2} --sizes {3} $n_threads > $out_dir/generate_out_{1}_t{2}_s{3}.log
" ::: "${algos[@]}" ::: "${times[@]}" ::: "${sizes[@]}"

