
cd "/Users/aglin/Python workspace/PreferenceDrivenOptimization/main/kp"

# violation + opti
echo "violation + opti"
python3 run_struct_perc.py

# use the classic SP
echo "use the classic SP"
python3 run_struct_perc.py --classic

cd "/Users/aglin/Python workspace/PreferenceDrivenOptimization/main/pctsp"

# violation + opti (disable ls violation)
echo "violation + opti (disable ls violation)"
python3 run_struct_perc.py --ls_violation

# use the classic SP
echo "use the classic SP"
python3 run_struct_perc.py --classic

# violation + ls
echo "violation + ls"
python3 run_struct_perc.py


