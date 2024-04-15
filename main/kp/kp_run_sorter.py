import csv

with open("kp_run") as file:
    lines = [list(map(lambda x: x.strip(), line.split(","))) for line in file.readlines() if "over" in line]
    lines = sorted(lines, key=lambda x: float(x[2]), reverse=True)

    with open("kp_run_sorted", "w") as file:
        writer = csv.writer(file)
        writer.writerows(lines)