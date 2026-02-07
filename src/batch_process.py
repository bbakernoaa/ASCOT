#!/user/bin/env python

import subprocess

import pandas as pd


def batch_process(start_date="2016-01-01", end_date="2022-02-01", freq="MS"):
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    processes = []
    for s in dates:
        start = s.strftime("%Y-%m-%d")
        end = (s + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
        output = s.strftime("data/%Y%m%d.dat")
        print(f"Launching dust.py for {start} to {end} -> {output}")
        p = subprocess.Popen(
            ["python", "./dust.py", "-s", start, "-e", end, "-d", "airnow", "-o", output]
        )
        processes.append(p)
    return processes


if __name__ == "__main__":
    batch_process()

# python ./dust.py -s '2022-02-24' -e '2022-02-25' -d 'airnow' -o test.csv
