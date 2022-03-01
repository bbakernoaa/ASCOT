#!/user/bin/env python

import os
import subprocess
import pandas as pd

dates = pd.date_range(start='2016-01-01',end='2022-02-01',freq='MS')


print(dates)
for s in dates:
    print(s)
    start=s.strftime("%Y-%m-%d")
    end = s + pd.DateOffset(months=1)
    print(end)
    end = end.strftime('%Y-%m-%d')
    output = s.strftime('data/%Y%m%d.dat')
    subprocess.Popen(['python','./dust.py','-s',start,'-e',end,'-d','airnow','-o',output])
    
# python ./dust.py -s '2022-02-24' -e '2022-02-25' -d 'airnow' -o test.csv
