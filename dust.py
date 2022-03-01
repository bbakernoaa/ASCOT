#!/usr/bin/env python
# """
#     Dust Detection Algorithm using Hourly EPA AQS/AIRNOW Surface Monitors
#     Barry D. Baker <Barry.Baker@noaa.gov>
#     History: Created   2017
#     Usage: See Readme.md for further explanation

# __author__ = "Barry D. Baker"
# __copyright__ = "NOAA Air Resources Laboratory"
# __version__ = "1.0.1"
# __maintainer__ = "Barry D. Baker, RTi"
# __last_modified__   = "2017/10/05"
# __email__  = "Barry.Baker@noaa.gov"
# __status__ = "Dev"

# """
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def start_end_duration(df, column='TR4'):
    import pandas as pd
    from numpy import NaN, timedelta64
    s = df.copy().sort_values(['siteid', 'time'])
    # get start end date based on TRUE and FALSE events
    s['VALUE'] = (s[column].shift() - s[column] != 0).astype('int').cumsum()
    start = s.groupby(['VALUE', 'siteid']).time_local.first().reset_index()
    end = s.groupby(['VALUE', 'siteid']).time_local.last().reset_index()
    start.rename(columns={'time_local': 'START_DATE'}, inplace=True)
    end.rename(columns={'time_local': 'END_DATE'}, inplace=True)
    s = s.merge(start, on=['siteid', 'VALUE'])
    s = s.merge(end, on=['siteid', 'VALUE'])
    s['START_DATE'].loc[s[column] == False] = pd.NaT
    s['END_DATE'].loc[s[column] == False] = pd.NaT
    # Calculate duration of event from start and end date
    s['DURATION'] = (s.END_DATE - s.START_DATE) / timedelta64(1, 'h') + 1.
    return s


def dust_algorithm(df, lower_threshold=100., upper_threshold=140.):
    import pandas as pd
    from numpy import inf, NaN
    print('Ensuring Data Integrity...')
    df['RATIO'] = df.PM25 / df.PM10
    df.replace(
        [inf, -inf], NaN,
        inplace=True)  # make sure real numbers...replace to NaN if not
    df.drop_duplicates(subset=['time_local', 'siteid', 'PM10'], inplace=True)
    df.index = df.time_local
    # first get 3hr rolling stats
    g = df.groupby('siteid')
    # 3hr rolling window
    print('Calculating PM10 Rolling Mean...')
    rmin = g.PM10.rolling(3, center=True).mean().bfill().reset_index()
    print('Calculating PM10 Rolling Max...')
    rmax = g.PM10.rolling(3, center=True).max().bfill().reset_index()
    print('Calculating WS Rolling Max...')
    wsroll = g.WS.rolling(3).max().reset_index()
    rmin.rename(columns={'PM10': 'PM10_3HR_ROLL_MIN'}, inplace=True)
    rmax.rename(columns={'PM10': 'PM10_3HR_ROLL_MAX'}, inplace=True)
    wsroll.rename(columns={'WS': 'WS_ROLL'}, inplace=True)
    print('Merging Rolling Windows...')
    df = df.reset_index(drop=True)
    df = df.merge(
        rmin[['PM10_3HR_ROLL_MIN', 'siteid', 'time_local']],
        on=['siteid', 'time_local'],
        how='left')
    df = df.merge(
        rmax[['PM10_3HR_ROLL_MAX', 'siteid', 'time_local']],
        on=['siteid', 'time_local'],
        how='left')
    df = df.merge(
        wsroll[['WS_ROLL', 'siteid', 'time_local']],
        on=['siteid', 'time_local'],
        how='left')
    print('Setting Upper and Lower limits')
    df['PM10_LOWER_THR'] = lower_threshold
    df['PM10_UPPER_THR'] = 180.
    df['PM10_98_THR'] = 85.
    print('Getting Dust Levels 1-6...')
    df['G1'] = (df.PM10_3HR_ROLL_MIN > df.PM10_LOWER_THR) & (
        df.PM10_3HR_ROLL_MAX >= df.PM10_UPPER_THR)
    df['G2'] = (df.PM10_3HR_ROLL_MIN >
                df.PM10_LOWER_THR) & (df.PM10_3HR_ROLL_MAX >= upper_threshold)
    df['G3'] = (df.PM10_3HR_ROLL_MIN > df.PM10_98_THR) & (df.PM10_3HR_ROLL_MAX
                                                          >= upper_threshold)
    df['T1'] = df.G2 & (df.RATIO <= 0.35)
    df['T2'] = df.G2 & (df.RATIO <= 0.26)
    df['T3'] = df.G2 & (df.RATIO <= 0.2)
    dfs = []
    # set the WS dust
    df['G2+WS'] = df.G2 & (df.WS_ROLL > 7.3)
    df['T2+WS'] = df.T2 & (df.WS_ROLL > 7.3)
    df['G3+WS'] = df.G3 & (df.WS_ROLL > 7.3)
    df['T3+WS'] = df.T3 & (df.WS_ROLL > 7.3)
    # Fill 1 hr gaps
    print('Filling Gaps')
    df.loc[df.shift(1).G1 & df.shift(-1).G1, 'G1'] = True
    df.loc[df.shift(1).G3 & df.shift(-1).G3, 'G3'] = True
    df.loc[df.shift(1).T1 & df.shift(-1).T1, 'T1'] = True
    df.loc[df.shift(1).T3 & df.shift(-1).T3, 'T3'] = True

    df.loc[df.shift(1).T2 & df.shift(-1).T2, 'T2'] = True
    df.loc[df.shift(1).G2 & df.shift(-1).G2, 'G2'] = True
    df.loc[df.shift(1)['T2+WS'] & df.shift(-1)['T2+WS'], 'T2+WS'] = True
    df.loc[df.shift(1)['G2+WS'] & df.shift(-1)['G2+WS'], 'G2+WS'] = True
    df.loc[df.shift(1)['T2+WS'] & df.shift(-1)['T2+WS'], 'T2+WS'] = True
    df.loc[df.shift(1)['G2+WS'] & df.shift(-1)['G2+WS'], 'G2+WS'] = True

    df.loc[df.shift(1).T3 & df.shift(-1).T3, 'T3'] = True
    df.loc[df.shift(1).G3 & df.shift(-1).G3, 'G3'] = True
    df.loc[df.shift(1)['T3+WS'] & df.shift(-1)['T3+WS'], 'T3+WS'] = True
    df.loc[df.shift(1)['G3+WS'] & df.shift(-1)['G3+WS'], 'G3+WS'] = True
    df.loc[df.shift(1)['T3+WS'] & df.shift(-1)['T3+WS'], 'T3+WS'] = True
    df.loc[df.shift(1)['G3+WS'] & df.shift(-1)['G3+WS'], 'G3+WS'] = True

    # make final dust product
    print('Finalizing Method...')
    df['DUST'] = False
    df.loc[df.T2 | df.G2 | df['G2+WS'] | df['T2+WS'], 'DUST'] = True

    # rename Species to method
    df.rename(columns={'variable': 'Method'}, inplace=True)
    df.Method = 'NONE'
    df.loc[df.DUST & df.G2 & df['T2'] & ~df['T2+WS'] & ~df['G2+WS'],
           'Method'] = 'T2'
    df.loc[df.DUST & df.G2 & ~df['T2'] & ~df['T2+WS'] & ~df['G2+WS'],
           'Method'] = 'G2'
    df.loc[df.DUST & df.G2 & df['T2'] & df['T2+WS'] & df['G2+WS'],
           'Method'] = 'T2+WS'
    df.loc[df.DUST & df.G2 & ~df['T2'] & ~df['T2+WS'] & df['G2+WS'],
           'Method'] = 'G2+WS'
    # get start date for dust
    print('Getting Start Date and Duration...')
    df = start_end_duration(df, column='DUST')
    df.drop('END_DATE', axis=1, inplace=True)
    df.loc[df.DURATION > 72., 'DUST'] = False
    df.loc[df.DURATION > 72., 'G1'] = False
    df.loc[df.DURATION > 72., 'G2'] = False
    df.loc[df.DURATION > 72., 'G3'] = False
    df.loc[df.DURATION > 72., 'G2+WS'] = False
    df.loc[df.DURATION > 72., 'G3+WS'] = False
    df.loc[df.DURATION > 72., 'T2+WS'] = False
    df.loc[df.DURATION > 72., 'T3+WS'] = False
    df.loc[df.DURATION > 72., 'T2'] = False
    df.loc[df.DURATION > 72., 'T3'] = False
    df.loc[df.DURATION > 72., 'T1'] = False
    df.loc[df.DURATION > 72., 'START_DATE'] = pd.NaT
    df.loc[df.DURATION > 72., 'DURATION'] = NaN

    print('Setting Quality Flag...')
    #df = get_quality(df)
    #df.drop('VALUE', axis=1, inplace=True)
    print('BOOOOOOOM!!!!! DUST ALGORITHM COMPLETE')
    return df


def get_quality(df):
    """ There are 3 quality levels:
    0 :: No dust
    1 :: low confidence
    2 :: medium confidence
    3 :: high confidence

    High confidence can occur if;
        1) Only Ganor Method and 3hr rolling max is > 300 ug/m3,
        2) If 2 or more measurements are used and 3hr rolling max is greater than 250
        3) if PM10, PM2.5 and WS are used in combination
    Medium confidence
        1) if only Ganor method detects but 3hr rolling max > 200
        2) if 2 or more measurements are used and 3hr rolling max > 180
    Low confidence
        1) Only Ganor method used
        2) 2 or more measurements are used
    """
    df['QC'] = 0
    # Low confidence
    df.loc[df.Method == 'G2', 'QC'] = 1  # requirement 1
    df.loc[df.Method == 'G2+WS', 'QC'] = 1  # requirement 2
    df.loc[df.Method == 'T2', 'QC'] = 1  # requirement 3
    # medium confidence
    df.loc[(df.Method == 'G2') & (df.PM10_3HR_ROLL_MAX > 200.), 'QC'] = 2  # 1
    df.loc[(df.Method == 'G2+WS') & (df.PM10_3HR_ROLL_MAX > 180.),
           'QC'] = 2  # 2
    df.loc[(df.Method == 'T2') & (df.PM10_3HR_ROLL_MAX > 180.), 'QC'] = 2  # 2
    # high confidence
    df.loc[(df.Method == 'G2') & (df.PM10_3HR_ROLL_MAX > 300.), 'QC'] = 3  # 1
    df.loc[(df.Method == 'G2+WS') & (df.PM10_3HR_ROLL_MAX > 250.),
           'QC'] = 3  # 2
    df.loc[(df.Method == 'T2') & (df.PM10_3HR_ROLL_MAX > 250.), 'QC'] = 3  # 2
    df.loc[(df.Method == 'T2+WS'), 'QC'] = 3  # 3
    df.loc[~df.DUST, 'QC'] = 0
    return df


def get_monthly_quantile(df, quantile, col):
    df.index = df.time_local
    g = df.groupby('siteid')
    temp = g[col].resample('M').apply(
        lambda x: x.quantile(quantile)).reset_index()
    temp2 = g[col].resample('M').apply(
        lambda x: x.quantile(quantile)).reset_index()
    temp.index = temp.time_local
    temp2.index = temp2.time_local
    temp3 = temp.groupby('siteid').resample('H').max().bfill()[
        col].reset_index()
    temp4 = temp2.groupby('siteid').resample('H').max().ffill()[
        col].reset_index()
    con = ~temp4.time_local.isin(temp3.time_local)
    return temp4[con].append(temp3)


def patch_co(df, col):
    test = df[col] & (df.CO_ROLL_MAX > df.CO_THR)
    testm1 = df[col] & test.shift(-1)
    testm2 = df[col] & test.shift(-2)
    test1 = test.shift(1) & df[col]
    test2 = test.shift(2) & df[col]
    test3 = test.shift(3) & df[col]
    test4 = test.shift(4) & df[col]
    test5 = test.shift(5) & df[col]
    test6 = test.shift(6) & df[col]
    test7 = test.shift(7) & df[col]
    test8 = test.shift(8) & df[col]
    test9 = test.shift(9) & df[col]
    test10 = test.shift(10) & df[col]
    test11 = test.shift(11) & df[col]
    test12 = test.shift(12) & df[col]
    return test | test1 | test2 | test3 | test4 | test5 | test6 | test7 | test9 | test10 | test8 | test11 | test12 | testm1 | testm2


def get_and_clean(start='2017-01-01', end='2017-01-28', path='.', **kwargs):
    from monetio.obs import aqs as a
#    from monetio.util.tools import long_to_wide
    import pandas as pd
    from numpy import NaN
    # import aqs
    import os
    # a.monitor_file = 'monitoring_site_locations.hdf'
    a.datadir = path
    # os.chdir(a.datadir)
    a.dates = pd.date_range(start=start, end=end, freq='H')
    print('STARTING')
    df = a.add_data(a.dates, param=['PM10', 'WIND', 'RHDP', 'PM2.5'], **kwargs)
    print('AFTER ADD DATA')
    df.loc[df.obs < 0, 'obs'] = NaN
    df = long_to_wide(df.copy())
    df.dropna(subset=['PM10'], inplace=True)
    df.rename(columns={'PM2.5': 'PM25'}, inplace=True)
    # pm10 = a.df.groupby('variable').get_group('PM10').copy()
    # ws = a.df.groupby('variable').get_group('WS').copy()
    # wdir = a.df.groupby('variable').get_group('WD').copy()
    # rh = a.df.groupby('variable').get_group('RH').copy()
    # pm25 = a.df.groupby('variable').get_group('PM2.5').copy()
    #
    # print 'Filtering NaNs...'
    # pm10.loc[pm10.obs < 0, 'obs'] = NaN
    # #co.loc[co.obs < 0, 'obs'] = NaN
    # ws.loc[ws.obs < 0, 'obs'] = NaN
    # wdir.loc[wdir.obs < 0, 'obs'] = NaN
    # rh.loc[rh.obs < 0, 'obs'] = NaN
    # print 'Renaming Columns....'
    # pm10.rename(columns={'obs': 'PM10'}, inplace=True)
    # #co.rename(columns={'obs': 'CO'}, inplace=True)
    # ws.rename(columns={'obs': 'WS'}, inplace=True)
    # wdir.rename(columns={'obs': 'WD'}, inplace=True)
    # rh.rename(columns={'obs': 'RHUM'}, inplace=True)
    # print 'Merging Raw Datasets...'
    # df = pm10.dropna(subset=['PM10']).merge(ws[['WS', 'siteid', 'time_local']], on=['siteid', 'time_local'], how='left')
    # #df = df.merge(co[['CO', 'siteid', 'time_local']], on=['siteid', 'time_local'], how='left')
    # df = df.merge(wdir[['WD', 'siteid', 'time_local']], on=['siteid', 'time_local'], how='left')
    # df = df.merge(rh[['RHUM', 'siteid', 'time_local']], on=['siteid', 'time_local'], how='left')
    # if pm25.empty:
    #     df['PM25'] = NaN
    # else:
    #     print 'Merging PM2.5....'
    #     pm25.rename(columns={'obs': 'PM25'}, inplace=True)
    #     pm25.loc[pm25.PM25 < 0, 'PM25'] = NaN
    #     df = df.merge(pm25[['PM25', 'siteid', 'time_local']], on=['siteid', 'time_local'], how='left')
    # # os.chdir(a.cwd)
    # return df.drop_duplicates(subset=['time_local', 'siteid', 'PM10']).sort_values(['siteid', 'time_local'])
    return df


def get_and_clean_airnow(start='2017-06-01',
                         end='2017-06-28',
                         path='/data/aqf2/barryb/AIRNOW'):
    from monetio.obs import airnow as a
#    from monetio.util.tools import long_to_wide
    import os
    import pandas as pd
    from numpy import NaN
    #a = airnow.AirNow()
#    a.monitor_file = 'monitoring_site_locations.hdf'
    a.datadir = path
    print(start,end)
    a.dates = pd.date_range(start=start, end=end, freq='H')
    df = a.add_data(a.dates, download=False,wide_fmt=True,n_procs=4)
#    df = long_to_wide(df.copy())
    df.rename(columns={'PM2.5': 'PM25'}, inplace=True)
    # pm10 = a.df.groupby('variable').get_group('PM10')
    # co = a.df.groupby('variable').get_group('CO')
    # ws = a.df.groupby('variable').get_group('WS')
    # wdir = a.df.groupby('variable').get_group('WD')
    # pm25 = a.df.groupby('variable').get_group('PM2.5')
    # rh = a.df.groupby('variable').get_group('RHUM')
    # srad = a.df.groupby('variable').get_group('SRAD')
    # print 'Filtering NaNs...'
    # pm10.loc[pm10.obs < 0, 'obs'] = NaN
    # co.loc[co.obs < 0, 'obs'] = NaN
    # ws.loc[ws.obs < 0, 'obs'] = NaN
    # wdir.loc[wdir.obs < 0, 'obs'] = NaN
    # rh.loc[rh.obs < 0, 'obs'] = NaN
    # srad.loc[srad.obs < 0, 'obs'] = NaN
    # print 'Renaming Columns....'
    # pm10.rename(columns={'obs': 'PM10'}, inplace=True)
    # co.rename(columns={'obs': 'CO'}, inplace=True)
    # ws.rename(columns={'obs': 'WS'}, inplace=True)
    # wdir.rename(columns={'obs': 'WD'}, inplace=True)
    # rh.rename(columns={'obs': 'RHUM'}, inplace=True)
    # srad.rename(columns={'obs': 'SRAD'}, inplace=True)
    # print 'Merging Raw Datasets...'
    # df = pm10.dropna(subset=['PM10']).merge(ws[['WS', 'siteid', 'time_local']], on=['siteid', 'time_local'], how='left')
    # df = df.merge(co[['CO', 'siteid', 'time_local']], on=['siteid', 'time_local'], how='left')
    # df = df.merge(wdir[['WD', 'siteid', 'time_local']], on=['siteid', 'time_local'], how='left')
    # df = df.merge(rh[['RHUM', 'siteid', 'time_local']], on=['siteid', 'time_local'], how='left')
    # df = df.merge(srad[['SRAD', 'siteid', 'time_local']], on=['siteid', 'time_local'], how='left')
    # if pm25.empty:
    #     df['PM25'] = NaN
    # else:
    #     print 'Merging PM2.5....'
    #     pm25.rename(columns={'obs': 'PM25'}, inplace=True)
    #     pm25.loc[pm25.PM25 < 0, 'PM25'] = NaN
    #     df = df.merge(pm25[['PM25', 'siteid', 'time_local']], on=['siteid', 'time_local'], how='left')
    # os.chdir(a.cwd)
    # return df.drop_duplicates(subset=['time_local', 'siteid', 'PM10']).sort_values(['siteid', 'time_local'])
    df.dropna(subset=['PM10'], inplace=True)
    return df


def main():
    import sys
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='convert nemsio file to netCDF4 file', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--start', help='start date; YYYY-MM-DD', type=str, required=True)
    parser.add_argument('-e', '--end', help='start date; YYYY-MM-DD', type=str, required=True)
    parser.add_argument('-d', '--data', help='airnow or aqs', type=str, required=False, default='airnow')
    parser.add_argument('-o', '--output', help='output filename', type=str, required=True)
    args = parser.parse_args()
    # get data from command line
    start = args.start
    end = args.end
    data = args.data
    output = args.output

    print('Getting Data...')
    if data.lower() == 'airnow':
        df = get_and_clean_airnow(start=start, end=end, path='.')
    else:
        df = get_and_clean(start=start, end=end, path='.', download=True)

    dust = dust_algorithm(df)
    # G1,G2,G3,T1,T2,T3,G2+WS,T2+WS,G3+WS,T3+WS,DUST,Method,VALUE,START_DATE,DURATION
    dust = dust.loc[(dust.G1 == True) | (dust.G2 == True) | (dust.G3 == True) | (dust.G1 == True) | (dust.T2 == True) | (dust.T3 == True) | (dust['G2+WS'] == True ) | (dust['T2+WS'] == True) | (dust.DUST == True)]
    
    dust.to_csv(output, index=None)


if __name__ == "__main__":
    main()
