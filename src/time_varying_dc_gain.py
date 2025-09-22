"""

computes time-varying DC gains for Solar PV, Wind Turbine and Battery

"""

import numpy as np
import pandas as pd


def get_wind_solar_dc_gains(get_probability_and_prices_distribution=True):
    """
    assumption: generation data is in 15 min intervals (see Fingrid website)

    returns production per 15-minute (meaned over the week) as DC gain
    """

    import pandas as pd

    df = pd.read_csv('data/data_wind_solar_2024_25.csv', sep=';')
    df.drop('endTime', inplace=True, axis=1)
    df.rename(columns={'startTime': 'Datum'}, inplace=True)
    df["Datum"] = pd.to_datetime(df["Datum"], format='mixed')
    df.set_index('Datum', inplace=True)
    df.columns = ['Wind', 'Solar']
    # get mean value
    df_mean = df.groupby([df.index.dayofweek, df.index.hour, df.index.minute]).mean()
    # normalize to 1
    df_mean = df_mean / df_mean.max()

    final_dfs = []
    if get_probability_and_prices_distribution:
        paths = ['data/data_ffr_fcr_procurement_2024_25.csv', 'data/data_ffr_fcr_d_price_2024_25.csv']
        for k, path in enumerate(paths):
            df = pd.read_csv(path, sep=';')
            df.drop('endTime', inplace=True, axis=1)
            df.rename(columns={'startTime': 'Datum'}, inplace=True)
            df["Datum"] = pd.to_datetime(df["Datum"], format='mixed')
            df.set_index('Datum', inplace=True)
            df = df.groupby([df.index.dayofweek, df.index.hour, df.index.minute]).mean()
            df.index = pd.MultiIndex.from_tuples(df.index, names=['Wochentag', 'Stunde', 'Minute'])
            if k==0: df = df / df.sum()
            # next, split up every row into 4 values for 15 min intervals
            df = pd.DataFrame(np.repeat(df.values, 4, axis=0), columns=df.columns)
            df.index = pd.RangeIndex(start=0, stop=len(df), step=1)
            if k==0: df = df / df.sum()
            final_dfs.append(df)

        return df_mean, final_dfs[0], final_dfs[1]

    return df_mean