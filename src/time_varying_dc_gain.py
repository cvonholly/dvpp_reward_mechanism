"""

computes time-varying DC gains for Solar PV, Wind Turbine and Battery

"""

import numpy as np
import pandas as pd


def get_wind_solar_dc_gains(get_probability_distribution=True):
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

    if get_probability_distribution:
        df = pd.read_csv('data/data_ffr_fcr_procurement_2024_25.csv', sep=';')
        df.drop('endTime', inplace=True, axis=1)
        df.rename(columns={'startTime': 'Datum'}, inplace=True)
        df["Datum"] = pd.to_datetime(df["Datum"], format='mixed')
        df.set_index('Datum', inplace=True)
        df_mean_probs = df.groupby([df.index.dayofweek, df.index.hour, df.index.minute]).mean()
        df_mean_probs.index = pd.MultiIndex.from_tuples(df_mean_probs.index, names=['Wochentag', 'Stunde', 'Minute'])
        df_prob = df_mean_probs / df_mean_probs.sum()
        # next, split up every row into 4 values for 15 min intervals
        df_prob_15min = pd.DataFrame(np.repeat(df_prob.values, 4, axis=0), columns=df_prob.columns)
        df_prob_15min.index = pd.RangeIndex(start=0, stop=len(df_prob_15min), step=1)
        df_prob_15min = df_prob_15min / df_prob_15min.sum()

        return df_mean, df_prob_15min
    
    return df_mean