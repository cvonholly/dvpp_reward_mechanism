"""

computes time-varying DC gains for Solar PV, Wind Turbine and Battery

"""

import numpy as np
import pandas as pd

def get_wind_solar_dc_gains(get_probability_and_prices_distribution=True,
                            path='data/data_wind_solar_2024_25.csv',
                            hourly_average=True):
    """
    gets production data for entire year
    - hourly_average: if True, averages the data to hourly values

    returns:
    - production per 15-minute as DC gain, meaned by max value
    - (optional) probability distribution of FFR and FCR procurement
    - (optional) price distribution of FFR and FCR-D prices
    """
    df = pd.read_csv(path, sep=';')
    df.drop('endTime', inplace=True, axis=1)
    df.rename(columns={'startTime': 'Datum'}, inplace=True)
    df["Datum"] = pd.to_datetime(df["Datum"], format='mixed')
    df.set_index('Datum', inplace=True)
    df.columns = ['Wind', 'Solar']
    # normalize by max values
    max_vals = df.max()
    df_mean = df / max_vals
    # remove time zone info
    time_stamps = [ts.tz_localize(None) for ts in df_mean.index]
    df_mean.index = time_stamps
    if hourly_average:
        df_mean = df_mean.resample('h').mean()

    final_dfs = []
    if get_probability_and_prices_distribution:
        paths = ['data/data_ffr_fcr_procurement_2024_25.csv', 'data/data_ffr_fcr_d_price_2024_25.csv']
        for k, path in enumerate(paths):
            # this data is already in hourly format
            df = pd.read_csv(path, sep=';')
            df.drop('endTime', inplace=True, axis=1)
            df.rename(columns={'startTime': 'Datum',
                               'Fast Frequency Reserve FFR, price': 'FFR_price',
                               'Frequency Containment Reserve for Disturbances upwards regulation, hourly market prices': 'FCR_D_up_price'}, 
                               inplace=True)
            df["Datum"] = pd.to_datetime(df["Datum"], format='mixed')
            df.set_index('Datum', inplace=True)
            if k==0: df = df / df.sum()
            # next, split up every row into 4 values for 15 min intervals
            if not hourly_average:
                df = pd.DataFrame(np.repeat(df.values, 4, axis=0), columns=df.columns)
            df.index = [ts.tz_localize(None) for ts in df.index]
            #  df.index = pd.RangeIndex(start=0, stop=len(df), step=1)
            if k==0: df = df / df.sum()
            final_dfs.append(df)
        # ensure they have same size
        minlen = min([len(d) for d in final_dfs])
        minlen = min(minlen, len(df_mean))
        df_mean = df_mean.iloc[:minlen]
        for df in final_dfs:
            df = df.iloc[:minlen]
            if not hourly_average:
                df.index = df_mean.index
                
        return df_mean, final_dfs[0], final_dfs[1]

    return df_mean

def datetime_to_idx(datetime_idx):
    """
    convert datetime to index range 0-671 representing:
        day of week (0-6) * 24 * 4 + hour of day (0-23) * 4 + quarter of hour (0-3)
    """
    day_of_week = datetime_idx.weekday()  # Monday=0, Sunday=6
    hour_of_day = datetime_idx.hour
    quarter_of_hour = datetime_idx.minute // 15
    index = day_of_week * 24 * 4 + hour_of_day * 4 + quarter_of_hour
    return index


def get_wind_solar_dc_gains_weekly(get_probability_and_prices_distribution=True):
    """
    means production data over a typical week

    returns production per 15-minute (meaned over the week) as DC gain
    """

    import pandas as pd

    df = pd.read_csv('data/data_wind_solar_2024_25.csv', sep=';')
    df.drop('endTime', inplace=True, axis=1)
    df.rename(columns={'startTime': 'Datum'}, inplace=True)
    df["Datum"] = pd.to_datetime(df["Datum"], format='mixed')
    df.set_index('Datum', inplace=True)
    df.columns = ['Wind', 'Solar']
    # get mean and std value
    df_mean = df.groupby([df.index.dayofweek, df.index.hour, df.index.minute]).mean()
    df_std = df.groupby([df.index.dayofweek, df.index.hour, df.index.minute]).std()
    # normalize by max values
    max_vals = df.max()
    df_mean = df_mean / max_vals
    df_std = df_std / max_vals

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

        return df_mean, df_std, final_dfs[0], final_dfs[1]

    return df_mean, df_std
