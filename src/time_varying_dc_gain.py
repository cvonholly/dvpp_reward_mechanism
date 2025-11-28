"""

computes time-varying DC gains for Solar PV, Wind Turbine and Battery

"""

import numpy as np
import pandas as pd

def get_wind_solar_dc_gains(path='data/data_wind_solar_2024_25.csv',
                            hourly_average=False):
    """
    gets probabilities and prices
    
    - hourly_average: if True, averages the data to hourly values

    returns:
    - probability distribution of FFR and FCR procurement
    - price distribution of FFR and FCR-D prices
    """

    final_dfs = {}
    paths = ['data/data_ffr_fcr_procurement_2024_25.csv', 'data/data_ffr_fcr_d_price_2024_25.csv']
    for k, path in enumerate(paths):
        # this data is already in hourly format
        df = pd.read_csv(path, sep=';')
        df.drop('endTime', inplace=True, axis=1)
        df.rename(columns={'startTime': 'Datum',
                            'Fast Frequency Reserve FFR, price': 'FFR_price',
                            'Frequency Containment Reserve for Disturbances upwards regulation, hourly market prices': 'FCR_D_up_price'}, 
                            inplace=True)
        df['Datum'] = pd.to_datetime(df["Datum"], format='mixed')
        df.set_index('Datum', inplace=True)
        if k==0: df = df / df.sum()
        # next, split up every row into 4 values for 15 min intervals
        if not hourly_average:
            df = pd.DataFrame(np.repeat(df.values, 4, axis=0), columns=df.columns)
        else:
            df.index = [ts.tz_localize(None) for ts in df.index]
        if k==0: df = df / df.sum()
        final_dfs[k] = df

    return final_dfs[0], final_dfs[1]


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
