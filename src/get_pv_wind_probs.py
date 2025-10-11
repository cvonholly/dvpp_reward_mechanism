import pandas as pd
import numpy as np
from scipy.stats import beta, weibull_min
from datetime import timedelta


def get_pv_wind_probs(day, df_path='../data/data_wind_solar_2024_25.csv'):
    """
    get hourly beta distribution parameters for PV and Wind generation
    
    params:
        day: pd.Timestamp, day to center the ±1 month window around
    """
    # format df
    df = pd.read_csv(df_path, sep=';')
    df.drop('endTime', inplace=True, axis=1)
    df.rename(columns={'startTime': 'Datum'}, inplace=True)
    df["Datum"] = pd.to_datetime(df["Datum"], format='mixed')
    df["Datum"] = df['Datum'].dt.tz_localize(None)
    df.set_index('Datum', inplace=True)
    df.columns = ['Wind', 'Solar']

    # normalize wind and solar to [0,1]
    df['Wind'] = df['Wind'].clip(lower=0) / (df['Wind'].max() + 1e-9)
    df['Solar'] = df['Solar'].clip(lower=0) / (df['Solar'].max() + 1e-9)

    # Select ±1 month window
    start = day - pd.DateOffset(months=1)
    end = day + pd.DateOffset(months=1)
    df_window = df.loc[start:end]
    # remove non-finfite values
    df_window = df_window[np.isfinite(df_window).all(axis=1)]

    # Solar: fit Beta distribution
    df_s = df_window['Solar']
    df_s = df_s.groupby([df_s.index.hour]).apply(list)

    def get_alpha_beta_mean(data, thresh=0.03):
        if np.mean(data) < thresh:
            return 0.0
        a, b, loc, scale = beta.fit(data)
        mean = beta.mean(a, b, loc=loc, scale=scale)
        return mean

    df_s_mean = df_s.apply(lambda x: get_alpha_beta_mean(x))

    # Wind: fit to mean
    df_w = df_window['Wind']
    df_w = df_w.groupby([df_w.index.hour]).mean()

    return df_s_mean, df_w

if __name__ == '__main__':
    # Example usage
    day = pd.Timestamp('2024-12-15')
    hourly_pv_mean, hourly_wind_mean = get_pv_wind_probs(day)
