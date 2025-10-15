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

    def get_alpha_beta_mean(data, thresh=0.05):
        if np.mean(data) < thresh:
            return np.mean(data)
        a, b, loc, scale = beta.fit(data)
        mean = beta.mean(a, b, loc=loc, scale=scale)
        return mean

    df_s_mean = df_s.apply(lambda x: get_alpha_beta_mean(x))

    # Wind: fit to mean
    df_w = df_window['Wind']
    df_w = df_w.groupby([df_w.index.hour]).mean()

    return df_s_mean, df_w

def get_errors(K, 
               hour, 
               path_prod='../data/data_wind_solar_2024_25.csv',
               path_forecast='../data/data_wind_solar_2024_25_forecast.csv'):
    """
    get K errors for PV and Wind generation for a some number of hours
    normalized by max production value

    params:
        K: int, number of errors to sample
        hour: int, number of hours to sample from
        path_prod: str, path to production data
        path_forecast: str, path to forecast data
    
    returns:
        errors_wind: list of K errors for wind generation
        errors_solar: list of K errors for solar generation
    """
    # get producition data
    df = pd.read_csv(path_prod, sep=';')
    df.drop('endTime', inplace=True, axis=1)
    df.rename(columns={'startTime': 'Datum'}, inplace=True)
    df["Datum"] = pd.to_datetime(df["Datum"], format='mixed')
    df["Datum"] = df['Datum'].dt.tz_localize(None)
    df.set_index('Datum', inplace=True)
    df.columns = ['Wind', 'Solar']
    # normalize [0,1]
    max_wind, max_solar = df['Wind'].max(), df['Solar'].max()
    df['Wind'] = df['Wind'].clip(lower=0) / max_wind
    df['Solar'] = df['Solar'].clip(lower=0) / max_solar
    # get forecast data
    df_forecast = pd.read_csv(path_forecast, sep=';')
    df_forecast.drop('endTime', inplace=True, axis=1)
    df_forecast.rename(columns={'startTime': 'Datum'}, inplace=True)
    df_forecast["Datum"] = pd.to_datetime(df_forecast["Datum"], format='mixed')
    df_forecast["Datum"] = df_forecast['Datum'].dt.tz_localize(None)
    df_forecast.set_index('Datum', inplace=True)
    df_forecast.columns = ['Wind_forecast', 'Solar_forecast']
    # normalize [0,1]
    df_forecast['Wind_forecast'] = df_forecast['Wind_forecast'].clip(lower=0) / max_wind
    df_forecast['Solar_forecast'] = df_forecast['Solar_forecast'].clip(lower=0) / max_solar

    # create errors
    error_wind = df_forecast['Wind_forecast'] - df['Wind']
    error_solar = df_forecast['Solar_forecast'] - df['Solar']
    # remove nan values
    error_wind = error_wind[~error_wind.isna()]
    error_solar = error_solar[~error_solar.isna()]
    # split by hour of day and get values
    hourly_error_wind = error_wind.groupby([error_wind.index.hour]).apply(list)[hour]
    hourly_error_solar = error_solar.groupby([error_solar.index.hour]).apply(list)[hour]

    # draw K samples
    k_err_wind = np.random.choice(hourly_error_wind, size=K).tolist()
    k_err_solar = np.random.choice(hourly_error_solar, size=K).tolist()

    return k_err_wind, k_err_solar
    




if __name__ == '__main__':
    # Example usage
    day = pd.Timestamp('2024-12-15')
    hourly_pv_mean, hourly_wind_mean = get_pv_wind_probs(day)
