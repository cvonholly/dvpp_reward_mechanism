import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# --- Configuration & Constants ---
# FILE_PATH = "dataexport_20251127T161056.csv"  # Ensure this file is in the working directory
CAPACITY_MW = 100.0

# Solar Constants
NOCT = 45.0             # Nominal Operating Cell Temperature (C)
TEMP_COEFF = -0.004     # -0.4% / C power loss
STC_TEMP = 25.0         # Standard Test Conditions Temperature (C)
STC_IRR = 1000.0        # Standard Test Conditions Irradiance (W/m2)
SNOW_THRESH = 0.5       # Snow depth (cm) threshold for covering panels
MELT_RATE = 0.5         # Snow melt rate (cm / deg C / hour)

# Wind Constants
R_SPECIFIC_AIR = 287.05 # J/(kg*K)
RHO_STD = 1.225         # Standard air density (kg/m3)
V_CUT_IN = 3.0          # m/s
V_RATED = 13.0          # m/s
V_CUT_OUT = 25.0        # m/s

def load_and_clean_data(file_path):
    """
    Loads the CSV, parses timestamps, and standardizes column names.
    """
    # Skip metadata rows (header=9 based on inspection)
    df = pd.read_csv(file_path, header=9)
    
    # Map variable names to standardized keys
    # Note: Column names are specific to the provided file structure
    col_map = {
        df.columns[0]: 'timestamp',
        '63.01°N 24.14°O Temperature [2 m elevation corrected]': 'temp_c',       # Temperature
        '63.01°N 24.14°O Snowfall Amount': 'snow_fall_cm', # Snowfall Amount
        '63.01°N 24.14°O Wind Speed [100 m]': 'ws_100_kmh',   # Wind Speed [100 m]
        '63.01°N 24.14°O Shortwave Radiation': 'ghi_wm2',      # Shortwave Radiation
        '63.01°N 24.14°O Mean Sea Level Pressure [MSL]': 'pressure_hpa'  # MSL Pressure
    }
    df = df.rename(columns=col_map)
    
    # DateTime conversion
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%dT%H%M')
    
    # Unit conversions
    df['ws_100_ms'] = df['ws_100_kmh'] / 3.6  # km/h -> m/s
    
    return df

def apply_solar_model(df):
    """
    Calculates Solar Output considering Temperature and Snow Cover.
    """
    # 1. Cell Temperature Calculation (Sandia Model Approximation)
    # T_cell = T_amb + (GHI / 800) * (NOCT - 20)
    df['t_cell'] = df['temp_c'] + (df['ghi_wm2'] / 800.0) * (NOCT - 20.0)
    
    # 2. Temperature Corrected Power (before snow)
    # P = P_rated * (G/1000) * (1 + gamma * (T_cell - 25))
    thermal_factor = 1 + TEMP_COEFF * (df['t_cell'] - STC_TEMP)
    df['solar_raw_mw'] = CAPACITY_MW * (df['ghi_wm2'] / STC_IRR) * thermal_factor
    df['solar_raw_mw'] = df['solar_raw_mw'].clip(lower=0.0)

    # 3. Snow Accumulation & Melt Model (Iterative)
    snow_depth = np.zeros(len(df))
    current_snow = 0.0
    
    # Extract arrays for speed
    snow_fall_arr = df['snow_fall_cm'].values
    temp_arr = df['temp_c'].values
    
    for i in range(len(df)):
        # Accumulate snowfall
        current_snow += snow_fall_arr[i]
        
        # Melt if above freezing
        if temp_arr[i] > 0:
            melt_amount = MELT_RATE * temp_arr[i]
            current_snow = max(0.0, current_snow - melt_amount)
            
        snow_depth[i] = current_snow
        
    df['snow_depth_est'] = snow_depth
    
    # 4. Apply Snow Loss (Binary cut-off for simplicity)
    # If snow depth > threshold, output is 0.
    df['solar_mw'] = np.where(df['snow_depth_est'] > SNOW_THRESH, 0.0, df['solar_raw_mw'])
    
    return df

def apply_wind_model(df):
    """
    Calculates Wind Output considering Air Density and Turbine Power Curve.
    """
    # 1. Calculate Air Density (rho)
    # rho = P / (R * T_kelvin)
    # P in Pa (hPa * 100), T in Kelvin
    df['rho'] = (df['pressure_hpa'] * 100) / (R_SPECIFIC_AIR * (df['temp_c'] + 273.15))
    
    # 2. Define Power Curve Function (Normalized 0-1)
    def power_curve_coeff(v):
        if v < V_CUT_IN:
            return 0.0
        elif V_CUT_IN <= v < V_RATED:
            # Cubic interpolation region
            return ((v**3 - V_CUT_IN**3) / (V_RATED**3 - V_CUT_IN**3))
        elif V_RATED <= v < V_CUT_OUT:
            return 1.0
        else: # v >= V_CUT_OUT
            return 0.0
            
    # Vectorize for performance
    vec_curve = np.vectorize(power_curve_coeff)
    base_fraction = vec_curve(df['ws_100_ms'])
    
    # 3. Apply Density Correction
    # Power is proportional to density: P_act = P_curve * (rho / rho_std)
    df['wind_mw'] = CAPACITY_MW * base_fraction * (df['rho'] / RHO_STD)
    
    # Clip to rated capacity (Electrical limit)
    df['wind_mw'] = df['wind_mw'].clip(upper=CAPACITY_MW)
    
    return df

def generate_energy_profile(start_time, end_time, file_path):
    """
    Main execution function.
    
    Args:
        start_time (str): Start datetime in 'YYYY-MM-DD' format.
        end_time (str): End datetime in 'YYYY-MM-DD' format.
        
    Returns:
        pd.DataFrame: Hourly production profile filtered by the time range.
    """
    # 1. Load Data
    print("Loading data...")
    df = load_and_clean_data(file_path)
    
    # 2. Filter by Time Range
    mask = (df['timestamp'] >= pd.to_datetime(start_time)) & \
           (df['timestamp'] <= pd.to_datetime(end_time))
    df_filtered = df.loc[mask].copy().reset_index(drop=True)
    
    if df_filtered.empty:
        raise ValueError("No data found for the specified date range.")
        
    print(f"Processing {len(df_filtered)} hours of data...")
    
    # 3. Run Models
    df_filtered = apply_solar_model(df_filtered)
    df_filtered = apply_wind_model(df_filtered)
    
    # 4. Final Cleanup
    result = df_filtered[['timestamp', 'solar_mw', 'wind_mw']].round(2)
    
    return result

# --- Example Usage ---
if __name__ == "__main__":
    # Define inputs
    START_DATE = '2023-01-01'
    END_DATE = '2023-12-31'

    file_paths = ['data/meteoblue/meteoblue_forecasted_100m.csv', 'data/meteoblue/meteoblue_realized.csv']
    
    forecasted = generate_energy_profile(START_DATE, END_DATE, file_paths[0])
    realized = generate_energy_profile(START_DATE, END_DATE, file_paths[1])

    # first, look for  wrong values in dataframes, i.e. greater then 100 or smaller then 0 or nan
    for df, label in zip([forecasted, realized], ['Forecasted', 'Realized']):
        if df[['solar_mw', 'wind_mw']].isnull().values.any():
            # count how many and if solar or wind
            if df['solar_mw'].isnull().sum() > 0:
                print(f"Warning: {df['solar_mw'].isnull().sum()} NaN values found in solar data.")
            if df['wind_mw'].isnull().sum() > 0:
                print(f"Warning: {df['wind_mw'].isnull().sum()} NaN values found in wind data.")
        if (df[['solar_mw', 'wind_mw']] < 0).values.any():
            print(f"Warning: Negative values found in {label} data.")
        if (df[['solar_mw', 'wind_mw']] > CAPACITY_MW).values.any():
            print(f"Warning: Values exceeding capacity found in {label} data.")
    
    # Calculate totals
    total_solar_forecasted = forecasted['solar_mw'].sum() / 1000 # GWh
    total_wind_forecasted = forecasted['wind_mw'].sum() / 1000   # GWh
    total_solar_realized = realized['solar_mw'].sum() / 1000     # GWh
    total_wind_realized = realized['wind_mw'].sum() / 1000         # GWh
    
    print(f"\nTotal Solar Forecasted: {total_solar_forecasted:.2f} GWh")
    print(f"Total Wind Forecasted:  {total_wind_forecasted:.2f} GWh")
    print(f"Total Solar Realized: {total_solar_realized:.2f} GWh")
    print(f"Total Wind Realized:  {total_wind_realized:.2f} GWh")

    # calcualte forecast error dataset
    forecast_error = pd.DataFrame()
    forecast_error['solar_error_mw'] = forecasted['solar_mw'] - realized['solar_mw']
    forecast_error['wind_error_mw'] = forecasted['wind_mw'] - realized['wind_mw']
    # print statistics on error
    print(f"\nSolar Forecast Error Statistics (MW):")
    print(forecast_error['solar_error_mw'].describe())
    print(f"\nWind Forecast Error Statistics (MW):")
    print(forecast_error['wind_error_mw'].describe())

    
    # plot a week of data for comparison
    plt.figure(figsize=(12,6))
    # pick dates in summer
    mask_week = (forecasted['timestamp'] >= '2023-06-01') & (forecasted['timestamp'] < '2023-06-08')
    plt.plot(forecasted.loc[mask_week, 'timestamp'], forecasted.loc[mask_week, 'solar_mw'], label='Solar Forecasted', color='orange')
    plt.plot(realized.loc[mask_week, 'timestamp'], realized.loc[mask_week, 'solar_mw'], label='Solar Realized', color='red', linestyle='--')
    plt.plot(forecasted.loc[mask_week, 'timestamp'], forecasted.loc[mask_week, 'wind_mw'], label='Wind Forecasted', color='blue')
    plt.plot(realized.loc[mask_week, 'timestamp'], realized.loc[mask_week, 'wind_mw'], label='Wind Realized', color='cyan', linestyle='--')

    plt.legend()
    plt.title('PV and Wind Power Production: Forecasted vs Realized (First Week of June 2023)')
    plt.xlabel('Date')
    plt.ylabel('Power (MW)')
    plt.show()
    