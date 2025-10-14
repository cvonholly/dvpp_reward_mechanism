"""
Parallelized version of run_dvpp_simulation.py using joblib for better pickling support
"""

import control as ct
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import os

from get_max_reward import simulate_devices_and_limits
from src.get_controllers import get_pi_params
from src.get_device_systems import get_bess_energy_sys, get_special_pfs, get_time_constants
from src.time_varying_dc_gain import get_wind_solar_dc_gains, datetime_to_idx
from src.get_required_services import *
from src.get_pv_wind_probs import get_pv_wind_probs


def simulate_single_scenario(service, ts, input_data, requirement_curve, i, idx,
                            create_io_dict_func, save_path, prices, wind_solar_dc_gains,
                            pi_params, time_constants, dpfs, STATIC_PF, adaptive_func,
                            make_PV_Wind_stochastic, power_ratings_dict, change_roles_for_services,
                            calc_1st_stage_reward, include_battery_uncertainty, save_pics,
                            pf_name, Sx, df_bat_path):
    """
    Worker function to simulate a single scenario for a specific service and timestamp.
    This function will be executed in parallel.
    
    Note: Recreates IO_dict inside the function to avoid pickling issues.
    """
    
    print(f'========================\n Simulating {service} for scenario {i+1}/{Sx}, time {idx} \n')
    
    # Preliminary calculations
    price = prices.loc[idx, 'FFR_price'] if service == 'FFR' else prices.loc[idx, 'FCR_D_up_price']
    my_path = save_path + '/' + service.replace('-', '_')
    save_pics_i = save_pics(i) if callable(save_pics) else save_pics
    
    # Load battery data if needed (inside worker to avoid pickling DataFrame)
    if include_battery_uncertainty and df_bat_path:
        df_bat = pd.read_csv(df_bat_path, sep=';', index_col=0)
    else:
        df_bat = None
    
    result = {
        'service': service,
        'idx': idx,
        'i': i,
        'forecasted_value': None,
        'value': None,
        'dvpp_info': {}
    }
    
    # 1st stage: forecasted reward
    if calc_1st_stage_reward:
        # Create IO_dict fresh in this process
        IO_dict = create_io_dict_func()
        
        if change_roles_for_services:
            for name, new_type in change_roles_for_services.get(service, {}).items():
                IO_dict[name] = (IO_dict[name][0], new_type, IO_dict[name][2])
        
        day = idx
        hour = day.hour
        forecast_PV, forecast_Wind = get_pv_wind_probs(day, df_path='data/data_wind_solar_2024_25.csv')
        forecast_PV, forecast_Wind = forecast_PV[hour], forecast_Wind[hour]
        dc_gain_Wind = forecast_Wind * power_ratings_dict['Wind']
        dc_gain_PV = forecast_PV * power_ratings_dict['PV']
        
        IO_dict['Wind'] = (IO_dict['Wind'][0], IO_dict['Wind'][1], dc_gain_Wind)
        IO_dict['PV'] = (IO_dict['PV'][0], IO_dict['PV'][1], dc_gain_PV)
        
        if include_battery_uncertainty and df_bat is not None:
            idx_bat = datetime_to_idx(idx)
            soc = df_bat.loc[idx_bat, 'Battery_SOC (MWh)']
            IO_dict['BESS'] = (get_bess_energy_sys(e_max=soc), IO_dict['BESS'][1], IO_dict['BESS'][2])
        else:
            soc = 4
        
        VALUE, ENERGY, PEAK_POWER = simulate_devices_and_limits(
            IO_dict=IO_dict,
            pi_params=pi_params,
            input_service_max=input_data,
            curve_service_min=requirement_curve,
            title=f'FORECAST {service}',
            T_MAX=ts[-1],
            save_path=my_path,
            pf_name=pf_name,
            x_scenario=i+1,
            price=price,
            dpfs=dpfs,
            save_pics=save_pics_i,
            adaptive_func=adaptive_func,
            service=service
        )
        
        result['forecasted_value'] = VALUE
        my_names = list(power_ratings_dict.keys())
        idx_grand_coalition = [k for k in VALUE.keys() if len(k) == len(my_names)][0]
        result['dvpp_info']['forecasted'] = {
            'time_stamps': idx,
            'forecasted_dc_gain': sum(IO_dict[k][2] for k in IO_dict.keys()),
            'forecasted_reward': VALUE[idx_grand_coalition],
            'bess_soc': soc
        }
        
        # Store for 2nd stage
        set_service_rating = {k: v / price for k, v in VALUE.items()}
    else:
        set_service_rating = None
    
    # 2nd stage: real-time operation
    # Create IO_dict fresh again
    IO_dict = create_io_dict_func()
    
    if change_roles_for_services:
        for name, new_type in change_roles_for_services.get(service, {}).items():
            IO_dict[name] = (IO_dict[name][0], new_type, IO_dict[name][2])
    
    if make_PV_Wind_stochastic:
        dc_gain_Wind = wind_solar_dc_gains['Wind'].loc[idx] * power_ratings_dict['Wind']
        dc_gain_PV = wind_solar_dc_gains['Solar'].loc[idx] * power_ratings_dict['PV']
        IO_dict['Wind'] = (IO_dict['Wind'][0], IO_dict['Wind'][1], dc_gain_Wind)
        IO_dict['PV'] = (IO_dict['PV'][0], IO_dict['PV'][1], dc_gain_PV)
    
    if include_battery_uncertainty and df_bat is not None:
        idx_bat = datetime_to_idx(idx)
        soc = df_bat.loc[idx_bat, 'Battery_SOC (MWh)']
        IO_dict['BESS'] = (get_bess_energy_sys(e_max=soc), IO_dict['BESS'][1], IO_dict['BESS'][2])
    
    VALUE, ENERGY, PEAK_POWER = simulate_devices_and_limits(
        IO_dict=IO_dict,
        pi_params=pi_params,
        input_service_max=input_data,
        curve_service_min=requirement_curve,
        title=f'{service}',
        T_MAX=ts[-1],
        save_path=my_path,
        pf_name=pf_name,
        x_scenario=i+1,
        price=price,
        dpfs=dpfs,
        save_pics=save_pics_i,
        adaptive_func=adaptive_func,
        service=service,
        set_service_rating=set_service_rating
    )
    
    result['value'] = VALUE
    my_names = list(power_ratings_dict.keys())
    idx_grand_coalition = [k for k in VALUE.keys() if len(k) == len(my_names)][0]
    result['dvpp_info']['real'] = {
        'real_dc_gain': sum(IO_dict[k][2] for k in IO_dict.keys()),
        'real_reward': VALUE[idx_grand_coalition]
    }
    
    return result


def run_dvpp_simulation(create_io_dict,
                        save_path='pics/new',
                        services_input={'FCR': get_fcr(), 'FFR': get_ffr(), 'FFR-FCR': get_ffr_fcr(), 'FCR-D': get_fcr_d()},
                        Sx=1,
                        make_PV_Wind_stochastic=False,
                        STATIC_PF=False,
                        save_pics=True,
                        adaptive_func={},
                        change_roles_for_services={},
                        calc_1st_stage_reward=False,
                        include_battery_uncertainty=False,
                        save_dvpp_info=False,
                        run_one_week_simulation=False,
                        hourly_average=True,
                        n_jobs=-1,
                        backend='loky'):
    """
    Parallelized version of run_dvpp_simulation using joblib.
    
    Additional parameters:
    n_jobs: Number of parallel jobs. -1 means use all CPUs, -2 means all but one, etc.
    backend: 'loky' (default, most compatible), 'multiprocessing', or 'threading'
            Use 'loky' for best pickling support
    """
    
    # Time series of 15-minute dc gain
    wind_solar_dc_gains, probs_15_min, prices = get_wind_solar_dc_gains(hourly_average=hourly_average)
    
    print(prices.head())
    
    # Get all orders of coalitions
    all_values = {}
    forecasted_values = {}
    power_ratings_dict = {name: vals[2] for name, vals in create_io_dict().items()}
    my_names = list(power_ratings_dict.keys())
    
    pi_params = get_pi_params()
    time_constants = get_time_constants()
    
    # Get dynamic participation factors
    dpfs = {}
    dpfs_set = get_special_pfs()
    for name in my_names:
        if name in dpfs_set.keys():
            dpfs[name] = dpfs_set[name]
        else:
            dpfs[name] = ct.tf([1], [time_constants[name], 1])
    
    # Set name for participation factors
    pf_name = 'SPF' if STATIC_PF else 'DPF'
    pf_name = 'ADPF' if (adaptive_func != {} and STATIC_PF == 'ADPF') else pf_name
    
    # Set scenarios
    if Sx > 1:
        ps = probs_15_min.iloc[:, 0].values
        selected_indices = np.random.choice(len(wind_solar_dc_gains), size=Sx, p=ps)
        time_stamps = wind_solar_dc_gains.index[selected_indices]
    elif run_one_week_simulation:
        start_date = run_one_week_simulation[0]
        end_date = run_one_week_simulation[1]
        mask = (wind_solar_dc_gains.index >= start_date) & (wind_solar_dc_gains.index < end_date)
        time_stamps = wind_solar_dc_gains[mask].index
    elif not run_one_week_simulation and Sx == 1:
        price = prices[prices > 0].mean()
        prices = pd.DataFrame([price], columns=prices.columns, index=[0])
        time_stamps = [0]
    
    # Store battery file path instead of DataFrame
    df_bat_path = 'data/battery_soc_profile.csv' if include_battery_uncertainty else None
    
    # Prepare DVPP info storage
    print(time_stamps)
    dvpps_info = {s: pd.DataFrame(pd.NA, index=time_stamps,
                              columns=['time_stamps', 'forecasted_dc_gain', 'real_dc_gain', 'FFR_price', 'FCR-D up_price',
                                       'forecasted_reward', 'real_reward', 'bess_soc']) for s in services_input.keys()}
    if save_dvpp_info:
        for s in dvpps_info.keys():
            dvpps_info[s]['FFR_price'] = prices.loc[time_stamps, 'FFR_price'].values
            dvpps_info[s]['FCR-D up_price'] = prices.loc[time_stamps, 'FCR_D_up_price'].values
    
    # Prepare task list
    tasks = []
    for service, (ts, input_data, requirement_curve) in services_input.items():
        for i, idx in enumerate(time_stamps):
            tasks.append((
                service, ts, input_data, requirement_curve, i, idx,
                create_io_dict, save_path, prices, wind_solar_dc_gains,
                pi_params, time_constants, dpfs, STATIC_PF, adaptive_func,
                make_PV_Wind_stochastic, power_ratings_dict, change_roles_for_services,
                calc_1st_stage_reward, include_battery_uncertainty, save_pics,
                pf_name, Sx, df_bat_path
            ))
    
    print(f"\nRunning {len(tasks)} simulations using {n_jobs if n_jobs > 0 else os.cpu_count()} parallel jobs...")
    print(f"Backend: {backend}")
    
    # Run simulations in parallel using joblib
    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=10)(
        delayed(simulate_single_scenario)(*task) for task in tasks
    )
    
    # Aggregate results
    for result in results:
        service = result['service']
        i = result['i']
        idx = result['idx']
        
        if calc_1st_stage_reward and result['forecasted_value'] is not None:
            forecasted_values[(service, i)] = result['forecasted_value']
            if save_dvpp_info and 'forecasted' in result['dvpp_info']:
                for key, value in result['dvpp_info']['forecasted'].items():
                    dvpps_info[service].loc[idx, key] = value
        
        if Sx == 1:
            all_values[service] = result['value']
        else:
            all_values[(service, i)] = result['value']
        
        if save_dvpp_info and 'real' in result['dvpp_info']:
            for key, value in result['dvpp_info']['real'].items():
                dvpps_info[service].loc[idx, key] = value
    
    # Save values and Shapley values
    df = pd.DataFrame.from_dict(all_values, orient='index')
    df.to_csv(f'{save_path}/values_{pf_name}.csv', float_format='%.5f')
    
    if calc_1st_stage_reward:
        df_forecast = pd.DataFrame.from_dict(forecasted_values, orient='index')
        df_forecast.to_csv(f'{save_path}/values_forecasted_{pf_name}.csv', float_format='%.5f')
    
    if save_dvpp_info:
        for s, infos in dvpps_info.items():
            infos.to_csv(f'{save_path}/dvpp_info_{pf_name}_{s}.csv', index=False)
    
    print(f"\nSimulation complete! Results saved to {save_path}")
    
    return all_values, forecasted_values if calc_1st_stage_reward else None