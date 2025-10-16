"""

This script simulates a Solar-Wind-BESS DVPP in a Baysian Coalitional Game

Setup:
- Finnish Grid
- Services: FFR, FCR, FCR-D up and combination
- Device and Size:
    - Solar PV (LPF)
    - Wind Turbine (LPF)
    - Battery (HPF/BPF)
    - Hydro (LPF)
    - Supercapacitor (HPF)
- DC gains:
    - Solar PV: STATIC
    - Wind Turbine: STATIC
    - Battery: 1 until energy limit reached, then decaying fast
    - Hydro: constant
    - Supercapacitor: 1 until energy limit reached, then decaying fast
- Time Frame: 1 Week
"""

import control as ct
import numpy as np
import pandas as pd

from get_max_reward import simulate_devices_and_limits
from src.get_controllers import get_pi_params
from src.get_device_systems import get_bess_energy_sys, get_special_pfs, get_time_constants, get_special_pfs

# import procduction data
from src.time_varying_dc_gain import get_wind_solar_dc_gains, datetime_to_idx
from src.get_required_services import *

from src.get_pv_wind_probs import get_errors, get_prod_forecast_data
from src. get_optimal_bid import get_optimal_bid


def run_bcd_dvpp_sim(create_io_dict,
                    save_path='pics/new',
                    services_input={'FCR': get_fcr(), 'FFR': get_ffr(), 'FFR-FCR': get_ffr_fcr(), 'FCR-D': get_fcr_d()},
                    Sx=1,   # number of scenarios to average over
                    STATIC_PF=False,
                    save_pics=True,
                    adaptive_func={},  # adaptive dynamic participation factor function
                    change_roles_for_services={},
                    include_battery_uncertainty=False,
                    save_dvpp_info=False,
                    time_slots=False,
                    hourly_average=True,
                    K_errors=10
                    ):
    """
    create_io_dict: dict of devices with entries:
        {(name): (ct.NonlinearIOSystem, device_type, rating)}
    save_path: path to save the data frames and plots
    services_input: dict of services with entries:
        {(service_name): (time_series, max_input_curve, min_requirement_curve)}
    Sx: number of scenarios to simulate
    STATIC_PF: if True, use static participation factors, else dynamic
    save_pics: if True, save the plots (not recommended for >8 scenarios)
    adaptive_func: dict of adaptive functions for devices with entries:
        {(name): function(t) -> [0, 1]}
        where the functions ideally add up to 1 at each time step
    change_roles_for_services:
        dict of dicts with entries:
            {(service_name): {(device_name): new_device_type}}
    include_battery_uncertainty:
        if True, include uncertainty in battery SoC
    save_dvpp_info:
        if True, save info about the DVPP (devices, ratings, controllers, ...) to csv
    time_slots:
        specify time slots
    hourly_average:
        get hourly average in data
    K_errors:
        number of fore-casted parameters to get for ex-ante calculation
    ------------
    """
    # time series of 15-minute dc gain
    wind_solar_dc_gains, probs_15_min, prices = get_wind_solar_dc_gains(hourly_average=hourly_average)
    _, wind_solar_forecast = get_prod_forecast_data()

    # get all orders of coalitions
    all_values = {}
    forecasted_values = {}
    power_ratings_dict = {name: vals[2] for name, vals in create_io_dict().items()}
    my_names = list(power_ratings_dict.keys())

    pi_params = get_pi_params()   # get all pi parameters
    time_constants = get_time_constants()  # get dict with time constants

    # get dynamic participation factors
    dpfs = {}
    dpfs_set = get_special_pfs()  # get dict with adpfs
    for name in my_names:
        if name in dpfs_set.keys():
            dpfs[name] = dpfs_set[name]
        else:
            dpfs[name] = ct.tf([1], [time_constants[name], 1])   # get 1st order transfer functions / reference

    # set name for participation factors
    pf_name = 'SPF' if STATIC_PF else 'DPF'
    pf_name = 'ADPF' if (adaptive_func!={} and STATIC_PF=='ADPF') else pf_name

    # set scenarios
    if time_slots:
        time_stamps = time_slots
    else:
        ps = probs_15_min.iloc[:, 0].values # select FFR probabilities for now
        # select Sx scenarios based on probabilities
        selected_indices = np.random.choice(len(wind_solar_dc_gains), size=Sx, p=ps)
        # convert indices to time stamps
        time_stamps = wind_solar_dc_gains.index[selected_indices]
    
    # get errors for time slots
    k_errs = {}
    for t in time_stamps:
        k_errs[t] = get_errors(K_errors, t.hour)  # tuple of wind_errors, solar_errors

    if include_battery_uncertainty:
        df_bat = pd.read_csv('data/battery_soc_profile.csv', sep=';', index_col=0)
    
    # save dvpp info
    print(time_stamps)
    dvpps_info = {s: pd.DataFrame(pd.NA, index=time_stamps,
                              columns=['time_stamps', 'forecasted_dc_gain', 'real_dc_gain', 'FFR_price', 'FCR-D up_price',
                                       'forecasted_reward', 'real_reward', 'bess_soc']) for s in services_input.keys()}
    if save_dvpp_info:
        for s in dvpps_info.keys():
            dvpps_info[s]['FFR_price'] = prices.loc[time_stamps, 'FFR_price'].values
            dvpps_info[s]['FCR-D up_price'] = prices.loc[time_stamps, 'FCR_D_up_price'].values

    # simulate scenarios
    for service, (ts, input, requirement_curve) in services_input.items():
        #
        # todo: in future use different probabilities for different services, however now for comaprison reasons, use same
        #
        for i, t in enumerate(time_stamps):
            print(f'========================\n Simulating {service} for scenario {i+1}/{Sx}, time {t} \n')

            # preliminary calculations
            price = prices.loc[t, 'FFR_price'] if service=='FFR' else prices.loc[t, 'FCR_D_up_price']
            my_path = save_path + '/' + service.replace('-', '_')
            save_pics_i = save_pics(i) if callable(save_pics) else save_pics
            bids = []

            # get forecasted values
            wind_fc, solar_fc = wind_solar_forecast.loc[t][['Wind_forecast', 'Solar_forecast']]
            for k, err in enumerate(k_errs[t]):
                IO_dict = create_io_dict()    # reset io dict for each run
                if change_roles_for_services:
                    for name, new_type in change_roles_for_services.get(service, {}).items():
                        IO_dict[name] = (IO_dict[name][0], new_type, IO_dict[name][2])
                # simulate for this error
                dc_gain_Wind = (wind_fc + err[0]) * power_ratings_dict['Wind']
                dc_gain_PV = (solar_fc + err[1]) * power_ratings_dict['PV']
                # adjust current io_dict
                IO_dict['Wind'] = (IO_dict['Wind'][0], IO_dict['Wind'][1], dc_gain_Wind)
                IO_dict['PV'] = (IO_dict['PV'][0], IO_dict['PV'][1], dc_gain_PV)
                if include_battery_uncertainty:
                    # get battery SoC for time
                    idx_bat = datetime_to_idx(time_stamps[i])
                    soc = df_bat.loc[idx_bat, 'Battery_SOC (MWh)']
                    IO_dict['BESS'] = (get_bess_energy_sys(e_max=soc), IO_dict['BESS'][1], IO_dict['BESS'][2])
                else:
                    soc = 4   # todo: get from somewhere else

                # service response - forecast
                VALUE, _, _ = simulate_devices_and_limits(
                                            IO_dict=IO_dict,
                                            pi_params=pi_params,
                                            input_service_max=input,
                                            curve_service_min=requirement_curve,
                                            title=f'FORECAST {service}',
                                            T_MAX=ts[-1],
                                            save_path=my_path,
                                            pf_name=pf_name,
                                            x_scenario=i+1,
                                            price=price,  # specify scenario if needed from 1...Sx,
                                            dpfs=dpfs,
                                            save_pics=False,
                                            adaptive_func=adaptive_func,
                                            service=service
                )
                idx_grand_coalition = [k for k in VALUE.keys() if len(k)==len(my_names)][0]
                val_grand_coalition = VALUE[idx_grand_coalition]
                bids.append(val_grand_coalition / price)  # append to bids the acheived price
                forecasted_values[(service, i, k)] = VALUE
                if save_dvpp_info:
                    dvpps_info[service].loc[t, ['time_stamps', 'forecasted_dc_gain', 'forecasted_reward', 'bess_soc']] = \
                            [time_stamps[i], sum(IO_dict[k][2] for k in IO_dict.keys()), VALUE[idx_grand_coalition], soc]

            # 2. choose optimal bids for grand coalition
            probs = [1/K_errors for kk in range(K_errors)]
            b_star = get_optimal_bid(bids, probs)

            # 3. stage: real-time operation
            IO_dict = create_io_dict()    # reset io dict for each service
            if change_roles_for_services:
                for name, new_type in change_roles_for_services.get(service, {}).items():
                    IO_dict[name] = (IO_dict[name][0], new_type, IO_dict[name][2])
            # get realized values
            dc_gain_Wind = wind_solar_dc_gains['Wind'].loc[t] * power_ratings_dict['Wind']
            dc_gain_PV = wind_solar_dc_gains['Solar'].loc[t] * power_ratings_dict['PV']
            # adjust current io_dict
            IO_dict['Wind'] = (IO_dict['Wind'][0], IO_dict['Wind'][1], dc_gain_Wind)
            IO_dict['PV'] = (IO_dict['PV'][0], IO_dict['PV'][1], dc_gain_PV)
            if include_battery_uncertainty:
                # get battery SoC for time
                idx_bat = datetime_to_idx(time_stamps[i])
                soc = df_bat.loc[idx_bat, 'Battery_SOC (MWh)']
                IO_dict['BESS'] = (get_bess_energy_sys(e_max=soc), IO_dict['BESS'][1], IO_dict['BESS'][2])

            
            # if we have 2. stage, we have to follow the bid we have made
            set_service_rating = {k: b_star for k, v in VALUE.items()}
            min_service_rating = b_star * .99  # margin for error

            # service response - real
            VALUE, _, _ = simulate_devices_and_limits(
                                        IO_dict=IO_dict,
                                        pi_params=pi_params,
                                        input_service_max=input,
                                        curve_service_min=requirement_curve,
                                        title=f'{service}',
                                        T_MAX=ts[-1],
                                        save_path=my_path,
                                        pf_name=pf_name,
                                        x_scenario=i+1,
                                        price=price,  # specify scenario if needed from 1...Sx,
                                        dpfs=dpfs,
                                        save_pics=save_pics_i,
                                        adaptive_func=adaptive_func,
                                        service=service,
                                        set_service_rating=set_service_rating,
                                        min_service_rating=min_service_rating
            )
            if Sx == 1:
                all_values[service] = VALUE
            else:
                all_values[(service, i)] = VALUE
            if save_dvpp_info:
                idx_grand_coalition = [k for k in VALUE.keys() if len(k)==len(my_names)][0]
                dvpps_info[service].loc[t, ['real_dc_gain', 'real_reward']] = \
                        [sum(IO_dict[k][2] for k in IO_dict.keys()), VALUE[idx_grand_coalition]]

    # save values and shapely values
    df = pd.DataFrame.from_dict(all_values, orient='index')
    df.to_csv(f'{save_path}/values_{pf_name}.csv', float_format='%.5f')

    df_forecast = pd.DataFrame.from_dict(forecasted_values, orient='index')
    df_forecast.to_csv(f'{save_path}/values_forecasted_{pf_name}.csv', float_format='%.5f')
    
    if save_dvpp_info:
        for s, infos in dvpps_info.items():
            infos.to_csv(f'{save_path}/dvpp_info_{pf_name}_{s}.csv', index=False)
