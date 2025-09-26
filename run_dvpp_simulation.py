"""

This script simulates a Solar-Wind-BESS DVPP with production patterns to participate in FFR and FCR services

The reward mechanisms are evaluated, namely Shapely Value and Nash Bargaining Solution

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
from src.get_device_systems import get_time_constants, get_adpfs

# import procduction data
from src.time_varying_dc_gain import get_wind_solar_dc_gains
from src.get_required_services import *


def run_dvpp_simulation(create_io_dict,
                        save_path='pics/new',
                        services_input={'FCR': get_fcr(), 'FFR': get_ffr(), 'FFR-FCR': get_ffr_fcr(), 'FCR-D': get_fcr_d()},
                        service_diff = 0.1,  # minimum fraction of capacity that can be provided as service (1MW)
                        Sx=1,   # number of scenarios to average over
                        make_PV_Wind_stochastic=False,   # take dc gain from probability distribution
                        STATIC_PF=False,
                        save_pics=True,
                        adaptive_func={},  # adaptive dynamic participation factor function
                        change_roles_for_services={},
                        set_service_rating={}
                        ):
    """
    create_io_dict: dict of devices with entries:
        {(name): (ct.NonlinearIOSystem, device_type, rating)}
    save_path: path to save the data frames and plots
    services_input: dict of services with entries:
        {(service_name): (time_series, max_input_curve, min_requirement_curve)}
    service_diff: fraction of minimum service to supply. at 1 MW -> 0.1 * 1 MW = 0.1 MW
    Sx: number of scenarios to simulate
    make_PV_Wind_stochastic: if True, take dc gain from probability distribution
    STATIC_PF: if True, use static participation factors, else dynamic
    save_pics: if True, save the plots (not recommended for >8 scenarios)
    adaptive_func: dict of adaptive functions for devices with entries:
        {(name): function(t) -> [0, 1]}
        where the functions ideally add up to 1 at each time step
    ------------
    """
    
    
    wind_solar_dc_gains, probs_15_min, prices = get_wind_solar_dc_gains()  # time series of 15-minute dc gain

    # get all orders of coalitions
    all_values = {}
    power_ratings_dict = {name: vals[2] for name, vals in create_io_dict().items()}
    my_names = list(power_ratings_dict.keys())

    pi_params = get_pi_params()   # get all pi parameters
    
    time_constants = get_time_constants()  # get dict with time constants

    # get dynamic participation factors
    dpfs = {}
    dpfs_set = get_adpfs()  # get dict with adpfs
    for name in my_names:
        if name in dpfs_set.keys():
            dpfs[name] = dpfs_set[name]
        else:
            dpfs[name] = ct.tf([1], [time_constants[name], 1])   # get 1st order transfer functions / reference

    # set scenarios
    if Sx > 1:
        ps = probs_15_min.iloc[:, 0].values # select FFR probabilities for now
        selected_indices = np.random.choice(len(wind_solar_dc_gains), size=Sx, p=ps)
        # save selected scenarios to csv
        df_scenarios =pd.DataFrame(selected_indices, columns=['selected_indices'])
        # add prices
        df_scenarios['FFR price EUR/MW/h'] = df_scenarios['selected_indices'].apply(lambda idx: prices.iloc[idx, 0])
        df_scenarios['FCR-D up price EUR/MW/h'] = df_scenarios['selected_indices'].apply(lambda idx: prices.iloc[idx, 1])
        # save
        df_scenarios.to_csv(f'{save_path}/selected_scenarios.csv', index=False)
    if Sx == 1:
        # set price to average non-zero price
        price = prices[prices>0].mean()
        prices = pd.DataFrame([price], columns=prices.columns, index=[0])
        selected_indices = [0]

    
    for service, (ts, input, requirement_curve) in services_input.items():
        #
        # todo: in future use different probabilities for different services, however now for comaprison reasons, use same
        #
        # if Sx > 1:
        #     ps = probs_15_min.iloc[:, 0].values if service=='FFR' else probs_15_min.iloc[:, 1].values
        #     selected_indices = np.random.choice(len(wind_solar_dc_gains), size=Sx, p=ps)
        for i, idx in enumerate(selected_indices):
            print(f'========================\n Simulating {service} for scenario {i+1}/{Sx}, index {idx} \n')
            IO_dict = create_io_dict()    # reset io dict for each service
            if change_roles_for_services:
                for name, new_type in change_roles_for_services.get(service, {}).items():
                    IO_dict[name] = (IO_dict[name][0], new_type, IO_dict[name][2])
            if make_PV_Wind_stochastic:   # adjust dc gain
                dc_gain_Wind = wind_solar_dc_gains['Wind'].iloc[idx] * power_ratings_dict['Wind']
                dc_gain_PV = wind_solar_dc_gains['Solar'].iloc[idx] * power_ratings_dict['PV']
                # adjust current io_dict
                IO_dict['Wind'] = (IO_dict['Wind'][0], IO_dict['Wind'][1], dc_gain_Wind)
                IO_dict['PV'] = (IO_dict['PV'][0], IO_dict['PV'][1], dc_gain_PV)
            price = prices.iloc[idx, 0] if service=='FFR' else prices.iloc[idx, 1]
            my_path = save_path + '/' + service.replace('-', '_')

            if Sx > 1:
                # set save_pics_i to function, otherwise false
                save_pics_i = save_pics(i)
            else:
                save_pics_i = save_pics

            # service response
            VALUE, ENERGY, PEAK_POWER = simulate_devices_and_limits(
                                        IO_dict=IO_dict,
                                        pi_params=pi_params,
                                        input_service_max=input,
                                        curve_service_min=requirement_curve,
                                        title=f'{service}',
                                        service_diff=service_diff,
                                        T_MAX=ts[-1],
                                        save_path=my_path,
                                        STATIC_PF=STATIC_PF,
                                        x_scenario=i+1,
                                        price=price,  # specify scenario if needed from 1...Sx,
                                        dpfs=dpfs,
                                        save_pics=save_pics_i,
                                        adaptive_func=adaptive_func,
                                        service=service,
                                        set_service_rating=set_service_rating
            )
            if Sx == 1:
                all_values[service] = VALUE
            else:
                all_values[(service, i)] = VALUE
        
    # save values and shapely values
    pf_name = 'stat' if STATIC_PF else 'dyn'
    df = pd.DataFrame.from_dict(all_values, orient='index')
    df.to_csv(f'{save_path}/values_{pf_name}.csv', float_format='%.5f')
