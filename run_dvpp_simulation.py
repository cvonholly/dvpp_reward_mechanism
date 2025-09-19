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
from src.get_device_systems import get_time_constants

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
                        adaptive_func={}  # adaptive dynamic participation factor function
                        ):
    
    wind_solar_dc_gains, probs_15_min, prices = get_wind_solar_dc_gains()  # time series of 15-minute dc gain

    # get all orders of coalitions
    all_values = {}
    power_ratings_dict = {name: vals[2] for name, vals in create_io_dict().items()}
    my_names = list(power_ratings_dict.keys())

    pi_params = get_pi_params()   # get all pi parameters
    
    time_constants = get_time_constants()  # get dict with time constants

    Gs_diff = {k: ct.tf([1], [time_constants[k], 1]) for k in my_names}   # get 1st order transfer functions / reference

    # set scenarios
    if Sx > 1:
        ps = probs_15_min.iloc[:, 0].values
        selected_indices = np.random.choice(len(wind_solar_dc_gains), size=Sx, p=ps)
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
            if make_PV_Wind_stochastic:   # adjust dc gain
                dc_gain_Wind = wind_solar_dc_gains['Wind'].iloc[idx] * power_ratings_dict['Wind']
                dc_gain_PV = wind_solar_dc_gains['Solar'].iloc[idx] * power_ratings_dict['PV']
                # adjust current io_dict
                IO_dict['Wind'] = (IO_dict['Wind'][0], IO_dict['Wind'][1], dc_gain_Wind)
                IO_dict['PV'] = (IO_dict['PV'][0], IO_dict['PV'][1], dc_gain_PV)
            price = prices.iloc[idx, 0] if service=='FFR' else prices.iloc[idx, 1]
            my_path = save_path + '/' + service.replace('-', '_')

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
                                        Gs_diff=Gs_diff,
                                        save_pics=save_pics,
                                        adaptive_func=adaptive_func
            )
            if Sx == 1:
                all_values[service] = VALUE
            else:
                all_values[(service, i)] = VALUE
        
    # save values and shapely values
    pf_name = 'stat' if STATIC_PF else 'dyn'
    df = pd.DataFrame.from_dict(all_values, orient='index')
    df.to_csv(f'{save_path}/values_{pf_name}.csv', float_format='%.5f')
